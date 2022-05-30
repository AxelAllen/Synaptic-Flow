import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from Utils import load
from train import *
from prune import *
import wandb

import sam.sam as sam

def run(args):
    if not args.save:
        print("This experiment requires an expid.")
        quit()

    if args.wandb:
        wandb.login()
        wandb.init(
            project="synflow",
            name=f"{args.expid}",
            group=f"{args.groupid}",
            config=vars(args)
            )

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset, args.image_size)
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.image_size, args.prune_dataset_ratio * num_classes)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers, args.image_size, args.dataset_length)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers, args.image_size, args.dataset_length)

    ## Model ##
    print('Creating {} model.'.format(args.model))
    if args.model_class == 'transformer':
        model = load.model(args.model, args.model_class).load_model(args.model,
                                                                    input_shape,
                                                                    args.patch_size,
                                                                    num_classes,
                                                                    args.pretrained,
                                                                    args.weights_path).to(device)
        '''
        if args.freeze_parameters:
            model.freeze_parameters(freeze_classifier=args.freeze_classifier)
            model.count_parameters()
        '''
    else:
        model = load.model(args.model, args.model_class)(input_shape,
                                                         num_classes,
                                                         args.pretrained).to(device)
    loss = nn.CrossEntropyLoss().to(device)
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    if args.sam:
        opt_kwargs.update({'lr': args.lr, 'weight_decay': args.weight_decay})
        optimizer = sam.SAM(generator.trainable_parameters(model, args.freeze_parameters, args.freeze_classifier), opt_class, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer, milestones=args.lr_drops,
                                                         gamma=args.lr_drop_rate)
    else:
        optimizer = opt_class(generator.trainable_parameters(model, args.freeze_parameters, args.freeze_classifier), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)



    ## Save Original ##
    torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

    ## Train-Prune Loop ##
    generator.count_trainable_parameters(model, args.freeze_parameters, args.freeze_classifier)
    generator.count_prunable_parameters(model)

    for compression in args.compression_list:
        for level in args.level_list:
            print('{} compression ratio, {} train-prune levels'.format(compression, level))

            '''
            # Reset Model, Optimizer, and Scheduler
            model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=device))
            optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
            scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))
            '''

            for l in range(level):

                # Pre Train Model
                pre_result = pre_train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                test_loader, device, args.pre_epochs, args.verbose, use_wandb=args.wandb)
                if args.wandb:
                    pre_result_logs = wandb.Table(dataframe=pre_result)
                    wandb.log({"pre_result": pre_result_logs})

                # Prune Model
                sparsity = 10 ** (-float(compression))
                print('Pruning for {} epochs.'.format(args.prune_epochs))
                prune_result = prune_loop(model, args.pruner, prune_loader, loss, device, sparsity,
                                          args.compression_schedule, args.mask_scope, args.prune_epochs,
                                          args.reinitialize, args.prune_train_mode, args.shuffle, args.invert, use_wandb=args.wandb)
                if args.wandb:
                    prune_result_logs = wandb.Table(dataframe=prune_result)
                    wandb.log({"prune_result": prune_result_logs})

                '''
                # Reset Model's Weights
                original_dict = torch.load("{}/model.pt".format(args.result_dir), map_location=device)
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
                scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))
                '''
            # Train Model
            post_result = post_train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                          test_loader, device, args.post_epochs, args.verbose, use_wandb=args.wandb)
            if args.wandb:
                post_result_logs = wandb.Table(dataframe=post_result)
                wandb.log({"post_result": post_result_logs})
            
            ## Display Results ##
            frames = [post_result.head(1), post_result.tail(1)]
            train_result = pd.concat(frames, keys=['Post-Prune', 'Final'])
            print("Train results:\n", train_result)
            if args.prune_epochs > 0:
                pruner = load.pruner(args.pruner)(sparsity)
                importance_scores = pruner.score(model, prune_loader, loss, device, args.prune_bias)
                prune_result = metrics.summary(model, importance_scores)
                print("Prune results:\n", prune_result)
            glob_sparsity = metrics.global_sparsity(model, args.prune_bias)
            print(f"Parameter Sparsity: {round(100 * glob_sparsity, 2)}%")

            '''
            total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
            possible_params = prune_result['size'].sum()
            total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
            possible_flops = prune_result['flops'].sum()          
            print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
            print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))
            '''
            
            # Save Data
            post_result.to_pickle("{}/post-train-{}-{}-{}.pkl".format(args.result_dir, args.pruner, str(compression),  str(level)))
            prune_result.to_pickle("{}/compression-{}-{}-{}.pkl".format(args.result_dir, args.pruner, str(compression), str(level)))

            # Reset Model's Weights
            original_dict = torch.load("{}/model.pt".format(args.result_dir), map_location=device)
            original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
            model_dict = model.state_dict()
            model_dict.update(original_weights)
            model.load_state_dict(model_dict)

            # Reset Optimizer and Scheduler
            optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
            scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))

    if args.wandb:
        wandb.finish()
