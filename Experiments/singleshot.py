import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils.metrics import global_sparsity
from train import *
from prune import *
import sam.sam as sam
import wandb

# from pthflops import count_ops

def run(args):
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

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    if args.model_class == 'transformer':
        model = load.model(args.model, args.model_class).load_model(args.model,
                                                                    input_shape,
                                                                    args.patch_size,
                                                                    num_classes,
                                                                    args.pretrained,
                                                                    device,
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
        print(f"Sharpness Aware Minimization disabled. Using base optimizer <{args.optimizer}>")
        optimizer = opt_class(generator.trainable_parameters(model, args.freeze_parameters, args.freeze_classifier), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)


    ## Pre-Train ##
    generator.count_trainable_parameters(model, args.freeze_parameters, args.freeze_classifier)
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    pre_result = pre_train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                 test_loader, device, args.pre_epochs, args.verbose, use_wandb=args.wandb)
    if args.wandb:
        pre_result_logs = wandb.Table(dataframe=pre_result)
        wandb.log({"pre_result": pre_result_logs})

    ## Prune ##
    generator.count_prunable_parameters(model)
    print('Pruning for {} epochs.'.format(args.prune_epochs))
    sparsity = 10 ** (-float(args.compression))
    prune_result = prune_loop(model, args.pruner, prune_loader, loss, device, sparsity,
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert, use_wandb=args.wandb)
    if args.wandb:
        prune_result_logs = wandb.Table(dataframe=prune_result)
        wandb.log({"prune_result": prune_result_logs})


    generator.initialize_weights(model, "classifier")
    ## Re-define optimizer to update whole model ##
    '''
    if args.sam:
        opt_kwargs.update({'lr': args.lr, 'weight_decay': args.weight_decay})
        optimizer = sam.SAM(generator.prunable_parameters(model), opt_class, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer, milestones=args.lr_drops,
                                                         gamma=args.lr_drop_rate)
    else:
        print(f"Sharpness Aware Minimization disabled. Using base optimizer <{args.optimizer}>")
        optimizer = opt_class(generator.prunable_parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
    '''

    ## Post-Train ##
    generator.count_trainable_parameters(model, args.freeze_parameters, args.freeze_classifier)
    print('Post-Training for {} epochs.'.format(args.post_epochs))
    post_result = post_train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                  test_loader, device, args.post_epochs, args.verbose, use_wandb=args.wandb)
    if args.wandb:
        post_result_logs = wandb.Table(dataframe=post_result)
        wandb.log({"post_result": post_result_logs})

    ## Count Flops ##
    # (data, _) = next(iter(train_loader))
    # ops, all_data = count_ops(model, data)

    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])

    # total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    # possible_params = prune_result['size'].sum()
    # total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    # possible_flops = prune_result['flops'].sum()
    glob_sparsity = global_sparsity(model, args.prune_bias)
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print(f"Parameter Sparsity: {round(100 * glob_sparsity, 2)}%")
    # print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    # print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))
    if args.wandb:
        wandb.finish()


