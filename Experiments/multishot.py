import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *
import sam.sam as sam

def run(args, ngpus_per_node):
    if not args.save:
        print("This experiment requires an expid.")
        quit()

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    input_shape, num_classes = load.dimension(args.dataset)
    if args.model_class == 'transformer':
        model = load.model(args.model, args.model_class).load_model(args.model,
                                                                    input_shape,
                                                                    num_classes,
                                                                    args.pretrained)
    else:
        model = load.model(args.model, args.model_class)(input_shape,
                                                         num_classes,
                                                         args.dense_classifier,
                                                         args.pretrained)

    use_cuda = torch.cuda.is_available()
    if args.distributed and use_cuda:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.train_batch_size = int(args.batch_size / ngpus_per_node)
            args.test_batch_size = int(args.batch_size / ngpus_per_node)
            args.prune_batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and use_cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    if args.sam:
        opt_kwargs.update({'lr': args.lr, 'weight_decay': args.weight_decay})
        optimizer = sam.SAM(generator.parameters(model), opt_class, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer, milestones=args.lr_drops,
                                                         gamma=args.lr_drop_rate)
    else:
        print(f"Sharpness Aware Minimization disabled. Using base optimizer {args.optimizer}")
        optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops,
                                                         gamma=args.lr_drop_rate)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    prune_loader, prune_sampler = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers,
                                                  args.distributed,
                                                  args.prune_dataset_ratio * num_classes)
    train_loader, train_sampler = load.dataloader(args.dataset, args.train_batch_size, True, args.workers,
                                                  args.distributed)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers, args.distributed)

    ## Save Original ##
    torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

    ## Train-Prune Loop ##
    for compression in args.compression_list:
        for level in args.level_list:
            print('{} compression ratio, {} train-prune levels'.format(compression, level))
            
            # Reset Model, Optimizer, and Scheduler
            model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=torch.device("cuda")))
            optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=torch.device("cuda")))
            scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=torch.device("cuda")))
            
            for l in range(level):

                # Pre Train Model
                train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                test_loader, train_sampler, args.gpu, args.pre_epochs, args.verbose, args.distributed)

                # Prune Model
                pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
                sparsity = (10**(-float(compression)))**((l + 1) / level)
                prune_loop(model, loss, pruner, prune_loader, prune_sampler, args.gpu, sparsity,
                           args.compression_schedule, args.mask_scope, args.prune_epochs, args.distributed,
                           args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)

                # Reset Model's Weights
                original_dict = torch.load("{}/model.pt".format(args.result_dir), map_location=torch.device("cuda"))
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=torch.device("cuda")))
                scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=torch.device("cuda")))

            # Prune Result
            prune_result = metrics.summary(model, 
                                           pruner.scores,
                                           lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual)) # metrics.flop(model, input_shape, device),
            # Train Model
            post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                test_loader, train_sampler, args.gpu, args.pre_epochs, args.verbose, args.distributed)
            
            ## Display Results ##
            frames = [post_result.head(1), post_result.tail(1)]
            train_result = pd.concat(frames, keys=['Post-Prune', 'Final'])
            prune_result = metrics.summary(model, 
                                   pruner.scores,
                                   lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual)) # metrics.flop(model, input_shape, device),
            total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
            possible_params = prune_result['size'].sum()
            # total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
            # possible_flops = prune_result['flops'].sum()
            print("Train results:\n", train_result)
            print("Prune results:\n", prune_result)
            print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
            # print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))
            
            # Save Data
            post_result.to_pickle("{}/post-train-{}-{}-{}.pkl".format(args.result_dir, args.pruner, str(compression),  str(level)))
            prune_result.to_pickle("{}/compression-{}-{}-{}.pkl".format(args.result_dir, args.pruner, str(compression), str(level)))


