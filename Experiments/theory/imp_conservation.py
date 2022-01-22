import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Layers import layers
from Utils import load
from Utils import generator
from train import *
from prune import *
import sam.sam as sam

# Experiment Hyperparameters
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
    train_loader, train_sampler = load.dataloader(args.dataset, args.train_batch_size, True, args.workers,
                                                  args.distributed)

    ## Compute Layer Name and Inv Size ##
    def layer_names(model):
        names = []
        inv_size = []
        for name, module in model.named_modules():
            if isinstance(module, (layers.Linear, layers.Conv2d)):
                num_elements = np.prod(module.weight.shape)
                if module.bias is not None:
                    num_elements += np.prod(module.bias.shape)
                names.append(name)
                inv_size.append(1.0/num_elements)
        return names, inv_size

    ## Compute Average Mag Score ##
    def average_mag_score(model):
        average_scores = []
        for module in model.modules():
            if isinstance(module, (layers.Linear, layers.Conv2d)):
                W = module.weight.detach().cpu().numpy()
                W_score = W**2
                score_sum = W_score.sum()
                num_elements = np.prod(W.shape)

                if module.bias is not None:
                    b = module.bias.detach().cpu().numpy()
                    b_score = b**2
                    score_sum += b_score.sum()
                    num_elements += np.prod(b.shape)

                average_scores.append(np.abs(score_sum / num_elements))
        return average_scores

    ## Train and Save Data ##
    _, inv_size = layer_names(model)
    Wscore = []
    for epoch in tqdm(range(args.post_epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        Wscore.append(average_mag_score(model))
        train(model, loss, optimizer, train_loader, args.gpu, epoch, args.verbose)
        scheduler.step()

    np.save('{}/{}'.format(args.result_dir,'inv-size'), inv_size)
    np.save('{}/score'.format(args.result_dir), np.array(Wscore))
