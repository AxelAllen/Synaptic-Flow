import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Layers import layers
from Utils import load
from Utils import generator
from train import *
from prune import *

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

    ## Data ##
    data_loader, sampler = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.distributed,
                                     args.prune_dataset_ratio * num_classes)

    torch.save(model.state_dict(), "{}/model.pt".format(args.result_dir))


    ## Compute per Neuron Score ##
    def unit_score_sum(model, scores, compute_linear=False):
        in_scores = []
        out_scores = []
        for name, module in model.named_modules():
            if (isinstance(module, layers.Linear) and compute_linear):
                W = module.weight
                b = module.bias

                W_score = scores[id(W)].detach().cpu().numpy()
                b_score = scores[id(b)].detach().cpu().numpy()

                in_scores.append(W_score.sum(axis=1) + b_score)
                out_scores.append(W_score.sum(axis=0))
            if isinstance(module, layers.Conv2d):
                W = module.weight
                W_score = scores[id(W)].detach().cpu().numpy()
                in_score = W_score.sum(axis=(1,2,3)) 
                out_score = W_score.sum(axis=(0,2,3))

                if module.bias is not None:
                    b = module.bias
                    b_score = scores[id(b)].detach().cpu().numpy()
                    in_score += b_score
                
                in_scores.append(in_score)
                out_scores.append(out_score)

        in_scores = np.concatenate(in_scores[:-1])
        out_scores = np.concatenate(out_scores[1:])
        return in_scores, out_scores

    ## Loop through Pruners and Save Data ##
    unit_scores = []
    pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    sparsity = 10**(-float(args.compression))
    prune_loop(model, loss, pruner, data_loader, sampler, args.gpu, sparsity,
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.distributed, args.reinitialize, args.prune_train_mode)
    unit_score = unit_score_sum(model, pruner.scores)
    unit_scores.append(unit_score)
    np.save('{}/{}'.format(args.result_dir, args.pruner), unit_score)
