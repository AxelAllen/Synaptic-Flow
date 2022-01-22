import argparse
import json
import os
import random
import warnings

from Experiments import singleshot
from Experiments import multishot
from Experiments.theory import unit_conservation
from Experiments.theory import layer_conservation
from Experiments.theory import imp_conservation
from Experiments.theory import schedule_conservation

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

parser = argparse.ArgumentParser(description='Network Compression')
# Training Hyperparameters
training_args = parser.add_argument_group('training')
training_args.add_argument('--dataset', type=str, default='cifar100',
                    choices=['mnist','cifar10','cifar100','tiny-imagenet','imagenet'],
                    help='dataset (default: mnist)')
training_args.add_argument('--model', type=str, default='ViT-B_16', choices=['fc','conv',
                    'vgg11','vgg11-bn','vgg13','vgg13-bn','vgg16','vgg16-bn','vgg19','vgg19-bn',
                    'resnet18','resnet20','resnet32','resnet34','resnet44','resnet50',
                    'resnet56','resnet101','resnet110','resnet110','resnet152','resnet1202',
                    'wide-resnet18','wide-resnet20','wide-resnet32','wide-resnet34','wide-resnet44','wide-resnet50',
                    'wide-resnet56','wide-resnet101','wide-resnet110','wide-resnet110','wide-resnet152',
                    'wide-resnet1202', 'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'R50+ViT-B_16'],
                    help='model architecture (default: fc)')
training_args.add_argument('--model-class', type=str, default='transformer', choices=['default','lottery','tinyimagenet','imagenet', 'transformer'],
                    help='model class (default: default)')
training_args.add_argument('--dense-classifier', type=bool, default=False,
                    help='ensure last layer of model is dense (default: False)')
training_args.add_argument('--pretrained', type=bool, default=False,
                    help='load pretrained weights (default: False)')
training_args.add_argument('--optimizer', type=str, default='adam', choices=['sgd','momentum','adam','rms'],
                    help='optimizer (default: adam)')
training_args.add_argument('--sam', action='store_true',
                           help='Whether to use sharpness aware optimization')
training_args.add_argument('--train-batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
training_args.add_argument('--test-batch-size', type=int, default=256,
                    help='input batch size for testing (default: 256)')
training_args.add_argument('--pre-epochs', type=int, default=1,
                    help='number of epochs to train before pruning (default: 0)')
training_args.add_argument('--post-epochs', type=int, default=10,
                    help='number of epochs to train after pruning (default: 10)')
training_args.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
training_args.add_argument('--lr-drops', type=int, nargs='*', default=[],
                    help='list of learning rate drops (default: [])')
training_args.add_argument('--lr-drop-rate', type=float, default=0.1,
                    help='multiplicative factor of learning rate drop (default: 0.1)')
training_args.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
# Pruning Hyperparameters
pruning_args = parser.add_argument_group('pruning')
pruning_args.add_argument('--pruner', type=str, default='synflow',
                    choices=['rand','mag','snip','grasp','synflow'],
                    help='prune strategy (default: rand)')
pruning_args.add_argument('--compression', type=float, default=1.0,
                    help='quotient of prunable non-zero prunable parameters before and after pruning (default: 1.0)')
pruning_args.add_argument('--prune-epochs', type=int, default=1,
                    help='number of iterations for scoring (default: 1)')
pruning_args.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear','exponential'],
                    help='whether to use a linear or exponential compression schedule (default: exponential)')
pruning_args.add_argument('--mask-scope', type=str, default='global', choices=['global','local'],
                    help='masking scope (global or layer) (default: global)')
pruning_args.add_argument('--prune-dataset-ratio', type=int, default=10,
                    help='ratio of prune dataset size and number of classes (default: 10)')
pruning_args.add_argument('--prune-batch-size', type=int, default=256,
                    help='input batch size for pruning (default: 256)')
pruning_args.add_argument('--prune-bias', type=bool, default=False,
                    help='whether to prune bias parameters (default: False)')
pruning_args.add_argument('--prune-batchnorm', type=bool, default=False,
                    help='whether to prune batchnorm layers (default: False)')
pruning_args.add_argument('--prune-residual', type=bool, default=False,
                    help='whether to prune residual connections (default: False)')
pruning_args.add_argument('--prune-train-mode', type=bool, default=False,
                    help='whether to prune in train mode (default: False)')
pruning_args.add_argument('--reinitialize', type=bool, default=False,
                    help='whether to reinitialize weight parameters after pruning (default: False)')
pruning_args.add_argument('--shuffle', type=bool, default=False,
                    help='whether to shuffle masks after pruning (default: False)')
pruning_args.add_argument('--invert', type=bool, default=False,
                    help='whether to invert scores during pruning (default: False)')
pruning_args.add_argument('--pruner-list', type=str, nargs='*', default=[],
                    help='list of pruning strategies for singleshot (default: [])')
pruning_args.add_argument('--prune-epoch-list', type=int, nargs='*', default=[],
                    help='list of prune epochs for singleshot (default: [])')
pruning_args.add_argument('--compression-list', type=float, nargs='*', default=[],
                    help='list of compression ratio exponents for singleshot/multishot (default: [])')
pruning_args.add_argument('--level-list', type=int, nargs='*', default=[],
                    help='list of number of prune-train cycles (levels) for multishot (default: [])')
## Experiment Hyperparameters ##
parser.add_argument('--experiment', type=str, default='singleshot',
                    choices=['singleshot','multishot','unit-conservation',
                    'layer-conservation','imp-conservation','schedule-conservation'],
                    help='experiment name (default: example)')
parser.add_argument('--expid', type=str, default='',
                    help='name used to save results (default: "")')
parser.add_argument('--result-dir', type=str, default='Results/vit/data',
                    help='path to directory to save results (default: "Results/vit/data")')
parser.add_argument('--gpu', type=int, default=None,
                    help='number of GPU device to use (default: None)')
parser.add_argument('--workers', type=int, default='4',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true',
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')
parser.add_argument('--verbose', action='store_true',
                    help='print statistics during training and testing')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
args = parser.parse_args()

def main():
    ## Construct Result Directory ##
    if args.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(args, 'save', False)
    else:
        result_dir = '{}/{}/{}'.format(args.result_dir, args.experiment, args.expid)
        setattr(args, 'save', True)
        setattr(args, 'result_dir', result_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            val = ""
            while val not in ['yes', 'no']:
                val = input("Experiment '{}' with expid '{}' exists.  Overwrite (yes/no)? ".format(args.experiment, args.expid))
            if val == 'no':
                quit()

    ## Save Args ##
    if args.save:
        with open(args.result_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    ## For more efficient memory usage ##
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    ## Run Experiment ##
    if args.experiment == 'singleshot':
        singleshot.run(args, ngpus_per_node)
    if args.experiment == 'multishot':
        multishot.run(args, ngpus_per_node)
    if args.experiment == 'unit-conservation':
        unit_conservation.run(args, ngpus_per_node)
    if args.experiment == 'layer-conservation':
        layer_conservation.run(args, ngpus_per_node)
    if args.experiment == 'imp-conservation':
        imp_conservation.run(args, ngpus_per_node)
    if args.experiment == 'schedule-conservation':
        schedule_conservation.run(args, ngpus_per_node)

if __name__ == '__main__':
    main()
