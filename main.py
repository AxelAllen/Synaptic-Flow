import argparse
import json
import os
from Experiments import singleshot
from Experiments import multishot

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Network Compression')
    # Training Hyperparameters
    general_args = parser.add_argument_group('general')
    training_args = parser.add_argument_group('training')
    model_args = parser.add_argument_group('model')
    pruning_args = parser.add_argument_group('pruning')

    ## Model Hyperparameters ##
    model_args.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','tiny-imagenet','imagenet'],
                        help='dataset (default: mnist)')
    model_args.add_argument('--model', type=str, default='ViT-B_16', choices=['fc','conv',
                        'vgg11','vgg11-bn','vgg13','vgg13-bn','vgg16','vgg16-bn','vgg19','vgg19-bn',
                        'resnet18','resnet20','resnet32','resnet34','resnet44','resnet50',
                        'resnet56','resnet101','resnet110','resnet110','resnet152','resnet1202',
                        'wide-resnet18','wide-resnet20','wide-resnet32','wide-resnet34','wide-resnet44','wide-resnet50',
                        'wide-resnet56','wide-resnet101','wide-resnet110','wide-resnet110','wide-resnet152',
                        'wide-resnet1202', 'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'R50+ViT-B_16'],
                        help='model architecture (default: ViT-B_16)')
    model_args.add_argument('--model-class', type=str, default='transformer', choices=['default','lottery','tinyimagenet','imagenet', 'transformer'],
                        help='model class (default: transformer)')
    model_args.add_argument('--dense-classifier', type=bool, default=False,
                        help='ensure last layer of model is dense (default: False)')
    model_args.add_argument('--pretrained', type=bool, default=False,
                        help='load pretrained weights (default: False)')
    model_args.add_argument('--optimizer', type=str, default='adam', choices=['sgd','momentum','adam','rms'],
                        help='optimizer (default: adam)')
    model_args.add_argument('--sam', action='store_true',
                               help='Whether to use sharpness aware optimization')
    model_args.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    model_args.add_argument('--pre-epochs', type=int, default=1,
                        help='number of epochs to train before pruning (default: 0)')
    model_args.add_argument('--post-epochs', type=int, default=10,
                        help='number of epochs to train after pruning (default: 10)')
    model_args.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    model_args.add_argument('--lr-drops', type=int, nargs='*', default=[],
                        help='list of learning rate drops (default: [])')
    model_args.add_argument('--lr-drop-rate', type=float, default=0.1,
                        help='multiplicative factor of learning rate drop (default: 0.1)')
    model_args.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    model_args.add_argument('--prune-dataset-ratio', type=int, default=10,
                              help='ratio of prune dataset size and number of classes (default: 10)')
    model_args.add_argument('--prune-batch-size', type=int, default=256,
                              help='input batch size for pruning (default: 256)')
    model_args.add_argument('--patch-size', type=int, default=4,
                            help='patch size (x, x)')
    model_args.add_argument('--emb-dim', type=int, default=768,
                            help='size of the embedding dimension')
    model_args.add_argument('--mlp-dim', type=int, default=3072,
                            help='size of the hidden MLP layer')
    model_args.add_argument('--num-heads', type=int, default=12,
                            help='number of attention heads per self-attention layer')
    model_args.add_argument('--num-layers', type=int, default=12,
                            help='number of self-attention layers')
    model_args.add_argument('--num_workers', type=int, default=4,
                            help='number of workers for a dataloader')
    ## Pruning Hyperparameters ##
    pruning_args.add_argument('--pruner', type=str, default='synflow',
                        choices=['rand','mag','snip','grasp','synflow'],
                        help='prune strategy (default: synflow)')
    pruning_args.add_argument('--compression', type=float, default=1.0,
                        help='quotient of prunable non-zero prunable parameters before and after pruning (default: 1.0)')
    pruning_args.add_argument('--prune-epochs', type=int, default=1,
                        help='number of iterations for scoring (default: 1)')
    pruning_args.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear','exponential'],
                        help='whether to use a linear or exponential compression schedule (default: exponential)')
    pruning_args.add_argument('--mask-scope', type=str, default='global', choices=['global','local'],
                        help='masking scope (global or layer) (default: global)')
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
    ## Trainer Hyperparameters ##
    training_args.add_argument('--accelerator', type=str, default='auto', choices=['cpu', 'gpu', 'tpu', 'ipu', 'auto'],
                               help='type of accelerator to use')
    training_args.add_argument('--amp-backend', type=str, default='native', choices=['native', 'apex'])
    training_args.add_argument('--auto-lr-find', type=bool, default=False,
                               help='automatic learning rate tuner')
    training_args.add_argument('--auto-scale-batch-size', type=bool, default=False,
                               help='automatic scaling of batch size')
    training_args.add_argument('--auto-select-gpus', type=bool, default=False,
                               help='automatically select the gpus to use')
    training_args.add_argument('--benchmark', type=bool, default=True,
                               help='if true enables cudnn.benchmark')
    training_args.add_argument('--enable-checkpointing', type=bool, default=False,
                               help='enables saving checkpoints')
    training_args.add_argument('--default-root-dir', type=str, default='Results/data',
                               help='default path for logs and weights')
    training_args.add_argument('--deterministic', type=bool, default=False,
                               help='PyTorch operations must use deterministic algorithms')
    training_args.add_argument('--fast-dev-run', type=bool, default=False,
                               help='for debugging')
    training_args.add_argument('--devices', type=str, default='auto', choices=["cpu", "gpu", "tpu", "ipu", "auto"],
                               help='what device to use. Alternatively can be an int defining number of devices to use')
    training_args.add_argument('--strategy', type=str, default='ddp', choices=['ddp', 'ddp_spawn'])
    training_args.add_argument('--resume-from-checkpoint', type=str, default=None,
                               help='path to a checkpoint from which to resume training')
    ## Experiment Hyperparameters ##
    general_args.add_argument('--experiment', type=str, default='singleshot',
                        choices=['singleshot','multishot'],
                        help='experiment name (default: singleshot)')
    general_args.add_argument('--prune', type=bool, default=False,
                              help='whether to prune the network')
    general_args.add_argument('--unit-conservation', type=bool, default=False,
                              help='whether to run unit conservation experiment')
    general_args.add_argument('--layer-conservation', type=bool, default=False,
                              help='whether to run layer conservation experiment')
    general_args.add_argument('--imp-conservation', type=bool, default=False,
                              help='whether to run imp conservation experiment')
    general_args.add_argument('--schedule-conservation', type=bool, default=False,
                              help='whether to run schedule conservation experiment')
    general_args.add_argument('--expid', type=str, default='',
                        help='name used to save results (default: "")')
    general_args.add_argument('--result-dir', type=str, default='Results/data',
                        help='path to directory to save results (default: "Results/vit/data")')
    general_args.add_argument('--save-checkpoint', type=bool, default=False,
                              help='whether to save checkpoints')
    general_args.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    general_args.add_argument('--verbose', action='store_true',
                        help='print statistics during training and testing')
    general_args.add_argument('--compression-list', type=float, nargs='*', default=[],
                              help='list of compression ratio exponents for singleshot/multishot (default: [])')
    general_args.add_argument('--level-list', type=int, nargs='*', default=[],
                              help='list of number of prune-train cycles (levels) for multishot (default: [])')
    general_args.add_argument('--log', type=bool, default=False,
                              help='whether to log the experiments')
    all_args = parser.parse_args()

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(all_args, a.dest, None) for a in group._group_actions}
        if group.title == 'training':
            train_args = vars(argparse.Namespace(**group_dict))
        if group.title == 'model':
            model_args = vars(argparse.Namespace(**group_dict))
        if group.title == 'pruning':
            pruning_args = argparse.Namespace(**group_dict)
        if group.title == 'general':
            general_args = argparse.Namespace(**group_dict)


    ## Construct Result Directory ##
    if general_args.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(general_args, 'save', False)
    else:
        result_dir = '{}/{}/{}'.format(general_args.result_dir, general_args.experiment, general_args.expid)
        setattr(general_args, 'save', True)
        setattr(general_args, 'result_dir', result_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            val = ""
            while val not in ['yes', 'no']:
                val = input("Experiment '{}' with expid '{}' exists.  Overwrite (yes/no)? ".format(general_args.experiment, general_args.expid))
            if val == 'no':
                quit()

    ## Save Args ##
    if general_args.save:
        with open(general_args.result_dir + '/args.json', 'w') as f:
            json.dump(general_args.__dict__, f, sort_keys=True, indent=4)

    args = {'general': general_args, 'training': train_args, 'model': model_args, 'pruning': pruning_args}

    ## Run Experiment ##
    if general_args.experiment == 'singleshot':
        singleshot.run(args)
    if general_args.experiment == 'multishot':
        multishot.run(args)

