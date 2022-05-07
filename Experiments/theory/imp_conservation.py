import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from train import *
from prune import *
from Models.vit import StdConv2d, LinearGeneral
from Models.resnet import StdConv2d as StdConv2d_
import sam.sam as sam

# Experiment Hyperparameters
def run(args):
    if not args.save:
        print("This experiment requires an expid.")
        quit()

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset, args.image_size)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers, args.image_size, args.dataset_length)

    ## Model, Loss, Optimizer ##
    if args.model_class == 'transformer':
        model = load.model(args.model, args.model_class).load_model(args.model,
                                                                    input_shape,
                                                                    args.patch_size,
                                                                    num_classes,
                                                                    args.pretrained,
                                                                    args.weights_path).to(device)

    else:
        model = load.model(args.model, args.model_class)(input_shape,
                                                         num_classes,
                                                         args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    if args.sam:
        opt_kwargs.update({'lr': args.lr, 'weight_decay': args.weight_decay})
        optimizer = sam.SAM(generator.trainable_parameters(model, args.freeze_parameters, args.freeze_classifier),
                            opt_class, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer, milestones=args.lr_drops,
                                                         gamma=args.lr_drop_rate)
    else:
        print(f"Sharpness Aware Minimization disabled. Using base optimizer <{args.optimizer}>")
        optimizer = opt_class(generator.trainable_parameters(model, args.freeze_parameters, args.freeze_classifier),
                              lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Compute Layer Name and Inv Size ##
    def layer_names(model):
        names = []
        inv_size = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, StdConv2d, LinearGeneral, StdConv2d_)):
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
            if isinstance(module, (nn.Linear, nn.Conv2d, StdConv2d, LinearGeneral, StdConv2d_)):
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
        Wscore.append(average_mag_score(model))
        train(model, loss, optimizer, train_loader, device, epoch, args.verbose)
        scheduler.step()

    np.save('{}/{}'.format(args.result_dir,'inv-size'), inv_size)
    np.save('{}/score'.format(args.result_dir), np.array(Wscore))
