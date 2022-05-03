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
    data_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.image_size,
                                   args.prune_dataset_ratio * num_classes)

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
                                                         args.dense_classifier,
                                                         args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()

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

    ## Compute Average Layer Score ##
    def average_layer_score(model, scores):
        average_scores = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, StdConv2d, LinearGeneral, StdConv2d_)):
                W = module.weight
                W_score = scores[id(W)].detach().cpu().numpy()
                score_sum = W_score.sum()
                num_elements = np.prod(W.shape)

                if module.bias is not None:
                    b = module.bias
                    b_score = scores[id(b)].detach().cpu().numpy()
                    score_sum += b_score.sum()
                    num_elements += np.prod(b.shape)

                average_scores.append(np.abs(score_sum / num_elements))
        return average_scores

    def score(model, dataloader, device, prune_bias=False):
        @torch.no_grad()
        def linearize(model):
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)
        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(device)
        output = model(input)
        maxflow = torch.sum(output)
        maxflow.backward()
        scores = {}

        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "bias" and prune_bias is False:
                    continue
                score = torch.clone(param.grad * param).detach().abs_()
                param.grad.data.zero_()
                scores.update({(module, pname): score})
        nonlinearize(model, signs)

        return scores


    ## Loop through Pruners and Save Data ##
    names, inv_size = layer_names(model)
    average_scores = []
    sparsity = 10**(-float(args.compression))
    scores = score(model, data_loader, device, args.prune_bias)
    average_score = average_layer_score(model, scores)
    average_scores.append(average_score)
    np.save('{}/{}'.format(args.result_dir, args.pruner), np.array(average_score))
    np.save('{}/{}'.format(args.result_dir,'inv-size'), inv_size)
