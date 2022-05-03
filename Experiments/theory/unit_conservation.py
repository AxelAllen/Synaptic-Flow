import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from train import *
from prune import *

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


    ## Compute per Neuron Score ##
    def unit_score_sum(model, scores, compute_linear=False):
        in_scores = []
        out_scores = []
        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) and compute_linear):
                W = module.weight
                b = module.bias

                W_score = scores[id(W)].detach().cpu().numpy()
                b_score = scores[id(b)].detach().cpu().numpy()

                in_scores.append(W_score.sum(axis=1) + b_score)
                out_scores.append(W_score.sum(axis=0))
            if isinstance(module, nn.Conv2d):
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
    unit_scores = []
    sparsity = 10**(-float(args.compression))
    scores = score(model. data_loader, device, args.prune_bias)
    unit_score = unit_score_sum(model, scores)
    unit_scores.append(unit_score)
    np.save('{}/{}'.format(args.result_dir, args.pruner), unit_score)
