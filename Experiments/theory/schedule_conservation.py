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
    torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))


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
        input_dim = list(data[0,:].shape)
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

        return scores, maxflow.item()

    def mask(parameters, scores, sparsity):
        mask = torch.nn.utils.parameters_to_vector(
            [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
            ]
        )
        importance_scores = torch.nn.utils.parameters_to_vector(
            [
                scores.get((module, name), getattr(module, name))
                for (module, name) in parameters
            ]
        )
        zero = torch.tensor([0.]).to(mask.device)
        one = torch.tensor([1.]).to(mask.device)
        k = int((1.0 - sparsity) * importance_scores.numel())
        cutsize = 0
        if not k < 1:
            cutsize = torch.sum(torch.topk(importance_scores, k, largest=False).values).item()
            threshold, _ = torch.kthvalue(importance_scores, k)
            mask.copy_(torch.where(importance_scores <= threshold, zero, one))
        return cutsize, mask

    @torch.no_grad()
    def apply_mask(parameters, mask):
        pointer = 0
        for module, name in parameters:
            param = getattr(module, name)
            # The length of the parameter
            num_param = param.numel()
            # Slice the mask, reshape it
            param_mask = mask[pointer: pointer + num_param].view_as(param)
            param.mul_(param_mask)
            pointer += num_param

    results = []
    for style in ['linear', 'exponential']:
        print(style)
        sparsity_ratios = []
        for i, exp in enumerate(args.compression_list):
            max_ratios = []
            for j, epochs in enumerate(args.prune_epoch_list):
                model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=device))
                parameters = list(generator.prunable_parameters(model))
                model.eval()
                ratios = []
                for epoch in tqdm(range(epochs)):
                    scores, maxflow = score(model, data_loader, device)
                    sparsity = 10**(-float(exp))
                    if style == 'linear':
                        sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
                    if style == 'exponential':
                        sparse = sparsity**((epoch + 1) / epochs)
                    cutsize, mask = mask(parameters, scores, sparse)
                    # apply_mask(parameters, mask)
                    ratios.append(cutsize / maxflow)
                max_ratios.append(max(ratios))
            sparsity_ratios.append(max_ratios)
        results.append(sparsity_ratios)
    np.save('{}/ratios'.format(args.result_dir), np.array(results))
