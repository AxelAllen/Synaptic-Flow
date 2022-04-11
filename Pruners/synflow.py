import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple


import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from Utils.generator import parameters, prunable

class SynFlow(prune.BasePruningMethod):

    '''
    options:
    global, structured, unstructured
    '''
    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        super(SynFlow, self).__init__()
        assert 1 >= amount >= 0
        self.sparsity = amount

    '''
    def compute_mask(self, importance_scores, default_mask):
        mask = default_mask.clone()
        k = int((1.0 - self.sparsity) * importance_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(importance_scores, k)
            for score in importance_scores:
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
        return mask
    '''

    def compute_mask(self, importance_scores, default_mask):
        mask = default_mask.clone()
        zero = torch.tensor([0.]).to(mask.device)
        one = torch.tensor([1.]).to(mask.device)
        k = int((1.0 - self.sparsity) * importance_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(importance_scores, k)
            mask.copy_(torch.where(importance_scores <= threshold, zero, one))
        return mask

    def shuffle(self, default_mask):
        shape = default_mask.shape
        mask = default_mask.clone()
        perm = torch.randperm(mask.nelement())
        mask = mask.reshape(-1)[perm].reshape(shape)
        return mask





def score(model, dataloader, device, prune_bias=False):

    scores = {}

    @torch.no_grad()
    def linearize(model):
        # model.double()
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    signs = linearize(model)

    (data, _) = next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(device)  # , dtype=torch.float64).to(device)
    output = model(input)
    torch.sum(output).backward()


    for module in filter(lambda p: prunable(p), model.modules()):
        for pname, param in module.named_parameters(recurse=False):
            if pname == "bias" and prune_bias is False:
                continue
            score = torch.clone(param.grad * param).detach().abs_()
            param.grad.data.zero_()
            scores.update({(module, pname): score})


    nonlinearize(model, signs)

    return scores


def apply_pruner(pruner, module, name, **kwargs):
    pruner.apply(
        module, name, **kwargs
    )
    return module

