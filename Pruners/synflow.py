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

    def compute_mask(self, importance_scores, default_mask):
        mask = default_mask.clone()
        k = int((1.0 - self.sparsity) * importance_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(importance_scores, k)
            for _, score in importance_scores.items():
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
        return mask

    def shuffle(self, default_mask):
        shape = default_mask.shape
        mask = default_mask.clone()
        perm = torch.randperm(mask.nelement())
        mask = mask.reshape(-1)[perm].reshape(shape)
        return mask





def score(model, dataloader, device):

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


    for name, module in filter(lambda p: prunable(p), model.named_modules()):
        param = module[name]
        score = torch.clone(param.grad * param).detach().abs_()
        param.grad.data.zero_()
        scores.update({(module, name): score})


    nonlinearize(model, signs)

    return scores


def apply_pruner(pruner, module, name, **kwargs):
    pruner.apply(
        module, name, **kwargs
    )
    return module

def global_unstructured(parameters, pruning_method, importance_scores, pruner, **kwargs):
    r"""
    Globally prunes tensors corresponding to all parameters in ``parameters``
    by applying the specified ``pruning_method``.
    Modifies modules in place by:
    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.
    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``
    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.
    """

    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = prune.PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            "the `pruning_method`. Found method {} of type {}".format(
                pruning_method, method.PRUNING_TYPE
            )
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    masked_parameters = compute_masked_parameters(pruner, parameters)



    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:

        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        prune.custom_from_mask(module, name, mask=param_mask)

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param


