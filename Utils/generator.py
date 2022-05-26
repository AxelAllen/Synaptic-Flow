import torch.nn as nn
from Models.tinyimagenet_resnet import Identity1d, Identity2d
from Models.resnet import StdConv2d
from Models.vit import LinearGeneral
from torch.nn.init import xavier_uniform_



def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (Identity1d, Identity2d))

def prunable(module, batchnorm=False, residual=False, layernorm=False):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (nn.Linear, nn.Conv2d, StdConv2d, LinearGeneral))
    if batchnorm:
        isprunable |= isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (Identity1d, Identity2d))
    if layernorm:
        isprunable |= isinstance(module, nn.LayerNorm)
    return isprunable

def parameters(model):
    for module in model.modules():
        for param in module.parameters(recurse=False):
            yield param

def trainable_parameters(model, freeze_parameters, freeze_classifier):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    if not freeze_parameters and not freeze_classifier:
        for module in filter(lambda p: trainable(p), model.modules()):
            for param in filter(lambda p: p.requires_grad, module.parameters(recurse=False)):
                yield param

    elif freeze_parameters and not freeze_classifier:
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                yield param
        else:
            last_layer = list(model.children())[-1]
            for param in last_layer.parameters():
                yield param

def prunable_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding just the
    parameter tensor."""
    for module in filter(lambda p: prunable(p), model.modules()):
        for param in module.parameters(recurse=False):
            yield param

def count_trainable_parameters(model, freeze_parameters, freeze_classifier):
    trainable_params = sum(p.numel() for p in trainable_parameters(model, freeze_parameters, freeze_classifier))
    total_params = sum(p.numel() for p in parameters(model))
    print(f"Trainable parameters: {trainable_params} / {total_params}")

def count_prunable_parameters(model):
    prunable_params = sum(p.numel() for p in prunable_parameters(model) if p.requires_grad)
    total_params = sum(p.numel() for p in parameters(model))
    print(f"Prunable parameters: {prunable_params} / {total_params}")

'''
def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param
'''

def initialize_weights(model, module=None):
    if module is not None:
        if hasattr(model, module) and module == "classifier":
            for param in model.classifier.parameters(recurse=False):
                xavier_uniform_(param)
    if module is None:
        for module_ in filter(lambda p: prunable(p), model.modules()):
            for param in module_.parameters(recurse=False):
                xavier_uniform_(param)

