import torch.nn as nn
from Models.tinyimagenet_resnet import Identity1d, Identity2d
from Models.resnet import StdConv2d
from Models.vit import LinearGeneral


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

def prunable(module, batchnorm=False, residual=False):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (nn.Linear, nn.Conv2d, StdConv2d, LinearGeneral))
    if batchnorm:
        isprunable |= isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (Identity1d, Identity2d))
    return isprunable

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in filter(lambda p: trainable(p), model.modules()):
        for param in filter(lambda p: p.requires_grad, module.parameters(recurse=False)):
            yield param

def prunable_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding just the
    parameter tensor."""
    for module in filter(lambda p: prunable(p), model.modules()):
        for param in filter(lambda p: p.requires_grad, module.parameters(recurse=False)):
            yield param
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