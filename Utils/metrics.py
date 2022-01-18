import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from prune import * 
from Layers import layers
from Models.vit import VisionTransformer, PositionEmbs, MlpBlock, SelfAttention, EncoderBlock, Encoder

_ignore_modules = [PositionEmbs, MlpBlock, SelfAttention, EncoderBlock, Encoder, nn.Dropout]

def summary(model, scores, flops, prunable):
    r"""Summary of compression results for a model.
    """
    rows = []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            pruned = prunable(module) and id(param) in scores.keys()
            if pruned:
                sparsity = getattr(module, pname+'_mask').detach().cpu().numpy().mean()
                score = scores[id(param)].detach().cpu().numpy()
            else:
                sparsity = 1.0
                score = np.zeros(1)
            shape = param.detach().cpu().numpy().shape
            flop = flops[name][pname]
            score_mean = score.mean()
            score_var = score.var()
            score_sum = score.sum()
            score_abs_mean = np.abs(score).mean()
            score_abs_var  = np.abs(score).var()
            score_abs_sum  = np.abs(score).sum()
            rows.append([name, pname, sparsity, np.prod(shape), shape, flop,
                         score_mean, score_var, score_sum, 
                         score_abs_mean, score_abs_var, score_abs_sum, 
                         pruned])

    columns = ['module', 'param', 'sparsity', 'size', 'shape', 'flops', 'score mean', 'score variance', 
               'score sum', 'score abs mean', 'score abs variance', 'score abs sum', 'prunable']
    return pd.DataFrame(rows, columns=columns)

def flop(model, input_shape, device):

    total = {}
    def count_flops(name):
        def hook(module, input, output):
            flops = {}
            if isinstance(module, layers.Linear) or isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                flops['weight'] = in_features * out_features
                if module.bias is not None:
                    flops['bias'] = out_features
            elif isinstance(module, layers.LinearGeneral):
                in_features = module.weight.shape[0][0]
                out_features = module.weight.shape[1][0] * module.weight.shape[1][1]
                flops['weight'] = in_features * out_features
                if module.bias is not None:
                    flops['bias'] = module.bias.shape[0][0] * module.bias.shape[0][1]
            elif isinstance(module, layers.Conv2d) or isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = int(np.prod(module.kernel_size))
                output_size = output.size(2) * output.size(3)
                flops['weight'] = in_channels * out_channels * kernel_size * output_size
                if module.bias is not None:
                    flops['bias'] = out_channels * output_size
            elif isinstance(module, layers.BatchNorm1d) or isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    flops['weight'] = module.num_features
                    flops['bias'] = module.num_features
            elif isinstance(module, layers.BatchNorm2d) or isinstance(module, nn.BatchNorm2d):
                output_size = output.size(2) * output.size(3)
                if module.affine:
                    flops['weight'] = module.num_features * output_size
                    flops['bias'] = module.num_features * output_size
            elif isinstance(module, layers.LayerNorm) or isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    flops['weight'] = module.normalized_shape[0]
                    flops['bias'] = module.normalized_shape[0]
            elif isinstance(module, layers.Identity1d):
                flops['weight'] = module.num_features
            elif isinstance(module, layers.Identity2d):
                output_size = output.size(2) * output.size(3)
                flops['weight'] = module.num_features * output_size
            elif hasattr(model, 'cls_token') and isinstance(module, nn.Parameter):
                in_features = module.shape[2]
                out_features = model.num_classes
                flops['weight'] = in_features * out_features
            elif hasattr(model, 'PositionEmbs') and hasattr(model.PositionEmbs, 'pos_embedding') and isinstance(module, nn.Parameter):
                in_features = module.shape[1]-1
                out_features = module.shape[2]
                flops['weight'] = in_features * out_features
            elif any(isinstance(module, mod) for mod in _ignore_modules):
                flops['weight'] = 0
                if hasattr(module, 'bias') and module.bias is not None:
                    flops['bias'] = 0
            total[name] = flops
        return hook
    
    for name, module in model.named_modules():
        module.register_forward_hook(count_flops(name))

    input = torch.ones([1] + list(input_shape)).to(device)
    model(input)

    return total


# def conservation(model, scores, batchnorm, residual):
#     r"""Summary of conservation results for a model.
#     """
#     rows = []
#     bias_flux = 0.0
#     mu = 0.0
#     for name, module in reversed(list(model.named_modules())):
#         if prunable(module, batchnorm, residual):
#             weight_flux = 0.0
#             for pname, param in module.named_parameters(recurse=False):
                
#                 # Get score
#                 score = scores[id(param)].detach().cpu().numpy()
                
#                 # Adjust batchnorm bias score for mean and variance
#                 if isinstance(module, (layers.Linear, layers.Conv2d)) and pname == "bias":
#                     bias = param.detach().cpu().numpy()
#                     score *= (bias - mu) / bias
#                     mu = 0.0
#                 if isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d)) and pname == "bias":
#                     mu = module.running_mean.detach().cpu().numpy()
                
#                 # Add flux
#                 if pname == "weight":
#                     weight_flux += score.sum()
#                 if pname == "bias":
#                     bias_flux += score.sum()
#             layer_flux = weight_flux
#             if not isinstance(module, (layers.Identity1d, layers.Identity2d)):
#                 layer_flux += bias_flux
#             rows.append([name, layer_flux])
#     columns = ['module', 'score flux']

#     return pd.DataFrame(rows, columns=columns)

