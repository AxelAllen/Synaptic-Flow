import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from Utils.generator import prunable
import wandb

def global_sparsity(model, prunable_parameters=None, prune_bias=False):
    r"""Returns remaining and total number of prunable parameters.
    """
    zero_params, total_params = 0, 0
    if prunable_parameters is None:
        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters():
                if pname == "weight" or (pname == "bias" and prune_bias is True):
                    zero_params += float(torch.sum(param == 0.))
                    total_params += float(param.nelement())
    else:
        for param, _ in prunable_parameters:
            zero_params += float(torch.sum(param.weight == 0.))
            total_params += float(param.weight.nelement())
    sparsity = zero_params / total_params
    # remaining_params = total_params - zero_params
    # if np.abs(remaining_params - total_params*sparsity) >= 5:
    #     print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
    #     quit()
    return sparsity

def summary(model, scores):
    r"""Summary of compression results for a model.
    """
    rows = []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            pruned = prunable(module) and (module, pname) in scores.keys()
            if pruned:
                zero_params = float(torch.sum(param == 0))
                total_params = float(param.nelement())
                sparsity = 1 - (zero_params / total_params)
                score = scores[(module, pname)].detach().cpu().numpy()
            else:
                sparsity = 0.0
                score = np.zeros(1)
            shape = param.detach().cpu().numpy().shape
            score_mean = score.mean()
            score_var = score.var()
            score_sum = score.sum()
            score_abs_mean = np.abs(score).mean()
            score_abs_var = np.abs(score).var()
            score_abs_sum = np.abs(score).sum()
            rows.append([name, pname, sparsity, np.prod(shape), shape,
                        score_mean, score_var, score_sum,
                        score_abs_mean, score_abs_var, score_abs_sum,
                        pruned])
    columns = ['module', 'param', 'sparsity', 'size', 'shape', 'score mean', 'score variance',
               'score sum', 'score abs mean', 'score abs variance', 'score abs sum', 'prunable']
    return pd.DataFrame(rows, columns=columns)

