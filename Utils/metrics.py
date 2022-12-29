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
                sparsity = 1.0
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

## Compute per Neuron Score ##
def unit_score_sum(model, scores, concatenate_scores=False, layers=1, prune_bias=False):
    in_scores = []
    out_scores = []
    if hasattr(model, 'bert'):
        params = {}
        for ii in range(layers):
            params.update({f"layer[{ii}].attention.self.query": (model.bert.encoder.layer[ii].attention.self.query, 'weight')})
            params.update({f"layer[{ii}].attention.self.key": (model.bert.encoder.layer[ii].attention.self.key, 'weight')})
            params.update({f"layer[{ii}].attention.self.value": (model.bert.encoder.layer[ii].attention.self.value, 'weight')})
            params.update({f"layer[{ii}].attention.output.dense": (model.bert.encoder.layer[ii].attention.output.dense, 'weight')})
            params.update({f"layer[{ii}].intermediate.dense": (model.bert.encoder.layer[ii].intermediate.dense, 'weight')})
            params.update({f"layer[{ii}].output.dense": (model.bert.encoder.layer[ii].output.dense, 'weight')})

        params.update({"pooler.dense": (model.bert.pooler.dense, 'weight')})

        for module_name, (param, pname) in params.items():
            score = scores[(param, pname)]
            in_scores.append((module_name, score.sum(axis=1).detach().cpu().numpy()))
            out_scores.append((module_name, score.sum(axis=0).detach().cpu().numpy()))

    elif hasattr(model, 'reformer'):
        params = []
        for ii in range(6):

            if ii % 2 == 0:
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.query, "weight"))
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.key, "weight"))
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.value, "weight"))
            else:
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.query_key, "weight"))
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.value, "weight"))

            params.append((model.reformer.encoder.layers[ii].attention.self_attention.value, "weight"))
            params.append((model.reformer.encoder.layers[ii].attention.output.dense, "weight"))
            params.append((model.reformer.encoder.layers[ii].feed_forward.dense.dense, "weight"))
            params.append((model.reformer.encoder.layers[ii].feed_forward.output.dense, "weight"))
        for param, pname in params:
            score = scores[(param, pname)]
            in_scores.append(score.sum(axis=1).detach().cpu().numpy())
            out_scores.append(score.sum(axis=0).detach().cpu().numpy())
    else:
        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "bias" and prune_bias is False:
                    continue
                score = scores[(module, pname)]
                if isinstance(module, nn.Linear):
                    in_scores.append(score.sum(axis=1).detach().cpu().numpy())
                    out_scores.append(score.sum(axis=0).detach().cpu().numpy())
                if isinstance(module, nn.Conv2d):
                    in_scores.append(score.sum(axis=(1, 2, 3)).detach().cpu().numpy())
                    out_scores.append(score.sum(axis=(0, 2, 3)).detach().cpu().numpy())

    if concatenate_scores:
        in_scores = np.concatenate(in_scores[:-1][1])
        out_scores = np.concatenate(out_scores[1:][1])

    return in_scores, out_scores

## Compute Average Layer Score ##
def average_layer_score(scores, prunable_parameters):

    layerwise_scores = {}
    for i, (module, name) in enumerate(prunable_parameters):
        W = module.weight
        W_score = scores[(module, name)]
        score_sum = W_score.sum().detach().cpu().numpy()
        num_elements = np.prod(W.shape)
        inv_size = 1.0 / num_elements
        average_score = np.abs(score_sum / num_elements)
        layerwise_scores.update({i: {
            'average_score': average_score,
            'inv_size': inv_size
            }})
    return layerwise_scores

