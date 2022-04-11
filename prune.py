from tqdm import tqdm
import torch
import numpy as np
import torch.nn.utils.prune as prune
from Pruners.synflow import score, SynFlow
from Utils.generator import prunable
from Utils.metrics import global_sparsity, summary


def prune_loop(model, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, prune_bias=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    importance_scores = None
    # Prune model
    for epoch in tqdm(range(epochs)):
        importance_scores = score(model, dataloader, device, prune_bias)
        sparse = sparsity
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        params = []
        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "bias" and prune_bias is False:
                    continue
                params.append((module, pname))
        prune.global_unstructured(parameters=params, pruning_method=SynFlow, importance_scores=importance_scores,
                                amount=sparse,)

    # make pruning permanent
    for module in filter(lambda p: prunable(p), model.modules()):
        if hasattr(module, 'weight'):
            prune.remove(module, "weight")
        if hasattr(module, "bias") and prune_bias is True:
            prune.remove(module, "bias")

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Confirm sparsity level
    glob_sparsity = global_sparsity(model, prune_bias)
    assert round(glob_sparsity, 2) == round(sparsity, 2)
    print(f"Global sparsity after pruning: {round(100 * glob_sparsity, 2)}%")

    summary_results = summary(model, importance_scores)
    return summary_results
