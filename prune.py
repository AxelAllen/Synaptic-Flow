from tqdm import tqdm
import torch
import numpy as np
import prune_
from Pruners.pruners_ import *
from Utils.generator import prunable
from Utils import load
from Utils.metrics import global_sparsity, summary


def prune_loop(model, prune_class, dataloader, loss, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, prune_bias=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    pruner = load.pruner(prune_class)(sparsity)
    prune_method = load.pruner(prune_class)
    print(f"Pruning with {pruner.__class__.__name__} pruner.")
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    importance_scores = None
    # Prune model
    with torch.autograd.set_detect_anomaly(True):
        for epoch in tqdm(range(epochs)):
            importance_scores = pruner.score(model, dataloader, loss, device, prune_bias)
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
            prune_.global_unstructured(parameters=params, pruning_method=prune_method, importance_scores=importance_scores,
                                    amount=sparse)

    # make pruning permanent
    if epochs > 0:
        for module in filter(lambda p: prunable(p), model.modules()):
            if hasattr(module, 'weight'):
                prune_.remove(module, "weight")
            if hasattr(module, "bias") and prune_bias is True:
                prune_.remove(module, "bias")

        # Reainitialize weights
        if reinitialize:
            model._initialize_weights()

        # Confirm sparsity level
        glob_sparsity = global_sparsity(model, prune_bias)
        assert round(glob_sparsity, 2) == round(sparsity, 2)
        print(f"Global sparsity after pruning: {round(100 * glob_sparsity, 2)}%")

        summary_results = summary(model, importance_scores)
        return summary_results
    else:
        return 0
