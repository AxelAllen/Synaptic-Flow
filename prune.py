from tqdm import tqdm
import torch
import numpy as np
from Pruners.synflow import score, SynFlow
from Utils.generator import prunable
import torch.nn.utils.prune as prune
from Utils.metrics import stats, summary


def prune_loop(model, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    importance_scores = None
    # Prune model
    for epoch in tqdm(range(epochs)):
        importance_scores = score(model, dataloader, device)
        sparse = sparsity
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        params = []
        for name, module in model.named_modules():
            if prunable(module):
                params.append((module, name))
        prune.global_unstructured(parameters=params, pruning_method=pruner, importance_scores=importance_scores,
                                amount=sparse)

    # make pruning permanent
    for name, module in model.named_modules():
        if prunable(module):
            prune.remove(module, name)

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Confirm sparsity level
    global_sparsity, remaining_params, total_params = stats(model)
    assert global_sparsity == sparsity
    print(f"Global sparsity after pruning: {100 * global_sparsity}%")
    if np.abs(remaining_params - total_params*sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()

    summary_results = summary(model, importance_scores, lambda p: prunable(p))
    return summary_results
