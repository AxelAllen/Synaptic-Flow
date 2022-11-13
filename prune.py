from tqdm import tqdm
import torch
import numpy as np
import prune_
from Pruners.pruners_ import *
from Utils.generator import prunable
from Utils import load
from Utils.metrics import global_sparsity, summary, unit_score_sum, average_layer_score
import wandb

def prune_loop(model, prune_class, dataloader, loss, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, prune_bias=False, use_wandb=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    pruner = load.pruner(prune_class)(sparsity)
    prune_method = load.pruner(prune_class)
    print(f"Pruning with {pruner.__class__.__name__} pruner.")
    # Set model to train or eval mode
    model.to(device)
    model.train()
    # if not train_mode:
    #     model.eval()
    all_summary_results = {}
    importance_scores = None
    # Prune model
    # print(torch.cuda.memory_allocated(device=device))
    # with torch.autograd.set_detect_anomaly(True):

    ## Prunable parameters ##
    params = []
    if hasattr(model, 'bert'):
        for ii in range(12):
            params.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
            params.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
            params.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
            params.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
            params.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
            params.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

        params.append((model.bert.pooler.dense, 'weight'))
    elif hasattr(model, 'reformer'):
        for ii in range(6):
            if ii % 2 == 0:
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.query, "weight"))
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.key, "weight"))
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.value, "weight"))
            else:
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.query_key, "weight"))
                params.append((model.reformer.encoder.layers[ii].attention.self_attention.value, "weight"))
            params.append((model.reformer.encoder.layers[ii].attention.output.dense, "weight"))
            params.append((model.reformer.encoder.layers[ii].feed_forward.dense.dense, "weight"))
            params.append((model.reformer.encoder.layers[ii].feed_forward.output.dense, "weight"))
    else:
        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "bias" and prune_bias is False:
                    continue
                params.append((module, pname))


    for epoch in tqdm(range(epochs)):
        importance_scores = pruner.score(model, dataloader, loss, device, prune_bias)
        if epoch == 0:
            in_scores, out_scores = unit_score_sum(model, importance_scores)
            unit_scores = (in_scores, out_scores)
            average_layers_scores = average_layer_score(importance_scores, params)

        #sparse = sparsity
        sparse = 1.0 - (sparsity ** ((epoch + 1) / epochs))
        if schedule == 'exponential':
            sparse = 1.0 - (sparsity**((epoch + 1) / epochs))
        elif schedule == 'linear':
             sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        if use_wandb:
            wandb.log({"sparsity": sparse})

        ## Prune ##
        prune_.global_unstructured(parameters=params, pruning_method=prune_method, importance_scores=importance_scores,
                                amount=1-sparse)


        ## Make pruning permanent ##
        if hasattr(model, 'bert') or hasattr(model, 'reformer'):
            for module, _ in params:
                if hasattr(module, 'weight'):
                    prune_.remove(module, "weight")
        else:
            for module in filter(lambda p: prunable(p), model.modules()):
                if hasattr(module, 'weight'):
                    prune_.remove(module, "weight")
                if hasattr(module, "bias") and prune_bias is True:
                    prune_.remove(module, "bias")


        summary_results = summary(model, importance_scores)
        all_summary_results.update({epoch: summary_results})

        # Reainitialize weights
        if reinitialize:
            model._initialize_weights()

        # Confirm sparsity level
        glob_sparsity = global_sparsity(model, prunable_parameters=None, prune_bias=False)
        # assert round(glob_sparsity, 2) == round(sparsity, 2)
        print(f"Global sparsity after pruning: {round(100 * glob_sparsity, 2)}%")
        if use_wandb:
            wandb.log({"global_sparsity_after_pruning": glob_sparsity})

    return all_summary_results, unit_scores, average_layers_scores
