import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from Utils.generator import parameters

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
            zero = torch.tensor([0.]).to(mask.device)
            one = torch.tensor([1.]).to(mask.device)
            mask.copy_(torch.where(score <= threshold, zero, one))
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

    for _, p in parameters(model):
        scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
        p.grad.data.zero_()

    nonlinearize(model, signs)

    return scores


def synflow_unstructured(module, name, amount, importance_scores=None):
    SynFlow.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module


sparsity = 0.1
prune.global_unstructured(parameters=None, pruning_method=synflow_unstructured(), importance_scores=score(), amount=sparsity)