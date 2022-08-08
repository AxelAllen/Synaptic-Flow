import torch
from torch import nn
import prune_
import torch.nn.functional as F
from Utils.generator import prunable
import types

class Pruner(prune_.BasePruningMethod):
    '''
        options:
        global, structured, unstructured
        '''
    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        super(Pruner, self).__init__()
        assert 1 >= amount >= 0
        self.sparsity = amount


    def compute_mask(self, importance_scores, default_mask):
        mask = default_mask.clone()
        zero = torch.tensor([0.]).to(mask.device)
        one = torch.tensor([1.]).to(mask.device)
        k = int((1.0 - self.sparsity) * importance_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(importance_scores, k)
            mask.copy_(torch.where(importance_scores <= threshold, zero, one))
        return mask

    def shuffle(self, default_mask):
        shape = default_mask.shape
        mask = default_mask.clone()
        perm = torch.randperm(mask.nelement())
        mask = mask.reshape(-1)[perm].reshape(shape)
        return mask

class SynFlow(Pruner):

    def __init__(self, amount):
        super(SynFlow, self).__init__(amount)

    def score(self, model, dataloader, loss,  device, prune_bias=False):

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

        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "bias" and prune_bias is False:
                    continue
                score = torch.clone(param.grad * param).detach().abs_()
                param.grad.data.zero_()
                scores.update({(module, pname): score})

        nonlinearize(model, signs)

        return scores

class SynFlowBERT(Pruner):

    def __init__(self, amount):
        super(SynFlowBERT, self).__init__(amount)

    def score(self, model, dataloader, loss,  device, prune_bias=False):

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

        batch = next(iter(dataloader))


        input = torch.ones(batch["input_ids"].shape).long().to(device)
        attn_mask = torch.ones(batch["attention_mask"].shape).to(device)

        output = model(input_ids=input,
                       attention_mask=attn_mask)
        '''
        output = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       token_type_ids=batch["token_type_ids"],
                       labels=batch["labels"])
        '''
        logits = output.logits
        torch.sum(logits).backward()

        parameters_to_prune = []
        for ii in range(12):
            parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
            parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
            parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
            parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
            parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
            parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

        parameters_to_prune.append((model.bert.pooler.dense, 'weight'))

        '''
        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "bias" and prune_bias is False:
                    continue
                score = torch.clone(param.grad * param).detach().abs_()
                param.grad.data.zero_()
                scores.update({(module, pname): score})
        '''

        for module, pname in parameters_to_prune:
            param = module.weight
            score = torch.clone(param.grad * param).detach().abs_()
            param.grad.data.zero_()
            scores.update({(module, pname): score})

        nonlinearize(model, signs)

        return scores


class Random(Pruner):
    def __init__(self, amount):
        super(Random, self).__init__(amount)

    def score(self, model, dataloader, loss,  device, prune_bias=False):
        scores = {}
        if hasattr(model, 'bert'):
            parameters_to_prune = []
            for ii in range(12):
                parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

            parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
            for module, pname in parameters_to_prune:
                param = module.weight
                scores.update({(module, pname): torch.randn_like(param)})
        else:
            for module in filter(lambda p: prunable(p), model.modules()):
                for pname, param in module.named_parameters(recurse=False):
                    if pname == "bias" and prune_bias is False:
                        continue
                    scores.update({(module, pname): torch.randn_like(param)})
        return scores



class Magnitude(Pruner):
    def __init__(self, amount):
        super(Magnitude, self).__init__(amount)

    def score(self, model, dataloader, loss, device, prune_bias=False):
        scores = {}
        if hasattr(model, 'bert'):
            parameters_to_prune = []
            for ii in range(12):
                parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
                parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

            parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
            for module, pname in parameters_to_prune:
                param = module.weight
                scores.update({(module, pname): torch.clone(param.data).detach().abs_()})
        else:
            for module in filter(lambda p: prunable(p), model.modules()):
                for pname, param in module.named_parameters(recurse=False):
                    if pname == "bias" and prune_bias is False:
                        continue
                    scores.update({(module, pname): torch.clone(param.data).detach().abs_()})
        return scores

# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):

    def __init__(self, amount):
        super(SNIP, self).__init__(amount)

    def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

    def score(self, model, dataloader, loss, device, prune_bias=False):

        scores = {}

        # allow masks to have gradient
        for module in filter(lambda p: prunable(p), model.modules()):
            if hasattr(module, "weight"):
                module.weight_mask = nn.Parameter(torch.ones_like(module.weight))
                module.weight_mask.requires_grad = True

            # Override the forward methods:
            if isinstance(module, nn.Conv2d):
                module.forward = types.MethodType(self.snip_forward_conv2d, module)

            if isinstance(module, nn.Linear):
                module.forward = types.MethodType(self.snip_forward_linear, module)

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
            for module in filter(lambda p: prunable(p), model.modules()):
                for pname, param in module.named_parameters(recurse=False):
                    if pname == "weight":
                        score = module.weight_mask.grad.detach().abs()
                        scores.update({(module, pname): score})
                        param.grad.data.zero_()
                        module.weight_mask.grad.data.zero_()
                        module.weight_mask.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
        norm = torch.sum(all_scores)
        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "weight":
                    scores[(module, pname)] = scores[(module, pname)].div_norm()

# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, amount):
        super(GraSP, self).__init__(amount)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, dataloader, loss,  device, prune_bias=False):

        scores = {}
        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [param for param in filter(lambda p: prunable(p), model.parameters())],
                                        create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [param for param in filter(lambda p: prunable(p), model.parameters())],
                                        create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "bias" and prune_bias is False:
                    continue
                score = torch.clone(param.grad * param.data).detach()
                scores.update({(module, pname): score})
                param.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for module in filter(lambda p: prunable(p), model.modules()):
            for pname, param in module.named_parameters(recurse=False):
                if pname == "bias" and prune_bias is False:
                    continue
                scores[(module, pname)] = scores[(module, pname)].div_norm()