import torch
import torch.nn as nn


class TopKNCALoss(nn.Module):
    def __init__(self, dim, reduction='mean', ret_top_values=False, ret_top_indices=False):
        super(TopKNCALoss, self).__init__()

        self.dim = dim
        self.reduction = reduction
        self.ret_top_values = ret_top_values
        self.ret_top_indices = ret_top_indices

    def forward(self, inputs, k):
        values, indices = torch.topk(inputs, k=k, dim=self.dim)

        top_sum = torch.sum(values, dim=self.dim)
        loss = - torch.log(top_sum)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        ret = loss

        if self.ret_top_values or self.ret_top_indices:
            ret = [ret, ]
            if self.ret_top_values:
                ret.append(values)
            if self.ret_top_indices:
                ret.append(indices)
            ret = tuple(ret)

        return ret
