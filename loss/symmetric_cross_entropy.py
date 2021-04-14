import torch.nn as nn
import torch.nn.functional as F


class SymmetricCrossEntropy(nn.Module):
    def __init__(self):
        super(SymmetricCrossEntropy, self).__init__()

    def forward(self, x, label):
        ce_loss = F.cross_entropy(x, label)

        num_class = x.size(1)
        one_hot = F.one_hot(label, num_class)
        sce_loss = F.softmax(x, dim=1) * F.log_softmax(one_hot, dim=1)
        sce_loss = sce_loss.sum(dim=1).mean()

        return ce_loss + sce_loss
