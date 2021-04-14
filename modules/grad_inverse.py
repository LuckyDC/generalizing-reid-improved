import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd


class GradInverse(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return -grad_output


class GradInversion(nn.Module):
    def __init__(self):
        super(GradInversion, self).__init__()

    def forward(self, x):
        return GradInverse.apply(x)
