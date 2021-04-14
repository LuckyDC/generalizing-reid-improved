import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class SplitBatchNorm(_BatchNorm):
    def __init__(self, source_ratio, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SplitBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.source_ratio = source_ratio

        self.register_buffer('running_mean_s', torch.zeros(num_features))
        self.register_buffer('running_var_s', torch.ones(num_features))

    def forward(self, x):

        if self.training:
            source_size = int(self.source_ratio * x.size(0))
            target_size = x.size(0) - source_size
            source_split, target_split = x.split((source_size, target_size), 0)

            source_result = F.batch_norm(source_split, self.running_mean_s, self.running_var_s, self.weight,
                                         self.bias, True, self.momentum, self.eps)
            target_result = F.batch_norm(target_split, self.running_mean, self.running_var, self.weight,
                                         self.bias, True, self.momentum, self.eps)
            source_result = source_result[:source_size]
            target_result = target_result[-target_size:]

            return torch.cat([source_result, target_result], dim=0)
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                not self.track_running_stats, self.momentum, self.eps)
