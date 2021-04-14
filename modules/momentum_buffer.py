import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class MomentumBuffer(nn.Module):
    def __init__(self, size, normalize=False):
        super(MomentumBuffer, self).__init__()

        self.register_buffer('buffer', torch.empty(size))
        self.normalize = normalize

    def update(self, x, ids, mom):
        if dist.is_initialized():
            x_list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
            ids_list = [torch.empty_like(ids) for _ in range(dist.get_world_size())]

            dist.all_gather(x_list, x)
            dist.all_gather(ids_list, ids)
            dist.barrier()

            x = torch.cat(x_list, dim=0)
            ids = torch.cat(ids_list, dim=0)

        new = mom * self.buffer[ids] + (1 - mom) * x
        if self.normalize:
            new = F.normalize(new)
        self.buffer.index_copy_(0, ids, new)
