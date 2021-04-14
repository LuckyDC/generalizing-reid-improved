import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.nn import init


class MemoryLinearFunc(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, memory, target, content=None, momentum=0.1):

        if dist.is_initialized():
            world_size = dist.get_world_size()
            input_list = [torch.empty_like(inputs) for _ in range(world_size)]
            target_list = [torch.empty_like(target) for _ in range(world_size)]

            dist.barrier()
            dist.all_gather(input_list, inputs)
            dist.all_gather(target_list, target)

            all_input = torch.cat(input_list, dim=0)
            all_target = torch.cat(target_list, dim=0)

            if content is not None:
                content_list = [torch.empty_like(content) for _ in range(world_size)]
                dist.all_gather(content_list, content)
                all_content = torch.cat(content_list, dim=0)
            else:
                all_content = all_input

            dist.barrier()
            ctx.save_for_backward(memory, all_content, all_target)
        else:
            ctx.save_for_backward(memory, content, target)

        ctx.momentum = momentum
        return torch.mm(inputs, memory.t())

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        memory, content, target = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(memory)

        momentum = ctx.momentum
        memory[target] *= momentum
        memory[target] += (1 - momentum) * content
        memory[target] /= torch.norm(memory[target], p=2, dim=1, keepdim=True)

        return grad_input, None, None, None, None


class MemoryLinear(nn.Module):
    def __init__(self, num_instances, num_features, momentum=0.1):
        super(MemoryLinear, self).__init__()

        self.num_instances = num_instances
        self.num_features = num_features
        self.momentum = momentum

        self.register_buffer('memory', torch.zeros(num_instances, num_features))
        self.reset_buffers()

    def set_momentum(self, value):
        self.momentum = value

    def reset_buffers(self):
        init.normal_(self.memory, std=1.0)
        self.memory.copy_(F.normalize(self.memory))

    def update_memory(self, x, target, momentum=None):
        if momentum is None:
            momentum = self.momentum

        if dist.is_initialized():
            dist.barrier()
            x_list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(x_list, x)
            x = torch.cat(x_list, dim=0)

            target_list = [torch.empty_like(target) for _ in range(dist.get_world_size())]
            dist.all_gather(target_list, target)
            dist.barrier()

            target = torch.cat(target_list, dim=0)

        mem = self.memory[target]
        mem = mem.mul(momentum).add(x.data, alpha=1 - momentum)
        self.memory.index_copy_(0, target, F.normalize(mem))

    def forward(self, x, target, content=None):
        """
        The forward pass of memory linear function.
        :param x: The operand of inner product.
        :param target: The indices of input feature.
        :param content: The actual features to update memory. If None, take x as content.
        :return: The inner product of input and memory.
        """
        return MemoryLinearFunc.apply(x, self.memory, target, content, self.momentum)
