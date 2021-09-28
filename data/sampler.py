import math
from collections import defaultdict

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, dataset, batch_size, num_instance):
        assert batch_size % num_instance == 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.p_size = batch_size // num_instance
        self.k_size = num_instance

        self.id2idx = defaultdict(list)
        for i, identity in enumerate(dataset.ids):
            self.id2idx[identity].append(i)

    def __len__(self):
        return self.dataset.num_id * self.k_size

    def __iter__(self):
        sample_list = []

        id_perm = np.random.permutation(self.dataset.num_id)
        for start in range(0, self.dataset.num_id, self.p_size):
            selected_ids = id_perm[start:start + self.p_size]

            sample = []
            for identity in selected_ids:
                if len(self.id2idx[identity]) < self.k_size:
                    s = np.random.choice(self.id2idx[identity], size=self.k_size, replace=True)
                else:
                    s = np.random.choice(self.id2idx[identity], size=self.k_size, replace=False)

                sample.extend(s)

            sample_list.extend(sample)

        return iter(sample_list)


class CrossDatasetRandomSampler(Sampler):
    def __init__(self, source_dataset, target_dataset, batch_size):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.batch_size = batch_size

        self.source_size = len(source_dataset)
        self.target_size = len(target_dataset)

    def __len__(self):
        return self.target_size * 2

    def __iter__(self):
        perm = []
        half_bs = self.batch_size // 2

        source_perm = np.random.permutation(self.source_size)
        if self.source_size >= self.target_size:
            source_perm = source_perm[:self.target_size]
        else:
            pad = np.random.choice(source_perm, self.target_size - self.source_size, replace=False)
            source_perm = np.concatenate([source_perm, pad])
        source_perm = source_perm.tolist()

        target_perm = np.random.permutation(self.target_size) + self.source_size
        target_perm = target_perm.tolist()

        for i in range(math.ceil(self.target_size / half_bs)):
            perm.extend(source_perm[i * half_bs:(i + 1) * half_bs])
            perm.extend(target_perm[i * half_bs:(i + 1) * half_bs])

        return iter(perm)


class CrossDatasetDistributedSampler(Sampler):
    def __init__(self, source_dataset, target_dataset, batch_size, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
            batch_size *= num_replicas

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = None

        self.source_size = len(source_dataset)
        self.target_size = len(target_dataset)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        perm = []
        half_bs = self.batch_size // 2

        source_perm = np.random.permutation(self.source_size)
        if self.source_size >= self.target_size:
            source_perm = source_perm[:self.target_size]
        else:
            pad = np.random.choice(source_perm, self.target_size - self.source_size, replace=False)
            source_perm = np.concatenate([source_perm, pad])
        source_perm = source_perm.tolist()

        target_perm = np.random.permutation(self.target_size) + self.source_size
        target_perm = target_perm.tolist()

        for i in range(math.ceil(self.target_size / half_bs)):
            perm.extend(source_perm[i * half_bs:(i + 1) * half_bs])
            perm.extend(target_perm[i * half_bs:(i + 1) * half_bs])

        perm = perm[self.rank::self.num_replicas]

        return iter(perm)

    def __len__(self):
        return int(np.round(self.target_size * 2 / self.num_replicas))

    def set_epoch(self, epoch):
        self.epoch = epoch
