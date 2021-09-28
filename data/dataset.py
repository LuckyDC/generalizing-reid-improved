import os
import torch
import bisect
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, recursive=False, label_organize=False, aug_view=False):
        if recursive:
            image_list = glob(os.path.join(root, "**", "*.jpg"), recursive=recursive) + \
                         glob(os.path.join(root, "**", "*.png"), recursive=recursive)
        else:
            image_list = glob(os.path.join(root, "*.jpg")) + glob(os.path.join(root, "*.png"))

        self.image_list = list(filter(lambda x: int(os.path.basename(x).split("_")[0]) != -1, image_list))
        self.image_list.sort()

        ids = []
        cam_ids = []
        for img_path in self.image_list:
            splits = os.path.basename(img_path).split("_")
            ids.append(int(splits[0]))

            if root.lower().find("msmt") != -1:
                cam_id = int(splits[2])
            else:
                cam_id = int(splits[1][1:]) if root.lower().find("ve") != -1 else int(splits[1][1])

            cam_ids.append(cam_id - 1)

        if label_organize:
            # organize identity label
            unique_ids = set(ids)
            label_map = dict(zip(unique_ids, range(len(unique_ids))))

            ids = map(lambda x: label_map[x], ids)
            ids = list(ids)

        self.ids = ids
        self.cam_ids = cam_ids
        self.num_id = len(set(ids))

        self.transform = transform
        self.aug_view = aug_view

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        batch = {}

        img_path = self.image_list[item]
        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)
        batch.update({'id': label,
                      'cam_id': cam,
                      'path': img_path,
                      'index': item})

        img = Image.open(img_path)
        if self.transform is not None:
            data = self.transform(img)
            batch.update({'img': data})

            if self.aug_view:
                aug = self.transform(img)
                batch.update({'aug': aug})

        return batch


class ImageListFile(Dataset):
    def __init__(self, path, prefix=None, transform=None, label_organize=False, aug_view=False):
        if not os.path.isfile(path):
            raise ValueError("The file %s does not exist." % path)

        txt = np.loadtxt(path, delimiter=" ", dtype=np.str)
        image_list = list(txt if txt.ndim == 1 else txt[:, 0])

        if prefix is not None:
            image_list = map(lambda x: os.path.join(prefix, x), image_list)

        self.image_list = list(filter(lambda x: int(os.path.basename(x).split("_")[0]) != -1, image_list))
        self.image_list.sort()

        ids = []
        cam_ids = []
        for img_path in self.image_list:
            splits = os.path.basename(img_path).split("_")
            ids.append(int(splits[0]))

            if path.lower().find("msmt") != -1:
                cam_id = int(splits[2])
            else:
                cam_id = int(splits[1][1])

            cam_ids.append(cam_id - 1)

        if label_organize:
            # organize identity label
            unique_ids = set(ids)
            label_map = dict(zip(unique_ids, range(len(unique_ids))))

            ids = map(lambda x: label_map[x], ids)
            ids = list(ids)

        self.cam_ids = cam_ids
        self.ids = ids
        self.num_id = len(set(ids))

        self.transform = transform
        self.aug_view = aug_view

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        batch = {}

        img_path = self.image_list[item]
        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)
        batch.update({'id': label,
                      'cam_id': cam,
                      'path': img_path,
                      'index': item})

        img = Image.open(img_path)
        if self.transform is not None:
            data = self.transform(img)
            batch.update({'img': data})

            if self.aug_view:
                aug = self.transform(img)
                batch.update({'aug': aug})

        return batch


class CrossDataset(Dataset):
    def __init__(self, source_dataset, target_dataset):
        super(CrossDataset, self).__init__()

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.source_size = len(self.source_dataset)
        self.target_size = len(target_dataset)

        self.num_source_cams = len(set(source_dataset.cam_ids))
        self.num_target_cams = len(set(target_dataset.cam_ids))

        self.num_source_id = source_dataset.num_id

    def __len__(self):
        return self.source_size + self.target_size

    def __getitem__(self, idx):
        # From source dataset
        if idx < self.source_size:
            sample = self.source_dataset[idx]
            sample['cam_id'].add_(self.num_target_cams)

            return sample
        # From target dataset
        else:
            idx = idx - self.source_size
            sample = self.target_dataset[idx]
            return sample


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()

        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])

        self.cam_ids = []
        for d in datasets:
            self.cam_ids.extend(d.cam_ids)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        sample = self.datasets[dataset_idx][sample_idx]
        sample['index'] = torch.tensor(idx, dtype=torch.long)

        return sample
