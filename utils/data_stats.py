import os
import random
from glob import glob
from collections import defaultdict
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

random.seed(0)


def get_dataset_stats(root_or_file):
    id2num = {}
    id2paths = defaultdict(list)

    if os.path.isdir(root_or_file):
        fnames = glob(os.path.join(root_or_file, '**/*.jpg'), recursive=True)
    elif os.path.isfile(root_or_file):
        fnames = [x.split(' ')[0] for x in open(root_or_file, 'r').readlines()]
    else:
        raise TypeError('The specified string does not point to a valid directory or file.')

    for name in fnames:
        idx = int(os.path.basename(name).split('_')[0])
        id2paths[idx].append(name)

    for k, v in id2paths.items():
        id2num[k] = len(v)

    num_img = len(fnames)
    num_id = len(id2paths)

    print('Number of images: {}.'.format(num_img))
    print('Number of identities: {}.'.format(num_id))
    print('Max number of identity images: {}.'.format(max(id2num.values())))
    print('Min number of identity images: {}.'.format(min(id2num.values())))

    bin2id = defaultdict(list)
    for idx, num in id2num.items():
        bin2id[num // 5].append(idx)

    # Imbalance
    selected_ids = []
    for ids in bin2id.values():
        selected_ids.extend(ids[:10])

    selected_id2paths = {}
    for idx, paths in id2paths.items():
        if idx in selected_ids:
            selected_id2paths[idx] = paths

    n = 0
    for idx in selected_ids:
        n += len(id2paths[idx])
    print(n)
    print(len(selected_ids))

    dirname = os.path.dirname(root_or_file) if os.path.isfile(root_or_file) else root_or_file
    f = open(os.path.join(dirname, 'Imbalance.txt'), 'w')
    for paths in selected_id2paths.values():
        f.writelines([p + '\n' for p in paths])

    plt.hist([n for i, n in id2num.items() if i in selected_ids], bins=50)
    plt.savefig('imbalance.jpg')
    plt.close()

    # Balance-2
    selected_id2paths = {}
    for idx, paths in id2paths.items():
        if len(paths) > 23:
            selected_id2paths[idx] = random.sample(paths, 20)

    dirname = os.path.dirname(root_or_file) if os.path.isfile(root_or_file) else root_or_file
    f = open(os.path.join(dirname, 'Balance2.txt'), 'w')
    for paths in selected_id2paths.values():
        f.writelines([p + '\n' for p in paths])

    nums = [len(paths) for paths in selected_id2paths.values()]
    print(sum(nums))
    print(len(selected_id2paths))

    plt.hist(nums, bins=50)
    plt.savefig('balance2.jpg')
    plt.close()

    # Balance
    selected_ids = []
    for b, ids in bin2id.items():
        if 30 < b * 5 < 75:
            selected_ids.extend(ids)

    selected_id2paths = {}
    for idx, paths in id2paths.items():
        if idx in selected_ids:
            selected_id2paths[idx] = paths

    n = 0
    for idx in selected_ids:
        n += len(id2paths[idx])
    print(n)
    print(len(selected_ids))

    dirname = os.path.dirname(root_or_file) if os.path.isfile(root_or_file) else root_or_file
    f = open(os.path.join(dirname, 'Balance.txt'), 'w')
    for paths in selected_id2paths.values():
        f.writelines([p + '\n' for p in paths])

    plt.hist([n for i, n in id2num.items() if i in selected_ids], bins=50)
    plt.savefig('balance.jpg')
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)

    args = parser.parse_args()

    get_dataset_stats(args.root)
