import os
import torch
import torch.distributed as dist
from torch import optim
from torch.nn import parallel
from runx.logx import logx

from data import get_cross_domain_train_loader
from data import get_test_loader
from data import get_train_loader
from engine import get_trainer
from models.model import Model


def train(cfg):
    # Training data loader
    if not cfg.joint_training:  # single domain
        train_loader = get_train_loader(root=os.path.join(cfg.source.root, cfg.source.train),
                                        batch_size=cfg.batch_size,
                                        image_size=cfg.image_size,
                                        random_flip=cfg.random_flip,
                                        random_crop=cfg.random_crop,
                                        random_erase=cfg.random_erase,
                                        color_jitter=cfg.color_jitter,
                                        padding=cfg.padding,
                                        num_workers=4)
    else:  # cross domain
        source_root = os.path.join(cfg.source.root, cfg.source.train)
        target_root = os.path.join(cfg.target.root, cfg.target.train)

        train_loader = get_cross_domain_train_loader(source_root=source_root,
                                                     target_root=target_root,
                                                     batch_size=cfg.batch_size,
                                                     random_flip=cfg.random_flip,
                                                     random_crop=cfg.random_crop,
                                                     random_erase=cfg.random_erase,
                                                     color_jitter=cfg.color_jitter,
                                                     padding=cfg.padding,
                                                     image_size=cfg.image_size,
                                                     num_workers=8)

    # Evaluation data loader
    query_loader = None
    gallery_loader = None
    if cfg.eval_interval > 0:
        query_loader = get_test_loader(root=os.path.join(cfg.target.root, cfg.target.query),
                                       batch_size=512,
                                       image_size=cfg.image_size,
                                       num_workers=4)

        gallery_loader = get_test_loader(root=os.path.join(cfg.target.root, cfg.target.gallery),
                                         batch_size=512,
                                         image_size=cfg.image_size,
                                         num_workers=4)

    # Model
    num_classes = cfg.source.num_id
    cam_ids = train_loader.dataset.target_dataset.cam_ids if cfg.joint_training else train_loader.dataset.cam_ids
    num_instances = len(train_loader.dataset.target_dataset) if cfg.joint_training else len(train_loader.dataset)

    model = Model(num_classes=num_classes,
                  drop_last_stride=cfg.drop_last_stride,
                  joint_training=cfg.joint_training,
                  num_instances=num_instances,
                  cam_ids=cam_ids,
                  neg_proto=cfg.neg_proto,
                  neighbor_mode=cfg.neighbor_mode,
                  neighbor_eps=cfg.neighbor_eps,
                  threshold=cfg.threshold,
                  scale=cfg.scale,
                  mix_st=cfg.mix_st,
                  alpha=cfg.alpha,
                  nn_filter=cfg.nn_filter,
                  momentum=cfg.momentum,
                  loss_factor=cfg.loss_factor)
    # ids=train_loader.dataset.target_dataset.ids)
    model.cuda()

    # Optimizer
    ft_params = model.backbone.parameters()
    new_params = [param for name, param in model.named_parameters() if not name.startswith("backbone.")]
    param_groups = [{'params': ft_params, 'lr': cfg.ft_lr},
                    {'params': new_params, 'lr': cfg.new_params_lr}]

    if cfg.optimizer_type == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.wd)
    else:
        optimizer = optim.Adam(param_groups, weight_decay=cfg.wd)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=cfg.lr_step,
                                                  gamma=0.1)

    # Convert model for mixed precision distributed training
    if dist.is_initialized():
        model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

    # Training engine
    engine = get_trainer(model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         enable_amp=cfg.fp16,
                         log_period=cfg.log_period,
                         save_interval=10,
                         eval_interval=cfg.eval_interval,
                         query_loader=query_loader,
                         gallery_loader=gallery_loader)

    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    import yaml
    import argparse
    import random
    import numpy as np
    from pprint import pformat
    from yacs.config import CfgNode
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/cross_domain.yml")
    parser.add_argument("--local_rank", type=int, default=None)
    args = parser.parse_args()

    # Initialize distributed training
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        torch.distributed.init_process_group(backend="nccl", rank=args.local_rank, world_size=num_gpus)
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True

    # Load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), yaml.SafeLoader)
    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    source_cfg = dataset_cfg.get(cfg.source_dataset)
    target_cfg = dataset_cfg.get(cfg.target_dataset)
    cfg.source = CfgNode()
    cfg.target = CfgNode()
    for k, v in source_cfg.items():
        cfg.source[k] = v
    for k, v in target_cfg.items():
        cfg.target[k] = v

    cfg.batch_size = cfg.batch_size // torch.cuda.device_count()
    cfg.freeze()

    # Set random seed
    seed = 0 if args.local_rank is None else args.local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set up logger
    logx.initialize(logdir='logs/{}'.format(cfg.prefix), hparams=cfg, tensorboard=True,
                    global_rank=args.local_rank if dist.is_initialized() else 0)
    logx.msg(pformat(cfg))

    train(cfg)
