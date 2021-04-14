import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation
from collections import defaultdict
from runx.logx import logx

from models.resnet import resnet50
from utils.calc_acc import calc_acc
from loss.nn_loss import NNLoss
from modules.memory_linear import MemoryLinear
from modules.momentum_buffer import MomentumBuffer
from modules.split_bn import SplitBatchNorm

num_parts = 2


def trans_dist(x, norm, bn):
    """
    Transfer the distribution across two sub-batches.
    We use this function to adapt the memory stored in the target domain to the joint domain
    for the cross-domain classifier composition.
    This operation makes little effect on the performance.
    """
    weight = bn.weight.unsqueeze(0)
    mean_s, var_s = bn.running_mean_s, bn.running_var_s
    mean_t, var_t = bn.running_mean, bn.running_var
    mean_s, std_s = mean_s.unsqueeze(0), var_s.add(1e-5).sqrt().unsqueeze(0)
    mean_t, std_t = mean_t.unsqueeze(0), var_t.add(1e-5).sqrt().unsqueeze(0)

    scale = weight / std_s
    mean_diff = mean_t - mean_s
    x = x.mul(norm.unsqueeze(1))
    x.mul_(std_t).div_(std_s).add_(scale * mean_diff)

    return F.normalize(x)


class Model(nn.Module):

    def __init__(self, num_classes=None, drop_last_stride=False, joint_training=False, neighbor_mode=1, **kwargs):
        super(Model, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.joint_training = joint_training
        self.neighbor_mode = neighbor_mode

        self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride)
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.global_neck = SplitBatchNorm(0.5, 2048)
        self.global_neck.bias.requires_grad_(False)

        self.part_pool = nn.Sequential(nn.AdaptiveAvgPool2d((num_parts, 1)), nn.Flatten(2))
        self.part_necks = nn.ModuleList([SplitBatchNorm(0.5, 2048) for _ in range(num_parts)])
        for neck in self.part_necks:
            neck.bias.requires_grad_(False)

        if kwargs.get('eval'):
            return

        # ----------- Tasks for source domain --------------
        if num_classes is not None:
            self.global_classifier = nn.Linear(2048, num_classes, bias=False)
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)

            self.part_classifier = nn.ModuleList([nn.Linear(2048, num_classes, bias=False) for _ in range(num_parts)])

        # ----------- Tasks for target domain --------------
        if self.joint_training:
            cam_ids = kwargs.get('cam_ids')
            num_instances = kwargs.get('num_instances')

            # Identities captured by each camera
            uid2cam = zip(range(num_instances), cam_ids)
            self.cam2uid = defaultdict(list)
            for uid, cam in uid2cam:
                self.cam2uid[cam].append(uid)

            # Components for neighborhood consistency
            self.nn_loss = NNLoss(dim=1)
            self.memory_linear = MemoryLinear(num_instances, 2048)
            self.norm_mem = MomentumBuffer(num_instances)
            self.part_memory = nn.ModuleList([MemoryLinear(num_instances, 2048) for _ in range(num_parts)])
            self.part_norm = nn.ModuleList([MomentumBuffer(num_instances) for _ in range(num_parts)])

            # Hyper-parameter for neighborhood consistency
            self.scale = kwargs.get('scale')
            self.neighbor_eps = kwargs.get('neighbor_eps')
            self.thresh = kwargs.get('threshold')

            # Hyper-parameter for mixing
            self.mix_st = kwargs.get('mix_st')
            alpha = kwargs.get('alpha')
            self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)
            self.mix_factor = None

            ids = kwargs.get('ids')
            self.id2uid = defaultdict(list)
            for i, idx in enumerate(ids):
                self.id2uid[idx].append(i)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {}
        backbone_dict = self.backbone.state_dict(destination, 'backbone.' + prefix, keep_vars)
        global_bn_dict = self.global_neck.state_dict(destination, 'global_neck.' + prefix, keep_vars)
        part_bn_dict = self.part_necks.state_dict(destination, 'part_bn_dict.' + prefix, keep_vars)

        state_dict.update(backbone_dict)
        state_dict.update(global_bn_dict)
        state_dict.update(part_bn_dict)
        return state_dict

    def generate_mix_data(self, inputs, **kwargs):
        half_batch_size = inputs.size(0) // 2
        input_s = inputs[:half_batch_size]
        input_t = inputs[half_batch_size:]

        # Sample mixture factor for cross-domain mixup
        mix_factor = self.beta_dist.sample().item()
        self.mix_factor = mix_factor

        mixed_st = mix_factor * input_s + (1 - mix_factor) * input_t
        return mixed_st

    def forward(self, batch, **kwargs):
        inputs = batch['img']
        epoch = kwargs.get('epoch')
        batch_size = inputs.size(0)
        half_batch_size = batch_size // 2

        # Compose training batch for mixture training (activated except the initial epoch)
        if self.training and epoch > 1 and self.mix_st:
            mixed_st = self.generate_mix_data(inputs, **kwargs)
            inputs = torch.cat([mixed_st, inputs[half_batch_size:]], dim=0)

        # Extract features
        feats = self.backbone(inputs)
        global_feats = self.avg_pool(feats)
        global_feats = self.global_neck(global_feats)
        part_feats = self.part_pool(feats)
        part_feats = [self.part_necks[i](p) for i, p in enumerate(part_feats.unbind(-1))]

        # ----------------------- Evaluation Process ------------------------------ #
        if not self.training:
            feats = [F.normalize(p) for p in part_feats]
            feats.append(F.normalize(global_feats))
            return torch.cat(part_feats, dim=1)

        # ----------------------- Single Domain Training ---------------------------- #
        labels = batch['id']
        if not self.joint_training:
            return self.source_train_forward(global_feats, labels)

        # ----------------------- Cross domain Training ------------------------------- #
        loss = 0
        metric = {}
        cam_ids = batch['cam_id']
        img_ids = batch['index']
        label_s = labels[:half_batch_size]

        # Target task
        global_feat_t = global_feats[half_batch_size:batch_size]
        part_feat_t = [p[half_batch_size:batch_size] for p in part_feats]
        target_loss, target_metric = self.target_train_forward(global_feat_t, part_feat_t,
                                                               cam_ids=cam_ids, img_ids=img_ids,
                                                               labels=labels[half_batch_size:], **kwargs)
        loss += target_loss if epoch > 5 else 0 * target_loss  # Only activated after 5 epochs
        metric.update(target_metric)

        # Tasks for source data or mixed data
        global_feat_s = global_feats[:half_batch_size]
        part_feat_s = [p[:half_batch_size] for p in part_feats]
        # Source task
        if epoch == 1 or not self.mix_st:
            source_loss, source_metric = self.source_train_forward(global_feat_s, part_feat_s, label_s)
            loss += source_loss
            metric.update(source_metric)
        # Cross-domain mixed task
        else:
            img_ids_t = img_ids[half_batch_size:]
            mix_loss, mix_metric = self.mixed_st_forward(global_feat_s, part_feat_s, label_s, img_ids_t)
            loss += mix_loss
            metric.update(mix_metric)

        return loss, metric

    # Tasks for source domain
    def source_train_forward(self, global_feat, part_feat, labels):
        # Global identification loss
        cls_score = self.global_classifier(global_feat)
        loss = self.id_loss(cls_score.float(), labels)

        # Part identification loss
        for i, p in enumerate(part_feat):
            part_score = self.part_classifier[i](p)
            loss += self.id_loss(part_score.float(), labels)

        metric = {'id_ce': loss.item(),
                  'id_acc': calc_acc(cls_score.data, labels.data, ignore_index=-1)}
        return loss, metric

    # Tasks for target domain
    def target_train_forward(self, global_feat, part_feat, **kwargs):
        epoch = kwargs.get('epoch')
        img_ids = kwargs.get('img_ids')[-global_feat.size(0):]

        # Global memory
        mom = 0.5
        self.memory_linear.set_momentum(mom if epoch > 5 else 0)
        self.norm_mem.update(global_feat.norm(p=2, dim=1), img_ids, mom=0)
        # Part memory
        for i, mem in enumerate(self.part_memory):
            mem.set_momentum(mom if epoch > 5 else 0)
            self.part_norm[i].update(part_feat[i].norm(p=2, dim=1), img_ids, mom=0)

        global_feat = F.normalize(global_feat)
        part_feat = [F.normalize(p) for p in part_feat]

        # Camera-agnostic neighborhood loss
        if self.neighbor_mode == 0:
            loss, metric = self.cam_agnostic_mem_loss(global_feat, **kwargs)
        # Camera-aware neighborhood loss (intra_loss and inter_loss)
        else:
            loss, metric = self.cam_aware_mem_loss(global_feat, self.memory_linear, **kwargs)
            for i, p in enumerate(part_feat):
                loss_p, metric_p = self.cam_aware_mem_loss(p, self.part_memory[i], **kwargs)
                loss += loss_p

        return loss, metric

    def mixed_st_forward(self, global_feat, part_feat, labels, img_ids):
        cls_score = self.global_classifier(global_feat)

        # # Involve negative virtual prototype  TODO
        # mask = (~ self.mask_instance) * (~ self.mask_neighbor_intra) * (~ self.mask_neighbor_inter)
        # num_neg = min(500, mask.sum(dim=1).min().item())
        # agent = []
        # for i in range(feat.size(0)):
        #     indices = mask[i].nonzero(as_tuple=False).squeeze().tolist()
        #     indices_selected = random.sample(indices, num_neg)
        #     indices_selected.insert(0, img_ids[i].item())
        #     ag = self.memory_linear.memory[indices_selected]
        #     ag = self.trans_dist(ag, self.norm_mem.buffer[indices_selected])
        #     agent.append(ag)
        # agent = torch.stack(agent, dim=0)
        # sim_agent = torch.einsum('nd,nkd->nk', feat, agent)
        # sim_agent = sim_agent.mul(self.global_classifier.weight.data[labels].norm(dim=1, keepdim=True))
        # cls_score = torch.cat([cls_score, sim_agent], dim=1).float()

        # Global cross-domain mixup
        agent = self.memory_linear.memory[img_ids]
        agent = trans_dist(agent, self.norm_mem.buffer[img_ids], self.global_neck)
        sim_agent = global_feat.mul(agent).sum(dim=1, keepdim=True)
        sim_agent = sim_agent.mul(self.global_classifier.weight.data[labels].norm(dim=1, keepdim=True))
        cls_score = torch.cat([cls_score, sim_agent], dim=1).float()

        num_src_class = self.global_classifier.weight.size(0)
        virtual_label = labels.clone().fill_(num_src_class)
        loss = self.mix_factor * self.id_loss(cls_score, labels)
        loss += (1 - self.mix_factor) * self.id_loss(cls_score, virtual_label)

        # Part-level cross-domain mixup
        for i, p in enumerate(part_feat):
            part_score = self.part_classifier[i](p)

            part_agent = self.part_memory[i].memory[img_ids]
            part_agent = trans_dist(part_agent, self.part_norm[i].buffer[img_ids], self.part_necks[i])
            part_sim_agent = p.mul(part_agent).sum(dim=1, keepdim=True)
            part_sim_agent = part_sim_agent.mul(self.part_classifier[i].weight.data[labels].norm(dim=1, keepdim=True))
            cls_score = torch.cat([part_score, part_sim_agent], dim=1).float()

            loss += self.mix_factor * self.id_loss(cls_score, labels)
            loss += (1 - self.mix_factor) * self.id_loss(cls_score, virtual_label)

        metric = {'mix_st': loss.item()}
        return loss, metric

    def cam_aware_mem_loss(self, feat, memory_linear, **kwargs):
        metric = {}
        cam_ids = kwargs.get('cam_ids')[-feat.size(0):]
        img_ids = kwargs.get('img_ids')[-feat.size(0):]

        sim = memory_linear(feat, img_ids).float()
        sim_exp = torch.exp(sim * self.scale)

        # Calculate mask for intra-camera matching and inter-camera matching
        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), img_ids, cam_ids, sim.device)

        mask_neighbor = torch.zeros_like(sim)
        for i, idx in enumerate(kwargs.get('labels')):
            uids = self.id2uid[idx.item()].copy()
            uids.remove(img_ids[i].item())
            mask_neighbor[i, uids] = 1

        mask_gt_neighbor_intra = mask_neighbor * mask_intra
        mask_gt_neighbor_inter = mask_neighbor * mask_inter

        # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
        # Compute masks for intra-camera neighbors
        sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
        mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        num_neighbor_intra = mask_neighbor_intra.sum(dim=1)

        # Activate intra-camera candidates
        sim_exp_intra = sim_exp * mask_intra
        score_intra = sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)
        score_intra = score_intra.clamp_min(1e-8)
        intra_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
        intra_loss = intra_loss.div(num_neighbor_intra)
        metric.update({'intra': intra_loss.mean().item()})

        # Weighting intra-camera neighborhood consistency
        weight_intra = sim.data * mask_neighbor_intra
        weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        intra_loss = intra_loss.mul(weight_intra).mean()

        # Instance consistency
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log().mean()
        metric.update({'ins': ins_loss.item()})

        # -------------------------- Inter-Camera Neighborhood Loss --------------------------#
        # Compute masks for inter-camera neighbors
        sim_inter = (sim.data + 1) * mask_inter - 1
        nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        mask_neighbor_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        num_neighbor_inter = mask_neighbor_inter.sum(dim=1)

        # Activate inter-camera candidates
        sim_exp_inter = sim_exp * mask_inter
        score_inter = sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True)
        score_inter = score_inter.clamp_min(1e-8)
        inter_loss = -score_inter.log().mul(mask_neighbor_inter).sum(dim=1)
        inter_loss = inter_loss.div(num_neighbor_inter)
        metric.update({'inter': inter_loss.mean().item()})

        # Weighting inter-camera neighborhood consistency
        weight_inter = sim.data * mask_neighbor_inter
        weight_inter = weight_inter.sum(dim=1) / num_neighbor_inter
        weight_inter = torch.where(weight_inter > self.thresh, 1, 0)
        inter_loss = inter_loss.mul(weight_inter).mean()

        loss = ins_loss + intra_loss * 1.0 + inter_loss * 0.8

        self.mask_instance = mask_instance.bool()
        self.mask_intra = mask_intra.bool()
        self.mask_neighbor_intra = mask_neighbor_intra.bool()
        self.mask_inter = mask_inter.bool()
        self.mask_neighbor_inter = mask_neighbor_inter.bool()

        return loss, metric

    def cam_agnostic_mem_loss(self, feat, **kwargs):
        img_ids = kwargs.get('img_ids')[feat.size(0):]

        sim = self.memory_linear(feat, img_ids).float()
        mask_instance = torch.zeros_like(sim)
        mask_instance[torch.arange(sim.size(0)), img_ids] = 1

        sim_neighbor = (sim.data + 1) * (1 - mask_instance) - 1
        nearest = sim_neighbor.max(dim=1, keepdim=True)[0]
        mask_neighbor = torch.gt(sim_neighbor, nearest * self.neighbor_eps)
        num_neighbor = mask_neighbor.sum(dim=1)

        weight = sim.data * mask_neighbor
        weight = weight.sum(dim=1) / num_neighbor
        weight = torch.where(weight > self.thresh, weight, torch.zeros_like(weight))

        score = F.log_softmax(sim * self.scale, dim=1)
        loss = -score.mul(mask_neighbor).sum(dim=1).div(num_neighbor)
        metric = {'neighbor': loss.mean().item()}

        loss = loss.mul(weight).mean()
        loss -= score.masked_select(mask_instance.bool()).mean()
        return loss, metric

    def compute_mask(self, size, img_ids, cam_ids, device):
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter
