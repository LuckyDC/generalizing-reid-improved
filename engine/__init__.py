import os
import torch
import numpy as np
import torch.distributed as dist
from runx.logx import logx
from ignite.engine import Events
from ignite.handlers import Timer
from torch.nn.functional import normalize

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from utils.eval_cmc import eval_rank_list


def get_trainer(model, optimizer, lr_scheduler=None, enable_amp=False, log_period=10, save_interval=10,
                query_loader=None, gallery_loader=None, eval_interval=None):
    # Trainer
    trainer = create_train_engine(model, optimizer, enable_amp)

    # Evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if eval_interval > 0 and query_loader and gallery_loader:
        evaluator = create_eval_engine(model.module if dist.is_initialized() else model)

    # Metric
    timer = Timer(average=True)
    kv_metric = AutoKVMetric()

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
        epoch = engine.state.epoch

        if dist.is_initialized():
            engine.state.dataloader.sampler.set_epoch(epoch)

        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch
        logx.msg('Epoch[{}] completed.'.format(epoch))

        if lr_scheduler is not None:
            lr_scheduler.step()

        checkpoint_flag = (not dist.is_initialized() or dist.get_rank() == 0)
        if epoch % save_interval == 0 and checkpoint_flag:
            state_dict = model.module.state_dict() if dist.is_initialized() else model.state_dict()
            save_path = os.path.join(logx.logdir, 'checkpoint_ep{}.pt'.format(epoch))
            torch.save(state_dict, save_path)
            logx.msg("Model saved at {}".format(save_path))

        if evaluator and epoch % eval_interval == 0 and checkpoint_flag:
            torch.cuda.empty_cache()

            # Extract query and gallery features
            q_feats, q_ids, q_cam = run_eval(evaluator, query_loader)
            g_feats, g_ids, g_cam = run_eval(evaluator, gallery_loader)

            # Calculate evaluation metric
            distance = -torch.mm(normalize(q_feats), normalize(g_feats).transpose(0, 1)).numpy()
            rank_list = np.argsort(distance, axis=1)
            mAP, r1, r5, r10 = eval_rank_list(rank_list, q_ids, q_cam, g_ids, g_cam)
            logx.msg('mAP = %f , r1 precision = %f , r5 precision = %f , r10 precision = %f' % (mAP, r1, r5, r10))

            torch.cuda.empty_cache()
            del q_feats, q_cam, q_ids, g_feats, g_ids, g_cam, distance, rank_list

        if dist.is_initialized():
            dist.barrier()

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()
        kv_metric.update(engine.state.output)

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)

        if iter_in_epoch % log_period == 0:
            batch_size = engine.state.batch['img'].size(0)
            speed = batch_size / timer.value()

            if dist.is_initialized():
                speed *= dist.get_world_size()

            # log output information
            metric_dict = kv_metric.compute()
            msg = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed)
            for k in sorted(metric_dict.keys()):
                msg += "\t%s: %.4f" % (k, metric_dict[k])
            logx.msg(msg)

            if epoch > 5:
                logx.metric('train', metric_dict, iteration)

            kv_metric.reset()
            timer.reset()

    return trainer


def run_eval(evaluator, loader):
    evaluator.run(loader)

    feats = torch.cat(evaluator.state.feat_list, dim=0)
    ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
    cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()

    evaluator.state.feat_list.clear()
    evaluator.state.id_list.clear()
    evaluator.state.cam_list.clear()

    return feats, ids, cams
