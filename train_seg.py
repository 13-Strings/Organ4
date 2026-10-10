"""
Stage 3 —— 在 HTS-Seg 上做任务微调：球体分割。

编码器从 Stage2 的域适配权重加载（没有则回落到 Stage1 的 MAE 权重），
分割头随机初始化。训练用 BCE + Dice 混合损失，验证/测试用与 baseline 完全一致的指标。

用法:
    /data/user/miniconda3/envs/mumo/bin/python train_seg.py

断点续训: 直接重跑同一命令，从 {DIR_STAGE3}/ckpt/last.pt 恢复。

产物:
    {DIR_STAGE3}/ckpt/{last,best}.pt
    {DIR_STAGE3}/train_log.jsonl
    {DIR_STAGE3}/metrics.json
    {DIR_STAGE3}/eval/…                （AUTO_EVAL_AFTER_TRAIN=True 时自动跑）
"""
import setproctitle
setproctitle.setproctitle("python3") 
import config  # noqa: F401  —— 必须最先 import

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import metrics as M
import boundary
import utils
from data.hts import build_hts_datasets
from eval_seg import build_seg_model, evaluate_model, load_encoder_weights, \
    predict_loader


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------


def make_loss():
    """返回 (loss_fn, 说明字符串)。loss_fn(logits, mask, dist=None)。"""
    def loss_fn(logits, mask, dist=None):
        parts = []
        if 'bce' in config.SEG_LOSS:
            # pos_weight 必须和 logits 同设备，因此在调用时按需创建
            pos_weight = None
            if config.SEG_BCE_POS_WEIGHT != 1.0:
                pos_weight = torch.tensor([config.SEG_BCE_POS_WEIGHT],
                                          dtype=torch.float32,
                                          device=logits.device)
            if 'bnd' in config.SEG_LOSS and config.SEG_BND_MODE == 'band':
                # 边界带加权 BCE：把梯度集中在 GT 轮廓附近
                parts.append(('bce_bnd', boundary.boundary_weighted_bce(
                    logits, mask, dist,
                    sigma=config.SEG_BND_SIGMA, alpha=config.SEG_BND_ALPHA,
                    pos_weight=pos_weight)))
            else:
                parts.append(('bce', F.binary_cross_entropy_with_logits(
                    logits, mask, pos_weight=pos_weight)))
        if 'dice' in config.SEG_LOSS:
            parts.append(('dice', soft_dice_loss(logits, mask)))
        if 'bnd' in config.SEG_LOSS and config.SEG_BND_MODE == 'kervadec':
            parts.append(('bnd', boundary.kervadec_boundary_loss(
                torch.sigmoid(logits), mask, dist)))
        total = 0.0
        for name, v in parts:
            if name == 'dice':
                total = total + config.SEG_DICE_WEIGHT * v
            elif name in ('bce_bnd', 'bnd'):
                total = total + config.SEG_BND_WEIGHT * v
            else:
                total = total + v
        return total, {n: float(v.detach()) for n, v in parts}

    return loss_fn, config.SEG_LOSS


def soft_dice_loss(logits, mask, eps=1e-6):
    """1 - soft Dice，直接在前景（球体）概率上算，缓解类别极不均衡。"""
    prob = torch.sigmoid(logits)
    num = 2 * (prob * mask).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def main():
    utils.apply_cli_overrides()
    utils.seed_everything(config.SEG_SEED)

    out_dir = utils.ensure_dir(config.DIR_STAGE3)
    ckpt_dir = utils.ensure_dir(os.path.join(out_dir, 'ckpt'))
    utils.copy_config_snapshot(out_dir)
    log = utils.Logger(os.path.join(out_dir, 'train_log.jsonl'))
    device = config.DEVICE

    # # ---------------- 数据（划分与 baseline 一致） ----------------
    train_ds, val_ds, test_ds, meta = build_hts_datasets()
    utils.write_json(os.path.join(out_dir, 'split_meta.json'),
                     {k: v for k, v in meta.items() if k != 'test_stems'})
    log.log('Stage3 启动', subset=meta['subset'], n_total=meta['n_total'],
            n_train=meta['n_train'], n_val=meta['n_val'], n_test=meta['n_test'],
            img_size=meta['img_size'], device=device)

    train_loader = utils.build_loader(train_ds, config.SEG_BATCH_SIZE,
                                      shuffle=True, seed=config.SEG_SEED,
                                      drop_last=True)
    val_loader = utils.build_loader(val_ds, utils.eval_batch_size(), shuffle=False,
                                    seed=config.SEG_SEED)
    steps_per_epoch = len(train_loader)

    if steps_per_epoch == 0:
        raise SystemExit(
            f"steps_per_epoch=0：SEG_BATCH_SIZE={config.SEG_BATCH_SIZE} 比训练集"
            f"({len(train_ds)}) 还大，drop_last=True 会把所有 batch 丢掉，请调小。")

    total_steps = steps_per_epoch * config.SEG_EPOCHS
    warmup_steps = steps_per_epoch * config.SEG_WARMUP_EPOCHS

    # # ---------------- 模型 ----------------
    model = build_seg_model(device)
    if getattr(config, 'SEG_INIT', 'auto') == 'scratch':
        # 数据效率实验的对照组:强制随机初始化。必须先于 auto 分支判断 ——
        # 否则 Stage1/2 训好后对照组会静默加载预训练权重,两组都变成"预训练",
        # 整个实验失去意义。
        enc_path = None
        log.log('SEG_INIT=scratch:编码器强制随机初始化(预训练对照组)')
    else:
        enc_path = load_encoder_weights(model.encoder, device, log=log)
    log.log('模型就绪', trainable=utils.format_params(model),
            encoder_from=enc_path, frozen_layers=config.SEG_FREEZE_LAYERS)

    # 编码器 / 解码器分组学习率
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.seg_head.parameters())
    optimizer = torch.optim.AdamW([
        {'params': enc_params, 'lr': config.SEG_LR_ENCODER},
        {'params': head_params, 'lr': config.SEG_LR_DECODER},
    ], weight_decay=config.SEG_WEIGHT_DECAY, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda') if config.SEG_AMP else None

    loss_fn, loss_desc = make_loss()
    log.log('损失函数', loss=loss_desc,
            pos_weight=config.SEG_BCE_POS_WEIGHT,
            dice_weight=config.SEG_DICE_WEIGHT)

    # ---------------- 断点续训 ----------------
    # best 记的是验证集前景 Dice（∈[0,1]），用 -1.0 当"比任何结果都差"的哨兵，
    # 避免用 -inf 时 round() 报 OverflowError
    start_epoch, global_step, best, _ = 0, 0, -1.0, {}
    resume_path = utils.resolve_resume(config.RESUME, ckpt_dir)
    if resume_path:
        start_epoch, global_step, best, _ = utils.load_checkpoint(
            resume_path, model, optimizer, None, scaler)
        log.log('已恢复断点', path=resume_path, epoch=start_epoch,
                step=global_step, best=round(best, 5))
    else:
        log.log('未发现断点，从零开始')

    stepper = utils.OOMAdaptiveStep(optimizer, scaler=scaler, chunks=1,
                                    grad_clip=config.SEG_GRAD_CLIP, device=device)
    interrupt = utils.InterruptHandler()
    early_stop = utils.EarlyStopping(config.SEG_EARLY_STOP_PATIENCE,
                                     config.SEG_EARLY_STOP_MIN_DELTA,
                                     mode='max')
    early_stop.seed_best(best)
    if early_stop.enabled:
        log.log('早停已启用', patience=early_stop.patience,
                min_delta=early_stop.min_delta, mode=early_stop.mode,
                judge='验证集前景 Dice')
    torch.cuda.reset_peak_memory_stats(device)

    run, run_n = {}, 0
    t_last = time.time()
    early_stopped = False
    epoch = start_epoch - 1

    for epoch in range(start_epoch, config.SEG_EPOCHS):
        if interrupt.should_stop:
            break
        model.train()

        for batch in train_loader:
            _set_lr(optimizer, global_step, total_steps, warmup_steps)

            def step_loss(mb, _model=model, _dev=device):
                mb = utils.move_to_device(mb, _dev)
                logits = _model(mb['image'])
                loss, _parts = loss_fn(logits, mb['mask'],
                                       mb.get('dist', None))
                return loss

            loss_val, ok = stepper(step_loss, batch)
            if ok:
                run['loss'] = run.get('loss', 0.0) + loss_val
                run_n += 1
            global_step += 1

            if global_step % config.LOG_EVERY_STEPS == 0:
                now = time.time()
                log.log('train', epoch=epoch, step=global_step,
                        loss=round(run.get('loss', 0.0) / max(1, run_n), 5),
                        lr=f"{optimizer.param_groups[1]['lr']:.2e}",
                        sec_per_step=round((now - t_last) / config.LOG_EVERY_STEPS, 3),
                        mem=utils.gpu_mem_str(device))
                run, run_n, t_last = {}, 0, now

            if config.SAVE_EVERY_STEPS > 0 and \
                    global_step % config.SAVE_EVERY_STEPS == 0:
                utils.save_checkpoint(os.path.join(ckpt_dir, 'step_latest.pt'),
                                      model, optimizer, None, scaler, epoch,
                                      global_step, best)
                os.replace(os.path.join(ckpt_dir, 'step_latest.pt'),
                           os.path.join(ckpt_dir, f'step_{global_step:08d}.pt'))
                utils.cleanup_snapshots(ckpt_dir, config.KEEP_LAST_N)

            if interrupt.should_stop:
                break

        # ---- 每 epoch 在验证集上按 baseline 口径评估，取前景 Dice 最优 ----
        val = predict_loader(model, val_loader, device)
        val_report = M.evaluate_segmentation([g for g in val['gt']],
                                             [p for p in val['pred']])
        score = val_report['per_image']['fg_dice']['mean']
        # 边界敏感指标:区域指标(Dice)看不出轮廓误差,这里逐轮记录 ASSD/HD95/bIoU,
        # 用于证明边界损失确实改善了轮廓(而不是只把 Dice 抬了 0.001)。
        val_bnd = boundary.evaluate_boundary([g for g in val['gt']],
                                             [p for p in val['pred']], band_d=2)
        is_best = score > best
        if is_best:
            best = score
        # 先喂早停再写日志：否则 es_best / bad_epochs 都慢一轮，
        # 日志里那两行也就不再是"本轮早停依据"的快照。
        es_stopped = early_stop.step(score)
        log.log('epoch 结束', epoch=epoch,
                val_fg_dice=round(score, 5),
                val_jaccard=round(val_report['per_image']['jaccard']['mean'], 5),
                val_accuracy=round(val_report['per_image']['accuracy']['mean'], 5),
                val_assd=round(val_bnd['assd']['mean'], 4),
                val_hd95=round(val_bnd['hd95']['mean'], 4),
                val_biou2=round(val_bnd['biou2']['mean'], 4),
                best=round(best, 5), is_best=is_best,
                es_score=score, es_best=early_stop.best,
                bad_epochs=early_stop.bad_epochs,
                mem=utils.gpu_mem_str(device))
        t_last = time.time()

        if config.SAVE_EVERY_EPOCH:
            utils.save_checkpoint(os.path.join(ckpt_dir, 'last.pt'), model,
                                  optimizer, None, scaler, epoch + 1,
                                  global_step, best)
            if is_best:
                utils.save_checkpoint(os.path.join(ckpt_dir, 'best.pt'), model,
                                      optimizer, None, scaler, epoch + 1,
                                      global_step, best)

        if es_stopped:
            early_stopped = True
            log.log('早停触发', epoch=epoch, bad_epochs=early_stop.bad_epochs,
                    patience=early_stop.patience,
                    min_delta=early_stop.min_delta,
                    best_val_fg_dice=round(early_stop.best, 5))
            break

        if interrupt.should_stop:
            break

    # # ---------------- 收尾：用最优权重在测试集上评估 ----------------
    best_ckpt = os.path.join(ckpt_dir, 'best.pt')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, 'last.pt')

    summary = {
        'stage': 'stage3_hts',
        'epochs_done': min(epoch + 1, config.SEG_EPOCHS),
        'global_step': global_step,
        'best_val_fg_dice': best if np.isfinite(best) else None,
        'es_best': early_stop.best,
        'n_train': meta['n_train'], 'n_val': meta['n_val'],
        'n_test': meta['n_test'], 'subset': meta['subset'],
        'batch_size': config.SEG_BATCH_SIZE,
        'encoder_from': enc_path,
        'peak_gpu_gb': round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        'oom_skipped_steps': stepper.skip_count,
        'best_ckpt': best_ckpt,
        'interrupted': interrupt.should_stop,
        'early_stopped': early_stopped,
    }
    utils.write_json(os.path.join(out_dir, 'metrics.json'), summary)
    log.log('Stage3 训练完成', **summary)

    if config.AUTO_EVAL_AFTER_TRAIN:
        log.log('开始测试集评估')
        state = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model', state), strict=False)
        test_loader = utils.build_loader(test_ds, utils.eval_batch_size(),
                                         shuffle=False, seed=config.SEG_SEED)
        out = evaluate_model(model, test_loader, device,
                             out_dir=os.path.join(out_dir, 'eval'),
                             split_name='test', log=log)
        log.log('测试集结果',
                **{k: round(out['metrics']['per_image'][k]['mean'], 4)
                   for k in M.TABLE2_ORDER})
        utils.write_json(os.path.join(out_dir, 'test_summary.json'),
                         out['metrics']['per_image'])

    interrupt.restore()
    log.close()

# def main():
#     utils.apply_cli_overrides()
#     utils.seed_everything(config.SEED)

#     out_dir = utils.ensure_dir(config.DIR_STAGE3)
#     ckpt_dir = utils.ensure_dir(os.path.join(out_dir, 'ckpt'))
#     utils.copy_config_snapshot(out_dir)
#     log = utils.Logger(os.path.join(out_dir, 'train_log.jsonl'))
#     device = config.DEVICE

#     # # ---------------- 数据（划分与 baseline 一致） ----------------
#     train_ds, val_ds, test_ds, meta = build_hts_datasets()
#     # utils.write_json(os.path.join(out_dir, 'split_meta.json'),
#     #                  {k: v for k, v in meta.items() if k != 'test_stems'})
#     # log.log('Stage3 启动', subset=meta['subset'], n_total=meta['n_total'],
#     #         n_train=meta['n_train'], n_val=meta['n_val'], n_test=meta['n_test'],
#     #         img_size=meta['img_size'], device=device)

#     # train_loader = utils.build_loader(train_ds, config.SEG_BATCH_SIZE,
#     #                                   shuffle=True, seed=config.SEG_SEED,
#     #                                   drop_last=True)
#     # val_loader = utils.build_loader(val_ds, utils.eval_batch_size(), shuffle=False,
#     #                                 seed=config.SEG_SEED)
#     # steps_per_epoch = len(train_loader)

#     # if steps_per_epoch == 0:
#     #     raise SystemExit(
#     #         f"steps_per_epoch=0：SEG_BATCH_SIZE={config.SEG_BATCH_SIZE} 比训练集"
#     #         f"({len(train_ds)}) 还大，drop_last=True 会把所有 batch 丢掉，请调小。")

#     # total_steps = steps_per_epoch * config.SEG_EPOCHS
#     # warmup_steps = steps_per_epoch * config.SEG_WARMUP_EPOCHS

#     # # ---------------- 模型 ----------------
#     model = build_seg_model(device)
#     # enc_path = load_encoder_weights(model.encoder, device, log=log)
#     # log.log('模型就绪', trainable=utils.format_params(model),
#     #         encoder_from=enc_path, frozen_layers=config.SEG_FREEZE_LAYERS)

#     # # 编码器 / 解码器分组学习率
#     # enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
#     # head_params = list(model.seg_head.parameters())
#     # optimizer = torch.optim.AdamW([
#     #     {'params': enc_params, 'lr': config.SEG_LR_ENCODER},
#     #     {'params': head_params, 'lr': config.SEG_LR_DECODER},
#     # ], weight_decay=config.SEG_WEIGHT_DECAY, betas=(0.9, 0.999))
#     # scaler = torch.amp.GradScaler('cuda') if config.SEG_AMP else None

#     # loss_fn, loss_desc = make_loss()
#     # log.log('损失函数', loss=loss_desc,
#     #         pos_weight=config.SEG_BCE_POS_WEIGHT,
#     #         dice_weight=config.SEG_DICE_WEIGHT)

#     # # ---------------- 断点续训 ----------------
#     # # best 记的是验证集前景 Dice（∈[0,1]），用 -1.0 当"比任何结果都差"的哨兵，
#     # # 避免用 -inf 时 round() 报 OverflowError
#     # start_epoch, global_step, best, _ = 0, 0, -1.0, {}
#     # resume_path = utils.resolve_resume(config.RESUME, ckpt_dir)
#     # if resume_path:
#     #     start_epoch, global_step, best, _ = utils.load_checkpoint(
#     #         resume_path, model, optimizer, None, scaler)
#     #     log.log('已恢复断点', path=resume_path, epoch=start_epoch,
#     #             step=global_step, best=round(best, 5))
#     # else:
#     #     log.log('未发现断点，从零开始')

#     # stepper = utils.OOMAdaptiveStep(optimizer, scaler=scaler, chunks=1,
#     #                                 grad_clip=config.SEG_GRAD_CLIP, device=device)
#     interrupt = utils.InterruptHandler()
#     # early_stop = utils.EarlyStopping(config.SEG_EARLY_STOP_PATIENCE,
#     #                                  config.SEG_EARLY_STOP_MIN_DELTA,
#     #                                  mode='max')
#     # early_stop.seed_best(best)
#     # if early_stop.enabled:
#     #     log.log('早停已启用', patience=early_stop.patience,
#     #             min_delta=early_stop.min_delta, mode=early_stop.mode,
#     #             judge='验证集前景 Dice')
#     # torch.cuda.reset_peak_memory_stats(device)

#     # run, run_n = {}, 0
#     # t_last = time.time()
#     # early_stopped = False
#     # epoch = start_epoch - 1

#     # for epoch in range(start_epoch, config.SEG_EPOCHS):
#     #     if interrupt.should_stop:
#     #         break
#     #     model.train()

#         # for batch in train_loader:
#         #     _set_lr(optimizer, global_step, total_steps, warmup_steps)

#         #     def step_loss(mb, _model=model, _dev=device):
#         #         mb = utils.move_to_device(mb, _dev)
#         #         logits = _model(mb['image'])
#         #         loss, _parts = loss_fn(logits, mb['mask'])
#         #         return loss

#         #     loss_val, ok = stepper(step_loss, batch)
#         #     if ok:
#         #         run['loss'] = run.get('loss', 0.0) + loss_val
#         #         run_n += 1
#         #     global_step += 1

#         #     if global_step % config.LOG_EVERY_STEPS == 0:
#         #         now = time.time()
#         #         log.log('train', epoch=epoch, step=global_step,
#         #                 loss=round(run.get('loss', 0.0) / max(1, run_n), 5),
#         #                 lr=f"{optimizer.param_groups[1]['lr']:.2e}",
#         #                 sec_per_step=round((now - t_last) / config.LOG_EVERY_STEPS, 3),
#         #                 mem=utils.gpu_mem_str(device))
#         #         run, run_n, t_last = {}, 0, now

#         #     if config.SAVE_EVERY_STEPS > 0 and \
#         #             global_step % config.SAVE_EVERY_STEPS == 0:
#         #         utils.save_checkpoint(os.path.join(ckpt_dir, 'step_latest.pt'),
#         #                               model, optimizer, None, scaler, epoch,
#         #                               global_step, best)
#         #         os.replace(os.path.join(ckpt_dir, 'step_latest.pt'),
#         #                    os.path.join(ckpt_dir, f'step_{global_step:08d}.pt'))
#         #         utils.cleanup_snapshots(ckpt_dir, config.KEEP_LAST_N)

#         #     if interrupt.should_stop:
#         #         break

#         # ---- 每 epoch 在验证集上按 baseline 口径评估，取前景 Dice 最优 ----
#         # val = predict_loader(model, val_loader, device)
#         # val_report = M.evaluate_segmentation([g for g in val['gt']],
#         #                                      [p for p in val['pred']])
#         # score = val_report['per_image']['fg_dice']['mean']
#         # is_best = score > best
#         # if is_best:
#         #     best = score
#         # # 先喂早停再写日志：否则 es_best / bad_epochs 都慢一轮，
#         # # 日志里那两行也就不再是"本轮早停依据"的快照。
#         # es_stopped = early_stop.step(score)
#         # log.log('epoch 结束', epoch=epoch,
#         #         val_fg_dice=round(score, 5),
#         #         val_jaccard=round(val_report['per_image']['jaccard']['mean'], 5),
#         #         val_accuracy=round(val_report['per_image']['accuracy']['mean'], 5),
#         #         best=round(best, 5), is_best=is_best,
#         #         es_score=score, es_best=early_stop.best,
#         #         bad_epochs=early_stop.bad_epochs,
#         #         mem=utils.gpu_mem_str(device))
#         # t_last = time.time()

#         # if config.SAVE_EVERY_EPOCH:
#         #     utils.save_checkpoint(os.path.join(ckpt_dir, 'last.pt'), model,
#         #                           optimizer, None, scaler, epoch + 1,
#         #                           global_step, best)
#         #     if is_best:
#         #         utils.save_checkpoint(os.path.join(ckpt_dir, 'best.pt'), model,
#         #                               optimizer, None, scaler, epoch + 1,
#         #                               global_step, best)

#         # if es_stopped:
#         #     early_stopped = True
#         #     log.log('早停触发', epoch=epoch, bad_epochs=early_stop.bad_epochs,
#         #             patience=early_stop.patience,
#         #             min_delta=early_stop.min_delta,
#         #             best_val_fg_dice=round(early_stop.best, 5))
#         #     break

#         # if interrupt.should_stop:
#         #     break

#     # # ---------------- 收尾：用最优权重在测试集上评估 ----------------
#     best_ckpt = os.path.join(ckpt_dir, 'best.pt')
#     if not os.path.exists(best_ckpt):
#         best_ckpt = os.path.join(ckpt_dir, 'last.pt')

#     # summary = {
#     #     'stage': 'stage3_hts',
#     #     'epochs_done': min(epoch + 1, config.SEG_EPOCHS),
#     #     'global_step': global_step,
#     #     'best_val_fg_dice': best if np.isfinite(best) else None,
#     #     'es_best': early_stop.best,
#     #     'n_train': meta['n_train'], 'n_val': meta['n_val'],
#     #     'n_test': meta['n_test'], 'subset': meta['subset'],
#     #     'batch_size': config.SEG_BATCH_SIZE,
#     #     'encoder_from': enc_path,
#     #     'peak_gpu_gb': round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
#     #     'oom_skipped_steps': stepper.skip_count,
#     #     'best_ckpt': best_ckpt,
#     #     'interrupted': interrupt.should_stop,
#     #     'early_stopped': early_stopped,
#     # }
#     # utils.write_json(os.path.join(out_dir, 'metrics.json'), summary)
#     # log.log('Stage3 训练完成', **summary)

#     if config.AUTO_EVAL_AFTER_TRAIN:
#         log.log('开始测试集评估')
#         state = torch.load(best_ckpt, map_location='cpu', weights_only=False)
#         model.load_state_dict(state.get('model', state), strict=False)
#         test_loader = utils.build_loader(test_ds, utils.eval_batch_size(),
#                                          shuffle=False, seed=config.SEG_SEED)
#         out = evaluate_model(model, test_loader, device,
#                              out_dir=os.path.join(out_dir, 'eval'),
#                              split_name='test', log=log)
#         log.log('测试集结果',
#                 **{k: round(out['metrics']['per_image'][k]['mean'], 4)
#                    for k in M.TABLE2_ORDER})
#         utils.write_json(os.path.join(out_dir, 'test_summary.json'),
#                          out['metrics']['per_image'])

#     interrupt.restore()
#     log.close()


def _set_lr(optimizer, step, total_steps, warmup_steps):
    """编码器与分割头各自按同一进度做 warmup + cosine。"""
    optimizer.param_groups[0]['lr'] = utils.cosine_lr(
        step, total_steps, config.SEG_LR_ENCODER,
        config.SEG_LR_ENCODER * 0.02, warmup_steps)
    optimizer.param_groups[1]['lr'] = utils.cosine_lr(
        step, total_steps, config.SEG_LR_DECODER,
        config.SEG_LR_DECODER * 0.02, warmup_steps)


if __name__ == '__main__':
    sys.exit(main())
