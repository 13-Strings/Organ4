"""
Stage 1 —— 在 RxRx1 上做 MAE 自监督预训练，得到通用细胞形态表征编码器。

用法:
    cd /data/user/workspace/organ/code
    /data/user/miniconda3/envs/mumo/bin/python train_mae.py

断点续训:
    直接重跑同一条命令即可。会从 {DIR_STAGE1}/ckpt/last.pt 恢复
    （epoch、global_step、优化器、AMP scaler、全部随机数状态）。
    想强制从头训: 把 config.RESUME 改成 'none'，或删掉 ckpt 目录。

产物:
    {DIR_STAGE1}/ckpt/last.pt         断点（每 epoch 覆盖）
    {DIR_STAGE1}/ckpt/best.pt         验证损失最低的权重
    {DIR_STAGE1}/encoder.pt           只含编码器，供 Stage 2/3 加载
    {DIR_STAGE1}/train_log.jsonl      逐步日志
    {DIR_STAGE1}/metrics.json         训练摘要
"""
import setproctitle
setproctitle.setproctitle("python3") 
import config  # noqa: F401  —— 必须最先 import，才能让 CUDA 相关环境变量生效

import os
import sys
import time

import numpy as np
import torch

import utils
from data import build_rxrx1
from models import MAE


def build_optimizer(model, lr, weight_decay):
    """bias / LayerNorm / 位置编码不做权重衰减（MAE 原文做法）。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr, betas=tuple(config.MAE_BETAS))


def main():
    utils.apply_cli_overrides()
    utils.seed_everything(config.SEED)

    out_dir = utils.ensure_dir(config.DIR_STAGE1)
    ckpt_dir = utils.ensure_dir(os.path.join(out_dir, 'ckpt'))
    utils.copy_config_snapshot(out_dir)
    log = utils.Logger(os.path.join(out_dir, 'train_log.jsonl'))

    device = config.DEVICE
    log.log('Stage1 启动', device=device,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            seed=config.SEED, workers=config.NUM_WORKERS)

    # ---------------- 数据 ----------------
    # MAE_DATA='rxrx1'  用 RxRx1 大规模 2D 细胞荧光图(**非器官芯片**,通用形态预训练)
    # MAE_DATA='oocmix' 只用器官芯片图像(OOC 芯片明场 + HTS-Seg 芯片球体,共 5,252 张)
    # 两臂其余配置完全相同,差异只来自预训练语料 —— 用来回答"非要非芯片数据吗"。
    if getattr(config, 'MAE_DATA', 'rxrx1') == 'oocmix':
        from data.ooc_pretrain import build_ooc_pretrain as _build_stage1_ds
        in_chans = 3
    else:
        from data import build_rxrx1 as _build_stage1_ds
        in_chans = len(config.RXRX1_CHANNELS)
    train_ds = _build_stage1_ds(phase='train')
    loader = utils.build_loader(train_ds, config.MAE_BATCH_SIZE, shuffle=True,
                                seed=config.SEED, drop_last=True)
    # 验证信号用训练集的前若干 batch（MAE 不需要独立验证集），只建一次避免重复起 worker
    eval_loader = utils.build_loader(_build_stage1_ds(phase='eval'),
                                     config.MAE_BATCH_SIZE, shuffle=False,
                                     seed=config.SEED)
    steps_per_epoch = len(loader)

    if steps_per_epoch == 0:
        raise SystemExit(
            f"steps_per_epoch=0：MAE_BATCH_SIZE={config.MAE_BATCH_SIZE} 比训练集"
            f"({len(train_ds)}) 还大，drop_last=True 会把所有 batch 丢掉，请调小。")

    total_steps = steps_per_epoch * config.MAE_EPOCHS
    if config.MAE_MAX_STEPS > 0:
        total_steps = min(total_steps, config.MAE_MAX_STEPS)
    warmup_steps = steps_per_epoch * config.MAE_WARMUP_EPOCHS

    log.log('数据集就绪', n_sites=len(train_ds), steps_per_epoch=steps_per_epoch,
            total_steps=total_steps, warmup_steps=warmup_steps,
            channels=(config.RXRX1_CHANNELS if getattr(config, 'MAE_DATA', 'rxrx1') == 'rxrx1' else 'RGB(OOC+HTS-Seg)'))

    # ---------------- 模型 ----------------
    model = MAE(
        img_size=config.MAE_INPUT_SIZE, patch_size=config.MAE_PATCH_SIZE,
        in_chans=in_chans,
        enc_dim=config.MAE_ENC_DIM, enc_depth=config.MAE_ENC_DEPTH,
        enc_heads=config.MAE_ENC_HEADS,
        dec_dim=config.MAE_DEC_DIM, dec_depth=config.MAE_DEC_DEPTH,
        dec_heads=config.MAE_DEC_HEADS,
        mask_ratio=config.MAE_MASK_RATIO,
        norm_pix_loss=config.MAE_NORM_PIX_LOSS,
    ).to(device)

    optimizer = build_optimizer(model, config.MAE_LR, config.MAE_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=config.MAE_AMP) \
        if config.MAE_AMP else None

    log.log('模型就绪', encoder=utils.format_params(model.encoder),
            total=utils.format_params(model))

    # ---------------- 断点续训 ----------------
    start_epoch, global_step, best, _ = 0, 0, float('inf'), {}
    resume_path = utils.resolve_resume(config.RESUME, ckpt_dir)
    if resume_path is None:
        latest = utils.find_latest_snapshot(ckpt_dir)
        if latest:
            resume_path = latest
            log.log('未找到 last.pt，改用最近的 step 快照', path=latest)

    if resume_path:
        start_epoch, global_step, best, _ = utils.load_checkpoint(
            resume_path, model, optimizer, None, scaler)
        log.log('已恢复断点', path=resume_path, epoch=start_epoch,
                global_step=global_step, best=best)
    else:
        log.log('未发现断点，从零开始训练')

    stepper = utils.OOMAdaptiveStep(
        optimizer, scaler=scaler, chunks=config.MAE_ACCUM_STEPS,
        grad_clip=config.MAE_GRAD_CLIP, device=device)
    interrupt = utils.InterruptHandler()
    early_stop = utils.EarlyStopping(config.MAE_EARLY_STOP_PATIENCE,
                                     config.MAE_EARLY_STOP_MIN_DELTA,
                                     mode='min')
    early_stop.seed_best(best)
    if early_stop.enabled:
        log.log('早停已启用', patience=early_stop.patience,
                min_delta=early_stop.min_delta, mode=early_stop.mode)

    torch.cuda.reset_peak_memory_stats(device)

    # ---------------- 训练循环 ----------------
    model.train()
    running, running_n = 0.0, 0
    t_last = time.time()
    stop_all = False
    early_stopped = False
    epoch = start_epoch - 1

    for epoch in range(start_epoch, config.MAE_EPOCHS):
        if stop_all:
            break
        for batch in loader:
            if global_step >= total_steps:
                stop_all = True
                break

            # 手动设置本 step 的学习率（warmup + cosine）
            lr = utils.cosine_lr(global_step, total_steps, config.MAE_LR,
                                 config.MAE_MIN_LR, warmup_steps)
            for g in optimizer.param_groups:
                g['lr'] = lr

            def loss_fn(mb, _model=model, _dev=device):
                mb = utils.move_to_device(mb, _dev)
                loss, _, _ = _model(mb['image'])
                return loss

            loss_val, ok = stepper(loss_fn, batch)
            if ok:
                running += loss_val
                running_n += 1

            global_step += 1

            if global_step % config.LOG_EVERY_STEPS == 0:
                now = time.time()
                avg = running / max(1, running_n)
                log.log('train', epoch=epoch, step=global_step, loss=round(avg, 5),
                        lr=f'{lr:.2e}',
                        sec_per_step=round((now - t_last) / config.LOG_EVERY_STEPS, 3),
                        chunks=stepper.current_chunks,
                        mem=utils.gpu_mem_str(device))
                running, running_n, t_last = 0.0, 0, now

            if config.SAVE_EVERY_STEPS > 0 and \
                    global_step % config.SAVE_EVERY_STEPS == 0:
                utils.save_checkpoint(
                    os.path.join(ckpt_dir, 'step_latest.pt'), model, optimizer,
                    None, scaler, epoch, global_step, best)
                stamp = os.path.join(ckpt_dir, f'step_{global_step:08d}.pt')
                os.replace(os.path.join(ckpt_dir, 'step_latest.pt'), stamp)
                utils.cleanup_snapshots(ckpt_dir, config.KEEP_LAST_N)

            if interrupt.should_stop:
                break

        # ---- epoch 结束：验证 + 落盘 ----
        val_loss = evaluate(model, device, eval_loader)
        is_best = val_loss < best
        best = min(best, val_loss)
        # 先喂早停再写日志：否则 es_best / bad_epochs 都慢一轮，
        # 日志里那两行也就不再是"本轮早停依据"的快照。
        es_stopped = early_stop.step(val_loss)

        if config.SAVE_EVERY_EPOCH:
            utils.save_checkpoint(os.path.join(ckpt_dir, 'last.pt'), model,
                                  optimizer, None, scaler, epoch + 1,
                                  global_step, best)
            if is_best:
                utils.save_checkpoint(os.path.join(ckpt_dir, 'best.pt'), model,
                                      optimizer, None, scaler, epoch + 1,
                                      global_step, best)

        log.log('epoch 结束', epoch=epoch, val_loss=round(val_loss, 5),
                best=round(best, 5), is_best=is_best,
                es_score=val_loss, es_best=early_stop.best,
                bad_epochs=early_stop.bad_epochs,
                mem=utils.gpu_mem_str(device),
                eta_min=round((total_steps - global_step) *
                              (time.time() - t_last) / max(1, config.LOG_EVERY_STEPS) / 60, 1))
        t_last = time.time()

        if es_stopped:
            early_stopped = True
            log.log('早停触发', epoch=epoch, bad_epochs=early_stop.bad_epochs,
                    patience=early_stop.patience,
                    min_delta=early_stop.min_delta,
                    best_val_loss=round(early_stop.best, 5))
            break

        if interrupt.should_stop:
            break

    # ---------------- 收尾 ----------------
    final_path = os.path.join(ckpt_dir, 'best.pt')
    if not os.path.exists(final_path):
        final_path = os.path.join(ckpt_dir, 'last.pt')

    if os.path.exists(final_path):
        utils.load_checkpoint(final_path, model)
    export_encoder(model, os.path.join(out_dir, 'encoder.pt'),
                   keep_decoder=config.MAE_SAVE_DECODER)

    summary = {
        'stage': 'stage1_mae',
        'epochs_done': min(epoch + 1, config.MAE_EPOCHS),
        'global_step': global_step,
        'best_val_loss': best if np.isfinite(best) else None,
        'es_best': early_stop.best,
        'n_sites': len(train_ds),
        'batch_size': config.MAE_BATCH_SIZE,
        'channels': config.RXRX1_CHANNELS,
        'peak_gpu_gb': round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        'oom_skipped_steps': stepper.skip_count,
        'encoder_path': os.path.join(out_dir, 'encoder.pt'),
        'interrupted': interrupt.should_stop,
        'early_stopped': early_stopped,
    }
    utils.write_json(os.path.join(out_dir, 'metrics.json'), summary)
    log.log('Stage1 完成', **summary)
    interrupt.restore()
    log.close()


@torch.no_grad()
def evaluate(model, device, loader, max_batches=20):
    """固定用前若干 batch 的重建损失作验证信号（MAE 无需独立验证集）。"""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        img = batch['image'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.MAE_AMP):
            loss, _, _ = model(img)
        total += float(loss)
        n += 1
    model.train()
    return total / max(1, n)


def export_encoder(model, path, keep_decoder=False):
    """导出编码器权重（下游 Stage 2/3 直接 load）。"""
    state = model.encoder_state_dict()
    utils.ensure_dir(os.path.dirname(path))
    torch.save(state, path)
    if keep_decoder:
        torch.save({'encoder': state, 'model': model.state_dict()},
                   path.replace('.pt', '_full.pt'))


if __name__ == '__main__':
    sys.exit(main())
