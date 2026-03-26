"""
pretrain.py  —  ECG-JEPA pretraining on HEEDB (H200 × 8, from scratch)

실행: ./run_h200.sh
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm
import yaml

from dataset import HEEDBDataset
from models  import build_model
from utils   import setup_ddp, is_main, all_reduce_mean, setup_logger


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    def __init__(self, d: dict):
        super().__init__({k: DotDict(v) if isinstance(v, dict) else v
                          for k, v in d.items()})

def load_config(path: str) -> DotDict:
    with open(path) as f:
        return DotDict(yaml.safe_load(f))

def apply_overrides(cfg: DotDict, overrides: list):
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip('-')
        val = overrides[i + 1]
        i += 2
        keys = key.split('.')
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        node[keys[-1]] = val

def cfg_to_dict(cfg) -> dict:
    if isinstance(cfg, dict):
        return {k: cfg_to_dict(v) for k, v in cfg.items()}
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, raw_model, optimizer, epoch, global_step, best_loss, cfg):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'encoder':        raw_model.encoder.state_dict(),
        'target_encoder': raw_model.target_encoder.state_dict(),
        'optimizer':      optimizer.state_dict(),
        'epoch':          epoch,
        'global_step':    global_step,
        'best_loss':      best_loss,
        'config':         cfg_to_dict(cfg),
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pretrain_heedb.yaml')
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config)
    if overrides:
        apply_overrides(cfg, overrides)

    local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # rank 0만 출력/tqdm 사용
    main_process = is_main()

    # rank 0이 아니면 stdout을 완전히 닫아서 tqdm 중복 방지
    #if not main_process:
    #    sys.stdout = open(os.devnull, 'w')
    #    sys.stderr = open(os.devnull, 'w')

    # ── 저장 디렉토리 & 로거 ──────────────────────────────────────────────────
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = cfg.train.save_dir or f'./weights/ecg_jepa_heedb_{ts}'
    logger   = setup_logger(save_dir)
    log      = logger.info

    log('=' * 70)
    log(f'  ECG-JEPA Pretraining  |  {world_size} GPU(s)  |  {ts}')
    log(f'  GPUs    : {os.environ.get("CUDA_VISIBLE_DEVICES", "all")}')
    log(f'  Save dir: {save_dir}')
    log('=' * 70)
    log('Config:\n' + yaml.dump(cfg_to_dict(cfg), default_flow_style=False).strip())

    # ── Dataset ───────────────────────────────────────────────────────────────
    t0      = time.time()
    dataset = HEEDBDataset(
        root_dir      = cfg.data.root_dir,
        csv_path      = cfg.data.csv_path,
        max_nan_ratio = cfg.data.max_nan_ratio,
        n_leads       = cfg.data.n_leads,
    )
    log(f'Dataset  : {len(dataset):,} samples  ({time.time()-t0:.1f}s)')

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader  = DataLoader(
        dataset,
        batch_size         = cfg.train.batch_size,
        sampler            = sampler,
        shuffle            = (sampler is None),
        num_workers        = cfg.train.num_workers,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = True,
        prefetch_factor    = 2,
    )

    iters_per_epoch = len(loader)
    eff_batch       = cfg.train.batch_size * world_size
    log(f'Batch    : {cfg.train.batch_size} per-GPU × {world_size} = {eff_batch} effective')
    log(f'Steps    : {iters_per_epoch:,}/epoch × {cfg.train.epochs} epochs = '
        f'{iters_per_epoch * cfg.train.epochs:,} total')

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg  = dict(cfg.model)
    model_name = model_cfg.pop('name')
    model = build_model(
        model_name,
        mask_scale = tuple(model_cfg.pop('mask_scale', [0.175, 0.225])),
        **model_cfg,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if world_size > 1 else model
    log(f'Model    : {model_name}  |  {raw_model.n_parameters()/1e6:.1f} M params')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                    if p.requires_grad and 'bias' not in n and p.dim() > 1]},
        {'params': [p for n, p in model.named_parameters()
                    if p.requires_grad and ('bias' in n or p.dim() == 1)],
         'WD_exclude': True, 'weight_decay': 0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))

    # ── Scheduler ─────────────────────────────────────────────────────────────
    total_iters = iters_per_epoch * cfg.train.epochs
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial      = total_iters,
        lr_min         = float(cfg.train.lr_min),
        warmup_lr_init = float(cfg.train.lr_min),
        warmup_t       = cfg.train.warmup_epochs * iters_per_epoch,
        cycle_limit    = 1,
        t_in_epochs    = False,
    )

    def get_momentum(step: int) -> float:
        return cfg.train.ema_start + step * (
            cfg.train.ema_end - cfg.train.ema_start) / total_iters

    scaler       = GradScaler()
    best_loss    = float('inf')
    loss_history = []
    global_step  = 0
    train_start  = time.time()

    # ── 전체 epoch tqdm (rank-0만) ────────────────────────────────────────────
    epoch_bar = tqdm(
        range(cfg.train.epochs),
        desc          = 'Training',
        unit          = 'epoch',
        dynamic_ncols = True,
        position      = 0,
        leave         = True,
    )

    for epoch in epoch_bar:
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = torch.tensor(0.0, device=device)
        t0         = time.time()

        # ── step tqdm (rank-0만) ──────────────────────────────────────────────
        step_bar = tqdm(
            loader,
            desc          = f'Epoch {epoch+1:04d}/{cfg.train.epochs}',
            unit          = 'step',
            dynamic_ncols = True,
            leave         = False,
            position      = 1,
        )

        for wave in step_bar:
            wave = wave.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = model(wave)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step_update(global_step)

            m = get_momentum(global_step)
            with torch.no_grad():
                for p_q, p_k in zip(raw_model.encoder.parameters(),
                                     raw_model.target_encoder.parameters()):
                    p_k.data.mul_(m).add_((1.0 - m) * p_q.detach().data)

            epoch_loss  += loss.detach()
            global_step += 1

            # 현재 step의 loss / lr 실시간 표시
            step_bar.set_postfix(
                loss = f'{loss.item():.4f}',
                lr   = f'{optimizer.param_groups[0]["lr"]:.2e}',
            )

        step_bar.close()

        # ── epoch 집계 ────────────────────────────────────────────────────────
        epoch_loss /= iters_per_epoch
        all_reduce_mean(epoch_loss)
        loss_val = epoch_loss.item()
        loss_history.append(loss_val)

        lr_now  = optimizer.param_groups[0]['lr']
        ep_time = time.time() - t0
        elapsed = time.time() - train_start
        eta     = str(timedelta(seconds=int(
            elapsed / (epoch + 1) * (cfg.train.epochs - epoch - 1))))
        is_best = loss_val < best_loss
        if is_best:
            best_loss = loss_val

        # epoch bar 업데이트
        epoch_bar.set_postfix(
            loss = f'{loss_val:.5f}',
            best = f'{best_loss:.5f}',
            lr   = f'{lr_now:.2e}',
            eta  = eta,
        )

        # 로그 파일 기록
        log(f'epoch {epoch+1:04d}/{cfg.train.epochs}  '
            f'loss={loss_val:.5f}  best={best_loss:.5f}  '
            f'lr={lr_now:.3e}  time={ep_time:.0f}s  eta={eta}'
            + ('  ★' if is_best else ''))

        # ── 체크포인트 ────────────────────────────────────────────────────────
        save_kw = dict(raw_model=raw_model, optimizer=optimizer, epoch=epoch,
                       global_step=global_step, best_loss=best_loss, cfg=cfg)

        if is_best:
            save_checkpoint(os.path.join(save_dir, 'best.pth'), **save_kw)
            log(f'  → best.pth  (loss={best_loss:.5f})')

        if (epoch + 1) % cfg.train.save_freq == 0:
            p = os.path.join(save_dir, f'epoch{epoch+1:04d}.pth')
            save_checkpoint(p, **save_kw)
            log(f'  → epoch{epoch+1:04d}.pth')

    epoch_bar.close()

    # ── 완료 ──────────────────────────────────────────────────────────────────
    save_checkpoint(os.path.join(save_dir, 'last.pth'),
                    raw_model=raw_model, optimizer=optimizer,
                    epoch=cfg.train.epochs-1, global_step=global_step,
                    best_loss=best_loss, cfg=cfg)
    total = str(timedelta(seconds=int(time.time() - train_start)))
    log('')
    log('=' * 70)
    log(f'  Done  |  total={total}  |  best loss={best_loss:.5f}')
    log(f'  Saved → {save_dir}/')
    log('=' * 70)

    with open(os.path.join(save_dir, 'loss_history.txt'), 'w') as f:
        f.write('epoch,loss\n')
        for i, l in enumerate(loss_history, 1):
            f.write(f'{i},{l:.6f}\n')

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()