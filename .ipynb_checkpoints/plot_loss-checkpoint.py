"""
plot_loss.py
─────────────────────────────────────────────────────────────────────────────
학습 결과 시각화 스크립트.

사용:
  python plot_loss.py --log_dir ./weights/ecg_jepa_heedb_20250326_120000

출력:
  loss_curve.png  — epoch별 loss 곡선
  콘솔에 학습 요약 출력
"""

import argparse
import os
import glob
import torch
import numpy as np


def parse_log(log_path: str) -> tuple[list, list]:
    """
    pretrain_{ts}.log 파일에서 epoch, loss 파싱.

    로그 형식:
      000001/0100  0.12345  6.000e-04   ...
    """
    epochs, losses = [], []
    with open(log_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # "epoch/total" 형태인 줄 찾기
            if '/' in parts[0]:
                try:
                    epoch = int(parts[0].split('/')[0])
                    loss  = float(parts[1])
                    epochs.append(epoch)
                    losses.append(loss)
                except (ValueError, IndexError):
                    continue
    return epochs, losses


def load_loss_csv(csv_path: str) -> tuple[list, list]:
    """loss_history.txt (epoch,loss 형식) 파싱."""
    epochs, losses = [], []
    with open(csv_path) as f:
        next(f)   # header skip
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                epochs.append(int(parts[0]))
                losses.append(float(parts[1]))
    return epochs, losses


def print_summary(log_dir: str, epochs: list, losses: list):
    """학습 결과 요약 출력."""
    if not losses:
        print('No loss data found.')
        return

    best_epoch = epochs[losses.index(min(losses))]
    best_loss  = min(losses)
    last_loss  = losses[-1]
    last_epoch = epochs[-1]

    # 체크포인트 목록
    ckpts = sorted(glob.glob(os.path.join(log_dir, '*.pth')))

    print('\n' + '='*60)
    print(f'  Training Summary')
    print('='*60)
    print(f'  Log dir      : {log_dir}')
    print(f'  Total epochs : {last_epoch}')
    print(f'  Best loss    : {best_loss:.5f}  (epoch {best_epoch})')
    print(f'  Last loss    : {last_loss:.5f}  (epoch {last_epoch})')
    print(f'\n  Checkpoints saved ({len(ckpts)}):')
    for p in ckpts:
        size_mb = os.path.getsize(p) / 1e6
        name    = os.path.basename(p)
        star    = ' ★ best' if name == 'best.pth' else \
                  ' (last)' if name == 'last.pth' else ''
        print(f'    {name:20s}  {size_mb:6.1f} MB{star}')
    print('='*60 + '\n')


def plot_loss(epochs: list, losses: list, save_path: str):
    """
    matplotlib이 있으면 loss 곡선 저장, 없으면 ASCII 그래프 출력.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, losses, linewidth=1.5, color='steelblue', label='train loss')

        best_idx = losses.index(min(losses))
        ax.scatter(epochs[best_idx], losses[best_idx],
                   color='crimson', s=80, zorder=5,
                   label=f'best: {losses[best_idx]:.5f} (ep{epochs[best_idx]})')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('SmoothL1 Loss')
        ax.set_title('ECG-JEPA Pretraining Loss')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f'Loss curve saved → {save_path}')

    except ImportError:
        # matplotlib 없으면 ASCII
        _ascii_plot(epochs, losses)


def _ascii_plot(epochs: list, losses: list, width: int = 60, height: int = 15):
    """터미널 ASCII 손실 곡선."""
    if not losses:
        return

    min_l, max_l = min(losses), max(losses)
    rng          = max_l - min_l or 1.0

    print(f'\n  Loss curve  (min={min_l:.5f}, max={max_l:.5f})')
    print(f'  {"─"*width}')

    grid = [[' '] * width for _ in range(height)]

    for i, (ep, l) in enumerate(zip(epochs, losses)):
        x = int((i / max(len(epochs) - 1, 1)) * (width - 1))
        y = int((1 - (l - min_l) / rng) * (height - 1))
        grid[y][x] = '●'

    for row in grid:
        print('  |' + ''.join(row) + '|')
    print(f'  {"─"*width}')
    print(f'  ep1{" "*(width-8)}ep{epochs[-1]}')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True,
                        help='체크포인트가 저장된 디렉토리 경로')
    args = parser.parse_args()

    log_dir = args.log_dir

    # loss_history.txt 우선, 없으면 .log 파싱
    csv_path = os.path.join(log_dir, 'loss_history.txt')
    if os.path.exists(csv_path):
        epochs, losses = load_loss_csv(csv_path)
    else:
        log_files = glob.glob(os.path.join(log_dir, 'pretrain_*.log'))
        if not log_files:
            print(f'No log files found in {log_dir}')
            return
        epochs, losses = parse_log(sorted(log_files)[-1])

    print_summary(log_dir, epochs, losses)
    plot_loss(epochs, losses, os.path.join(log_dir, 'loss_curve.png'))


if __name__ == '__main__':
    main()