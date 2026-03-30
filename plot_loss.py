"""
plot_loss.py
─────────────────────────────────────────────────────────────────────────────
학습 결과 시각화 스크립트. resume으로 생긴 여러 run 폴더를 합쳐서 표시.

사용 예시:
  # weights/ 디렉토리를 통째로 지정 → ecg_jepa_heedb_* 폴더 자동 탐색
  python plot_loss.py --weights_dir ./weights

  # 특정 폴더들을 직접 지정 (여러 개 가능)
  python plot_loss.py --log_dir ./weights/run1 ./weights/run2 ./weights/run3

  # 단일 폴더 (기존 방식 그대로)
  python plot_loss.py --log_dir ./weights/ecg_jepa_heedb_20260326_203353

출력:
  loss_curve.png  — epoch별 loss 곡선 (resume 경계선 포함)
  콘솔에 학습 요약 출력
"""

import argparse
import os
import re
import glob


# ─────────────────────────────────────────────────────────────────────────────
# 로그 파싱
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> list[dict]:
    """
    pretrain_{ts}.log 에서 epoch별 정보 파싱.

    로그 형식:
      2026-03-26 21:58:53  epoch 0001/100  loss=0.04489  best=0.04489  lr=3.080e-05  ...
    """
    records = []
    with open(log_path) as f:
        for line in f:
            if 'epoch' not in line or 'loss=' not in line:
                continue
            try:
                parts = line.split()

                # epoch 번호: "0001/100" 형태 토큰
                ep_token = next(p for p in parts if re.match(r'^\d+/\d+$', p))
                epoch = int(ep_token.split('/')[0])

                # loss=
                loss = float(next(p for p in parts if p.startswith('loss=')).split('=')[1])

                # lr= (없으면 None)
                lr_tok = next((p for p in parts if p.startswith('lr=')), None)
                lr = float(lr_tok.split('=')[1]) if lr_tok else None

                records.append({'epoch': epoch, 'loss': loss, 'lr': lr})
            except (StopIteration, ValueError, IndexError):
                continue
    return records


def load_run(log_dir: str) -> list[dict]:
    """디렉토리에서 가장 최신 pretrain_*.log 를 읽어 records 반환."""
    log_files = sorted(glob.glob(os.path.join(log_dir, 'pretrain_*.log')))
    if not log_files:
        return []
    return parse_log(log_files[-1])


# ─────────────────────────────────────────────────────────────────────────────
# 여러 run 병합
# ─────────────────────────────────────────────────────────────────────────────

def discover_runs(weights_dir: str, model_prefix: str = 'ecg_jepa_heedb_') -> list[str]:
    """
    weights_dir 아래에서 model_prefix로 시작하는 폴더를 타임스탬프 순으로 반환.
    """
    pattern = os.path.join(weights_dir, f'{model_prefix}*')
    dirs = sorted(glob.glob(pattern))  # 이름에 타임스탬프가 있어 lexicographic = 시간순
    return [d for d in dirs if os.path.isdir(d)]


def merge_runs(run_dirs: list[str]) -> tuple[list, list, list, list[int]]:
    """
    여러 run 디렉토리의 loss 기록을 epoch 기준으로 병합.

    - 같은 epoch이 여러 run에 존재하면 **나중 run** 데이터가 우선
    - resume 경계(새 run이 시작된 epoch)를 resume_epochs 로 반환

    Returns:
        epochs  : 정렬된 epoch 리스트
        losses  : 대응하는 loss 리스트
        lrs     : 대응하는 lr 리스트 (None 포함 가능)
        resume_epochs : resume이 일어난 epoch 번호 리스트
    """
    epoch_map: dict[int, dict] = {}   # epoch → record (나중 run이 덮어씀)
    resume_epochs: list[int] = []

    for i, d in enumerate(run_dirs):
        records = load_run(d)
        if not records:
            print(f'  [경고] 로그 없음: {d}')
            continue

        if i > 0:
            resume_epochs.append(records[0]['epoch'])

        for rec in records:
            epoch_map[rec['epoch']] = rec

    if not epoch_map:
        return [], [], [], []

    sorted_records = sorted(epoch_map.values(), key=lambda r: r['epoch'])
    epochs  = [r['epoch'] for r in sorted_records]
    losses  = [r['loss']  for r in sorted_records]
    lrs     = [r['lr']    for r in sorted_records]
    return epochs, losses, lrs, resume_epochs


# ─────────────────────────────────────────────────────────────────────────────
# 요약 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(run_dirs: list[str],
                  epochs: list, losses: list, resume_epochs: list[int]):
    if not losses:
        print('No loss data found.')
        return

    best_idx   = losses.index(min(losses))
    best_epoch = epochs[best_idx]
    best_loss  = losses[best_idx]
    last_loss  = losses[-1]
    last_epoch = epochs[-1]

    print('\n' + '='*65)
    print('  Training Summary')
    print('='*65)
    print(f'  Run 수        : {len(run_dirs)} 개')
    for i, d in enumerate(run_dirs):
        tag = ' (최초)' if i == 0 else f' (resume #{i})'
        print(f'    [{i+1}] {os.path.basename(d)}{tag}')
    print(f'  총 epoch      : {last_epoch}')
    print(f'  Best loss     : {best_loss:.5f}  (epoch {best_epoch:04d})')
    print(f'  Last loss     : {last_loss:.5f}  (epoch {last_epoch:04d})')
    if resume_epochs:
        print(f'  Resume 경계   : epoch {resume_epochs}')

    # 최신 run의 체크포인트
    latest_dir = run_dirs[-1]
    ckpts = sorted(glob.glob(os.path.join(latest_dir, '*.pth')))
    if ckpts:
        print(f'\n  Checkpoints ({os.path.basename(latest_dir)}):')
        for p in ckpts:
            size_mb = os.path.getsize(p) / 1e6
            name    = os.path.basename(p)
            star    = ' ★ best' if name == 'best.pth' else \
                      ' (last)' if name == 'last.pth' else ''
            print(f'    {name:22s}  {size_mb:6.1f} MB{star}')
    print('='*65 + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# 그래프 / ASCII
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(epochs: list, losses: list, lrs: list,
             resume_epochs: list[int], save_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        has_lr = any(lr is not None for lr in lrs)
        fig, axes = plt.subplots(2 if has_lr else 1, 1,
                                 figsize=(12, 8 if has_lr else 5),
                                 sharex=True)
        ax_loss = axes[0] if has_lr else axes
        ax_lr   = axes[1] if has_lr else None

        # ── Loss 곡선 ──
        ax_loss.plot(epochs, losses, linewidth=1.5,
                     color='steelblue', label='train loss')

        best_idx = losses.index(min(losses))
        ax_loss.scatter(epochs[best_idx], losses[best_idx],
                        color='crimson', s=80, zorder=5,
                        label=f'best: {losses[best_idx]:.5f} (ep {epochs[best_idx]:04d})')

        # resume 경계선
        for i, rep in enumerate(resume_epochs):
            ax_loss.axvline(x=rep, color='orange', linestyle='--',
                            linewidth=1.2, alpha=0.8,
                            label=f'resume #{i+1} (ep {rep:04d})' if i == 0 else f'resume #{i+1}')

        ax_loss.set_ylabel('SmoothL1 Loss')
        ax_loss.set_title('ECG-JEPA Pretraining Loss' +
                          (f'  [{len(resume_epochs)+1} runs merged]'
                           if resume_epochs else ''))
        ax_loss.legend(fontsize=8)
        ax_loss.grid(alpha=0.3)

        # ── LR 곡선 ──
        if ax_lr is not None:
            valid = [(e, lr) for e, lr in zip(epochs, lrs) if lr is not None]
            if valid:
                ep_v, lr_v = zip(*valid)
                ax_lr.plot(ep_v, lr_v, linewidth=1.2,
                           color='seagreen', label='learning rate')
                for rep in resume_epochs:
                    ax_lr.axvline(x=rep, color='orange',
                                  linestyle='--', linewidth=1.2, alpha=0.8)
            ax_lr.set_xlabel('Epoch')
            ax_lr.set_ylabel('Learning Rate')
            ax_lr.legend(fontsize=8)
            ax_lr.grid(alpha=0.3)
        else:
            ax_loss.set_xlabel('Epoch')

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f'Loss curve saved → {save_path}')

    except ImportError:
        _ascii_plot(epochs, losses, resume_epochs)


def _ascii_plot(epochs: list, losses: list, resume_epochs: list[int],
                width: int = 60, height: int = 15):
    if not losses:
        return

    min_l, max_l = min(losses), max(losses)
    rng = max_l - min_l or 1.0

    print(f'\n  Loss curve  (min={min_l:.5f}, max={max_l:.5f})')
    if resume_epochs:
        print(f'  Resume 경계 epoch: {resume_epochs}')
    print(f'  {"─"*width}')

    grid = [[' '] * width for _ in range(height)]
    for i, (ep, l) in enumerate(zip(epochs, losses)):
        x = int((i / max(len(epochs) - 1, 1)) * (width - 1))
        y = int((1 - (l - min_l) / rng) * (height - 1))
        grid[y][x] = '●'

    for row in grid:
        print('  |' + ''.join(row) + '|')
    print(f'  {"─"*width}')
    print(f'  ep{epochs[0]}{" "*(width-6)}ep{epochs[-1]}')
    print()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ECG-JEPA loss 시각화 (multi-run 병합 지원)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--weights_dir', type=str,
        help='weights/ 디렉토리 경로. ecg_jepa_heedb_* 폴더를 자동 탐색합니다.')
    group.add_argument(
        '--log_dir', type=str, nargs='+',
        help='로그 디렉토리 경로. 여러 개 지정 가능 (시간순으로 나열).')
    parser.add_argument(
        '--prefix', type=str, default='ecg_jepa_heedb_',
        help='--weights_dir 사용 시 탐색할 폴더 prefix (기본: ecg_jepa_heedb_)')
    parser.add_argument(
        '--out', type=str, default=None,
        help='출력 이미지 경로 (기본: 마지막 run 폴더/loss_curve.png)')
    args = parser.parse_args()

    # run 디렉토리 목록 결정
    if args.weights_dir:
        run_dirs = discover_runs(args.weights_dir, args.prefix)
        if not run_dirs:
            print(f'[오류] {args.weights_dir} 아래에 {args.prefix}* 폴더가 없습니다.')
            return
        print(f'발견된 run 폴더 ({len(run_dirs)}개):')
        for d in run_dirs:
            print(f'  {d}')
    else:
        run_dirs = args.log_dir

    # 병합
    epochs, losses, lrs, resume_epochs = merge_runs(run_dirs)

    if not epochs:
        print('[오류] 파싱된 loss 데이터가 없습니다.')
        return

    # 요약 출력
    print_summary(run_dirs, epochs, losses, resume_epochs)

    # 그래프 저장 경로
    save_path = args.out or os.path.join(run_dirs[-1], 'loss_curve.png')
    plot_loss(epochs, losses, lrs, resume_epochs, save_path)


if __name__ == '__main__':
    main()