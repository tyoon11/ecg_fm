"""
utils/checkpoint.py
─────────────────────────────────────────────────────────────────────────────
체크포인트 로드 유틸리티.

저장 구조:
    ckpt = {
        'encoder':        state_dict   ← downstream에서 주로 사용
        'target_encoder': state_dict   ← EMA 안정화된 버전
        'optimizer':      state_dict   ← resume용
        'epoch':          int
        'global_step':    int
        'best_loss':      float
        'config':         dict
    }

사용 패턴:
    # 1) encoder만 (downstream fine-tuning)
    encoder = load_encoder('best.pth', device='cuda')

    # 2) 학습 재개
    model, optimizer, start_epoch = resume_training('last.pth', model, optimizer)

    # 3) 체크포인트 내용 확인
    inspect_checkpoint('best.pth')
"""

import torch
from typing import Optional


def load_checkpoint(path: str, device: str = 'cpu') -> dict:
    """체크포인트 파일 로드 (raw dict 반환)."""
    ckpt = torch.load(path, map_location=device)
    return ckpt


def load_encoder(
    path:         str,
    model,                        # ECGJepa 인스턴스
    use_target:   bool = False,   # True → target_encoder 가중치 사용
    device:       str  = 'cpu',
    strict:       bool = True,
) -> None:
    """
    체크포인트에서 encoder 가중치만 로드.

    Args:
        path       : 체크포인트 파일 경로
        model      : ECGJepa (또는 BaseModel) 인스턴스
        use_target : True면 target_encoder 가중치 사용
                     (EMA로 안정화된 버전 — downstream에 더 좋을 수 있음)
        device     : 로드할 디바이스
        strict     : state_dict key 불일치 시 에러 여부

    왜 target_encoder가 더 나을 수 있는가?
        target_encoder는 EMA(지수 이동 평균)로 업데이트되어
        학습 노이즈가 평활화된 "더 보수적인" 표현을 갖습니다.
        downstream 태스크에 따라 encoder보다 성능이 좋은 경우가 있으므로
        두 버전을 모두 실험해볼 것을 권장합니다.
    """
    ckpt = load_checkpoint(path, device=device)

    key = 'target_encoder' if use_target else 'encoder'
    if key not in ckpt:
        raise KeyError(
            f"'{key}' not found in checkpoint. "
            f"Available keys: {list(ckpt.keys())}"
        )

    missing, unexpected = model.encoder.load_state_dict(
        ckpt[key], strict=strict)

    if missing:
        print(f'[load_encoder] missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'[load_encoder] unexpected keys ({len(unexpected)}): {unexpected[:5]}...')

    print(
        f'[load_encoder] loaded {"target_encoder" if use_target else "encoder"} '
        f'from {path}  (epoch={ckpt.get("epoch", "?")+1}, '
        f'best_loss={ckpt.get("best_loss", "?"):.5f})'
    )


def resume_training(
    path:      str,
    model,
    optimizer,
    device:    str = 'cuda',
) -> tuple[int, float]:
    """
    중단된 학습 재개.

    Returns:
        start_epoch : 이어서 시작할 epoch (0-indexed)
        best_loss   : 지금까지의 best loss
    """
    ckpt = load_checkpoint(path, device=device)

    # encoder + target_encoder 복원
    model.encoder.load_state_dict(ckpt['encoder'])
    model.target_encoder.load_state_dict(ckpt['target_encoder'])

    # optimizer 상태 복원
    optimizer.load_state_dict(ckpt['optimizer'])

    start_epoch = ckpt['epoch'] + 1
    best_loss   = ckpt.get('best_loss', float('inf'))

    print(
        f'[resume] Resumed from epoch {ckpt["epoch"]+1}  '
        f'(global_step={ckpt.get("global_step", "?")}, '
        f'best_loss={best_loss:.5f})'
    )
    return start_epoch, best_loss


def inspect_checkpoint(path: str):
    """체크포인트 내부 구조 출력 (디버깅용)."""
    ckpt = torch.load(path, map_location='cpu')

    print(f'\n{"="*60}')
    print(f'Checkpoint: {path}')
    print(f'{"="*60}')
    print(f'  epoch       : {ckpt.get("epoch", "?")+1}')
    print(f'  global_step : {ckpt.get("global_step", "?")}')
    print(f'  best_loss   : {ckpt.get("best_loss", "?"):.5f}')

    print(f'\n  encoder keys      : {len(ckpt["encoder"])} tensors')
    enc = ckpt['encoder']
    total_params = sum(v.numel() for v in enc.values())
    print(f'  encoder params    : {total_params/1e6:.1f} M')

    if 'target_encoder' in ckpt:
        print(f'  target_encoder    : included')

    if 'config' in ckpt:
        cfg = ckpt['config']
        print(f'\n  Training config:')
        for k, v in cfg.get('train', {}).items():
            print(f'    {k}: {v}')

    print(f'{"="*60}\n')