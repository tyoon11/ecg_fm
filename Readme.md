# ECG-JEPA Pretraining on HEEDB

12-lead ECG를 위한 Joint-Embedding Predictive Architecture(JEPA) 사전학습 코드입니다.  
H200 × 8 GPU 분산 학습 환경에 최적화되어 있으며, HEEDB 데이터셋을 기반으로 합니다.

---

## 프로젝트 구조

```
.
├── pretrain.py                  # 메인 사전학습 스크립트
├── run_h200.sh                  # 처음부터 학습 실행 스크립트
├── resume_h200.sh               # 체크포인트에서 학습 재개 스크립트
├── plot_loss.py                 # loss 곡선 시각화
│
├── configs/
│   └── pretrain_heedb.yaml      # 학습 하이퍼파라미터 설정
│
├── dataset/
│   ├── __init__.py
│   └── heedb_dataset.py         # HEEDB HDF5 데이터 로더
│
├── models/
│   ├── __init__.py              # 모델 레지스트리
│   ├── base_model.py            # BaseModel ABC + build_model()
│   └── ecg_jepa/
│       └── model.py             # ECGJepa, MaskTransformer, MaskTransformerPredictor
│
└── utils/
    ├── __init__.py
    ├── dist.py                  # DDP 헬퍼 (setup_ddp, is_main, all_reduce_mean)
    ├── logger.py                # rank-0 전용 로거
    └── checkpoint.py            # 체크포인트 저장/로드 유틸
```

---

## 모델 아키텍처

ECG-JEPA는 Student-Teacher JEPA 프레임워크를 12-lead ECG에 적용한 모델입니다.

- **입력**: 10초 12-lead ECG, 250Hz 샘플링 → T = 2500 time points
- **패치**: lead당 50 patches × 50 timepoints (c=12, p=50, t=50)
- **마스킹**: block 마스킹 (비율 17.5~22.5%) 또는 random 마스킹 (60~70%)
- **Student / Teacher**: Transformer encoder (depth=12, heads=16, dim=768), Teacher는 Student의 EMA
- **Predictor**: 소형 Transformer (depth=6, heads=12, dim=384), lead별 독립 처리
- **Loss**: Smooth L1 Loss (latent space에서 마스킹된 위치 예측)
- **총 파라미터**: ~87M

---

## 설정 파일 (`configs/pretrain_heedb.yaml`)

| 항목 | 값 |
|---|---|
| 데이터 경로 | `/home/irteam/opendata1/h5/heedb/v4.0` |
| Optimizer | AdamW |
| Learning Rate | 1.5e-4 (linear scaled) |
| Weight Decay | 0.05 |
| Batch Size | 96 per-GPU (effective 768) |
| Epochs | 100 |
| Warmup Epochs | 5 |
| LR Schedule | Cosine Decay |
| EMA | 0.996 → 1.0 |
| 체크포인트 저장 주기 | 5 epoch마다 |

---

## 실행 방법

### 처음부터 학습

```bash
bash run_h200.sh
```

설정 값을 CLI에서 override할 수도 있습니다:

```bash
bash run_h200.sh train.batch_size 128 train.lr 0.0002
```

### 학습 재개 (Resume)

```bash
# 어떤 체크포인트가 있는지 확인
bash resume_h200.sh

# 특정 체크포인트에서 재개
bash resume_h200.sh ./weights/ecg_jepa_heedb_YYYYMMDD_HHMMSS/epoch0010.pth
```

---

## 체크포인트

학습 중 아래 파일들이 `./weights/ecg_jepa_heedb_{timestamp}/` 에 자동 저장됩니다:

| 파일 | 저장 시점 |
|---|---|
| `epoch{N:04d}.pth` | 매 5 epoch마다 (`save_freq: 5`) |
| `best.pth` | loss가 갱신될 때마다 |
| `last.pth` | 학습 완료 시 |

체크포인트 내부 구조:
```python
{
    'encoder':        state_dict,   # downstream fine-tuning에 사용
    'target_encoder': state_dict,   # EMA 버전 (downstream에 더 좋을 수 있음)
    'optimizer':      state_dict,
    'epoch':          int,
    'global_step':    int,
    'best_loss':      float,
    'config':         dict,
}
```

---

## Downstream 사용

```python
from models import build_model
from utils  import load_encoder

model = build_model('ecg_jepa', c=12, p=50, t=50, ...)

# encoder 가중치만 로드 (fine-tuning용)
load_encoder('best.pth', model, use_target=False)

# target_encoder 사용 (EMA 안정화 버전, 경우에 따라 더 좋음)
load_encoder('best.pth', model, use_target=True)
```

체크포인트 내용 확인:
```python
from utils import inspect_checkpoint
inspect_checkpoint('./weights/ecg_jepa_heedb_XXXX/epoch0010.pth')
```

---

## 학습 결과 시각화

```bash
python plot_loss.py --log_dir ./weights/ecg_jepa_heedb_YYYYMMDD_HHMMSS
```

`loss_curve.png`와 학습 요약이 출력됩니다.

---

## 새 모델 추가

1. `BaseModel`을 상속하고 `model_name`을 정의합니다.
2. `models/__init__.py`에 import 한 줄을 추가합니다.
3. `build_model('새모델이름')` 으로 바로 사용 가능합니다.

```python
# models/my_model/model.py
class MyModel(BaseModel):
    model_name = "my_model"
    ...

# models/__init__.py
from .my_model.model import MyModel  # noqa
```

---

## 환경

- Python 3.10+
- PyTorch 2.x
- CUDA 12.x
- timm (CosineLRScheduler)
- H200 × 8 GPU (torchrun DDP)