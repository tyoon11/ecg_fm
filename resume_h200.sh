#!/bin/bash
# resume_h200.sh  —  ECG-JEPA pretraining resume on HEEDB, H200 × 8 GPU
# 사용 예시
# epoch0010.pth로 재개
# bash resume_h200.sh ./weights/ecg_jepa_heedb_20250327_XXXXXX/epoch0010.pth

set -euo pipefail

# ── 체크포인트 경로 인자 확인 ─────────────────────────────────────────────────
if [ -z "${1:-}" ]; then
    echo "Usage: bash resume_h200.sh <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  bash resume_h200.sh ./weights/ecg_jepa_heedb_20260326_203353/epoch0010.pth"
    echo ""
    echo "Available checkpoints:"
    ls ./weights/ecg_jepa_heedb_*/epoch*.pth ./weights/ecg_jepa_heedb_*/best.pth 2>/dev/null || echo "  (none found)"
    exit 1
fi

RESUME_PATH="$1"

# 체크포인트 파일 존재 확인
if [ ! -f "${RESUME_PATH}" ]; then
    echo "ERROR: checkpoint not found: ${RESUME_PATH}"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="configs/pretrain_heedb.yaml"
N_GPUS=4

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

echo "========================================================"
echo "  ECG-JEPA Resume      |  H200 × ${N_GPUS}  |  ${TS}"
echo "  Project   : ${PROJECT_DIR}"
echo "  Config    : ${CONFIG}"
echo "  GPUs      : ${CUDA_VISIBLE_DEVICES}"
echo "  Resuming  : ${RESUME_PATH}"
echo "========================================================"

torchrun \
    --nproc_per_node=${N_GPUS} \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    pretrain.py \
    --config "${CONFIG}" \
    --resume "${RESUME_PATH}" \
    "${@:2}" \
    2>&1 | tee "logs/resume_${TS}.log"