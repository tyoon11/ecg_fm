#!/bin/bash
# run_h200.sh  —  ECG-JEPA pretraining on HEEDB, H200 × 8 GPU
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="configs/pretrain_heedb.yaml"
N_GPUS=7

# ── GPU 지정 (0~7번 전체 사용) ───────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# ── PYTHONPATH: 프로젝트 루트 최우선 ─────────────────────────────────────────
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ── NCCL ─────────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# ── Python 경고 최소화 ────────────────────────────────────────────────────────
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

echo "========================================================"
echo "  ECG-JEPA Pretraining  |  H200 × ${N_GPUS}  |  ${TS}"
echo "  Project : ${PROJECT_DIR}"
echo "  Config  : ${CONFIG}"
echo "  GPUs    : ${CUDA_VISIBLE_DEVICES}"
echo "========================================================"

torchrun \
    --nproc_per_node=${N_GPUS} \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    pretrain.py \
    --config "${CONFIG}" \
    "$@" \
    2>&1 | tee "logs/pretrain_${TS}.log"