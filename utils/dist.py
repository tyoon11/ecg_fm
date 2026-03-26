"""utils/dist.py — DDP 헬퍼."""

import os
import torch
import torch.distributed as dist


def setup_ddp() -> tuple[int, int]:
    """
    torchrun 환경이면 DDP 초기화 후 (local_rank, world_size) 반환.
    일반 python 실행이면 (0, 1) 반환.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size()
    return 0, 1


def is_main() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_reduce_mean(tensor: 'torch.Tensor') -> 'torch.Tensor':
    """분산 환경에서 모든 rank의 평균값을 반환."""
    if get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor