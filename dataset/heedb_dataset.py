"""
dataset/heedb_dataset.py
─────────────────────────────────────────────────────────────────────────────
HEEDB Dataset  —  CSV 기반 lazy-loading + worker-local handle cache

성능 핵심:
  - h5py.File() open/close를 매 getitem마다 하면 파일 1개당 0.5~5ms 낭비
  - persistent_workers=True + worker-local LRU cache로 handle을 재사용
  - 파일 순서에 무관하게 동일 worker 내 재방문 시 거의 0ms

HEEDB HDF5 구조:
  ECG/segments/{sid}/signal  →  shape (12, T), dtype float16
  ECG/metadata attrs         →  fs (250 or 500), sig_len (2500 or 5000)
  sig_name 고정: I II III V1 V2 V3 V4 V5 V6 aVF aVL aVR

heedb_table.csv 주요 컬럼:
  filepath   "data/he110745030.h5"   (v4.0/ 기준 상대경로)
  fs         250 or 500
  sid        0 고정 (HEEDB seg_len=1)

nan_ratio 필터링을 하지 않는 이유:
  실측 결과 nan_ratio가 전부 0.0 이어서 필터링해도 제거되는 행이 없음.
  또한 nan_ratio 컬럼 포맷이 "np.float64(0.0)" 형태라
  ast.literal_eval로 파싱 불가 (11M 행 파싱 시 590초 소요).
  → filepath / fs / sid 3컬럼만 읽어 로드 시간 590초 → 10초 이내로 단축.
"""

import os
from collections import OrderedDict
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

TARGET_LEN        = 2500   # ECG-JEPA 고정값 (250 Hz × 10 s)
HANDLE_CACHE_SIZE = 128    # worker당 열어둘 최대 파일 핸들 수


# ─────────────────────────────────────────────────────────────────────────────
# Worker-local LRU 파일 핸들 캐시
# ─────────────────────────────────────────────────────────────────────────────

class _HandleCache:
    """
    h5py 파일 핸들을 LRU 방식으로 캐싱.

    DataLoader의 persistent_workers=True와 함께 사용할 때 효과 극대화.
    각 worker 프로세스마다 독립적으로 생성되므로 멀티프로세싱 안전.

    왜 worker당 독립 캐시인가?
      h5py.File 객체는 fork 이후 자식 프로세스에서 재사용하면
      SWMR 모드가 아닌 경우 파일 손상 위험이 있다.
      __getitem__ 안에서 처음 접근하면 해당 worker 프로세스에서만
      파일을 열기 때문에 안전하다.
    """

    def __init__(self, max_size: int = HANDLE_CACHE_SIZE):
        self._cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._max   = max_size

    def get(self, path: str) -> h5py.File:
        if path in self._cache:
            # LRU: 최근 접근을 맨 뒤로
            self._cache.move_to_end(path)
            return self._cache[path]

        # cache miss → 파일 열기
        if len(self._cache) >= self._max:
            # 가장 오래된 핸들 닫고 제거
            _, old_handle = self._cache.popitem(last=False)
            try:
                old_handle.close()
            except Exception:
                pass

        handle = h5py.File(path, 'r')
        self._cache[path] = handle
        return handle

    def close_all(self):
        for h in self._cache.values():
            try:
                h.close()
            except Exception:
                pass
        self._cache.clear()

    def __del__(self):
        self.close_all()


# ─────────────────────────────────────────────────────────────────────────────
# HEEDBDataset
# ─────────────────────────────────────────────────────────────────────────────

class HEEDBDataset(Dataset):
    """
    Args:
        root_dir  : /home/irteam/opendata1/h5/heedb/v4.0/
        csv_path  : heedb_table.csv 경로
                    None → root_dir/heedb_table.csv 자동 탐색
        n_leads   : 사용할 리드 수 (12 = 전체)

    사용하지 않는 인자:
        max_nan_ratio : nan_ratio가 전부 0.0이므로 필터링 불필요.
                        인터페이스 호환을 위해 인자만 남겨둠.
    """

    # sig_name 고정 순서 (HEEDB 스펙)
    # idx: 0=I  1=II  2=III  3=V1  4=V2  5=V3  6=V4  7=V5  8=V6  9=aVF  10=aVL  11=aVR
    LEAD_NAMES = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4',
                  'V5', 'V6', 'aVF', 'aVL', 'aVR']

    def __init__(
        self,
        root_dir:      str,
        csv_path:      Optional[str] = None,
        max_nan_ratio: float = 0.1,   # 현재 미사용 (하위 호환용)
        n_leads:       int   = 12,
    ):
        self.root_dir = root_dir
        self.n_leads  = n_leads

        # CSV 로드: filepath / fs / sid 3컬럼만 읽음 → ~10초
        csv_path = csv_path or os.path.join(root_dir, 'heedb_table.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"heedb_table.csv not found: {csv_path}")

        df = pd.read_csv(csv_path, usecols=['filepath', 'fs', 'sid'],
                         low_memory=False)
        df = df.dropna(subset=['filepath', 'fs', 'sid'])
        self.records = df.reset_index(drop=True)

        # worker-local 핸들 캐시: 메인 프로세스에서는 None
        # DataLoader worker가 fork 한 이후 처음 접근할 때 생성
        self._cache: Optional[_HandleCache] = None

        print(f'[HEEDBDataset] {len(self.records):,} records  (n_leads={n_leads})')

    # ── Dataset 인터페이스 ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        반환: (n_leads, 2500) float32 tensor
        """
        row     = self.records.iloc[idx]
        fpath   = os.path.join(self.root_dir, str(row['filepath']))
        sid     = int(row['sid'])
        orig_fs = int(row['fs'])

        wave = self._load_signal(fpath, sid)   # (12, T) float32

        # 12보다 적은 리드 사용할 경우 앞에서 슬라이싱
        if self.n_leads < 12:
            wave = wave[:self.n_leads]         # (n_leads, T)

        # 500 Hz → 250 Hz (2500 pts) 리샘플링
        # 이미 250 Hz이고 길이도 맞으면 스킵
        if orig_fs != 250 or wave.shape[1] != TARGET_LEN:
            wave = resample(wave, TARGET_LEN, axis=1).astype(np.float32)

        # NaN / inf → 0 (학습 안정성)
        if not np.isfinite(wave).all():
            wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(wave, dtype=torch.float32)   # (n_leads, 2500)

    # ── HDF5 신호 로드 (handle cache 사용) ───────────────────────────────────

    def _load_signal(self, fpath: str, sid: int) -> np.ndarray:
        """
        ECG/segments/{sid}/signal → (12, T) float32

        _HandleCache를 통해 파일 핸들 재사용:
          - 첫 접근: h5py.File 오픈 (0.5~5ms)
          - 재접근: 캐시 hit → ~0ms
          - 캐시 용량 초과: LRU eviction 후 오픈

        주의: self._cache는 worker 프로세스에서 처음 호출 시 초기화.
              메인 프로세스에서는 None 상태.
        """
        # worker 프로세스에서 첫 호출 시 캐시 초기화
        if self._cache is None:
            self._cache = _HandleCache(max_size=HANDLE_CACHE_SIZE)

        try:
            handle = self._cache.get(fpath)
            wave   = np.array(
                handle[f'ECG/segments/{sid}/signal'],
                dtype=np.float32,    # float16 → float32 즉시 변환
            )
        except Exception:
            # 파일 손상, 키 없음 등 → 0 패딩 (학습 중단 방지)
            return np.zeros((12, TARGET_LEN), dtype=np.float32)

        # shape 보정: (T, 12) → (12, T)  (방어적 처리)
        if wave.ndim == 2 and wave.shape[0] != 12 and wave.shape[1] == 12:
            wave = wave.T

        return wave   # (12, T)