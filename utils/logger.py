"""utils/logger.py — rank-0 전용 로거."""

import logging
import os
import sys
from datetime import datetime

from utils.dist import is_main


def setup_logger(save_dir: str, name: str = 'pretrain') -> logging.Logger:
    """
    rank-0에서만 파일 + stdout 출력.
    다른 rank에서는 NullHandler만 붙인 조용한 로거 반환.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []   # 중복 핸들러 방지

    if is_main():
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        fh = logging.FileHandler(os.path.join(save_dir, f'{name}_{ts}.log'))
        fh.setLevel(logging.INFO)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        fmt = logging.Formatter('%(asctime)s  %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)
    else:
        logger.addHandler(logging.NullHandler())

    return logger