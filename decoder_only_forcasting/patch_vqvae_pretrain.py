"""
Patch VQVAE + Transformer 预训练入口（forecasting 目录）
全部逻辑迁移至 src/training/patch_vqvae_pretrain_common.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.patch_vqvae_pretrain_common import run_pretrain

if __name__ == '__main__':
    run_pretrain()
