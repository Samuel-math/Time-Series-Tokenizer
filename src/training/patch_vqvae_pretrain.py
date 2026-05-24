"""Patch VQVAE + Transformer pretrain entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training.patch_vqvae_pretrain_common import run_pretrain


if __name__ == '__main__':
    run_pretrain()
