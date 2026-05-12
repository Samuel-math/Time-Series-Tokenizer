#!/usr/bin/env python3
"""Build channel groups by frequency-domain similarity.

This script is CPU-only. It reads the dataset csv, extracts compact spectral
features per channel, then greedily forms groups whose members have similar
frequency structure. This better matches patch reconstruction than global
Pearson correlation because it separates smooth/trend-heavy channels from
high-frequency or noisy channels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_FILES = {
    "ettm1": "ETTm1.csv",
    "ettm2": "ETTm2.csv",
    "etth1": "ETTh1.csv",
    "etth2": "ETTh2.csv",
    "electricity": "electricity.csv",
    "traffic": "traffic.csv",
    "weather": "weather.csv",
    "illness": "national_illness.csv",
    "exchange": "exchange_rate.csv",
}


def _resolve_csv(repo_root: Path, dset: str, csv_path: str | None) -> Path:
    if csv_path:
        p = Path(csv_path)
        return p if p.is_absolute() else repo_root / p
    name = DATASET_FILES[dset]
    candidates = [
        repo_root / "datasets" / name,
        repo_root / "datasets" / dset / name,
        repo_root / "data" / name,
        repo_root / "data" / dset / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find csv for {dset}. Tried: {candidates}")


def build_frequency_features(x: np.ndarray) -> np.ndarray:
    """Return one compact frequency feature vector per channel."""
    eps = 1e-8
    spectrum = np.abs(np.fft.rfft(x, axis=0)).astype(np.float32)
    power = spectrum ** 2
    total = power.sum(axis=0, keepdims=True) + eps
    prob = power / total

    n_bins = power.shape[0]
    freq = np.linspace(0.0, 1.0, n_bins, dtype=np.float32).reshape(-1, 1)
    low_end = max(1, n_bins // 4)
    mid_end = max(low_end + 1, n_bins // 2)

    low = power[:low_end].sum(axis=0) / total.squeeze(0)
    mid = power[low_end:mid_end].sum(axis=0) / total.squeeze(0)
    high = power[mid_end:].sum(axis=0) / total.squeeze(0)
    centroid = (prob * freq).sum(axis=0)
    bandwidth = np.sqrt((prob * (freq - centroid.reshape(1, -1)) ** 2).sum(axis=0))
    entropy = -(prob * np.log(prob + eps)).sum(axis=0) / np.log(n_bins + eps)
    flatness = np.exp(np.log(power + eps).mean(axis=0)) / (power.mean(axis=0) + eps)
    peak = power.argmax(axis=0).astype(np.float32) / max(n_bins - 1, 1)

    # Add cheap time-domain patch proxies so channels with similar spectra but
    # very different local roughness are less likely to be grouped together.
    abs_diff = np.abs(np.diff(x, axis=0)).mean(axis=0)
    std = x.std(axis=0)

    feats = np.stack(
        [low, mid, high, centroid, bandwidth, entropy, flatness, peak, abs_diff, std],
        axis=1,
    ).astype(np.float32)
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + eps)
    return feats


def cosine_similarity(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norm, 1e-8)
    sim = normalized @ normalized.T
    return np.clip(sim, -1.0, 1.0)


def greedy_similarity_groups(sim: np.ndarray, max_channels: int) -> list[list[int]]:
    n = sim.shape[0]
    remaining = set(range(n))
    groups: list[list[int]] = []
    strength = sim.mean(axis=1)

    while remaining:
        seed = max(remaining, key=lambda i: strength[i])
        group = [seed]
        remaining.remove(seed)

        while remaining and len(group) < max_channels:
            # Add the channel most similar to the current group's spectrum.
            nxt = max(remaining, key=lambda j: float(sim[j, group].mean()))
            group.append(nxt)
            remaining.remove(nxt)

        groups.append(group)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dset", required=True, choices=sorted(DATASET_FILES))
    ap.add_argument("--max_channels", type=int, required=True)
    ap.add_argument("--csv_path", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.7,
                    help="Use the first train_ratio portion of rows to compute frequency features.")
    ap.add_argument("--sample_rows", type=int, default=20000,
                    help="Uniformly subsample rows after train split for memory/speed; <=0 disables.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    csv_file = _resolve_csv(repo_root, args.dset, args.csv_path)
    df = pd.read_csv(csv_file)
    value_df = df.iloc[:, 1:]  # first column is date/time in benchmark csvs
    n_train = int(len(value_df) * args.train_ratio)
    value_df = value_df.iloc[:n_train]

    if args.sample_rows and args.sample_rows > 0 and len(value_df) > args.sample_rows:
        idx = np.linspace(0, len(value_df) - 1, args.sample_rows).astype(int)
        value_df = value_df.iloc[idx]

    x = value_df.to_numpy(dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = x / std

    features = build_frequency_features(x)
    sim = cosine_similarity(features)
    np.fill_diagonal(sim, 1.0)

    groups = greedy_similarity_groups(sim, args.max_channels)
    out = {
        "dset": args.dset,
        "method": "greedy_frequency_features",
        "csv_file": str(csv_file),
        "max_channels": args.max_channels,
        "n_channels": x.shape[1],
        "n_rows_used": x.shape[0],
        "feature_names": [
            "low_energy_ratio",
            "mid_energy_ratio",
            "high_energy_ratio",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_entropy",
            "spectral_flatness",
            "peak_frequency",
            "abs_diff_mean",
            "std",
        ],
        "groups": groups,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {len(groups)} groups to {out_path}")
    for i, g in enumerate(groups):
        print(f"group {i}: size={len(g)} channels={g}")


if __name__ == "__main__":
    main()
