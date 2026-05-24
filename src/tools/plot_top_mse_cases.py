"""
Plot worst-MSE forecast cases for a finetuned PatchVQVAETransformer checkpoint.

Example:
    python plot_top_mse_cases.py \
        --checkpoint saved_models/patch_vqvae_finetune/ettm2/patch_vqvae_finetune_cw192_tw96_model1.pth \
        --top_k 8
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add repo root to path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datautils import get_dls
from src.models.layers.revin import RevIN
from src.models.patch_vqvae_transformer import PatchVQVAETransformer


def parse_args():
    p = argparse.ArgumentParser(description='Plot top-MSE forecast cases')
    p.add_argument('--checkpoint', type=str, required=True, help='finetuned .pth checkpoint')
    p.add_argument('--out_dir', type=str, default=None, help='output dir; default: <checkpoint stem>_top_mse')
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', type=str, default=None, help='cuda/cpu; default auto')
    p.add_argument('--save_npz', type=int, default=1, help='save arrays for top cases')

    # Optional overrides. If omitted, values are read from checkpoint["args"].
    p.add_argument('--dset', type=str, default=None)
    p.add_argument('--context_points', type=int, default=None)
    p.add_argument('--target_points', type=int, default=None)
    p.add_argument('--scaler', type=str, default=None)
    p.add_argument('--features', type=str, default=None)
    p.add_argument('--revin', type=int, default=None)
    p.add_argument('--ar_step_size', type=int, default=None)
    p.add_argument('--pred_len', type=int, default=None)
    p.add_argument('--use_gumbel_softmax', type=int, default=None)
    p.add_argument('--gumbel_temperature', type=float, default=None)
    p.add_argument('--gumbel_hard', type=int, default=None)
    return p.parse_args()


def _arg(saved_args, cli_args, name, default=None):
    cli_v = getattr(cli_args, name)
    if cli_v is not None:
        return cli_v
    return saved_args.get(name, default)


def build_data_args(saved_args, cli_args):
    """Build the minimal args object expected by datautils.get_dls."""
    ns = argparse.Namespace()
    ns.dset = _arg(saved_args, cli_args, 'dset', 'ettm2')
    ns.dset_finetune = ns.dset
    ns.context_points = int(_arg(saved_args, cli_args, 'context_points', 192))
    ns.target_points = int(_arg(saved_args, cli_args, 'target_points', 96))
    ns.batch_size = int(cli_args.batch_size or saved_args.get('batch_size', 32))
    ns.num_workers = int(cli_args.num_workers)
    ns.scaler = _arg(saved_args, cli_args, 'scaler', 'standard')
    ns.features = _arg(saved_args, cli_args, 'features', 'M')
    return ns


def load_model(checkpoint, saved_args, cli_args, dls, device):
    config = copy.deepcopy(checkpoint['config'])
    config['n_channels'] = dls.vars
    config['use_gumbel_softmax'] = bool(_arg(saved_args, cli_args, 'use_gumbel_softmax', 1))
    config['gumbel_temperature'] = float(_arg(saved_args, cli_args, 'gumbel_temperature', 1.0))
    config['gumbel_hard'] = bool(_arg(saved_args, cli_args, 'gumbel_hard', 0))

    model = PatchVQVAETransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def collect_predictions(model, dls, saved_args, cli_args, data_args, device):
    use_revin = bool(_arg(saved_args, cli_args, 'revin', 1))
    revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if use_revin else None
    ar_step_size = _arg(saved_args, cli_args, 'ar_step_size', None)
    pred_len = _arg(saved_args, cli_args, 'pred_len', None)

    contexts, preds, targets = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in dls.test:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            raw_x = batch_x.detach().cpu()

            if revin:
                batch_x = revin(batch_x, 'norm')

            pred, _ = model.forward_finetune(
                batch_x,
                data_args.target_points,
                step_size=ar_step_size,
                pred_len=pred_len,
            )

            if revin:
                pred = revin(pred, 'denorm')

            contexts.append(raw_x)
            preds.append(pred.float().cpu())
            targets.append(batch_y.cpu())

    return (
        torch.cat(contexts, dim=0).numpy(),
        torch.cat(preds, dim=0).numpy(),
        torch.cat(targets, dim=0).numpy(),
    )


def plot_case(out_path, context, pred, target, sample_idx, sample_mse, per_channel_mse):
    num_channels = target.shape[1]
    total_len = context.shape[0] + target.shape[0]
    t_context = np.arange(context.shape[0])
    t_future = np.arange(context.shape[0], total_len)

    fig, axes = plt.subplots(num_channels + 1, 1, figsize=(13, 2.0 * num_channels + 2.5), sharex=False)
    if num_channels == 1:
        axes = np.array([axes[0], axes[1]])

    for c in range(num_channels):
        ax = axes[c]
        ax.plot(t_context, context[:, c], color='0.55', linewidth=1.0, label='context')
        ax.plot(t_future, target[:, c], color='black', linewidth=1.2, label='target')
        ax.plot(t_future, pred[:, c], color='tab:red', linewidth=1.2, label='pred')
        ax.axvline(context.shape[0] - 1, color='tab:blue', linestyle='--', linewidth=0.8)
        ax.set_ylabel(f'ch{c}\nMSE={per_channel_mse[c]:.4g}')
        ax.grid(alpha=0.25)
        if c == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=8)

    ax_bar = axes[-1]
    ax_bar.bar(np.arange(num_channels), per_channel_mse, color='tab:orange')
    ax_bar.set_ylabel('channel MSE')
    ax_bar.set_xlabel('channel')
    ax_bar.grid(axis='y', alpha=0.25)

    fig.suptitle(f'Top-MSE case | sample={sample_idx} | sample MSE={sample_mse:.6f}', y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    cli_args = parse_args()
    device = torch.device(cli_args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint_path = Path(cli_args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})

    data_args = build_data_args(saved_args, cli_args)
    dls = get_dls(data_args)
    model = load_model(checkpoint, saved_args, cli_args, dls, device)

    contexts, preds, targets = collect_predictions(model, dls, saved_args, cli_args, data_args, device)
    per_sample_channel_mse = ((preds - targets) ** 2).mean(axis=1)  # [N, C]
    per_sample_mse = per_sample_channel_mse.mean(axis=1)
    order = np.argsort(-per_sample_mse)
    top_indices = order[:max(1, cli_args.top_k)]

    out_dir = Path(cli_args.out_dir) if cli_args.out_dir else checkpoint_path.with_suffix('').parent / f'{checkpoint_path.stem}_top_mse'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        rows.append({
            'rank': rank,
            'sample_index': int(idx),
            'sample_mse': float(per_sample_mse[idx]),
            'worst_channel': int(np.argmax(per_sample_channel_mse[idx])),
            'worst_channel_mse': float(np.max(per_sample_channel_mse[idx])),
        })
        plot_case(
            out_dir / f'top{rank:02d}_sample{idx}_mse{per_sample_mse[idx]:.6f}.png',
            contexts[idx],
            preds[idx],
            targets[idx],
            int(idx),
            float(per_sample_mse[idx]),
            per_sample_channel_mse[idx],
        )

    pd.DataFrame(rows).to_csv(out_dir / 'top_mse_summary.csv', index=False)
    if cli_args.save_npz:
        np.savez_compressed(
            out_dir / 'top_mse_cases.npz',
            top_indices=top_indices,
            contexts=contexts[top_indices],
            preds=preds[top_indices],
            targets=targets[top_indices],
            per_sample_mse=per_sample_mse[top_indices],
            per_sample_channel_mse=per_sample_channel_mse[top_indices],
        )

    print(f'Saved top-{len(top_indices)} MSE plots to: {out_dir}')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
