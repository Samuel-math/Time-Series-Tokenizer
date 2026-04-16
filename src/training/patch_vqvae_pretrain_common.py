"""
Patch VQVAE + Transformer 渐进式预训练公共逻辑
（decoder_only_NTP / decoder_only_forcasting 共用）

入口脚本只需:
    sys.path.insert(0, repo_root)
    from src.training.patch_vqvae_pretrain_common import run_pretrain
    run_pretrain()
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datautils import get_dls
from src.models.patch_vqvae_transformer import (
    PatchVQVAETransformer, FlattenedVectorQuantizerEMA, get_model_config,
)
from src.models.layers.revin import RevIN


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description='Patch VQVAE Transformer 渐进式预训练')

    # 数据
    p.add_argument('--dset', type=str, default='ettm1')
    p.add_argument('--context_points', type=int, default=512)
    p.add_argument('--progressive_step_size', type=int, required=True,
                   help='渐进式预训练步长（patches）')
    p.add_argument('--progressive_max_stages', type=int, default=None,
                   help='最大阶段数，None 表示全部')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--scaler', type=str, default='standard')
    p.add_argument('--features', type=str, default='M')

    # 模型结构
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--embedding_dim', type=int, default=32)
    p.add_argument('--compression_factor', type=int, default=4)
    p.add_argument('--codebook_size', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--transformer_hidden_dim', type=int, default=None)
    p.add_argument('--commitment_cost', type=float, default=0.25)
    p.add_argument('--codebook_ema', type=int, default=1)
    p.add_argument('--disable_ema_update', type=int, default=1,
                   help='禁用 EMA 更新（1=禁用，用于稳定 recon_loss）')
    p.add_argument('--ema_decay', type=float, default=0.99)
    p.add_argument('--ema_eps', type=float, default=1e-5)
    p.add_argument('--num_hiddens', type=int, default=64)
    p.add_argument('--num_residual_layers', type=int, default=2)
    p.add_argument('--num_residual_hiddens', type=int, default=32)

    # VQVAE checkpoint
    p.add_argument('--vqvae_checkpoint', type=str, default=None,
                   help='预训练 VQVAE 路径（可选）')
    p.add_argument('--freeze_vqvae', type=int, default=1,
                   help='加载后冻结 VQVAE（1=冻结）')
    p.add_argument('--load_vq_weights', type=int, default=1,
                   help='是否加载 VQ 层权重（1=加载）')

    # Per-channel 码本
    p.add_argument('--per_channel_codebook', type=int, default=0,
                   help='每通道独立码本（1=启用，需与 vqvae-only 训练一致）')

    # RVQ 层数
    p.add_argument('--n_rq_layers', type=int, default=1,
                   help='残差向量量化层数（1=普通VQ，2=2层RVQ）')
    p.add_argument('--rq_layer_weights', type=float, nargs='+', default=None,
                   help='各 RVQ 层 pred_loss 的权重，顺序对应第0层、第1层…'
                        '（默认 None = 均等权重）。示例: --rq_layer_weights 1.0 0.5')

    # NMPP 模式
    p.add_argument('--use_raw_input', type=int, default=0,
                   help='1: NMPP 模式，Transformer 接收原始 patch，VQVAE 仅作为 teacher')

    # TSO 分解模式
    p.add_argument('--use_decomposition', type=int, default=0,
                   help='启用 Trend-Stochastic-Oscillatory 分解（1=启用）')
    p.add_argument('--codebook_size_trend', type=int, default=0,
                   help='趋势码本大小（0=与 codebook_size 相同）')
    p.add_argument('--codebook_size_osc', type=int, default=0,
                   help='振荡码本大小（0=与 codebook_size 相同）')
    p.add_argument('--stochastic_latent_dim', type=int, default=4,
                   help='随机分量 VAE 隐变量维度')
    p.add_argument('--trend_kernel_size', type=int, default=5,
                   help='趋势低通滤波器核大小')

    # 训练超参
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--revin', type=int, default=1)
    p.add_argument('--vq_weight', type=float, default=1.0)
    p.add_argument('--recon_weight', type=float, default=0.1)

    # 保存
    p.add_argument('--save_path', type=str, default='saved_models/patch_vqvae/')
    p.add_argument('--model_id', type=int, default=1)
    p.add_argument('--run_id', type=int, default=None)

    return p


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _progressive_loss(all_logits, all_target_indices, rq_layer_weights=None):
    """
    all_logits:        List[stage] of List[rq_layer] of [B, step_size, C, codebook_size]
    all_target_indices: List[stage] of List[rq_layer] of [B, step_size, C]
    rq_layer_weights:  List[float] | None — 各 RVQ 层的损失权重（None 表示均等）
    """
    n_layers = len(all_logits[0])
    if rq_layer_weights is None:
        weights = [1.0] * n_layers
    else:
        if len(rq_layer_weights) != n_layers:
            raise ValueError(
                f"rq_layer_weights 长度 ({len(rq_layer_weights)}) 与 RVQ 层数 ({n_layers}) 不匹配"
            )
        weights = list(rq_layer_weights)

    total_loss = 0.0
    total_weight = 0.0
    for logits_layers, tgt_layers in zip(all_logits, all_target_indices):
        for l, (logits_l, tgt_l) in enumerate(zip(logits_layers, tgt_layers)):
            B, P, C, K = logits_l.shape
            total_loss += weights[l] * F.cross_entropy(logits_l.reshape(-1, K), tgt_l.reshape(-1))
            total_weight += weights[l]

    return total_loss / (total_weight * len(all_logits) / n_layers)


# ---------------------------------------------------------------------------
# Train / validate epochs
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, revin, args, device, trainable_params):
    model.train()
    use_decomp = getattr(model, 'use_decomposition', False)
    totals = dict(loss=0., pred_loss=0., vq_loss=0., recon_loss=0.)
    n = 0

    use_raw = bool(args.use_raw_input) and not use_decomp
    compute_recon = args.recon_weight > 0 and not use_raw and not use_decomp
    vq_w    = 0. if use_raw else args.vq_weight
    recon_w = 0. if use_raw else args.recon_weight
    rq_weights = getattr(args, 'rq_layer_weights', None)

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if revin:
            batch_x = revin(batch_x, 'norm')
            batch_y = revin(batch_y, 'norm')
        batch_full = torch.cat([batch_x, batch_y], dim=1)

        if use_decomp:
            tl, tt, ol, ot, vq_loss = model.forward_progressive_pretrain_decomposed(
                batch_full, step_size=args.progressive_step_size,
                max_stages=args.progressive_max_stages,
            )
            t_pred = _progressive_loss(tl, tt, rq_weights)
            o_pred = _progressive_loss(ol, ot, rq_weights)
            pred_loss = t_pred + o_pred
            recon_loss = batch_x.new_tensor(0.0)
            loss = pred_loss + vq_w * vq_loss
        else:
            all_logits, all_tgt, vq_loss, recon_loss = model.forward_progressive_pretrain(
                batch_full,
                step_size=args.progressive_step_size,
                max_stages=args.progressive_max_stages,
                compute_recon_loss=compute_recon,
                use_raw_input=use_raw,
            )
            pred_loss = _progressive_loss(all_logits, all_tgt, rq_weights)
            loss = pred_loss + vq_w * vq_loss + recon_w * recon_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        totals['loss']       += loss.item()
        totals['pred_loss']  += pred_loss.item()
        totals['vq_loss']    += vq_loss.item()
        totals['recon_loss'] += recon_loss.item()
        n += 1

    scheduler.step()
    return {k: v / n for k, v in totals.items()}


def validate_epoch(model, dataloader, revin, args, device):
    model.eval()
    use_decomp = getattr(model, 'use_decomposition', False)
    totals = dict(loss=0., pred_loss=0., vq_loss=0., recon_loss=0.)
    n = 0

    use_raw = bool(args.use_raw_input) and not use_decomp
    compute_recon = args.recon_weight > 0 and not use_raw and not use_decomp
    vq_w    = 0. if use_raw else args.vq_weight
    recon_w = 0. if use_raw else args.recon_weight
    rq_weights = getattr(args, 'rq_layer_weights', None)

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if revin:
                batch_x = revin(batch_x, 'norm')
                batch_y = revin(batch_y, 'norm')
            batch_full = torch.cat([batch_x, batch_y], dim=1)

            if use_decomp:
                tl, tt, ol, ot, vq_loss = model.forward_progressive_pretrain_decomposed(
                    batch_full, step_size=args.progressive_step_size,
                    max_stages=args.progressive_max_stages,
                )
                t_pred = _progressive_loss(tl, tt, rq_weights)
                o_pred = _progressive_loss(ol, ot, rq_weights)
                pred_loss = t_pred + o_pred
                recon_loss = batch_x.new_tensor(0.0)
                loss = pred_loss + vq_w * vq_loss
            else:
                all_logits, all_tgt, vq_loss, recon_loss = model.forward_progressive_pretrain(
                    batch_full,
                    step_size=args.progressive_step_size,
                    max_stages=args.progressive_max_stages,
                    compute_recon_loss=compute_recon,
                    use_raw_input=use_raw,
                )
                pred_loss = _progressive_loss(all_logits, all_tgt, rq_weights)
                loss = pred_loss + vq_w * vq_loss + recon_w * recon_loss

            totals['loss']       += loss.item()
            totals['pred_loss']  += pred_loss.item()
            totals['vq_loss']    += vq_loss.item()
            totals['recon_loss'] += recon_loss.item()
            n += 1

    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _disable_flash_sdp():
    if not torch.cuda.is_available():
        return
    for fn in ('enable_flash_sdp', 'enable_mem_efficient_sdp'):
        if hasattr(torch.backends.cuda, fn):
            getattr(torch.backends.cuda, fn)(False)
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)
    print('✓ 已禁用 flash/memory-efficient attention（PyTorch 2.7+ 兼容）')


def _disable_ema(model):
    """冻结所有 VQ 模块的 EMA 更新"""
    def _disable_rvq(rvq_mod):
        for single_vq in rvq_mod.layers:
            if isinstance(single_vq, FlattenedVectorQuantizerEMA):
                single_vq._disable_ema_update = True

    if getattr(model, 'use_decomposition', False):
        for attr in ('trend_vq', 'osc_vq'):
            if hasattr(model, attr):
                _disable_rvq(getattr(model, attr))
        print('✓ 已禁用 EMA 更新（分解模式：trend_vq + osc_vq）')
    elif model.per_channel_codebook:
        for rvq_mod in model.vqs:
            _disable_rvq(rvq_mod)
        print('✓ 已禁用 EMA 更新（per-channel 模式）')
    elif hasattr(model, 'vq'):
        _disable_rvq(model.vq)
        print('✓ 已禁用 EMA 更新（shared 模式）')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pretrain():
    args = build_arg_parser().parse_args()
    print('Args:', args)

    # NMPP / 分解模式校验
    if args.use_raw_input and not getattr(args, 'use_decomposition', 0):
        if not args.vqvae_checkpoint:
            raise ValueError('NMPP (--use_raw_input=1) 需要指定 --vqvae_checkpoint')
        args.freeze_vqvae = 1
    if getattr(args, 'use_decomposition', 0) and not args.vqvae_checkpoint:
        raise ValueError('分解模式 (--use_decomposition=1) 需要指定 --vqvae_checkpoint')

    _disable_flash_sdp()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    save_dir = Path(args.save_path) / args.dset
    save_dir.mkdir(parents=True, exist_ok=True)

    # 模型命名
    use_decomp = bool(getattr(args, 'use_decomposition', 0))
    code_dim  = args.embedding_dim * (args.patch_size // args.compression_factor)
    step_size = args.progressive_step_size
    nmpp_sfx   = '_nmpp'  if args.use_raw_input         else ''
    perch_sfx  = '_perch' if args.per_channel_codebook  else ''
    rvq_sfx    = f'_rvq{args.n_rq_layers}' if getattr(args, 'n_rq_layers', 1) > 1 else ''
    decomp_sfx = '_tso' if use_decomp else ''
    rid_sfx    = f'_run{args.run_id}' if args.run_id is not None else ''
    model_name = (
        f'patch_vqvae_ps{args.patch_size}_cb{args.codebook_size}_cd{code_dim}'
        f'_l{args.n_layers}_in{args.context_points}_step{step_size}'
        f'{rid_sfx}_model{args.model_id}{perch_sfx}{rvq_sfx}{nmpp_sfx}{decomp_sfx}'
    )

    # 数据
    args.dset_pretrain = args.dset
    dls = get_dls(args)
    print(f'Channels: {dls.vars} | Train batches: {len(dls.train)} | Val batches: {len(dls.valid)}')

    # 如果提供了 VQVAE checkpoint，先从其 config 覆盖 VQVAE 结构参数，
    # 防止 num_residual_hiddens 等参数与命令行默认值不一致导致 size mismatch
    if args.vqvae_checkpoint:
        try:
            ckpt_meta = torch.load(args.vqvae_checkpoint, map_location='cpu')
            ckpt_cfg  = ckpt_meta.get('config', {})
            vqvae_keys = [
                'patch_size', 'embedding_dim', 'compression_factor',
                'codebook_size', 'num_hiddens', 'num_residual_layers',
                'num_residual_hiddens', 'commitment_cost',
                'codebook_ema', 'ema_decay', 'ema_eps',
                'codebook_size_trend', 'codebook_size_osc',
                'stochastic_latent_dim', 'trend_kernel_size', 'shape_alpha',
            ]
            overridden = []
            for k in vqvae_keys:
                if k in ckpt_cfg:
                    old_v = getattr(args, k, None)
                    new_v = ckpt_cfg[k]
                    if old_v != new_v:
                        setattr(args, k, new_v)
                        overridden.append(f'{k}: {old_v} → {new_v}')
            if overridden:
                print('\n[VQVAE config 自动覆盖]')
                for s in overridden:
                    print(f'  {s}')
        except Exception as e:
            print(f'[警告] 读取 checkpoint config 失败，使用命令行参数: {e}')

    # 模型
    config = get_model_config(args)
    config['n_channels'] = dls.vars
    use_decomp = bool(getattr(args, 'use_decomposition', 0))
    if use_decomp:
        config['use_decomposition'] = True
        config['codebook_size_trend'] = (
            args.codebook_size_trend if getattr(args, 'codebook_size_trend', 0) > 0
            else args.codebook_size
        )
        config['codebook_size_osc'] = (
            args.codebook_size_osc if getattr(args, 'codebook_size_osc', 0) > 0
            else args.codebook_size
        )
        config['stochastic_latent_dim'] = int(getattr(args, 'stochastic_latent_dim', 4))
        config['trend_kernel_size'] = int(getattr(args, 'trend_kernel_size', 5))
    model = PatchVQVAETransformer(config).to(device)

    # 加载预训练 VQVAE
    if args.vqvae_checkpoint:
        print(f'\n加载预训练 VQVAE: {args.vqvae_checkpoint}')
        model.load_vqvae_weights(
            args.vqvae_checkpoint,
            device,
            load_vq=bool(args.load_vq_weights),
            freeze=bool(args.freeze_vqvae),
        )

    # 禁用 EMA 更新
    if args.disable_ema_update:
        _disable_ema(model)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n参数: 总计 {total_p:,} | 可训练 {train_p:,} | 冻结 {total_p - train_p:,}')

    revin = RevIN(dls.vars, eps=1e-5, affine=True).to(device) if args.revin else None
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)

    best_val  = float('inf')
    train_losses, valid_losses = [], []
    no_improve, patience, saved = 0, 10, False

    print(f'\n开始预训练，共 {args.n_epochs} epoch（早停 patience={patience}）')
    print('=' * 80)

    for epoch in range(args.n_epochs):
        tr = train_epoch(model, dls.train, optimizer, scheduler, revin,
                         args, device, trainable_params)
        va = validate_epoch(model, dls.valid, revin, args, device)

        train_losses.append(tr['loss'])
        valid_losses.append(va['loss'])

        print(
            f"Epoch {epoch+1:3d}/{args.n_epochs} | "
            f"Train {tr['loss']:.4f} (Pred {tr['pred_loss']:.4f}  "
            f"VQ {tr['vq_loss']:.4f}  Recon {tr['recon_loss']:.4f}) | "
            f"Val {va['loss']:.4f} (Pred {va['pred_loss']:.4f})"
        )

        if epoch >= 3:
            if va['loss'] < best_val:
                best_val   = va['loss']
                no_improve = 0
                saved      = True
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'args':   vars(args),
                        'epoch':  epoch,
                        'train_loss': tr['loss'],
                        'val_loss':   va['loss'],
                    },
                    save_dir / f'{model_name}.pth',
                )
                print(f"  -> Best model saved (val_loss: {va['loss']:.4f})")
            elif saved:
                no_improve += 1
                if no_improve >= patience:
                    print(f'\n>>> 早停: val_loss 连续 {patience} epoch 未下降')
                    break

        # 定期打印码本利用率
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample = next(iter(dls.train))[0].to(device)
                if revin:
                    sample = revin(sample, 'norm')
                usage, _ = model.get_codebook_usage(sample)
                print(f'  -> Codebook usage: {usage * 100:.1f}%')

    pd.DataFrame({
        'epoch':       range(1, len(train_losses) + 1),
        'train_loss':  train_losses,
        'valid_loss':  valid_losses,
    }).to_csv(save_dir / f'{model_name}_history.csv', index=False)

    with open(save_dir / f'{model_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print('=' * 80)
    print(f'预训练完成。最佳验证损失: {best_val:.4f}')
    print(f'模型: {save_dir / model_name}.pth')
