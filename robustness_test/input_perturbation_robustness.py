#!/usr/bin/env python3
"""Input perturbation robustness test for forecasting checkpoints.

The script evaluates clean-trained checkpoints under Gaussian perturbations on
historical input windows only. Targets remain clean.

By default it keeps the original PatchVQVAETransformer behavior. Other model
architectures can be used by passing a Python import path via --model_ctor and
selecting the appropriate initialization / forward options.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import inspect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datautils import get_dls  # noqa: E402
from src.models.layers.revin import RevIN  # noqa: E402


DEFAULT_SIGMAS = "0.00,0.05,0.10,0.15,0.20,0.30"
DEFAULT_SEEDS = "42,43,44"
DEFAULT_TARGETS = "96,192,336,720"
DEFAULT_AR_STEPS = "3,6,6,9"
DEFAULT_PRED_LENS = "6,12,12,18"


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(',') if x.strip()]


def _resolve_path(path: str | None) -> str | None:
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def _build_checkpoint_map(args, targets: list[int] | None = None) -> dict[int, str]:
    ckpt_map = {}
    default_ckpt = _resolve_path(args.ckpt)
    if default_ckpt:
        for horizon in (targets or []):
            ckpt_map[horizon] = default_ckpt

    explicit = {
        96: args.ckpt96,
        192: args.ckpt192,
        336: args.ckpt336,
        720: args.ckpt720,
    }
    for horizon, path in explicit.items():
        resolved = _resolve_path(path)
        if resolved:
            ckpt_map[horizon] = resolved

    if args.checkpoint_map:
        p = Path(_resolve_path(args.checkpoint_map))
        with p.open() as f:
            payload = json.load(f)
        for k, v in payload.items():
            ckpt_map[int(k)] = _resolve_path(v)
    return ckpt_map


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_data_args(args, target_points: int):
    return SimpleNamespace(
        dset=args.dset,
        context_points=args.context_points,
        target_points=target_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=args.scaler,
        features=args.features,
        channel_start=args.channel_start,
        channel_end=args.channel_end,
        channel_indices=args.channel_indices,
        use_time_features=False,
    )


def compute_train_std(dataloader, device: torch.device) -> torch.Tensor:
    """Compute channel-wise train std over the dataloader input windows."""
    count = 0
    sum_x = None
    sum_x2 = None
    for batch_x, _ in dataloader:
        x = batch_x.to(device=device, dtype=torch.float32)
        flat = x.reshape(-1, x.shape[-1])
        if sum_x is None:
            sum_x = flat.sum(dim=0)
            sum_x2 = (flat ** 2).sum(dim=0)
        else:
            sum_x += flat.sum(dim=0)
            sum_x2 += (flat ** 2).sum(dim=0)
        count += flat.shape[0]
    mean = sum_x / max(count, 1)
    var = (sum_x2 / max(count, 1)) - mean ** 2
    std = torch.sqrt(var.clamp_min(1e-12)).view(1, 1, -1)
    return std


def _import_object(import_path: str):
    """Import ``package.module:object`` or ``package.module.object``."""
    if ':' in import_path:
        module_name, object_name = import_path.split(':', 1)
    else:
        module_name, object_name = import_path.rsplit('.', 1)
    obj = importlib.import_module(module_name)
    for part in object_name.split('.'):
        obj = getattr(obj, part)
    return obj


def _namespace_to_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, argparse.Namespace) or isinstance(value, SimpleNamespace):
        return vars(value)
    return value


def _merge_missing(dst: dict, src):
    src = _namespace_to_dict(src)
    if not isinstance(src, dict):
        return
    for nested_key in ('config', 'model_config', 'vqvae_config', 'transformer_config', 'hparams', 'hyper_parameters'):
        nested = _namespace_to_dict(src.get(nested_key))
        if isinstance(nested, dict):
            _merge_missing(dst, nested)
    for key, value in src.items():
        if key not in dst and key not in ('config', 'model_config', 'vqvae_config', 'transformer_config', 'hparams', 'hyper_parameters'):
            dst[key] = value


def _normalize_config(config: dict, checkpoint=None) -> dict:
    """Normalize common checkpoint config aliases used by older baselines."""
    normalized = copy.deepcopy(config)
    _merge_missing(normalized, normalized)
    if isinstance(checkpoint, dict):
        _merge_missing(normalized, checkpoint.get('args'))
        for key in ('model_args', 'model_kwargs', 'config_args'):
            _merge_missing(normalized, checkpoint.get(key))

    aliases = {
        'num_hiddens': ('block_hidden_size', 'hidden_size', 'num_hidden'),
        'num_residual_hiddens': ('res_hidden_size', 'residual_hidden_size'),
        'codebook_size': ('num_embeddings', 'n_embed', 'n_embeddings'),
    }
    for canonical, candidates in aliases.items():
        if canonical not in normalized:
            for candidate in candidates:
                if candidate in normalized:
                    normalized[canonical] = normalized[candidate]
                    break

    defaults = {
        'patch_size': 16,
        'embedding_dim': 64,
        'compression_factor': 4,
        'codebook_size': 256,
        'commitment_cost': 0.25,
        'num_hiddens': 128,
        'num_residual_layers': 2,
        'num_residual_hiddens': 64,
        'n_layers': 3,
        'n_heads': 4,
        'd_ff': 256,
        'dropout': 0.1,
    }
    for key, value in defaults.items():
        normalized.setdefault(key, value)
    return normalized


def _get_checkpoint_config(checkpoint, args) -> dict:
    if not isinstance(checkpoint, dict):
        return {}

    if args.model_config_key:
        if args.model_config_key not in checkpoint:
            raise KeyError(f"Config key '{args.model_config_key}' not found in checkpoint")
        config = checkpoint[args.model_config_key]
    else:
        config = None
        for key in ('config', 'model_config', 'hparams', 'hyper_parameters'):
            if key in checkpoint:
                config = checkpoint[key]
                break
        ckpt_args = _namespace_to_dict(checkpoint.get('args'))
        if config is None and isinstance(ckpt_args, dict):
            config = ckpt_args.get('config') or ckpt_args.get('model_config') or ckpt_args

    config = _namespace_to_dict(config)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError(f'Checkpoint config must be a dict/Namespace, got {type(config)}')
    return _normalize_config(config, checkpoint)


def _augment_config(config: dict, dls, args) -> dict:
    if bool(args.inject_n_channels):
        config['n_channels'] = dls.vars
    if bool(args.inject_gumbel_args):
        config['use_gumbel_softmax'] = bool(args.use_gumbel_softmax)
        config['gumbel_temperature'] = float(args.gumbel_temperature)
        config['gumbel_hard'] = bool(args.gumbel_hard)
    return config


def _filter_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    allowed = {
        name for name, p in signature.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


def _build_model(model_ctor, config: dict, args):
    mode = args.model_init_mode
    attempts = []

    def _try(label, fn):
        try:
            return fn()
        except TypeError as exc:
            attempts.append(f'{label}: {exc}')
            return None

    if mode in ('config', 'auto'):
        model = _try('config', lambda: model_ctor(config))
        if model is not None:
            return model
    if mode in ('kwargs', 'auto'):
        kwargs = _filter_kwargs(model_ctor, config)
        model = _try('kwargs', lambda: model_ctor(**kwargs))
        if model is not None:
            return model
    if mode in ('none', 'auto'):
        model = _try('none', lambda: model_ctor())
        if model is not None:
            return model

    raise TypeError(
        f'Failed to instantiate model with --model_init_mode {mode}. '
        f'Attempts: {" | ".join(attempts)}'
    )


def _looks_like_state_dict(value) -> bool:
    return isinstance(value, dict) and bool(value) and all(isinstance(k, str) for k in value.keys())


def _get_state_dict(checkpoint, args):
    if not isinstance(checkpoint, dict):
        return None
    if args.state_dict_key:
        if args.state_dict_key not in checkpoint:
            raise KeyError(f"State dict key '{args.state_dict_key}' not found in checkpoint")
        return checkpoint[args.state_dict_key]
    for key in ('model_state_dict', 'state_dict', 'model_state', 'net_state_dict',
                'network_state_dict', 'model', 'net', 'network', 'module'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    return None


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not prefix:
        return state_dict
    keys = list(state_dict.keys())
    if not keys or not all(k.startswith(prefix) for k in keys):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _load_state_dict(model, state_dict: dict, args):
    strict = bool(args.strict_load)
    prefixes = []
    if args.strip_state_dict_prefix == 'auto':
        prefixes = ['', 'module.', 'model.', 'net.', 'network.']
    elif args.strip_state_dict_prefix:
        prefixes = [''] + [p for p in args.strip_state_dict_prefix.split(',') if p]
    else:
        prefixes = ['']

    errors = []
    for prefix in prefixes:
        candidate = _strip_prefix_if_present(state_dict, prefix)
        try:
            return model.load_state_dict(candidate, strict=strict)
        except RuntimeError as exc:
            errors.append(f'prefix={prefix or "<none>"}: {exc}')

    raise RuntimeError('Failed to load state_dict. ' + ' | '.join(errors))


def _model_from_pickled_checkpoint(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in ('model', 'net', 'network', 'module'):
            value = checkpoint.get(key)
            if isinstance(value, torch.nn.Module):
                return value
    return None


def load_model(checkpoint_path: str, dls, args, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _augment_config(_get_checkpoint_config(checkpoint, args), dls, args)

    model = _model_from_pickled_checkpoint(checkpoint)
    if model is None:
        ctor_path = args.model_ctor
        if not ctor_path and isinstance(checkpoint, dict):
            ctor_path = checkpoint.get('model_ctor') or checkpoint.get('model_class')
        if not ctor_path:
            raise ValueError(
                'Checkpoint does not contain a pickled model. Please pass --model_ctor '
                '(for example src.models.patch_vqvae_transformer:PatchVQVAETransformer).'
            )
        model = _build_model(_import_object(ctor_path), config, args)
        state_dict = _get_state_dict(checkpoint, args)
        if state_dict is None:
            raise ValueError(f'Unsupported checkpoint format: no state_dict found in {checkpoint_path}')
        _load_state_dict(model, state_dict, args)

    model = model.to(device)
    model.eval()
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {'model': checkpoint}
    return model, checkpoint_dict, config


def _noise_like(x: torch.Tensor, seed: int, batch_idx: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) * 1_000_003 + int(batch_idx))
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def _unwrap_prediction(output, args) -> torch.Tensor:
    if args.prediction_key:
        if not isinstance(output, dict):
            raise TypeError('--prediction_key requires the model output to be a dict')
        output = output[args.prediction_key]
    elif isinstance(output, dict):
        for key in ('pred', 'prediction', 'preds', 'forecast', 'output', 'y_hat'):
            if key in output:
                output = output[key]
                break
        else:
            tensors = [v for v in output.values() if torch.is_tensor(v)]
            if not tensors:
                raise TypeError('Model output dict does not contain tensor predictions')
            output = tensors[0]
    elif isinstance(output, (tuple, list)):
        tensors = [v for v in output if torch.is_tensor(v)]
        if not tensors:
            raise TypeError('Model output tuple/list does not contain tensor predictions')
        output = tensors[0]

    if not torch.is_tensor(output):
        raise TypeError(f'Model prediction must be a tensor, got {type(output)}')
    return output


def _call_forward_finetune(model, x, target_len: int, ar_step_size: int | None, pred_len: int | None):
    kwargs = {}
    if ar_step_size is not None:
        kwargs['step_size'] = ar_step_size
    if pred_len is not None:
        kwargs['pred_len'] = pred_len
    try:
        return model.forward_finetune(x, target_len, **kwargs)
    except TypeError:
        return model.forward_finetune(x, target_len)


def predict(model, x, target_len: int, args, ar_step_size: int | None, pred_len: int | None) -> torch.Tensor:
    method_name = args.model_forward_method
    if method_name == 'auto':
        if hasattr(model, 'forward_finetune'):
            output = _call_forward_finetune(model, x, target_len, ar_step_size, pred_len)
        else:
            forward_takes_target_len = args.forward_takes_target_len == '1'
            output = model(x, target_len) if forward_takes_target_len else model(x)
    elif method_name == 'forward_finetune':
        output = _call_forward_finetune(model, x, target_len, ar_step_size, pred_len)
    elif method_name == 'forward':
        if args.forward_takes_target_len == 'auto':
            try:
                output = model(x, target_len)
            except TypeError:
                output = model(x)
        else:
            output = model(x, target_len) if args.forward_takes_target_len == '1' else model(x)
    else:
        method = getattr(model, method_name)
        if args.forward_takes_target_len == '0':
            output = method(x)
        else:
            try:
                output = method(x, target_len)
            except TypeError:
                output = method(x)
    return _unwrap_prediction(output, args)


def evaluate_with_noise(
    model,
    dataloader,
    revin,
    args,
    device: torch.device,
    sigma: float,
    seed: int,
    train_std: torch.Tensor | None,
    ar_step_size: int | None,
    pred_len: int | None,
):
    model.eval()
    preds = []
    targets = []
    train_std = train_std.to(device) if train_std is not None else None

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if args.noise_position == 'before_revin':
                eps = _noise_like(batch_x, seed, batch_idx)
                if args.noise_scale == 'train_std':
                    scale = train_std.to(dtype=batch_x.dtype) if train_std is not None else 1.0
                    batch_x = batch_x + float(sigma) * scale * eps
                else:
                    batch_x = batch_x + float(sigma) * eps

            if revin:
                batch_x = revin(batch_x, 'norm')

            if args.noise_position == 'after_revin':
                eps = _noise_like(batch_x, seed, batch_idx)
                if args.noise_scale == 'train_std':
                    scale = train_std.to(dtype=batch_x.dtype) if train_std is not None else 1.0
                    batch_x = batch_x + float(sigma) * scale * eps
                else:
                    batch_x = batch_x + float(sigma) * eps

            pred = predict(
                model,
                batch_x,
                args.target_points_current,
                args,
                ar_step_size=ar_step_size,
                pred_len=pred_len,
            )

            if revin:
                pred = revin(pred, 'denorm')

            preds.append(pred.float().cpu())
            targets.append(batch_y.float().cpu())

    preds_np = torch.cat(preds, dim=0).numpy()
    targets_np = torch.cat(targets, dim=0).numpy()
    mse = float(np.mean((preds_np - targets_np) ** 2))
    mae = float(np.mean(np.abs(preds_np - targets_np)))
    return mse, mae


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_sigma(sigma: float) -> str:
    return f'{float(sigma):g}'


def write_wide_metrics(path: Path, aggregate_rows: list[dict], targets: list[int], sigmas: list[float]):
    """Write a TSV table with two header rows: sigma groups, then MSE/MAE."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key = {
        (int(r['horizon']), float(r['sigma'])): r
        for r in aggregate_rows
    }
    with path.open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['horizon'] + [v for sigma in sigmas for v in (_fmt_sigma(sigma), '')])
        writer.writerow([''] + [metric for _ in sigmas for metric in ('MSE', 'MAE')])
        for horizon in targets:
            row = [horizon]
            for sigma in sigmas:
                metrics = by_key.get((int(horizon), float(sigma)), {})
                row.extend([metrics.get('mse_mean', ''), metrics.get('mae_mean', '')])
            writer.writerow(row)


def print_wide_metrics(aggregate_rows: list[dict], targets: list[int], sigmas: list[float]):
    by_key = {
        (int(r['horizon']), float(r['sigma'])): r
        for r in aggregate_rows
    }
    print('\t'.join(['horizon'] + [v for sigma in sigmas for v in (_fmt_sigma(sigma), '')]))
    print('\t'.join([''] + [metric for _ in sigmas for metric in ('MSE', 'MAE')]))
    for horizon in targets:
        row = [str(horizon)]
        for sigma in sigmas:
            metrics = by_key.get((int(horizon), float(sigma)), {})
            mse = metrics.get('mse_mean', '')
            mae = metrics.get('mae_mean', '')
            row.extend([
                f'{mse:.6f}' if isinstance(mse, float) else str(mse),
                f'{mae:.6f}' if isinstance(mae, float) else str(mae),
            ])
        print('\t'.join(row))


def main():
    parser = argparse.ArgumentParser(
        description='Input perturbation robustness test',
        allow_abbrev=False,
    )
    parser.add_argument('--dset', type=str, default='ettm1')
    parser.add_argument('--context_points', type=int, default=96)
    parser.add_argument('--targets', type=str, default=DEFAULT_TARGETS)
    parser.add_argument('--ckpt', '--checkpoint', dest='ckpt', type=str, default=None,
                        help='Single checkpoint path used for all requested targets unless overridden by --ckpt96/192/336/720')
    parser.add_argument('--ckpt96', type=str, default=None)
    parser.add_argument('--ckpt192', type=str, default=None)
    parser.add_argument('--ckpt336', type=str, default=None)
    parser.add_argument('--ckpt720', type=str, default=None)
    parser.add_argument('--checkpoint_map', type=str, default=None,
                        help='Optional JSON mapping horizon -> checkpoint path')

    # Generic model/checkpoint interface. Defaults preserve the original PatchVQVAE path.
    parser.add_argument('--model_ctor', type=str,
                        default='src.models.patch_vqvae_transformer:PatchVQVAETransformer',
                        help='Model class/factory import path, e.g. package.module:Class')
    parser.add_argument('--model_init_mode', type=str, default='config',
                        choices=['config', 'kwargs', 'none', 'auto'],
                        help='How to instantiate --model_ctor: ctor(config), ctor(**config), ctor(), or auto')
    parser.add_argument('--model_config_key', type=str, default=None,
                        help='Checkpoint key containing model config; auto-detect if omitted')
    parser.add_argument('--state_dict_key', type=str, default=None,
                        help='Checkpoint key containing state_dict; auto-detect if omitted')
    parser.add_argument('--strict_load', type=int, default=1,
                        help='Use strict=True when loading state_dict')
    parser.add_argument('--strip_state_dict_prefix', type=str, default='auto',
                        help='auto, empty string, or comma-separated prefixes to strip before loading')
    parser.add_argument('--model_forward_method', type=str, default='auto',
                        help='auto, forward_finetune, forward, or another method name')
    parser.add_argument('--forward_takes_target_len', type=str, default='auto', choices=['auto', '0', '1'],
                        help='Whether forward/custom method receives target_len as second argument')
    parser.add_argument('--prediction_key', type=str, default=None,
                        help='If model returns a dict, choose this key as prediction tensor')
    parser.add_argument('--inject_n_channels', type=int, default=1,
                        help='Add/override config["n_channels"] from dataloader')
    parser.add_argument('--inject_gumbel_args', type=int, default=1,
                        help='Add PatchVQVAE gumbel args into config for backward compatibility')

    parser.add_argument('--ar_steps', type=str, default=DEFAULT_AR_STEPS)
    parser.add_argument('--pred_lens', type=str, default=DEFAULT_PRED_LENS)
    parser.add_argument('--sigmas', type=str, default=DEFAULT_SIGMAS)
    parser.add_argument('--seeds', type=str, default=DEFAULT_SEEDS)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--scaler', type=str, default='standard')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--channel_start', type=int, default=None)
    parser.add_argument('--channel_end', type=int, default=None)
    parser.add_argument('--channel_indices', type=str, default=None)
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--use_gumbel_softmax', type=int, default=1)
    parser.add_argument('--gumbel_temperature', type=float, default=0.8)
    parser.add_argument('--gumbel_hard', type=int, default=0)
    parser.add_argument('--noise_position', type=str, default='after_revin',
                        choices=['before_revin', 'after_revin'])
    parser.add_argument('--noise_scale', type=str, default='unit',
                        choices=['unit', 'train_std'],
                        help='unit: sigma*randn_like; train_std: sigma*std_train*epsilon')
    parser.add_argument('--output_dir', type=str, default='robustness_test/results/ettm1_input_perturbation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    targets = _parse_csv_ints(args.targets)
    ar_steps = _parse_csv_ints(args.ar_steps)
    pred_lens = _parse_csv_ints(args.pred_lens)
    sigmas = _parse_csv_floats(args.sigmas)
    seeds = _parse_csv_ints(args.seeds)
    if len(targets) == 1 and len(ar_steps) > 1:
        ar_steps = [ar_steps[0]]
    if len(targets) == 1 and len(pred_lens) > 1:
        pred_lens = [pred_lens[0]]
    if len(ar_steps) not in (1, len(targets)):
        raise ValueError('--ar_steps length must be 1 or match --targets')
    if len(pred_lens) not in (1, len(targets)):
        raise ValueError('--pred_lens length must be 1 or match --targets')

    ckpt_map = _build_checkpoint_map(args, targets)
    missing = [h for h in targets if h not in ckpt_map]
    if missing:
        raise ValueError(f'Missing checkpoint(s) for target horizon(s): {missing}')

    device = torch.device(args.device)
    output_dir = Path(_resolve_path(args.output_dir))
    seed_rows = []
    aggregate_rows = []

    for idx, horizon in enumerate(targets):
        ar_step = ar_steps[idx] if len(ar_steps) > 1 else ar_steps[0]
        pred_len = pred_lens[idx] if len(pred_lens) > 1 else pred_lens[0]
        args.target_points_current = horizon

        data_args = make_data_args(args, horizon)
        dls = get_dls(data_args)
        revin = RevIN(dls.vars, eps=1e-5, affine=False).to(device) if args.revin else None
        train_std = compute_train_std(dls.train, device) if args.noise_scale == 'train_std' else None
        model, checkpoint, config = load_model(ckpt_map[horizon], dls, args, device)
        saved_args = _namespace_to_dict(checkpoint.get('args', {})) if isinstance(checkpoint, dict) else {}
        saved_args = saved_args if isinstance(saved_args, dict) else {}
        if ar_step < 0:
            ar_step = saved_args.get('ar_step_size', saved_args.get('progressive_step_size'))
        if pred_len < 0:
            pred_len = saved_args.get('pred_len', ar_step)

        print('=' * 80)
        print(f'Horizon={horizon} | checkpoint={ckpt_map[horizon]}')
        print(f'model_ctor={args.model_ctor or "<pickled/checkpoint>"} | forward={args.model_forward_method}')
        print(f'ar_step={ar_step} | pred_len={pred_len} | channels={dls.vars}')
        print(f'noise_position={args.noise_position} | noise_scale={args.noise_scale}')

        horizon_seed_rows = []
        baseline_mse = None
        for sigma in sigmas:
            for seed in seeds:
                set_seed(seed)
                mse, mae = evaluate_with_noise(
                    model, dls.test, revin, args, device,
                    sigma=sigma, seed=seed, train_std=train_std,
                    ar_step_size=ar_step, pred_len=pred_len,
                )
                row = {
                    'horizon': horizon,
                    'sigma': sigma,
                    'seed': seed,
                    'mse': mse,
                    'mae': mae,
                    'checkpoint': ckpt_map[horizon],
                    'ar_step_size': ar_step,
                    'pred_len': pred_len,
                    'noise_position': args.noise_position,
                    'noise_scale': args.noise_scale,
                }
                horizon_seed_rows.append(row)
                seed_rows.append(row)
                print(f'  sigma={sigma:.2f} seed={seed} | MSE={mse:.6f} MAE={mae:.6f}')

        sigma0 = [r['mse'] for r in horizon_seed_rows if abs(float(r['sigma'])) < 1e-12]
        baseline_mse = float(np.mean(sigma0)) if sigma0 else None
        for sigma in sigmas:
            rows = [r for r in horizon_seed_rows if abs(float(r['sigma']) - sigma) < 1e-12]
            mse_values = np.array([r['mse'] for r in rows], dtype=np.float64)
            mae_values = np.array([r['mae'] for r in rows], dtype=np.float64)
            rel = ((float(mse_values.mean()) - baseline_mse) / baseline_mse * 100.0) if baseline_mse else float('nan')
            aggregate_rows.append({
                'horizon': horizon,
                'sigma': sigma,
                'mse_mean': float(mse_values.mean()),
                'mse_std': float(mse_values.std(ddof=0)),
                'mae_mean': float(mae_values.mean()),
                'mae_std': float(mae_values.std(ddof=0)),
                'relative_mse_increase_pct': rel,
                'n_seeds': len(rows),
                'checkpoint': ckpt_map[horizon],
                'ar_step_size': ar_step,
                'pred_len': pred_len,
                'noise_position': args.noise_position,
                'noise_scale': args.noise_scale,
            })

    seed_fields = [
        'horizon', 'sigma', 'seed', 'mse', 'mae', 'relative_mse_increase_pct',
        'checkpoint', 'ar_step_size', 'pred_len', 'noise_position', 'noise_scale',
    ]
    # Fill seed-level relative degradation from aggregate baseline for convenience.
    baseline_by_h = {
        r['horizon']: r['mse_mean'] for r in aggregate_rows if abs(float(r['sigma'])) < 1e-12
    }
    for r in seed_rows:
        base = baseline_by_h.get(r['horizon'])
        r['relative_mse_increase_pct'] = ((r['mse'] - base) / base * 100.0) if base else float('nan')

    agg_fields = [
        'horizon', 'sigma', 'mse_mean', 'mse_std', 'mae_mean', 'mae_std',
        'relative_mse_increase_pct', 'n_seeds', 'checkpoint', 'ar_step_size',
        'pred_len', 'noise_position', 'noise_scale',
    ]
    wide_path = output_dir / 'aggregate_metrics_wide.tsv'
    write_csv(output_dir / 'seed_metrics.csv', seed_rows, seed_fields)
    write_csv(output_dir / 'aggregate_metrics.csv', aggregate_rows, agg_fields)
    write_wide_metrics(wide_path, aggregate_rows, targets, sigmas)
    print('=' * 80)
    print_wide_metrics(aggregate_rows, targets, sigmas)
    print('=' * 80)
    print(f'Saved seed metrics: {output_dir / "seed_metrics.csv"}')
    print(f'Saved aggregate metrics: {output_dir / "aggregate_metrics.csv"}')
    print(f'Saved wide metrics: {wide_path}')


if __name__ == '__main__':
    main()
