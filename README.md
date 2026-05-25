# Time DeCode

This repository provides the code implementation for **Time DeCode: Denoising Codebook-based Pre-training Framework for Time Series Analysis**.

Time DeCode is a denoising codebook-based pre-training framework for time series analysis. It is designed around three sources of noise in real-world time series pre-training:

- **Data-level noise**: noisy raw observations caused by measurement errors, stochastic fluctuations, and environmental disturbances.
- **Objective-level noise**: continuous-value reconstruction or prediction objectives may force the model to fit local random deviations.
- **Inference-level noise**: autoregressive rollout with a single prediction trajectory can accumulate errors under noisy temporal contexts.

To mitigate these issues, Time DeCode introduces three main components:

1. **Noise-Aware Discrete Tokenizer**: converts continuous time-series patches into discrete semantic codes. It uses sparse noise extraction to separate irregular local fluctuations from structural temporal patterns, and learns multi-layer codebooks for multi-frequency representations.
2. **Next Multi-Code Prediction (NMCP)**: reformulates pre-training from continuous-value prediction to future code-index prediction. The temporal model predicts multiple future code tokens from historical code sequences.
3. **Code-Space Prediction Calibration**: performs forecasting in the learned code space. Overlapping future code predictions are aggregated before decoding, improving inference stability and reducing error accumulation.

The current code supports long-term time series forecasting with MSE / MAE evaluation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If installing `torch` from `requirements.txt` fails, install the PyTorch wheel matching your CUDA version first, then install the remaining dependencies.

## Data Preparation

Put datasets under `datasets/` in the repository root. Typical file names are:

```text
datasets/ETTm1.csv
datasets/ETTm2.csv
datasets/ETTh1.csv
datasets/ETTh2.csv
datasets/electricity.csv
datasets/traffic.csv
datasets/weather.csv
```

Supported dataset names:

```text
ettm1 ettm2 etth1 etth2 electricity traffic weather illness exchange
```

`ecl` is accepted as an alias of `electricity` in the provided scripts.

## Quick Start: Full Pipeline

Run codebook training, pre-training, and fine-tuning for one dataset, one input length, and one prediction length:

```bash
bash scripts/single_run.sh --dset etth2 --input_len 96 --output_len 336
```

Specify GPU outside the script:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/single_run.sh --dset etth2 --input_len 96 --output_len 336
```

Common options:

```bash
--forecast_step_size N      # autoregressive step size in patches
--forecast_pred_len N       # number of future patches predicted per forward step
--max_channels_per_model N  # channel-group size for high-dimensional datasets
--force                     # rerun all stages
--no_resume                 # do not reuse the latest run
--no_channel_groups         # disable frequency-based channel grouping
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/single_run.sh \
  --dset weather \
  --input_len 96 \
  --output_len 720 \
  --forecast_step_size 12 \
  --forecast_pred_len 14
```

## Stage 1: Train Noise-Aware Codebook Tokenizer

This stage trains the VQ-VAE / RVQ tokenizer. It learns to map continuous patches into discrete codebook entries.

```bash
bash scripts/train_codebook_only.sh \
  --dset etth2 \
  --context_points 512
```

With common tokenizer settings:

```bash
PATCH_SIZE=8 \
COMPRESSION_FACTOR=4 \
EMBEDDING_DIM=64 \
CODEBOOK_SIZE=256 \
N_RQ_LAYERS=2 \
bash scripts/train_codebook_only.sh \
  --dset ettm2 \
  --context_points 512
```

For high-dimensional datasets, train on a channel slice:

```bash
bash scripts/train_codebook_only.sh \
  --dset traffic \
  --context_points 336 \
  --channel_start 0 \
  --channel_end 128
```

Useful environment variables:

```bash
PATCH_SIZE                # patch length
COMPRESSION_FACTOR        # latent compression ratio
EMBEDDING_DIM             # latent embedding dimension
CODEBOOK_SIZE             # number of entries per codebook
N_RQ_LAYERS               # number of residual quantization layers
SPARSE_WEIGHT             # sparse noise extraction weight
LAMBDA_ORD                # frequency-ordering regularization weight
ORTH_WEIGHT               # noise-codebook orthogonality weight
CB_EPOCHS                 # codebook training epochs
CB_LR                     # codebook learning rate
```

Default output:

```text
vqvae-only/saved_models/vqvae_only/<run_name>/<dset>/
```

Use the generated `.pth` checkpoint as `--vqvae_checkpoint` in the pre-training stage.

## Stage 2: Pre-train with Next Multi-Code Prediction

This stage loads the trained tokenizer and pre-trains a temporal backbone in code space.

```bash
bash scripts/pretrain_only.sh \
  --dset etth2 \
  --context_points 296 \
  --progressive_step_size 3 \
  --pred_len 6 \
  --vqvae_checkpoint /absolute/path/to/codebook.pth
```

Use `timefilter_lite` as temporal backbone:

```bash
TEMPORAL_BACKBONE=timefilter_lite \
TIMEFILTER_TOPK=8 \
bash scripts/pretrain_only.sh \
  --dset weather \
  --context_points 672 \
  --progressive_step_size 6 \
  --pred_len 6 \
  --vqvae_checkpoint /absolute/path/to/codebook.pth
```

Use multi-layer RVQ prediction weights:

```bash
N_RQ_LAYERS=2 \
RQ_LAYER_WEIGHTS="1.0 0.5" \
bash scripts/pretrain_only.sh \
  --dset ettm2 \
  --context_points 336 \
  --progressive_step_size 10 \
  --pred_len 12 \
  --vqvae_checkpoint /absolute/path/to/codebook.pth
```

Key arguments:

```bash
--progressive_step_size M  # context grows by M patches at each progressive stage
--pred_len N              # predict N future patches at each stage; N > M enables overlap
```

Default output:

```text
decoder_only_NTP/saved_models/patch_vqvae/<run_name>/<dset>/
```

Use the generated `.pth` checkpoint as `--pretrained_model` in the fine-tuning stage.

## Stage 3: Fine-tune and Forecast with Code-Space Calibration

This stage loads the pre-trained model, predicts future code representations, decodes them into continuous values, and reports MSE / MAE.

```bash
bash scripts/finetune_only.sh \
  --dset etth2 \
  --input_len 96 \
  --output_len 336 \
  --pretrained_model /absolute/path/to/pretrain.pth
```

Explicitly set autoregressive prediction parameters:

```bash
bash scripts/finetune_only.sh \
  --dset etth2 \
  --input_len 96 \
  --output_len 720 \
  --ar_step_size 4 \
  --pred_len 6 \
  --pretrained_model /absolute/path/to/pretrain.pth
```

Use Huber loss for fine-tuning:

```bash
TRAIN_LOSS=huber \
HUBER_DELTA=1.8 \
FINETUNE_LR=1e-4 \
bash scripts/finetune_only.sh \
  --dset etth2 \
  --input_len 96 \
  --output_len 720 \
  --pretrained_model /absolute/path/to/pretrain.pth
```

Default output:

```text
decoder_only_NTP/saved_models/patch_vqvae_finetune/<run_name>/<dset>/
```

## Complete Three-Stage Example

```bash
# 1. Train the noise-aware tokenizer / codebook
bash scripts/train_codebook_only.sh \
  --dset etth2 \
  --context_points 512

# 2. Pre-train with Next Multi-Code Prediction
bash scripts/pretrain_only.sh \
  --dset etth2 \
  --context_points 296 \
  --progressive_step_size 3 \
  --pred_len 6 \
  --vqvae_checkpoint /absolute/path/to/codebook.pth

# 3. Fine-tune for downstream forecasting
bash scripts/finetune_only.sh \
  --dset etth2 \
  --input_len 96 \
  --output_len 336 \
  --ar_step_size 7 \
  --pred_len 14 \
  --pretrained_model /absolute/path/to/pretrain.pth
```

If `--channel_start/--channel_end` or `--channel_indices` is used, keep the same channel configuration across all three stages.

## Run Existing Best Configurations

```bash
bash scripts/etth1_best.sh
bash scripts/etth2_best.sh
bash scripts/ettm1_best.sh
bash scripts/ettm2_best.sh
bash scripts/ECL_best.sh
bash scripts/traffic_best.sh
bash scripts/weather_best.sh
```

Some scripts support selecting prediction horizons:

```bash
HORIZONS="96 192" bash scripts/weather_best.sh
HORIZONS="96,192,336,720" bash scripts/traffic_best.sh
```

## Evaluation Results

For the full pipeline, logs and summaries are usually saved under:

```text
logs/<run_name>/summary.tsv
logs/<run_name>/summary_overall.tsv
```

Checkpoints are saved under:

```text
vqvae-only/saved_models/vqvae_only/<run_name>/
decoder_only_NTP/saved_models/patch_vqvae/<run_name>/
decoder_only_NTP/saved_models/patch_vqvae_finetune/<run_name>/
```

## Notes

1. Keep tokenizer-related hyperparameters consistent across codebook training, pre-training, and fine-tuning, especially `PATCH_SIZE`, `COMPRESSION_FACTOR`, `EMBEDDING_DIM`, `CODEBOOK_SIZE`, and `N_RQ_LAYERS`.
2. The fine-tuning model architecture is reconstructed from the `--pretrained_model` checkpoint config.
3. Scripts do not set GPU ids internally. Use `CUDA_VISIBLE_DEVICES=...` outside the command when needed.
4. `--force` or `FORCE_RETRAIN_*` may overwrite or remove existing artifacts with the same run name.
