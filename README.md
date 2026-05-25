# Time DeCode

本仓库是论文 **Time DeCode: Denoising Codebook-based Pre-training Framework for Time Series Analysis** 中提出方法的代码实现。

核心训练流程分为三步：

1. **Codebook training**：训练 VQ-VAE / RVQ 码本，把连续时间序列 patch 离散化为 code tokens。
2. **Pre-training**：基于码本 token 做 denoising / next-token style 的预训练。
3. **Fine-tuning**：加载预训练 checkpoint，在指定数据集、输入长度和预测长度上微调并评估 MSE / MAE。

## 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果 `torch` 安装失败，请根据你的 CUDA 版本先从 PyTorch 官方源单独安装 `torch`，再安装其它依赖。

## 数据准备

数据文件默认放在仓库根目录的 `datasets/` 下，例如：

```text
datasets/ETTm1.csv
datasets/ETTm2.csv
datasets/ETTh1.csv
datasets/ETTh2.csv
datasets/electricity.csv
datasets/traffic.csv
datasets/weather.csv
```

支持的数据集名称：

```text
ettm1 ettm2 etth1 etth2 electricity traffic weather illness exchange
```

脚本中也支持用 `ecl` 代替 `electricity`。

## 一键运行完整流程

如果只想跑一个数据集、一个输入长度、一个预测长度，直接用：

```bash
bash scripts/single_run.sh --dset etth2 --input_len 96 --output_len 336
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/single_run.sh --dset etth2 --input_len 96 --output_len 336
```

常用参数：

```bash
--forecast_step_size N      # finetune 自回归步长，单位 patch
--forecast_pred_len N       # 每次 forward 预测的 patch 数
--max_channels_per_model N  # 高维数据集通道分组大小
--force                     # 强制重跑完整流程
--no_resume                 # 不复用最近一次 run
--no_channel_groups         # 不使用频域通道分组
```

例子：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/single_run.sh \
  --dset weather \
  --input_len 96 \
  --output_len 720 \
  --forecast_step_size 12 \
  --forecast_pred_len 14
```

## 分阶段运行

### 1. 单独训练 Codebook

```bash
bash scripts/train_codebook_only.sh \
  --dset etth2 \
  --context_points 512
```

指定常用结构参数：

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

高维数据集可以指定通道范围：

```bash
bash scripts/train_codebook_only.sh \
  --dset traffic \
  --context_points 336 \
  --channel_start 0 \
  --channel_end 128
```

默认输出位置：

```text
vqvae-only/saved_models/vqvae_only/<run_name>/<dset>/
```

后续 pretrain 需要使用这里生成的 `.pth` 文件。

### 2. 单独 Pre-train

```bash
bash scripts/pretrain_only.sh \
  --dset etth2 \
  --context_points 296 \
  --progressive_step_size 3 \
  --pred_len 6 \
  --vqvae_checkpoint /absolute/path/to/codebook.pth
```

使用 `timefilter_lite`：

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

RVQ 权重示例：

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

默认输出位置：

```text
decoder_only_NTP/saved_models/patch_vqvae/<run_name>/<dset>/
```

后续 finetune 需要使用这里生成的 `.pth` 文件。

### 3. 单独 Fine-tune

```bash
bash scripts/finetune_only.sh \
  --dset etth2 \
  --input_len 96 \
  --output_len 336 \
  --pretrained_model /absolute/path/to/pretrain.pth
```

显式指定自回归参数：

```bash
bash scripts/finetune_only.sh \
  --dset etth2 \
  --input_len 96 \
  --output_len 720 \
  --ar_step_size 4 \
  --pred_len 6 \
  --pretrained_model /absolute/path/to/pretrain.pth
```

使用 Huber loss：

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

默认输出位置：

```text
decoder_only_NTP/saved_models/patch_vqvae_finetune/<run_name>/<dset>/
```

## 三阶段完整示例

```bash
# 1. Train codebook
bash scripts/train_codebook_only.sh \
  --dset etth2 \
  --context_points 512

# 2. Pre-train，替换为上一步生成的 codebook checkpoint
bash scripts/pretrain_only.sh \
  --dset etth2 \
  --context_points 296 \
  --progressive_step_size 3 \
  --pred_len 6 \
  --vqvae_checkpoint /absolute/path/to/codebook.pth

# 3. Fine-tune，替换为上一步生成的 pretrain checkpoint
bash scripts/finetune_only.sh \
  --dset etth2 \
  --input_len 96 \
  --output_len 336 \
  --ar_step_size 7 \
  --pred_len 14 \
  --pretrained_model /absolute/path/to/pretrain.pth
```

如果使用 `--channel_start/--channel_end` 或 `--channel_indices`，三阶段必须保持一致。

## 运行已有 best 配置

```bash
bash scripts/etth1_best.sh
bash scripts/etth2_best.sh
bash scripts/ettm1_best.sh
bash scripts/ettm2_best.sh
bash scripts/ECL_best.sh
bash scripts/traffic_best.sh
bash scripts/weather_best.sh
```

部分脚本支持只跑指定预测长度：

```bash
HORIZONS="96 192" bash scripts/weather_best.sh
HORIZONS="96,192,336,720" bash scripts/traffic_best.sh
```

## 查看结果

完整 pipeline 的日志和汇总结果通常保存在：

```text
logs/<run_name>/summary.tsv
logs/<run_name>/summary_overall.tsv
```

模型 checkpoint 默认保存在：

```text
vqvae-only/saved_models/vqvae_only/<run_name>/
decoder_only_NTP/saved_models/patch_vqvae/<run_name>/
decoder_only_NTP/saved_models/patch_vqvae_finetune/<run_name>/
```

## 注意事项

1. 三阶段的 VQ-VAE 结构参数要保持一致，尤其是 `PATCH_SIZE`、`COMPRESSION_FACTOR`、`EMBEDDING_DIM`、`CODEBOOK_SIZE`、`N_RQ_LAYERS`。
2. `finetune` 的模型结构由 `--pretrained_model` checkpoint 中的 config 决定。
3. 脚本内部不固定 GPU；需要指定 GPU 时在命令前加 `CUDA_VISIBLE_DEVICES=...`。
4. `--force` 或 `FORCE_RETRAIN_*` 可能覆盖/删除同名旧模型，使用前请确认。
