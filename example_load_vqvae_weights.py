"""
示例：如何使用VQVAETransformerPretrain加载预训练的VQVAE权重
"""

import torch
from src.models.vqvae_transformer import VQVAETransformerPretrain
from src.utils_vqvae import load_vqvae_config

# 示例1: 不加载VQVAE权重（随机初始化）
print("=" * 60)
print("示例1: 不加载VQVAE权重")
print("=" * 60)

vqvae_config = {
    'block_hidden_size': 128,
    'num_residual_layers': 2,
    'res_hidden_size': 64,
    'embedding_dim': 64,
    'num_embeddings': 256,
    'commitment_cost': 0.25,
    'compression_factor': 4
}

transformer_config = {
    'd_model': 128,
    'n_layers': 3,
    'n_heads': 8,
    'd_ff': 256,
    'dropout': 0.1,
    'attn_dropout': 0.1
}

# 不加载权重
model1 = VQVAETransformerPretrain(
    vqvae_config, 
    transformer_config,
    load_vqvae_weights=False
)
print(f"模型1参数总数: {sum(p.numel() for p in model1.parameters()):,}\n")


# 示例2: 加载预训练的VQVAE权重
print("=" * 60)
print("示例2: 加载预训练的VQVAE权重")
print("=" * 60)

# 从配置文件加载VQVAE配置
vqvae_config_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json'
vqvae_checkpoint_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth'

try:
    vqvae_config = load_vqvae_config(vqvae_config_path)
    
    # 加载权重
    model2 = VQVAETransformerPretrain(
        vqvae_config,
        transformer_config,
        load_vqvae_weights=True,
        vqvae_checkpoint_path=vqvae_checkpoint_path,
        device='cpu'
    )
    print(f"模型2参数总数: {sum(p.numel() for p in model2.parameters()):,}\n")
    
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
    print("请确保VQVAE模型路径正确\n")


# 示例3: 在预训练脚本中使用
print("=" * 60)
print("示例3: 在预训练脚本中使用")
print("=" * 60)
print("""
在 vqvae_transformer_pretrain.py 中，可以这样使用：

python vqvae_transformer_pretrain.py \\
    --vqvae_config_path saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json \\
    --vqvae_checkpoint saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth \\
    --dset_pretrain ettm1 \\
    --context_points 512 \\
    --batch_size 64 \\
    --n_epochs_pretrain 50

模型会自动检测checkpoint路径是否存在，如果存在则加载权重。
""")

print("=" * 60)
print("完成！")
print("=" * 60)

