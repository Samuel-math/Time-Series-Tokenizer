"""
示例脚本：如何分别加载VQVAE的encoder、codebook和decoder权重
并在后续训练中使用这些组件
"""

import torch
import json
import os
from src.models.vqvae import vqvae


def load_vqvae_components(checkpoint_path, device='cpu'):
    """
    从保存的VQVAE模型中分别加载encoder、codebook和decoder的权重
    
    Args:
        checkpoint_path: 保存的模型路径 (例如: 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth')
        device: 设备 ('cpu' 或 'cuda')
    
    Returns:
        encoder_state_dict: encoder的权重字典
        codebook_state_dict: codebook的权重字典 (vq._embedding)
        decoder_state_dict: decoder的权重字典
        vqvae_config: 模型配置信息
    """
    # 加载完整的模型
    model = torch.load(checkpoint_path, map_location=device)
    
    # 提取各个组件的state_dict
    encoder_state_dict = model.encoder.state_dict()
    codebook_state_dict = model.vq._embedding.state_dict()  # codebook就是vq中的_embedding
    decoder_state_dict = model.decoder.state_dict()
    
    # 如果模型有compression_factor属性，保存它
    compression_factor = model.compression_factor if hasattr(model, 'compression_factor') else None
    
    print(f"成功加载模型组件:")
    print(f"  - Encoder参数数量: {sum(p.numel() for p in encoder_state_dict.values())}")
    print(f"  - Codebook参数数量: {sum(p.numel() for p in codebook_state_dict.values())}")
    print(f"  - Decoder参数数量: {sum(p.numel() for p in decoder_state_dict.values())}")
    
    return encoder_state_dict, codebook_state_dict, decoder_state_dict, compression_factor


def load_vqvae_config(config_path):
    """
    加载VQVAE的配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        vqvae_config: 配置字典
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['vqvae_config']


def create_vqvae_from_components(vqvae_config, 
                                  encoder_state_dict=None,
                                  codebook_state_dict=None,
                                  decoder_state_dict=None,
                                  load_encoder=True,
                                  load_codebook=True,
                                  load_decoder=True,
                                  strict=True):
    """
    创建新的VQVAE模型，并选择性加载预训练的组件权重
    
    Args:
        vqvae_config: VQVAE配置字典
        encoder_state_dict: encoder的权重字典 (可选)
        codebook_state_dict: codebook的权重字典 (可选)
        decoder_state_dict: decoder的权重字典 (可选)
        load_encoder: 是否加载encoder权重
        load_codebook: 是否加载codebook权重
        load_decoder: 是否加载decoder权重
        strict: 是否严格匹配权重名称
    
    Returns:
        model: 创建的VQVAE模型
    """
    # 创建新模型
    model = vqvae(vqvae_config)
    
    # 选择性加载权重
    if load_encoder and encoder_state_dict is not None:
        try:
            model.encoder.load_state_dict(encoder_state_dict, strict=strict)
            print("✓ Encoder权重加载成功")
        except Exception as e:
            print(f"✗ Encoder权重加载失败: {e}")
            if not strict:
                # 尝试部分加载
                model_dict = model.encoder.state_dict()
                pretrained_dict = {k: v for k, v in encoder_state_dict.items() 
                                 if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                model.encoder.load_state_dict(model_dict)
                print(f"  部分加载了 {len(pretrained_dict)}/{len(encoder_state_dict)} 个参数")
    
    if load_codebook and codebook_state_dict is not None:
        try:
            model.vq._embedding.load_state_dict(codebook_state_dict, strict=strict)
            print("✓ Codebook权重加载成功")
        except Exception as e:
            print(f"✗ Codebook权重加载失败: {e}")
    
    if load_decoder and decoder_state_dict is not None:
        try:
            model.decoder.load_state_dict(decoder_state_dict, strict=strict)
            print("✓ Decoder权重加载成功")
        except Exception as e:
            print(f"✗ Decoder权重加载失败: {e}")
            if not strict:
                # 尝试部分加载
                model_dict = model.decoder.state_dict()
                pretrained_dict = {k: v for k, v in decoder_state_dict.items() 
                                 if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                model.decoder.load_state_dict(model_dict)
                print(f"  部分加载了 {len(pretrained_dict)}/{len(decoder_state_dict)} 个参数")
    
    return model


def freeze_components(model, freeze_encoder=False, freeze_codebook=False, freeze_decoder=False):
    """
    冻结模型组件的参数
    
    Args:
        model: VQVAE模型
        freeze_encoder: 是否冻结encoder
        freeze_codebook: 是否冻结codebook
        freeze_decoder: 是否冻结decoder
    """
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder已冻结")
    
    if freeze_codebook:
        for param in model.vq._embedding.parameters():
            param.requires_grad = False
        print("✓ Codebook已冻结")
    
    if freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("✓ Decoder已冻结")


# ========== 使用示例 ==========

def example_1_load_all_components():
    """
    示例1: 加载所有组件并创建新模型
    """
    print("\n" + "="*60)
    print("示例1: 加载所有组件")
    print("="*60)
    
    # 模型路径
    checkpoint_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth'
    config_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json'
    
    # 加载组件
    encoder_state, codebook_state, decoder_state, compression_factor = load_vqvae_components(
        checkpoint_path, device='cpu'
    )
    
    # 加载配置
    vqvae_config = load_vqvae_config(config_path)
    
    # 创建新模型并加载所有组件
    model = create_vqvae_from_components(
        vqvae_config,
        encoder_state_dict=encoder_state,
        codebook_state_dict=codebook_state,
        decoder_state_dict=decoder_state,
        load_encoder=True,
        load_codebook=True,
        load_decoder=True
    )
    
    return model


def example_2_load_only_encoder_and_codebook():
    """
    示例2: 只加载encoder和codebook，用于特征提取
    """
    print("\n" + "="*60)
    print("示例2: 只加载encoder和codebook")
    print("="*60)
    
    checkpoint_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth'
    config_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json'
    
    # 加载组件
    encoder_state, codebook_state, decoder_state, _ = load_vqvae_components(
        checkpoint_path, device='cpu'
    )
    
    # 加载配置
    vqvae_config = load_vqvae_config(config_path)
    
    # 只加载encoder和codebook
    model = create_vqvae_from_components(
        vqvae_config,
        encoder_state_dict=encoder_state,
        codebook_state_dict=codebook_state,
        decoder_state_dict=None,
        load_encoder=True,
        load_codebook=True,
        load_decoder=False  # 不加载decoder
    )
    
    # 冻结encoder和codebook，只训练decoder
    freeze_components(model, freeze_encoder=True, freeze_codebook=True, freeze_decoder=False)
    
    return model


def example_3_finetune_with_frozen_encoder():
    """
    示例3: 微调场景 - 冻结encoder，训练codebook和decoder
    """
    print("\n" + "="*60)
    print("示例3: 微调 - 冻结encoder")
    print("="*60)
    
    checkpoint_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth'
    config_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json'
    
    # 加载组件
    encoder_state, codebook_state, decoder_state, _ = load_vqvae_components(
        checkpoint_path, device='cpu'
    )
    
    # 加载配置
    vqvae_config = load_vqvae_config(config_path)
    
    # 加载所有组件
    model = create_vqvae_from_components(
        vqvae_config,
        encoder_state_dict=encoder_state,
        codebook_state_dict=codebook_state,
        decoder_state_dict=decoder_state,
        load_encoder=True,
        load_codebook=True,
        load_decoder=True
    )
    
    # 冻结encoder，允许codebook和decoder训练
    freeze_components(model, freeze_encoder=True, freeze_codebook=False, freeze_decoder=False)
    
    # 检查哪些参数需要训练
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n可训练参数: {trainable_params:,} / 总参数: {total_params:,}")
    
    return model


def example_4_use_in_training():
    """
    示例4: 在训练循环中使用加载的组件
    """
    print("\n" + "="*60)
    print("示例4: 在训练中使用")
    print("="*60)
    
    checkpoint_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth'
    config_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json'
    
    # 加载组件和配置
    encoder_state, codebook_state, decoder_state, _ = load_vqvae_components(
        checkpoint_path, device='cpu'
    )
    vqvae_config = load_vqvae_config(config_path)
    
    # 创建模型并加载权重
    model = create_vqvae_from_components(
        vqvae_config,
        encoder_state_dict=encoder_state,
        codebook_state_dict=codebook_state,
        decoder_state_dict=decoder_state
    )
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 创建优化器（可以选择性地只优化某些组件）
    # 方式1: 优化所有可训练参数
    optimizer = model.configure_optimizers(lr=vqvae_config['learning_rate'])
    
    # 方式2: 只优化特定组件（例如只优化decoder）
    # optimizer = torch.optim.Adam(model.decoder.parameters(), lr=vqvae_config['learning_rate'])
    
    # 方式3: 为不同组件设置不同学习率
    # optimizer = torch.optim.Adam([
    #     {'params': model.encoder.parameters(), 'lr': 1e-5},  # 较小的学习率
    #     {'params': model.vq.parameters(), 'lr': 1e-4},
    #     {'params': model.decoder.parameters(), 'lr': 1e-3}   # 较大的学习率
    # ])
    
    print(f"\n模型已准备好，可以开始训练")
    print(f"设备: {device}")
    print(f"优化器: {type(optimizer).__name__}")
    
    # 示例训练步骤（伪代码）
    # model.train()
    # for batch_x, _ in train_loader:
    #     batch_x = batch_x.to(device)
    #     B, L, C = batch_x.shape
    #     
    #     for ch in range(C):
    #         x_ch = batch_x[:, :, ch].view(B, L)
    #         loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, \
    #         encoding_indices, encodings = model.shared_eval(x_ch, optimizer, 'train')
    
    return model, optimizer


if __name__ == '__main__':
    # 检查模型文件是否存在
    checkpoint_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"警告: 模型文件不存在: {checkpoint_path}")
        print("请修改checkpoint_path为您的实际模型路径")
    else:
        # 运行示例
        print("VQVAE组件加载示例")
        print("="*60)
        
        # 示例1: 加载所有组件
        model1 = example_1_load_all_components()
        
        # 示例2: 只加载encoder和codebook
        # model2 = example_2_load_only_encoder_and_codebook()
        
        # 示例3: 微调场景
        # model3 = example_3_finetune_with_frozen_encoder()
        
        # 示例4: 在训练中使用
        # model4, optimizer = example_4_use_in_training()
        
        print("\n" + "="*60)
        print("所有示例完成！")
        print("="*60)

