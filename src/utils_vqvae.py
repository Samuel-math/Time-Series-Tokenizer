"""
VQVAE组件加载的实用工具函数
可以在训练脚本中直接导入使用
"""

import torch
import json
from src.models.vqvae import vqvae


def load_vqvae_checkpoint(checkpoint_path, device='cpu'):
    """
    从checkpoint加载VQVAE模型并返回各个组件的state_dict
    
    Args:
        checkpoint_path: 模型checkpoint路径
        device: 设备
    
    Returns:
        dict: 包含 'encoder', 'codebook', 'decoder', 'compression_factor' 的字典
    """
    model = torch.load(checkpoint_path, map_location=device)
    
    return {
        'encoder': model.encoder.state_dict(),
        'codebook': model.vq._embedding.state_dict(),
        'decoder': model.decoder.state_dict(),
        'compression_factor': getattr(model, 'compression_factor', None)
    }


def load_vqvae_config(config_path):
    """加载VQVAE配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['vqvae_config']


def init_vqvae_with_components(vqvae_config, 
                               checkpoint_path=None,
                               components_dict=None,
                               load_encoder=True,
                               load_codebook=True,
                               load_decoder=True,
                               freeze_encoder=False,
                               freeze_codebook=False,
                               freeze_decoder=False,
                               device='cpu'):
    """
    初始化VQVAE模型并加载预训练组件
    
    Args:
        vqvae_config: VQVAE配置字典
        checkpoint_path: checkpoint路径（如果提供，会从中加载组件）
        components_dict: 组件字典（如果提供，直接使用；否则从checkpoint_path加载）
        load_encoder: 是否加载encoder权重
        load_codebook: 是否加载codebook权重
        load_decoder: 是否加载decoder权重
        freeze_encoder: 是否冻结encoder
        freeze_codebook: 是否冻结codebook
        freeze_decoder: 是否冻结decoder
        device: 设备
    
    Returns:
        model: 初始化好的VQVAE模型
    """
    # 创建模型
    model = vqvae(vqvae_config)
    
    # 加载组件
    if checkpoint_path is not None:
        components = load_vqvae_checkpoint(checkpoint_path, device)
    elif components_dict is not None:
        components = components_dict
    else:
        components = None
    
    # 加载权重
    if components is not None:
        if load_encoder and 'encoder' in components:
            try:
                model.encoder.load_state_dict(components['encoder'], strict=True)
                print("✓ Encoder权重已加载")
            except Exception as e:
                print(f"✗ Encoder加载失败: {e}")
        
        if load_codebook and 'codebook' in components:
            try:
                model.vq._embedding.load_state_dict(components['codebook'], strict=True)
                print("✓ Codebook权重已加载")
            except Exception as e:
                print(f"✗ Codebook加载失败: {e}")
        
        if load_decoder and 'decoder' in components:
            try:
                model.decoder.load_state_dict(components['decoder'], strict=True)
                print("✓ Decoder权重已加载")
            except Exception as e:
                print(f"✗ Decoder加载失败: {e}")
    
    # 冻结组件
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
    
    model = model.to(device)
    return model


# ========== 使用示例 ==========

if __name__ == '__main__':
    # 示例：在训练脚本中使用
    
    # 方式1: 从checkpoint加载所有组件
    config_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json'
    checkpoint_path = 'saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth'
    
    vqvae_config = load_vqvae_config(config_path)
    
    # 加载所有组件
    model = init_vqvae_with_components(
        vqvae_config,
        checkpoint_path=checkpoint_path,
        load_encoder=True,
        load_codebook=True,
        load_decoder=True
    )
    
    # 方式2: 只加载encoder和codebook，冻结它们
    # model = init_vqvae_with_components(
    #     vqvae_config,
    #     checkpoint_path=checkpoint_path,
    #     load_encoder=True,
    #     load_codebook=True,
    #     load_decoder=False,
    #     freeze_encoder=True,
    #     freeze_codebook=True
    # )
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

