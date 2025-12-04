"""
VQVAE + Transformer 预训练脚本
"""

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.vqvae_transformer import VQVAETransformerPretrain
from src.learner import Learner
from src.callback.tracking import *
from src.callback.transforms import *
from src.callback.core import Callback
from src.metrics import *
from src.basics import set_device
from src.utils_vqvae import load_vqvae_config
from datautils import get_dls

import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='ettm1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# VQVAE config
parser.add_argument('--vqvae_config_path', type=str, 
                   default='saved_models/vqvae/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json')
parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                   help='Path to pretrained VQVAE checkpoint (optional)')
# Transformer config
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='number of Transformer heads')
parser.add_argument('--d_ff', type=int, default=256, help='Transformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='Transformer dropout')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
parser.add_argument('--transformer_parameter_path', type=str, default='model_config/', help='to store transformer parameter')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=50, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='vqvae_transformer', help='model type for saving')

args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = (
    'vqvae_transformer_pretrained'
    + '_cw' + str(args.context_points)
    + '_d' + str(args.d_model)
    + '_l' + str(args.n_layers)
    + '_h' + str(args.n_heads)
    + '_epochs-pretrain' + str(args.n_epochs_pretrain)
    + '_model' + str(args.pretrained_model_id)
)
args.save_path = 'saved_models/' + args.dset_pretrain + '/' + args.model_type + '/'
if not os.path.exists(args.save_path): 
    os.makedirs(args.save_path)

# get available GPU device
set_device()

def save_transformer_config(args, filename="transformer_config.json"):
    """
    将 Transformer 相关参数保存为 JSON 文件
    """
    # 1) 需要保存的 Transformer 参数
    transformer_config = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "attn_dropout": args.attn_dropout,
    }

    # 2) 创建路径（如果不存在）
    save_dir = args.transformer_parameter_path
    os.makedirs(save_dir, exist_ok=True)

    # 3) 文件完整路径
    save_path = os.path.join(save_dir, filename)

    # 4) 保存为 JSON
    with open(save_path, "w") as f:
        json.dump(transformer_config, f, indent=4)

    print(f"Transformer 参数已保存到 {save_path}")
save_transformer_config(f'{args.dset}_transformer_config.json')

class NextTokenPredictionCB(Callback):
    """
    Callback for next token prediction task
    处理 VQVAE+Transformer 的 next token prediction 损失计算
    """
    def __init__(self, compression_factor):
        super().__init__()
        self.compression_factor = compression_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.target_indices = None
        self.causal_mask = None
        
    def before_forward(self):
        """在forward之前计算target indices和causal mask"""
        x = self.learner.xb  # [B, seq_len, n_vars]
        B, L, C = x.shape
        model = self.learner.model  # 获取VQVAE+Transformer模型
        device = x.device
        
        # 计算压缩后的长度
        T_compressed = L // self.compression_factor
        
        # 创建causal mask用于next token prediction
        self.causal_mask = torch.triu(torch.ones(T_compressed, T_compressed, device=device), diagonal=1)
        self.causal_mask = self.causal_mask.masked_fill(self.causal_mask == 1, float('-inf'))
        
        # 获取真实的codebook indices（通过VQVAE encoder + VQ）
        target_indices = []
        with torch.no_grad():
            for ch in range(C):
                x_ch = x[:, :, ch].view(B, L)
                # 使用VQVAE encoder和VQ获取真实indices
                z = model.vqvae_encoder(x_ch, self.compression_factor)
                z = z.permute(0, 2, 1)  # [B, T/compressed, embedding_dim]
                _, _, _, _, encoding_indices, _ = model.vq(z.permute(0, 2, 1).contiguous())
                # encoding_indices: [B*T/compressed, 1]
                indices = encoding_indices.squeeze(-1).view(B, T_compressed)
                target_indices.append(indices)
        
        self.target_indices = torch.stack(target_indices, dim=2)  # [B, T/compressed, C]
        
        # 修改模型的forward调用，传入mask
        # 我们需要在forward中传入mask，但Learner框架不支持额外参数
        # 所以我们将mask存储为模型属性
        model._current_mask = self.causal_mask
        
    def after_forward(self):
        """在forward之后计算损失"""
        probs = self.learner.pred  # [B, codebook_size, T/compression_factor, C] - 现在是概率
        target_indices = self.target_indices  # [B, T/compressed, C]
        
        B, codebook_size, T_compressed, C = probs.shape
        
        # 计算损失（next token prediction）
        # 由于输出是概率，需要使用NLLLoss（负对数似然损失）
        # 或者将概率转换回logits
        nll_criterion = nn.NLLLoss(ignore_index=-1)
        loss = 0
        for ch in range(C):
            probs_ch = probs[:, :, :, ch].permute(0, 2, 1)  # [B, T/compressed, codebook_size]
            target_ch = target_indices[:, :, ch]  # [B, T/compressed]
            
            # Next token prediction: 预测位置i+1的token，使用位置i的context
            pred_probs = probs_ch[:, :-1, :].reshape(-1, codebook_size)  # [B*(T/compressed-1), codebook_size]
            target_tokens = target_ch[:, 1:].reshape(-1)  # [B*(T/compressed-1)]
            
            # 使用NLLLoss，需要取对数
            log_probs = torch.log(pred_probs + 1e-10)  # 添加小值避免log(0)
            loss += nll_criterion(log_probs, target_tokens)
        
        loss = loss / C
        self.learner.pred = loss.unsqueeze(0)  # 将损失作为pred返回


class NextTokenLoss(nn.Module):
    """自定义损失函数，用于next token prediction"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # pred 是标量损失值（在NextTokenPredictionCB中计算）
        # target 是dummy，不使用
        return pred


def get_model(c_in, args, vqvae_config, transformer_config, device='cpu'):
    """
    c_in: number of variables
    """
    # 创建VQVAE+Transformer模型
    model = VQVAETransformerPretrain(
        vqvae_config,
        transformer_config,
        load_vqvae_weights=args.vqvae_checkpoint is not None,
        vqvae_checkpoint_path=args.vqvae_checkpoint,
        device=device
    )
    
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    # get dataloader
    dls = get_dls(args)
    
    # 加载VQVAE配置
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    
    # Transformer配置
    transformer_config = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'attn_dropout': args.attn_dropout
    }
    
    model = get_model(dls.vars, args, vqvae_config, transformer_config)
    
    # get loss (dummy loss, actual loss computed in model)
    loss_func = NextTokenLoss()
    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [NextTokenPredictionCB(compression_factor=vqvae_config['compression_factor'])]
    
    # define learner
    learn = Learner(dls, model, 
                    loss_func, 
                    lr=args.lr, 
                    cbs=cbs,
                    )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    
    # 加载VQVAE配置
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    
    # Transformer配置
    transformer_config = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'attn_dropout': args.attn_dropout
    }
    
    # get model
    model = get_model(dls.vars, args, vqvae_config, transformer_config)
    
    # get loss (dummy loss, actual loss computed in model)
    loss_func = NextTokenLoss()
    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        NextTokenPredictionCB(compression_factor=vqvae_config['compression_factor']),
        SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model,                       
                    path=args.save_path)
    ]
    
    # define learner
    learn = Learner(dls, model, 
                    loss_func, 
                    lr=lr, 
                    cbs=cbs,
                    )                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)


if __name__ == '__main__':
    args.dset = args.dset_pretrain
    suggested_lr = find_lr()
    # Pretrain
    pretrain_func(suggested_lr)
    print('pretraining completed')
