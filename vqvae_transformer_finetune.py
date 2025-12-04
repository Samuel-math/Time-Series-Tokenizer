"""
VQVAE + Transformer 微调脚本
参考 patchtst_finetune.py 的结构，使用 Learner 框架
在微调完成后直接输出测试集结果
"""

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.vqvae_transformer import VQVAETransformerPretrain, VQVAETransformerFinetune
from src.models.vqvae import vqvae, Decoder
from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from src.utils_vqvae import load_vqvae_config
from datautils import get_dls

import argparse

parser = argparse.ArgumentParser()
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=1, help='do finetuning or not')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='ettm1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Pretrained model paths
parser.add_argument('--pretrained_model', type=str, required=True,
                   help='Path to pretrained VQVAE+Transformer model checkpoint')
parser.add_argument('--vqvae_config_path', type=str, required=True,
                   help='Path to VQVAE config file')
parser.add_argument('--vqvae_checkpoint', type=str, required=True,
                   help='Path to VQVAE checkpoint (for decoder)')
parser.add_argument('--transformer_config_path', type=str, default='', help='Path to transformer config file')
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='vqvae_transformer', help='model type for saving')

args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/vqvae_transformer_finetune/' + args.model_type + '/'
if not os.path.exists(args.save_path): 
    os.makedirs(args.save_path)

suffix_name = (
    '_cw' + str(args.context_points) 
    + '_tw' + str(args.target_points) 
    + '_epochs-finetune' + str(args.n_epochs_finetune) 
    + '_model' + str(args.finetuned_model_id)
)
args.save_finetuned_model = args.dset_finetune + '_vqvae_transformer_finetuned' + suffix_name

# get available GPU device
set_device()


class VQVAETransformerFinetuneWrapper(nn.Module):
    """
    包装类，使 VQVAETransformerFinetune 能够与 Learner 框架兼容
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        """
        Args:
            x: [B, context_len, n_vars] 输入时间序列
        
        Returns:
            pred: [B, target_len, n_vars] 预测结果
        """
        # 从dataloader获取target_len（通过模型属性传递）
        target_len = getattr(self, '_target_len', args.target_points)
        pred = self.model(x, target_len)
        return pred

def load_json_config(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


def load_transformer_config(args, pretrained_checkpoint=None, device='cpu'):
    """
    统一加载 transformer_config 的辅助函数
    优先从 checkpoint 加载，其次从配置文件加载
    
    Args:
        args: 命令行参数
        pretrained_checkpoint: 已加载的 checkpoint（可选，避免重复加载）
        device: 设备
    """
    # 如果未提供 checkpoint，则加载
    if pretrained_checkpoint is None:
        print(f"加载预训练Transformer模型: {args.pretrained_model}")
        pretrained_checkpoint = torch.load(args.pretrained_model, map_location=device)
    
    if isinstance(pretrained_checkpoint, dict) and "transformer_config" in pretrained_checkpoint:
        transformer_config = pretrained_checkpoint["transformer_config"]
        print("从 checkpoint 加载 transformer_config")
    elif hasattr(args, "transformer_config_path") and args.transformer_config_path and os.path.exists(args.transformer_config_path):
        transformer_config = load_json_config(args.transformer_config_path)
        print(f"从文件加载 transformer_config: {args.transformer_config_path}")
    else:
        raise ValueError(
            "找不到 Transformer 配置！请确保 checkpoint 包含 transformer_config，"
            "或通过 --transformer_config_path 指定配置文件。"
        )
    
    print("Transformer 配置:", transformer_config)
    return transformer_config


def get_model(c_in, args, vqvae_config, device='cpu'):
    """
    加载预训练模型 + VQVAE decoder + 构建 Finetune 模型
    c_in: number of input variables
    """
    print(f"加载预训练Transformer模型: {args.pretrained_model}")
    pretrained_checkpoint = torch.load(args.pretrained_model, map_location=device)

    # --------------------------
    # 1) 获取 transformer_config
    # --------------------------
    transformer_config = load_transformer_config(args, pretrained_checkpoint, device)

    # --------------------------
    # 2) 构建预训练模型骨架
    # --------------------------
    pretrained_model = VQVAETransformerPretrain(vqvae_config, transformer_config)

    # --------------------------
    # 3) 加载 state_dict
    # --------------------------
    if isinstance(pretrained_checkpoint, dict) and 'model_state_dict' in pretrained_checkpoint:
        pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'])
    elif isinstance(pretrained_checkpoint, dict):
        pretrained_model.load_state_dict(pretrained_checkpoint)
    else:
        if hasattr(pretrained_checkpoint, 'state_dict'):
            pretrained_model.load_state_dict(pretrained_checkpoint.state_dict())
        else:
            print("警告: checkpoint格式异常，尝试直接使用")
            pretrained_model = pretrained_checkpoint

    pretrained_model.eval()
    print("预训练Transformer模型已成功加载")

    # --------------------------
    # 4) 加载 VQVAE decoder
    # --------------------------
    print(f"加载VQVAE Decoder: {args.vqvae_checkpoint}")
    vqvae_model = torch.load(args.vqvae_checkpoint, map_location=device)
    decoder = vqvae_model.decoder
    print("VQVAE Decoder已加载")

    # --------------------------
    # 5) Finetune 模型
    # --------------------------
    finetune_model = VQVAETransformerFinetune(
        pretrained_model,
        decoder,
        vqvae_config,
        freeze_transformer=False,
        freeze_decoder=True,
        add_finetune_head=False,
    )

    # 包装成 learner-friendly 格式
    model = VQVAETransformerFinetuneWrapper(finetune_model)
    model._target_len = args.target_points

    print('number of trainable params:',
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def find_lr():
    # 1) dataloader
    dls = get_dls(args)

    # 2) 加载 VQVAE config
    vqvae_config = load_vqvae_config(args.vqvae_config_path)

    # 3) 构建 model
    model = get_model(dls.vars, args, vqvae_config)

    # 4) loss
    loss_func = torch.nn.MSELoss(reduction='mean')

    # 5) callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []

    # 6) learner
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=args.lr,
        cbs=cbs,
    )

    # 7) lr finder
    suggested_lr = learn.lr_finder()
    print("suggested_lr =", suggested_lr)
    return suggested_lr

def save_recorders(learn):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)


def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args)
    
    # 加载配置
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    
    # get model (transformer_config 会在 get_model 内部自动加载)
    model = get_model(dls.vars, args, vqvae_config)
    
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
    ]
    
    # define learner
    learn = Learner(dls, model, 
                    loss_func, 
                    lr=lr, 
                    cbs=cbs,
                    metrics=[mse]
                    )                            
    # fit the data to the model
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=0)
    save_recorders(learn)


def test_func(weight_path):
    # get dataloader
    dls = get_dls(args)
    
    # 加载配置
    vqvae_config = load_vqvae_config(args.vqvae_config_path)
    
    # get model (transformer_config 会在 get_model 内部自动加载)
    model = get_model(dls.vars, args, vqvae_config)
    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    
    learn = Learner(dls, model, cbs=cbs)
    out = learn.test(dls.test, weight_path=weight_path + '.pth', scores=[mse, mae])
    print('score:', out[2])
    
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1, -1), columns=['mse', 'mae']).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv', 
        float_format='%.6f', 
        index=False
    )
    return out


if __name__ == '__main__':
    if args.is_finetune:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr()
        finetune_func(suggested_lr)
        print('finetune completed')
        # Test - 直接输出测试集结果
        out = test_func(args.save_path + args.save_finetuned_model)
        print('----------- Complete! -----------')
    else:
        args.dset = args.dset_finetune
        weight_path = args.save_path + args.save_finetuned_model
        # Test only
        out = test_func(weight_path)
        print('----------- Complete! -----------')
