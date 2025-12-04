import argparse
import json
import numpy as np
import os
import pdb
import random
import time
import torch
from src.models.vqvae import vqvae
from time import gmtime, strftime
from datautils import *

def main(device, config, save_dir, logger, data_init_loc, args):
    # Create/overwrite checkpoints folder
    if os.path.exists(os.path.join(save_dir, 'checkpoints')):
        print('Checkpoint Directory Already Exists - files inside may be overwritten.')
    else:
        os.makedirs(os.path.join(save_dir, 'checkpoints'))


    # Start training
    vqvae_config, summary = start_training(
        device=device,
        vqvae_config=config['vqvae_config'],
        save_dir=save_dir,
        logger=logger,
        data_init_loc=data_init_loc,
        args=args
    )

    # Save config
    config['vqvae_config'] = vqvae_config
    print('CONFIG FILE TO SAVE:', config)

    # Create Configs folder
    if os.path.exists(os.path.join(save_dir, 'configs')):
        print('Saved Config Directory Already Exists - files inside may be overwritten.')
    else:
        os.makedirs(os.path.join(save_dir, 'configs'))

    # Save JSON config
    with open(os.path.join(save_dir, 'configs', 'config_file.json'), 'w+') as f:
        json.dump(config, f, indent=4)

    # Save Master file
    summary['log_path'] = os.path.join(save_dir)
    master['summaries'] = summary
    print('MASTER FILE:', master)
    with open(os.path.join(save_dir, 'master.json'), 'w') as f:
        json.dump(master, f, indent=4)


def start_training(device, vqvae_config, save_dir, logger, data_init_loc, args):
    summary = {}

    if 'general_seed' not in vqvae_config:
        vqvae_config['seed'] = random.randint(0, 9999)

    general_seed = vqvae_config['general_seed']
    summary['general_seed'] = general_seed

    torch.manual_seed(general_seed)
    random.seed(general_seed)
    np.random.seed(general_seed)
    torch.backends.cudnn.deterministic = True

    summary['data initialization location'] = data_init_loc
    summary['device'] = device

    # Setup model
    model = vqvae(vqvae_config)
    print('Total trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if vqvae_config['pretrained']:
        model = torch.load(vqvae_config['pretrained'])

    summary['vqvae_config'] = vqvae_config

    # Train
    start_time = time.time()
    model = train_model(model, device, vqvae_config, save_dir, logger, args=args)

    torch.save(model, os.path.join(save_dir, 'checkpoints/final_model.pth'))

    summary['total_time'] = round(time.time() - start_time, 3)
    return vqvae_config, summary


def train_model(model, device, vqvae_config, save_dir, logger, args):
    optimizer = model.configure_optimizers(lr=vqvae_config['learning_rate'])

    model.to(device)
    start_time = time.time()

    print('BATCHSIZE:', args.batch_size)
    dls = get_dls(args)
    train_loader, vali_loader = dls.train, dls.valid

    best_val_loss = float('inf')

    for epoch in range(args.n_epochs_pretrain):
        # ========= Training =========
        model.train()
        train_losses = []

        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)  # [B, L, C]
            
            B, L, C = batch_x.shape
        
            for ch in range(C):
                # 取出第 ch 个通道
                x_ch = batch_x[:, :, ch]       # [B, L]
        
                # 变成模型需要的输入格式
                x_ch = x_ch.view(B, L)
        
                loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, \
                encoding_indices, encodings = model.shared_eval(
                    x_ch, optimizer, 'train'
                )
        
                train_losses.append(loss.item())

            mean_train_loss = sum(train_losses) / len(train_losses)

        # ========= Validation =========
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, _ in vali_loader:
                batch_x = batch_x.to(device)
                B, L, C = batch_x.shape
            
                for ch in range(C):
                    # 取出第 ch 个通道
                    x_ch = batch_x[:, :, ch]       # [B, L]
            
                    # 变成模型需要的输入格式
                    x_ch = x_ch.view(B, L)
    
                    val_loss, val_vq_loss, val_recon_error, val_x_recon, \
                    val_perplexity, val_embedding_weight, val_encoding_indices, \
                    val_encodings = model.shared_eval(
                        x_ch, optimizer, 'val'
                    )
    
                    val_losses.append(val_loss.item())
    
            mean_val_loss = sum(val_losses) / len(val_losses)

        # ========= Logging =========
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss = {mean_train_loss:.6f} | "
            f"val_loss = {mean_val_loss:.6f}"
        )

        # ========= Save best model =========
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            ckpt_path = os.path.join(save_dir, 'checkpoints/best_model.pth')
            torch.save(model, ckpt_path)
            print(f"New best model saved at epoch {epoch} (val_loss={mean_val_loss:.6f})")

    print('Total training time:', round(time.time() - start_time, 3))
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset and dataloader
    parser.add_argument('--dset_pretrain', type=str, default='ettm1')
    parser.add_argument('--context_points', type=int, default=512)
    parser.add_argument('--target_points', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--scaler', type=str, default='standard')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
    
    parser.add_argument('--config_path', type=str, default='model_config/ettm1_vqvae.json')
    parser.add_argument('--model_init_num_gpus', type=int, default=0)
    parser.add_argument('--data_init_cpu_or_gpu', type=str)
    parser.add_argument('--save_path', type=str, default='saved_models/vqvae/')
    parser.add_argument('--base_path', type=str, default=False)
    parser.add_argument('--n_epochs_pretrain', type=int, default=100)

    args = parser.parse_args()
    args.dset = args.dset_pretrain

    # Load config
    config_file = args.config_path
    print('Config folder:\t {}'.format(config_file))

    with open(config_file, 'r') as f:
        config = json.load(f)
    print(' Running Config:', config_file)

    # Prepare save folder
    save_folder_name = (
        str(args.dset) + '/'
        + 'vqvae' + str(config['vqvae_config']['embedding_dim'])
        + '_CW' + str(config['vqvae_config']['num_embeddings'])
        + '_CF' + str(config['vqvae_config']['compression_factor'])
        + '_BS' + str(args.batch_size)
        + '_ITR' + str(config['vqvae_config']['num_training_updates'])
    )
    save_dir = args.save_path + save_folder_name

    master = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'config file': config_file,
        'save directory': save_dir,
        'gpus': args.model_init_num_gpus,
    }

    # No comet logger anymore
    comet_logger = None

    # Setup device
    if torch.cuda.is_available() and args.model_init_num_gpus >= 0:
        assert args.model_init_num_gpus < torch.cuda.device_count()
        device = f'cuda:{args.model_init_num_gpus}'
    else:
        device = 'cpu'

    if args.data_init_cpu_or_gpu == 'gpu':
        data_init_loc = device
    else:
        data_init_loc = 'cpu'

    # Call main
    main(device, config, save_dir, comet_logger, data_init_loc, args)
