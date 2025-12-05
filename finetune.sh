dset=ettm1
python vqvae_transformer_finetune.py \
    --dset_finetune ${dset} \
    --context_points 512 \
    --target_points 96 \
    --batch_size 64 \
    --n_epochs_finetune 50 \
    --init_mode vqvae_only \
    --pretrained_model saved_models/${dset}/vqvae_transformer/vqvae_transformer_pretrained_cw512_d128_l3_h8_epochs-pretrain30_model1.pth \
    --vqvae_config_path saved_models/vqvae/${dset}/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json \
    --vqvae_checkpoint saved_models/vqvae/${dset}/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth \
    --transformer_config_path model_config/ettm1_transformer_config.json