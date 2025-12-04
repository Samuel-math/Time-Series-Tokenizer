dset='ettm1'
python vqvae_transformer_pretrain.py \
    --dset_pretrain $dset \
    --context_points 512 \
    --batch_size 64 \
    --n_epochs_pretrain 30 \
    --vqvae_config_path saved_models/vqvae/ettm1/vqvae64_CW256_CF4_BS64_ITR15000/configs/config_file.json \
    --vqvae_checkpoint saved_models/vqvae/ettm1/vqvae64_CW256_CF4_BS64_ITR15000/checkpoints/best_model.pth \
    --transformer_parameter_path "model_config/${dset}/"\
    --d_model 128 \
    --n_layers 3 \
    --n_heads 8 \
    --d_ff 256 \
    --dropout 0.3