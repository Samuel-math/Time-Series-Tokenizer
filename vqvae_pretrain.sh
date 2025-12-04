dset=ettm1
python vqvae_pretrain.py \
    --dset_pretrain $dset \
    --context_points 512 \
    --batch_size 64 \
    --n_epochs_pretrain 50 \
    --config_path model_config/ettm1_vqvae.json \
    --save_path saved_models/vqvae/