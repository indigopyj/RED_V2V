# CUDA_VISIBLE_DEVICES=2 python train_RED.py --dataroot /home/nas4_dataset/vision/ \
# --name RED_10l1_1gan_10out_20lpips_sn_shift --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 10.0 \
# --lpips --lambda_lpips 20.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm --shift_param

# CUDA_VISIBLE_DEVICES=3 python train_RED.py --dataroot /home/nas4_dataset/vision/ \
# --name RED_10l1_1gan_20out_20lpips_sn_shift --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 20.0 \
# --lpips --lambda_lpips 20.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm --shift_param

# CUDA_VISIBLE_DEVICES=7 python train_RED.py --dataroot /home/nas4_dataset/vision/ \
# --name RED_10l1_1gan_20out_50lpips_sn_shift --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 20.0 \
# --lpips --lambda_lpips 50.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm --shift_param

# CUDA_VISIBLE_DEVICES=4 python train_RED.py --dataroot /home/nas4_dataset/vision/ \
# --name RED_10l1_1gan_10out_20lpips_sn_shift_ssim --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 10.0 \
# --lpips --lambda_lpips 20.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm --shift_param --ssim

# CUDA_VISIBLE_DEVICES=5 python train_RED.py --dataroot /home/nas4_dataset/vision/ \
# --name RED_10l1_1gan_20out_20lpips_sn_shift_ssim --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 20.0 \
# --lpips --lambda_lpips 20.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm --shift_param --ssim

CUDA_VISIBLE_DEVICES=6 python train_RED.py --dataroot /home/nas4_dataset/vision/ \
--name RED_10l1_1gan_20out_50lpips_sn_shift_ssim --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
--gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
--tensorboard_dir tensorboard/ \
--lambda_L1 10.0 \
--lambda_L1_out 20.0 \
--lpips --lambda_lpips 50.0 \
--n_convs 8 \
--act_GAN_loss --spectral_norm --shift_param --ssim