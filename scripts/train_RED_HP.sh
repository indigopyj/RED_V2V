#python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
#--name RED_v2c --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
#--gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \ --tensorboard_dir tensorboard/
#--lambda_L1 1.0 \
#--lambda_L1_out 10 \
#--lpips --lambda_lpips 10 \
#--n_convs 8 \
#--act_GAN_loss \
#--Temporal_GAN_loss --lambda_D_T 1.0 \
#--lambda_feature 1.0 \ 

# lambda_L1 : 기준 프레임의 activation과 red가 예측한 difference map을 반영한 activation에 걸리는 l1 loss
# lambda_L1_out: main generator output과 red output간에 걸리는 l1 loss
# lambda_lpips : perceptual loss
# n_convs : red network의 conv block 개수
# Temporal_GAN_loss: (기준프레임 output, real output)과 (기준프레임 output, fake output)을 discriminator에 넣는 loss
# labmda_feature : feature matching loss

# CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_1l1_1gan --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 1.0 \
# --lambda_L1_out 0.0 \
# --n_convs 8 \
# --act_GAN_loss \

# CUDA_VISIBLE_DEVICES=0 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 0.0 \
# --n_convs 8 \
# --act_GAN_loss

# CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_0.1l1_1gan --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 0.1 \
# --lambda_L1_out 0.0 \
# --n_convs 8 \
# --act_GAN_loss

# CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 0.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=2 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_1out --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 1.0 \
# --n_convs 8 \
# --act_GAN_loss

# CUDA_VISIBLE_DEVICES=3 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_5l1_1gan_5out --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 5.0 \
# --lambda_L1_out 5.0 \
# --n_convs 8 \
# --act_GAN_loss

# CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_1out_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 1.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=0 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_5out_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 5.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_10out_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 10.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=0 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_1out_sn_dl2 --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 1.0 \
# --n_convs 8 \
# --RED_n_layers_D 2 \
# --act_GAN_loss --spectral_norm # receptive field ; 34

# CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_1l1_1gan_1out_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 1.0 \
# --lambda_L1_out 1.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=2 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_10out_1lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 10.0 \
# --lpips --lambda_lpips 1.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=2 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_10out_10lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 10.0 \
# --lpips --lambda_lpips 10.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=3 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_10out_5lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 10.0 \
# --lpips --lambda_lpips 5.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm # best until now

# CUDA_VISIBLE_DEVICES=4 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_5out_5lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 5.0 \
# --lpips --lambda_lpips 5.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=4 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_5out_10lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 5.0 \
# --lpips --lambda_lpips 10.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=3 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_10out_20lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 10.0 \
# --lpips --lambda_lpips 20.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=2 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_10l1_1gan_20out_20lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 10.0 \
# --lambda_L1_out 20.0 \
# --lpips --lambda_lpips 20.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

# CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_5l1_1gan_10out_10lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
# --tensorboard_dir tensorboard/ \
# --lambda_L1 5.0 \
# --lambda_L1_out 10.0 \
# --lpips --lambda_lpips 10.0 \
# --n_convs 8 \
# --act_GAN_loss --spectral_norm

CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
--name RED_10l1_1gan_10out_50lpips_sn --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
--gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
--tensorboard_dir tensorboard/ \
--lambda_L1 10.0 \
--lambda_L1_out 10.0 \
--lpips --lambda_lpips 50.0 \
--n_convs 8 \
--act_GAN_loss --spectral_norm