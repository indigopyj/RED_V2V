# CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv_GANloss --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --no_dropout --batchSize 8 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8 --GAN_loss --tensorboard_dir tensorboard/

# CUDA_VISIBLE_DEVICES=0 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv_GANloss_actl1 --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --no_dropout --batchSize 8 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8 --GAN_loss --actl1 --tensorboard_dir tensorboard/

#CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
#--name RED_v2c_lpips_8conv_GANloss_feature --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
#--no_dropout --batchSize 16 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
#--lpips --n_convs 8 --GAN_loss --lambda_feature 0.01 --tensorboard_dir tensorboard/

# CUDA_VISIBLE_DEVICES=4 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv_feature --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --no_dropout --batchSize 16 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8 --lambda_feature 0.01 --tensorboard_dir tensorboard/

# CUDA_VISIBLE_DEVICES=1,2 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv_gan_2tgan --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0,1 --no_dropout --pool_size 0 --batchSize 16 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8 --GAN_loss --Temporal_GAN_loss --lambda_D_T 2.0 --tensorboard_dir tensorboard/ 

# CUDA_VISIBLE_DEVICES=2,1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv_shift --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0,1 --no_dropout --pool_size 0 --batchSize 16 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8 --shift_param --tensorboard_dir tensorboard/ 

# CUDA_VISIBLE_DEVICES=4 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv_gan_after_tgan --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 16 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8 --GAN_loss --Temporal_GAN_loss --lambda_D_T 1.0 --tensorboard_dir tensorboard/ 

# CUDA_VISIBLE_DEVICES=2 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv_shift_tanh_1 --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8 --shift_param --actl1 --tensorboard_dir tensorboard/ 

# CUDA_VISIBLE_DEVICES=0,2 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv_shift_tanh_shiftx1_2 --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --gpu_ids 0,1 --no_dropout --pool_size 0 --batchSize 16 --niter_decay 200 --niter 2500 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8 --shift_param --actl1 --lambda_shift 1.0 --tensorboard_dir tensorboard/ 

CUDA_VISIBLE_DEVICES=0,2 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
--name RED_v2c_lpips_8conv_shift_tanh_shiftx1_2 --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
--gpu_ids 0,1 --no_dropout --pool_size 0 --batchSize 4 --niter_decay 200 --niter 2500 --save_latest_freq 1 --save_epoch_freq 300 \
--lpips --n_convs 8 --shift_param --actl1 --lambda_shift 1.0 --tensorboard_dir tensorboard/ --continue_train --which_epoch 1500

# CUDA_VISIBLE_DEVICES=2,3 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --no_dropout --batchSize 8 --niter_decay 300 --niter 1500 --save_latest_freq 10000 --save_epoch_freq 300 --lpips --n_convs 8 --continue_train --which_epoch 1200 --epoch_count 1201
