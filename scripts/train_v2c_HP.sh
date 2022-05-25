python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
--name RED_v2c --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
--gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \ --tensorboard_dir tensorboard/
--lambda_L1 1.0 \
--lambda_L1_out 10 \
--lpips --lambda_lpips 10 \
--n_convs 8 \
--act_GAN_loss \
--Temporal_GAN_loss --lambda_D_T 1.0 \
--lambda_feature 1.0 \ 
