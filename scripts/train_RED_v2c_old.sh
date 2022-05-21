# CUDA_VISIBLE_DEVICES=0 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
# --name RED_v2c_lpips_8conv --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
# --no_dropout --batchSize 8 --niter_decay 200 --niter 1600 --save_latest_freq 10000 --save_epoch_freq 300 \
# --lpips --n_convs 8

CUDA_VISIBLE_DEVICES=1 python train_RED.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
--name RED_v2c_lpips --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
--no_dropout --batchSize 8 --niter_decay 200 --niter 1300 --save_latest_freq 10000 --save_epoch_freq 300 \
--lpips --n_convs 5
