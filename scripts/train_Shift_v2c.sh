CUDA_VISIBLE_DEVICES=0,1 python train_shift.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
--name Shift_v2c_10 --dataset_mode video \
--gpu_ids 0,1 --no_dropout --pool_size 0 --batchSize 8 --niter_decay 200 --niter 500 --save_latest_freq 1000 --save_epoch_freq 200  --shift_param --tensorboard_dir tensorboard/