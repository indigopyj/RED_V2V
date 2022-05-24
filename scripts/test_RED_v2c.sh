# CUDA_VISIBLE_DEVICES=1 python test_RED.py \
#  --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
#  --name RED_v2c_lpips_8conv_feature --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
#  --how_many 10 --no_dropout --n_convs 8 --results_dir ./results/

 CUDA_VISIBLE_DEVICES=1 python test_RED.py \
 --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
 --name RED_v2c_lpips_8conv_shift_tanh_shiftx1_2 --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
 --how_many 10 --no_dropout --n_convs 8 --results_dir ./results/ --shift_param --which_epoch 2700