# CUDA_VISIBLE_DEVICES=1 python test_RED.py \
#  --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
#  --name RED_v2c_lpips_8conv_feature --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
#  --how_many 10 --no_dropout --n_convs 8 --results_dir ./results/

CUDA_VISIBLE_DEVICES=6 python test_RED.py \
 --dataroot /home/nas4_dataset/vision/ \
 --name RED_10l1_1gan_10out_20lpips_sn_shift --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
 --how_many 10 --no_dropout --n_convs 8 --shift_param --results_dir ./results/ --which_epoch 2100

# CUDA_VISIBLE_DEVICES=6 python test_RED.py \
#  --dataroot /home/nas4_dataset/vision/ \
#  --name RED_10l1_1gan_20out_20lpips_sn_shift --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
#  --how_many 10 --no_dropout --n_convs 8 --shift_param --results_dir ./results/ --which_epoch latest