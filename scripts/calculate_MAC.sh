# # RED setting
# python calculate.py \
#  --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
#  --name RED_v2c_lpips --dataset_mode video --main_G_path checkpoints/v2c_experiment/2_net_G_A.pth \
#  --no_dropout --n_convs 5

# original generator
 python calculate.py \
 --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
--model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 512 --loadSizeH 256 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout --which_epoch 2