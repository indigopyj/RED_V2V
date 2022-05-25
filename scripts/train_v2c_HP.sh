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

# lambda_L1 : 기준 프레임의 activation과 red가 예측한 difference map을 반영한 activation에 걸리는 l1 loss
# lambda_L1_out: main generator output과 red output간에 걸리는 l1 loss
# lambda_lpips : perceptual loss
# n_convs : red network의 conv block 개수
# Temporal_GAN_loss: (기준프레임 output, real output)과 (기준프레임 output, fake output)을 discriminator에 넣는 loss
# labmda_feature : feature matching loss