import cv2
import numpy as np
import os
# v2c
# save_path = "comparison_result_v2c"
# cyclegan_path = "/home/nas4_user/yeojeongpark/compression/pytorch-CycleGAN-and-pix2pix/results/cyclegan_v2c/test_latest/fake"
# unsup_path = "/home/nas4_user/yeojeongpark/compression/Unsup_Recycle_GAN/results/RED_10l1_1gan_10out_20lpips_sn/test_2400/real"
# red_path = "/home/nas4_user/yeojeongpark/compression/Unsup_Recycle_GAN/results/RED_10l1_1gan_10out_20lpips_sn/test_2400/fake"
# input_path = "/home/nas4_user/yeojeongpark/compression/Unsup_Recycle_GAN/results/RED_10l1_1gan_10out_20lpips_sn/test_2400/input"
# num_videos = 10
# frames = 200
# width = 512
# obama trump
save_path = "comparison_result_o2t"
cyclegan_path = "/home/nas4_user/yeojeongpark/compression/pytorch-CycleGAN-and-pix2pix/results/ott_cyclegan_new/test_latest/fake"
unsup_path = "/home/nas4_user/yeojeongpark/compression/Unsup_Recycle_GAN/results/RED_face_obamatrump/test_latest/real"
red_path = "/home/nas4_user/yeojeongpark/compression/Unsup_Recycle_GAN/results/RED_face_obamatrump/test_latest/fake"
input_path = "/home/nas4_user/yeojeongpark/compression/Unsup_Recycle_GAN/results/RED_face_obamatrump/test_latest/input"
num_videos = 1
frames = 100
width = 256

os.makedirs(save_path, exist_ok=True)
for i in range(1, num_videos+1):
    video = cv2.VideoWriter(os.path.join(save_path, str(i)+'.mp4'), fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=10, frameSize=(width * 4, 256))
    for j in range(frames):
        #img_name = "%03d_%d.png" % (i, j)
        img_name = "testA_%d.png" % j
        input_im = cv2.imread(os.path.join(input_path, img_name))
        cycle_im = cv2.imread(os.path.join(cyclegan_path, img_name))
        unsup_im = cv2.imread(os.path.join(unsup_path, img_name))
        red_im = cv2.imread(os.path.join(red_path, img_name))
        
        cat_img = np.concatenate((input_im, cycle_im, unsup_im, red_im), axis=1)
        video.write(cat_img)
    print("write %d.mp4"  % i)
