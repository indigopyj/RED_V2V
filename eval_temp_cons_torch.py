import numpy as np
import cv2
import os
import sys
import torch
import glob
import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from util.eval import *
import natsort
#import models.flow_models as flow_models
import argparse


exps = [
        'source',
        #path-to-frames,
        ]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--alpha", type=float, default=-50.0)
parser.add_argument('--mode', type=str, default='main',
                                 help='chooses how datasets are loaded. [main | RED | cycleGAN]')
parser.add_argument("--mask_path", type=str, default='precomputed_masks/')
parser.add_argument("--img_path", type=str, default='results/model_name/test_latest/real/')
parser.add_argument("--viper_path", type=str, default="/home/nas4_dataset/vision/Viper")
opt = parser.parse_args()
# ############################# NOTE : YOU SHOULD FIX HERE
# mode_list = { # you can set your own path (path_of_masks_you_want_to_save, path_of_translated_images_you_want_to_evaluate)
#     "main": ("precomputed_masks_v2c_main/", "results/RED_10l1_1gan_10out_20lpips_sn/test_2400/real/*_*.png"),
#     "RED": ("precomputed_masks_v2c_RED/", "results/RED_10l1_1gan_10out_20lpips_sn/test_2400/fake/*_*.png"),
#     "cycleGAN" : ("precomputed_masks_v2c_cyclegan", "/home/nas4_user/yeojeongpark/compression/pytorch-CycleGAN-and-pix2pix/results/viper2cityscapes_cyclegan/val_45/fake/*.png")
# }
# #############################
masks_path = opt.mask_path
viper_path = opt.viper_path
if not os.path.exists(masks_path):
    os.makedirs(masks_path)

if opt.mode == "cyclegan":
    w, h = 256, 256
else:
    w, h = 512, 256
original_w, original_h = 1920, 1080

for exp in exps:
    errs = []
    if "source" in exp:
        #files = sorted(glob.glob(os.path.join("results/v2c_experiment/test_2/images/*0_real_A.png")))
        files = natsort.natsorted(glob.glob(os.path.join(opt.img_path, "*.png")))
    else:
        files = sorted(glob.glob(os.path.join("results", exp, "images/*0_fake_B.png")))
        print((os.path.join("results", exp)))
        if len(files) == 0:
            files = sorted(glob.glob(os.path.join("results", exp, "images/*0_centered_fake_B1.png")))
            
    for i in tqdm.tqdm(range(len(files))):
        if i == len(files) - 1 : break
        try:
            f = files[i]
            ### load input images
            filename1 = f
            filename2 = files[i+1]
            #filename2 = f.replace("0_real", "1_real").replace("0_fake", "1_fake")
            if not os.path.exists(filename1) or not os.path.exists(filename2):
                continue
            
            img1 = cv2.resize(read_img(filename1), (w, h))
            img2 = cv2.resize(read_img(filename2), (w, h))

            ### load flow
            video_ind = f.split("/")[-1].split("_")[0]
            video_ind2 = filename2.split("/")[-1].split("_")[0]
            if video_ind != video_ind2:
                continue
            img_ind = "%05d" % (int(filename2.split("/")[-1].split("_")[1].split(".")[0]))
            d = np.load(os.path.join(viper_path, "val/flowbw/", video_ind, video_ind + "_" + img_ind + ".npz"))
            u   = d['u']
            v   = d['v']
            gt_flow = (np.array([u, v]).transpose(1,2,0))

            # resize img and flow to the size
            # img1 = img2tensor(cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR)).to(device)
            # img2 = img2tensor(cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)).to(device)
            img1 = img2tensor(img1).to(device)
            img2 = img2tensor(img2).to(device)
            gt_flow = resize_flow(img2tensor(gt_flow).to(device), w, h, original_w, original_h)

            # process nan value in flow
            gt_flow[torch.isnan(gt_flow)] = 0
            ## warp img2
            img_21, warp_mask = warp(img2, gt_flow, "forward")
            img_21[torch.isnan(img_21)] = 0
            warp_mask = tensor2img(warp_mask)
            img1_ = tensor2img(img1)
            # img2_ = tensor2img(img2)
            img_21 = tensor2img(img_21)

            if "source" in exp:
                # estimate the occlusion mask
                mask = np.exp(opt.alpha * np.sum(np.square(img_21 - img1_), -1))
                with open(os.path.join(masks_path, video_ind + "_" + img_ind + ".npy"), 'wb') as fn:
                    np.save(fn, mask)

            else:
                with open(os.path.join(masks_path, video_ind + "_" + img_ind + ".npy"), 'rb') as f:
                    mask = np.load(f)


            diff = np.sum(np.abs(img_21 - img1_), -1)
            diff_masked = np.multiply(diff, mask)

            err = (np.mean(diff_masked))
            # print(err)
            errs.append(err)
        except FileNotFoundError as e:
            print(e)
        # except Exception as e:
        #     print(e)
    print("exp:", exp, "err:",sum(errs)/len(errs))