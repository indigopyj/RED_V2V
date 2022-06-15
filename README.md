# Efficient Inference for Video-to-Video Translation based on Temporal Redundancy Reduction

# 0. Virtual Environment setting

1) Create a new conda environment

```bash
conda create -n red python=3.8
```

2) Install the packages you need

```bash
pip install -r requirements.txt
```

# 1. Dataset Preparation

### Viper to Cityscapes

Viper dataset can be downloaded [here](https://playing-for-benchmarks.org/download/). We used ‘jpg’ image dataset for training/validation due to the limited space and **backward optical flow** for evaluating warping error. We used validation set for both validation and test phase. 

Cityscapes dataset is available [here](https://www.cityscapes-dataset.com/downloads/). Note that we need to deal with video data, so we downloaded “[leftImg8bit_sequence_trainvaltest.zip (324GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=14)”. Don’t be confused by downloading image dataset.

### Face to Face

*Face to Face dataset* is provided by RecycleGAN and can be downloaded [here](https://www.dropbox.com/s/s6kzovbrevin5tr/faces.tar.gz?dl=0). A single image within this dataset is composed of consecutive 3 frames. So we cropped it into a single frame when data-preprocessing.

# 2. Training

There are two steps for training. First, train an original generator. Then, train our method, RED with the original model.

Pretrained models are available below.

1. [Viper-to-Cityscapes original model](https://davian-lab.quickconnect.to/d/s/p532tgVkC7RVvjzKG43eXmm78MLmQ2Sy/W3F70eTb5zlDZWAsFSLHGzRpjHbmqMCk-WLkAxhmumAk)
    1. [Viper-to-Cityscapes RED model](https://davian-lab.quickconnect.to/d/s/p532ugFUzqykV8hDvr5Q35TP2qBfElbk/Xz7UnRDXr4IuNSWOIOZtcq5pjfygFWzY-WriAlwOumAk)
2. [Obama-to-Trump original model](https://davian-lab.quickconnect.to/d/s/p5xjE9W6O7e8KopBgsTrL32ZFZ40FF2T/R7yCII_paOcO9g1n3q_Wbwpgy1m97tZ--e7vgSpdjmQk)
    1. [Obama-to-Trump RED model](https://davian-lab.quickconnect.to/d/s/p5xhW6mDzER2dYjcF0m5qNeCihB2kjLX/FsO-jM60rqOKJbqxWs_MpaMEOaQ420xX-JrxgZKRjmQk)
3. [Oliver-to-Colbert original model](https://davian-lab.quickconnect.to/d/s/p6tnR9Il4YU04KHAVbeT0uw1oWT4HJmH/-ZwBD9HAfcwhIraCvZjym4hu4jPYlMDo-FLzAi5cdmgk)
    1. [Oliver-to-Colbert RED model](https://davian-lab.quickconnect.to/d/s/p6toa23IVyJllY5aKHP6oLSqSsFGJdKz/u_z4lXdVEQz9KcAE3wubDIs_Yey-2BRB-V7zgc6Edmgk)

### Original Generator

**Viper-to-Cityscapes**

```bash
python train.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 542 --loadSizeH 286 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 --lambda_spa_unsup_A 10 --lambda_spa_unsup_B 10 --lambda_unsup_cycle_A 10 --lambda_unsup_cycle_B 10 --lambda_cycle_A 0 --lambda_cycle_B 0 --lambda_content_A 1 --lambda_content_B 1 --batchSize 1 --noise_level 0.001  --niter_decay 0 --niter 2
```

**Obama-to-Trump**

```bash
python train.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name obama_to_trump --loadSizeW 768 --loadSizeH 256 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 --lambda_spa_unsup_A 10 --lambda_spa_unsup_B 10 --lambda_unsup_cycle_A 10 --lambda_unsup_cycle_B 10 --lambda_cycle_A 0 --lambda_cycle_B 0 --lambda_content_A 1 --lambda_content_B 1 --noise_level 0.001  --niter_decay 50 --niter 50 --batchSize 4 --lr 0.0002
```

### RED

**Viper-to-Cityscapes**

```bash
python train_RED.py --dataroot path/to/data \
--name RED_v2c --dataset_mode video --main_G_path v2c_experiment/latest_net_G_A.pth \
--gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
--tensorboard_dir tensorboard/ \
--lambda_L1 10.0 \
--lambda_L1_out 10.0 \
--lpips --lambda_lpips 20.0 \
--act_GAN_loss --spectral_norm
```

**Obama-to-Trump**

```bash
python train_RED.py --dataroot path/to/data \
--name RED_obama_to_trump --dataset_mode video --main_G_path obama_to_trump/latest_net_G_A.pth \
--gpu_ids 0 --no_dropout --pool_size 0 --batchSize 8 --actl1 --niter_decay 200 --niter 2500 --save_epoch_freq 300 --save_latest_freq 10000 \
--tensorboard_dir tensorboard/ --lambda_L1 10.0 --lambda_L1_out 10.0 --lpips --lambda_lpips 50.0 \
--act_GAN_loss --spectral_norm --phase train \
--loadSizeH 256 --loadSizeW 768 --fineSizeH 256 --fineSizeW 256 --max_interval 10
```

# 2. Test

Note that we test only 200 frames per video.

### Original Generator

**Viper-to-Cityscapes**

```bash
python test.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 512 --loadSizeH 256 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout --which_epoch 2
```

**Obama-to-Trump**

```bash
python test.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name obama_to_trump --loadSizeW 256 --loadSizeH 256 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout
```

### RED

**Viper-to-Cityscapes**

```bash
python test_RED.py \
 --dataroot path/to/data/ \
 --name v2c_red --dataset_mode video --main_G_path v2c_experiment/latest_net_G_A.pth \
 --how_many num_of_videos_you_want_to_test --no_dropout --results_dir ./results/ --which_epoch 2400
```

**Obama-to-Trump**

```bash
python test_RED.py --dataroot path/to/data/ \
--name RED_obama_to_trump --dataset_mode video --main_G_path obama_to_trump/model.pth --no_dropout --results_dir ./results/ --loadSizeH 256 --loadSizeW 256 --fineSizeH 256 --fineSizeW 256 --max_interval 10
```

# 3. Evaluate

## FID

For evaluating FID score, we use 'pytorch-fid' from official PyTorch package. To compute the FID score between two datasets, where images of each dataset are contained in an individual folder:

```
pip install pytorch-fid
python -m pytorch_fid path/to/real_dataset path/to/generated_dataset
```

In this paper, we use 200 frames of Oliver-to-Colbert test dataset as real dataset and our 200 result images as generated dataset.

## Warping Error

NOTE : we recommend you to do this step after testing the model and generating translated frames.

To evaluate warping error, the ground truth optical flow is required.

You can download GT flow of Viper dataset according to the description from **[Section 1](https://github.com/indigopyj/RED_V2V#1-dataset-preparation)**.

When you want to calculate warping error with images generated by the *original generator*, run this code:

```bash
python eval_temp_cons_torch.py --mode main --mask_path precomputed_mask_main/ --img_path ./results/RED_model_name/test_2400/real/
```

Or, if you want to calculate warping error with images generated by *RED*, run this code:

```bash
python eval_temp_cons_torch.py --mode RED --mask_path precomputed_mask_RED/ --img_path ./results/RED_model_name/test_2400/fake/
```

You can set the same “img_path” for both two cases, because both the images generated by the original model and ones generated by RED are created together during test time. But note that you should set the following path with “**real**” or “**fake**” to distinguish them.

# 4. Etc

- The novel codes that we write by ourselves:
    - train_RED.py
    - test_RED.py
    - models/RED_model.py
    - models/modules/RED_modules.py
    - models/modules/shift_modules.py
    - data/video_dataset.py
- The pretrained VGG weights is available [here](https://davian-lab.quickconnect.to/d/s/p54tQnnBHylHD78i9SwrVEYnOMbPAioH/P0TaFBnGc8qYHexocB1fY5922gYnVmKO-TrYAgeuzmAk) for computing perceptual loss during training.