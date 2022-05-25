import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch
from util.util import scale_img


class VideoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # if opt.split == "":
        if opt.phase == "test" or opt.phase == "val":
            phase = "val"
        else:
            phase = "train"
        self.dir_A = os.path.join(opt.dataroot, 'Viper', phase, 'img')
        self.dir_B = os.path.join(opt.dataroot, "Cityscapes_sequence", "leftImg8bit", phase)
            
        # if opt.phase == "train":
        #     self.dir_A = os.path.join(opt.dataroot, "train/A")
        #     self.dir_B = os.path.join(opt.dataroot, "train/B")
        # if opt.phase == "test":
        #     self.dir_A = os.path.join(opt.dataroot, "val/A")
        #     self.dir_B = os.path.join(opt.dataroot, "val/B")
        # else:
        #     self.dir_A = os.path.join(opt.dataroot, opt.split, "A")
        #     self.dir_B = os.path.join(opt.dataroot, opt.split, "B")

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.seq_list = sorted(os.listdir(self.dir_A))

        # self.transform = get_transform(opt)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        seq_path = os.path.join(self.dir_A, self.seq_list[index])
        A_path = sorted(os.listdir(seq_path))
        interval = torch.randint(1, self.opt.max_interval, [1]).item()
        if self.opt.phase != 'train':
            idx1 = torch.randint(0, len(A_path) - self.opt.max_interval, [1]).item()
        else:
            idx1 = torch.randint(0, len(A_path) - interval, [1]).item()
        img_root = seq_path
        if self.opt.phase == "test":
            return {'seq_path' : seq_path }
        img1 = Image.open(os.path.join(img_root, A_path[idx1])).convert("RGB") # change
        img2 = Image.open(os.path.join(img_root, A_path[idx1 + interval])).convert("RGB") #change

        # A = self.transform(A_img)
        # B = self.transform(B_img)
        # get the triplet from A
        img1 = scale_img(img1, self.opt, self.transform)
        img2 = scale_img(img2, self.opt, self.transform)
        # if self.opt.resize_mode == "scale_shortest":
        #     w, h = img1.size
        #     if w >= h: 
        #         scale = self.opt.loadSize / h
        #         new_w = int(w * scale)
        #         new_h = self.opt.loadSize
        #     else:
        #         scale = self.opt.loadSize / w
        #         new_w = self.opt.loadSize
        #         new_h = int(h * scale)
                
        #     img1 = img1.resize((new_w, new_h), Image.BICUBIC)
        # elif self.opt.resize_mode == "square":
        #     img1 = img1.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        # elif self.opt.resize_mode == "rectangle":
        #     img1 = img1.resize((self.opt.loadSizeW, self.opt.loadSizeH), Image.BICUBIC)
        # elif self.opt.resize_mode == "none":
        #     pass
        # else:
        #     raise ValueError("Invalid resize mode!")

        # img1 = self.transform(img1)

        # w = img1.size(2)
        # h = img1.size(1)
        # if self.opt.crop_mode == "square":
        #     fineSizeW, fineSizeH = self.opt.fineSize, self.opt.fineSize
        # elif self.opt.crop_mode == "rectangle":
        #     fineSizeW, fineSizeH = self.opt.fineSizeW, self.opt.fineSizeH
        # elif self.opt.crop_mode == "none":
        #     fineSizeW, fineSizeH = w, h
        # else:
        #     raise ValueError("Invalid crop mode!")

        # w_offset = random.randint(0, max(0, w - fineSizeW - 1))
        # h_offset = random.randint(0, max(0, h - fineSizeH - 1))

        # img1 = img1[:, h_offset:h_offset + fineSizeH, w_offset:w_offset + fineSizeW]


        # if self.opt.resize_mode == "scale_shortest":
        #     w, h = img2.size
        #     if w >= h: 
        #         scale = self.opt.loadSize / h
        #         new_w = int(w * scale)
        #         new_h = self.opt.loadSize
        #     else:
        #         scale = self.opt.loadSize / w
        #         new_w = self.opt.loadSize
        #         new_h = int(h * scale)
                
        #     img2 = img2.resize((new_w, new_h), Image.BICUBIC)
        # elif self.opt.resize_mode == "square":
        #     img2 = img2.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        # elif self.opt.resize_mode == "rectangle":
        #     img2 = img2.resize((self.opt.loadSizeW, self.opt.loadSizeH), Image.BICUBIC)
        # elif self.opt.resize_mode == "none":
        #     pass
        # else:
        #     raise ValueError("Invalid resize mode!")

        # img2 = self.transform(img2)

        # w = img2.size(2)
        # h = img2.size(1)
        # if self.opt.crop_mode == "square":
        #     fineSizeW, fineSizeH = self.opt.fineSize, self.opt.fineSize
        # elif self.opt.crop_mode == "rectangle":
        #     fineSizeW, fineSizeH = self.opt.fineSizeW, self.opt.fineSizeH
        # elif self.opt.crop_mode == "none":
        #     fineSizeW, fineSizeH = w, h
        # else:
        #     raise ValueError("Invalid crop mode!")
        # w_offset = random.randint(0, max(0, w - fineSizeW - 1))
        # h_offset = random.randint(0, max(0, h - fineSizeH - 1))

        # img2 = img2[:, h_offset:h_offset + fineSizeH,
        #      w_offset:w_offset + fineSizeW]
        
        
        img_path = A_path[idx1]
        #######
        # input_nc = self.opt.input_nc
        # output_nc = self.opt.output_nc

        # if input_nc == 1:  # RGB to gray
        #    tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #    A = tmp.unsqueeze(0)

        # if output_nc == 1:  # RGB to gray
        #    tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #    B = tmp.unsqueeze(0)
        return {'img1': img1, 'img2': img2, "img1_paths": A_path[idx1], "img_root": img_root}

    def __len__(self):
        return len(self.seq_list)

    def name(self):
        return 'VideoDataset'
