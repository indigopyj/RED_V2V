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
        if 'ObamaTrump' in opt.dataroot or 'OliverColbert' in opt.dataroot:
            if opt.phase == "train":
                self.dir_A = os.path.join(opt.dataroot, "trainA")
                self.dir_B = os.path.join(opt.dataroot, "trainB")
            if opt.phase == "test" or opt.phase == "val":
                self.dir_A = os.path.join(opt.dataroot, "testA")
                self.dir_B = os.path.join(opt.dataroot, "testB")
        else:
            
            if opt.phase == "test" or opt.phase == "val":
                phase = "val"
            else:
                phase = "train"
            self.dir_A = os.path.join(opt.dataroot, 'Viper', phase, 'img')
            self.dir_B = os.path.join(opt.dataroot, "Cityscapes_sequence", "leftImg8bit", phase)

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
        if 'ObamaTrump' in self.opt.dataroot or 'OliverColbert' in self.opt.dataroot:
            seq_path = self.dir_A
            
        if self.opt.phase == "test":
            return {'seq_path' : seq_path }
        
        A_path = sorted([f for f in os.listdir(seq_path) if f.endswith(".jpg") or f.endswith(".png")])
        interval = torch.randint(1, self.opt.max_interval, [1]).item()
        if self.opt.phase != 'train':
            idx1 = torch.randint(0, len(A_path) - self.opt.max_interval, [1]).item()
        else:
            idx1 = torch.randint(0, len(A_path) - interval, [1]).item()
        img_root = seq_path
        
        img1 = Image.open(os.path.join(img_root, A_path[idx1])).convert("RGB") # change
        img2 = Image.open(os.path.join(img_root, A_path[idx1 + interval])).convert("RGB") #change

        # get the triplet from A
        img1 = scale_img(img1, self.opt, self.transform)
        img2 = scale_img(img2, self.opt, self.transform)
        
        img_path = A_path[idx1]

        return {'img1': img1, 'img2': img2, "img1_paths": A_path[idx1], "img_root": img_root}

    def __len__(self):
        return len(self.seq_list)

    def name(self):
        return 'VideoDataset'
