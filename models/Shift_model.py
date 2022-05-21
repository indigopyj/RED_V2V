import copy
import os

import torch
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from models import networks
from .base_model import BaseModel
from models.modules.shift_modules import Learnable_Shift
from util.util import tensor2im, save_image
import cv2
import numpy as np
from torch.nn import functional as F
from torch import nn
from torchprofile import profile_macs
from util.image_pool import ImagePool
from models.networks import init_weights
from util.util import warp

def init_net(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    return net


def create_eval_dataloader(opt, phase="val"):
    opt = copy.deepcopy(opt)
    opt.isTrain = False
    opt.serial_batches = True
    opt.phase = phase # 고쳐야됨
    dataloader = CreateDataLoader(opt)
    dataloader = dataloader.load_data()
    return dataloader


class ShiftModel(BaseModel):
    def __init__(self, opt):
        #assert opt.isTrain
        super(ShiftModel, self).__init__()
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['shift']
        
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['Shift']

        #self.netR = REDNet(input_nc=256+3+3, ngfs=[256], n_convs=self.opt.n_convs, shift_param=self.opt.shift_param)
        self.net_Shift = Learnable_Shift(input_nc=6, n_channel=64, n_convs=2)
        self.net_Shift = init_net(self.net_Shift, opt.gpu_ids)
        

        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL1_out = torch.nn.L1Loss()
            
        
        if opt.isTrain:
            # if self.opt.GAN_loss:
            #     self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
            #                                 opt.which_model_netD,
            #                                 opt.n_layers_D, opt.norm, False, opt.init_type, self.gpu_ids)
                
            #     self.fake_B_pool = ImagePool(opt.pool_size)
            
            # if self.opt.Temporal_GAN_loss:
            #     self.netD_T_A = networks.define_D(opt.output_nc * 2, opt.ndf,
            #                                 opt.which_model_netD,
            #                                 opt.n_layers_D, opt.norm, False, opt.init_type, self.gpu_ids) # temporal discriminator
            
            #     self.loss_names += ['D_T']
            # if self.opt.lambda_feature > 0:
            #     self.loss_names += ['feature']
            #     hook_list = [12, 15, 17, 20] # 12,15 : resnetblock / 17, 20: decoder conv_norm
            #     self.mapping_layers = ['model.%d' % i for i in hook_list]
            #     self.hooks = []
            #     self.main_acts = {}
            #     self.RED_acts = {}
            # if self.opt.GAN_loss and self.isTrain:
            #     #self.criterionGAN = networks.GANLoss(use_lsgan=False, tensor=self.Tensor)
            #     self.criterionGAN = networks.GANLoss_custom(gan_mode='vanilla').to(self.device)
            # if self.opt.lpips:
            #     from models import lpips
            #     self.netLPIPS = lpips.PerceptualLoss(model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], use_gpu=True,)
            
            # if opt.continue_train:
            #     self.load_network(self.netR, 'RED', self.opt.which_epoch)
            #     self.load_network(self.netD_A, 'D_A', self.opt.which_epoch)  
                
            if self.opt.shift_param:
                self.opt.actl1 = True
                self.loss_names += ['shift']
                self.criterionL1_shift = torch.nn.L1Loss()
                
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.net_Shift.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer)
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
            # if self.opt.GAN_loss:
            #     self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #     self.optimizers.append(self.optimizer_D_A)
            # if self.opt.Temporal_GAN_loss:
            #     self.optimizer_D_T_A = torch.optim.Adam(self.netD_T_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #     self.optimizers.append(self.optimizer_D_T_A)
            #     self.loss_D_T = 0.0
            
        self.eval_dataloader = create_eval_dataloader(self.opt, "val")
        

    def setup_with_G(self, opt, model, verbose=True):
        if len(self.opt.gpu_ids) > 1:
            self.modelG = model.netG_A.module
        else:
            self.modelG = model.netG_A
        if not opt.isTrain or opt.continue_train:
            self.load_network(self.netR, "RED", opt.which_epoch)
        if self.opt.main_G_path != None:
            self.load_pretrained_network(self.modelG, self.opt.main_G_path)
        if self.opt.phase == "train":
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        for param in self.modelG.parameters():
            param.requires_grad = False
        print("freeze main generator")
    
    def hook_acts(self, acts):
        def get_activation(mem, name):
            def get_output_hook(module, input, output):
                mem[name + str(output.device)] = output

            return get_output_hook

        def add_hook(net, mem, mapping_layers):
            for n, m in net.named_modules():
                if n in mapping_layers:
                    hook = m.register_forward_hook(get_activation(mem, n))
                    self.hooks.append(hook)

        add_hook(self.modelG, acts, self.mapping_layers)
    
    def unregister_forward_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def calc_feature_loss(self):
        losses = []
        for i, n in enumerate(self.mapping_layers):
            RED_acts = tuple([self.RED_acts[key] for key in sorted(self.RED_acts.keys()) if n in key])
            main_acts = [self.main_acts[key] for key in sorted(self.main_acts.keys()) if n in key]
            loss = [F.mse_loss(Ract, Mact) for Ract, Mact in zip(RED_acts, main_acts)]
            #loss = loss.sum()
            setattr(self, 'loss_feature%d' % i, loss)
            losses.append(loss[0])
        return sum(losses)
    

    def set_input(self, input):
        self.img1 = input['img1'].to(self.device)
        self.img2 = input['img2'].to(self.device)
        self.img1_paths = input['img1_paths']
        self.image_root = input['img_root']
    
    def set_test_input(self, img_paths): # load an image from img_paths and preprocess it as a train input.
        from PIL import Image
        import torchvision.transforms as transforms
        from util.util import scale_img
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        self.next_img_paths = img_paths
        self.next_img = []
        for i in range(len(img_paths)):
            next_img = Image.open(img_paths[i]).convert("RGB")
            next_img = scale_img(next_img, self.opt, self.transform).unsqueeze(0)
            self.next_img.append(next_img)
        self.next_img = torch.cat(self.next_img, dim=0).to(self.device)

    def forward(self):
        # img1 : reference frame
        # img2 : next frame
        # [B, 256, 72, 136]
        B = self.img1.shape[0]
        self.height = 64
        self.width = 128
        #self.real_diff = self.real_act - self.ref_act
        self.img1_resized = F.interpolate(self.img1, size=(self.height, self.width), mode='bicubic')
        self.img2_resized = F.interpolate(self.img2, size=(self.height, self.width), mode='bicubic')
        # if shifted, self.ref_act is warped and self.flow is not None
        self.shift_param = self.net_Shift(torch.cat((self.img1_resized, self.img2_resized) , 1))
        B, C, H, W = self.img1_resized.shape
        shift_lr = self.shift_param[:, 0].expand(self.height, self.width, B).permute(2,0,1) # shift right(+) or left(-)
        shift_ud = self.shift_param[:, 1].expand(self.height, self.width, B).permute(2,0,1) # shift up(+) or down(-)
        self.flow = torch.ones([B, 2, self.height, self.width])
        self.flow[:, 0, :, :] = shift_lr
        self.flow[:, 1, :, :] = shift_ud
        self.flow = self.flow.float().cuda()
        #warped_ref_act = warp(ref_act, flow)
        self.warped_img1 = warp(self.img1_resized, self.flow)
            

    def backward(self):
        self.loss_shift = self.criterionL1_shift(self.warped_img1, self.img2_resized)
        self.loss_shift.backward()

    
        
    def optimize_parameters(self, steps):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        
        
            
            
        
    def save(self, label):
        self.save_network(self.net_Shift, 'Shift', label, self.gpu_ids)


    def evaluate_model(self, step):
        self.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        save_dir = os.path.join(self.save_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.net_Shift.eval()
        
        
        with torch.no_grad(): 
            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
                self.set_input(data_i) # load random video and random index image
                
                for j in range(1, self.opt.max_interval):
                    # self.img1 : reference frame(past frame)
                    # self.next_img : next image
                    img2_paths = []
                    for batch_idx in range(len(self.img1_paths)):
                        img1_name, img1_ext = os.path.splitext(self.img1_paths[batch_idx])
                        img2_idx = int(img1_name.split("_")[1]) + j
                        img2_name =  "%s_%05d%s" %(img1_name.split("_")[0], img2_idx, img1_ext)
                        img2_path = os.path.join(self.image_root[batch_idx], img2_name) 
                        img2_paths.append(img2_path)
                    self.set_test_input(img2_paths) # load an image (img1 + interval) as a batch : self.next_img
                    img1_resized = F.interpolate(self.img1, size=(self.height, self.width), mode='bicubic')
                    nextimg_resized = F.interpolate(self.next_img, size=(self.height, self.width), mode='bicubic')
                    shift_params = self.net_Shift(torch.cat((img1_resized, nextimg_resized) , 1))
                    B, C, H, W = img1_resized.shape
                    shift_lr = shift_params[:, 0].expand(H, W, B).permute(2,0,1) # shift left(+) or right(-)
                    shift_ud = shift_params[:, 1].expand(H, W, B).permute(2,0,1) # shift up(+) or down(-)
                    flow = torch.ones([B, 2, H, W])
                    flow[:, 0, :, :] = shift_lr
                    flow[:, 1, :, :] = shift_ud
                    flow = flow.float().cuda()
                    #warped_ref_act = warp(ref_act, flow)
                    warped_img1 = warp(img1_resized, flow)
                        
                    for k in range(len(self.img1_paths)):
                        img1_name, ext = os.path.splitext(self.img1_paths[k])
                        name = f"{img1_name}_{j}{ext}" # interval_originalname
                        input1_im = tensor2im(img1_resized, idx=k)
                        input2_im = tensor2im(nextimg_resized, idx=k)
                        fake = tensor2im(warped_img1, idx=k)
                        cat_img = np.concatenate((input1_im, input2_im, fake), axis=1)
                        save_image(cat_img, os.path.join(save_dir, '%s' % name), create_dir=True)

        self.net_Shift.train()
    
    def test_model(self, result_path):
        def make_heatmap(activation):
            a = activation.cpu().numpy()
            output = np.abs(a)
            output = np.sum(output,axis=1).squeeze()
            output /= output.max()
            output *= 255
            output = 255 - output.astype('uint8')
            heatmap = cv2.applyColorMap(output, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (512,256))
            #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            return heatmap
        
        os.makedirs(result_path, exist_ok=True)
        
        self.netR.eval()
        self.test_dataloader = create_eval_dataloader(self.opt, "test")

        with torch.no_grad():
            for seq_idx, seq_i in enumerate(tqdm(self.test_dataloader, desc='Eval       ', position=2, leave=False)):
                if seq_idx >= self.opt.how_many:  # only apply our model to opt.how_many videos.
                    break
               
                vid_name = seq_i['seq_path'][0].split("/")[-1]
                video = cv2.VideoWriter(os.path.join(result_path, vid_name+'.mp4'), fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=10, frameSize=(self.opt.fineSizeW*3, self.opt.fineSizeH))
                img_list = sorted(os.listdir(seq_i['seq_path'][0]))[:200] # 300:  limit length of test video
                for i, data_i in enumerate(img_list):
                    data_path = os.path.join(seq_i['seq_path'][0], data_i)
                    data_path = [data_path] # one batch setting
                    self.set_test_input(data_path)
                    # self.next_img : current frame
                    # self.reference_img : past frame(picked every interval)
                    if i % self.opt.max_interval == 0:
                        reference_img = self.next_img
                        fake_im = self.modelG(self.next_img)
                        #real_act = self.modelG.model[:self.opt.layer_idx](self.next_img)
                        #fake_im = self.modelG.model[self.opt.layer_idx:](real_act)
                        real_im = fake_im
                        activations = self.modelG.model[:self.opt.layer_idx](reference_img)
                        ref_resized = F.interpolate(reference_img, size=(self.height, self.width), mode='bicubic')
                        # hm = make_heatmap(activations)
                        # hm1 = make_heatmap(real_act)
                        # hm2 = np.zeros((256, 512, 3), np.uint8)
                    else:
                        nextimg_resized = F.interpolate(self.next_img, size=(self.height, self.width), mode='bicubic')
                        fake_diff, warped_activations, _ = self.netR(ref_resized, nextimg_resized, activations, 0)
                        #hm = make_heatmap(fake_diff)
                        #real_act = self.modelG.model[:self.opt.layer_idx](self.next_img)
                        fake_act = warped_activations + fake_diff
                        
                        #hm2 = make_heatmap(fake_act)
                        real_im = self.modelG.model(self.next_img)
                        #real_im = self.modelG.model[self.opt.layer_idx:](real_act)
                        fake_im = self.modelG.model[self.opt.layer_idx:](fake_act)
                    
                        # hm1 = make_heatmap(real_act)
                        # hm2 = make_heatmap(fake_act)
                    name = f"{vid_name}_{i}.png"
                    input_im = tensor2im(self.next_img)
                    real_im = tensor2im(real_im)
                    fake_im = tensor2im(fake_im)
                    #cat_img = np.concatenate((fake_im, hm1, hm2, hm), axis=1) # real_output, real_act, fake_act, diff
                    cat_img = np.concatenate((input_im, real_im, fake_im), axis=1)
                    #cat_img = np.concatenate((input_im, real_im, fake_im, hm), axis=1)
                    save_image(input_im, os.path.join(result_path, 'input', '%s' % name), create_dir=True)
                    save_image(real_im, os.path.join(result_path, 'real', '%s' % name), create_dir=True)
                    save_image(fake_im, os.path.join(result_path, 'fake', '%s' % name), create_dir=True)
                    cat_img = cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR)
                    video.write(cat_img)
    
    
    def profile(self, verbose=True):
        self.test_dataloader = create_eval_dataloader(self.opt, "test")
        self.seq = next(iter(self.test_dataloader))
        img_list = sorted(os.listdir(self.seq['seq_path'][0]))
        img1 = img_list[0]
        data_path = os.path.join(self.seq['seq_path'][0], img1)
        self.set_test_input([data_path])
        activations = self.modelG.model[:self.opt.layer_idx](self.next_img)
        ref_resized = F.interpolate(self.next_img, size=(self.height, self.width), mode='bicubic')
        img2 = img_list[1]
        data_path = os.path.join(self.seq['seq_path'][0], img2)
        self.set_test_input([data_path])
        nextimg_resized = F.interpolate(self.next_img, size=(self.height, self.width), mode='bicubic')
        
        if isinstance(self.netR, nn.DataParallel):
            netR = self.netR.module
        else:
            netR = self.netR
        with torch.no_grad():
            idx = torch.tensor(0)
            macs = profile_macs(netR, (torch.cat((ref_resized, nextimg_resized, activations), 1), idx))
        params = 0
        for p in netR.parameters():
            params += p.numel()
        if verbose:
            print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)
        return macs, params
    
    def get_current_errors(self):
        from collections import OrderedDict
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def generate_flow(H, W, scale_level=30.0, shift_level=50.0):
        import random
        scale_x = random.unit(-scale_level, scale_level)
        scale_y = random.unit(-scale_level, scale_level)
        shift_x = random.randint(-shift_level, shift_level)
        shift_y = random.randint(-shift_level, shift_level)
        
        x_range = np.arange(- H/2 * 0.01, H/2 * 0.01, 0.01)
        y_range = np.arange(- W/2 * 0.01, W/2 * 0.01, 0.01)
        
        xx = np.tile(x_range.reshape(1,-1), (H,1)) * scale_x + shift_x
        yy = np.tile(y_range.reshape(-1,1), (1,W)) * scale_y + shift_y
        
        flow = np.ones([H,W,2])
        flow[:,:,0] = xx
        flow[:,:,1] = yy
        flow = torch.from_numpy(flow.transpose((2, 0, 1))).float()

        return flow, (scale_x, scale_y), (shift_x, shift_y)