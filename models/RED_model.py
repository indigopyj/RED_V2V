import copy
import os

import torch
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from models import networks
from .base_model import BaseModel
from models.modules.RED_modules import REDNet
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
    #opt.serial_batches = True
    opt.phase = phase # 고쳐야됨
    dataloader = CreateDataLoader(opt)
    dataloader = dataloader.load_data()
    return dataloader


class REDModel(BaseModel):
    def __init__(self, opt):
        #assert opt.isTrain
        super(REDModel, self).__init__()
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['l1', 'l1_out']
        if opt.GAN_loss or opt.act_GAN_loss:
            self.loss_names += ['G_A', 'D_A']
        if opt.lpips:
            self.loss_names += ['lpips']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['img1', 'img2', 'fake_diff', 'real_diff']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['R']

        self.netR = REDNet(input_nc=256+3+3, ngfs=[256], n_convs=self.opt.n_convs, shift_param=self.opt.shift_param)
        print(self.netR)
        self.netR = init_net(self.netR, opt.gpu_ids)
        

        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL1_out = torch.nn.L1Loss()
            
        
        if opt.isTrain:
            if self.opt.GAN_loss:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, False, opt.init_type, self.gpu_ids)
                
                self.fake_B_pool = ImagePool(opt.pool_size)
                
            elif self.opt.act_GAN_loss:
                if self.opt.spectral_norm:
                    self.netD_A = networks.define_D(256 + 256 + 3 + 3, 512,
                                                "sn_basic",
                                                opt.RED_n_layers_D, opt.norm, False, opt.init_type, self.gpu_ids)    
                else:
                    self.netD_A = networks.define_D(256 + 256 + 3 + 3, 512,
                                                "activation",
                                                opt.RED_n_layers_D, opt.norm, False, opt.init_type, self.gpu_ids)
                self.fake_B_pool = ImagePool(opt.pool_size)
            
            if self.opt.Temporal_GAN_loss:
                self.netD_T_A = networks.define_D(3 + 3, 512,
                                            "activation",
                                            opt.n_layers_D, opt.norm, False, opt.init_type, self.gpu_ids) # temporal discriminator
            
                self.loss_names += ['D_T']
            if self.opt.lambda_feature > 0:
                self.loss_names += ['feature']
                hook_list = [12, 15, 17, 20] # 12,15 : resnetblock / 17, 20: decoder conv_norm
                self.mapping_layers = ['model.%d' % i for i in hook_list]
                self.hooks = []
                self.main_acts = {}
                self.RED_acts = {}
            if self.opt.GAN_loss and self.isTrain:
                #self.criterionGAN = networks.GANLoss(use_lsgan=False, tensor=self.Tensor)
                self.criterionGAN = networks.GANLoss_custom(gan_mode='vanilla').to(self.device)
            elif self.opt.act_GAN_loss and self.isTrain:
                self.criterionGAN = networks.GANLoss_custom(gan_mode='lsgan').to(self.device)
            if self.opt.lpips:
                from models import lpips
                self.netLPIPS = lpips.PerceptualLoss(model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], use_gpu=True,)
            
            if opt.continue_train:
                self.load_network(self.netR, 'RED', self.opt.which_epoch)
                if self.opt.GAN_loss or self.opt.act_GAN_loss:
                    self.load_network(self.netD_A, 'D_A', self.opt.which_epoch)  
                if self.opt.Temporal_GAN_loss:
                    self.load_network(self.netD_T_A, 'D_T_A', self.opt.which_epoch) 
                
            if self.opt.shift_param:
                self.opt.actl1 = True
                self.loss_names += ['shift']
                self.criterionL1_shift = torch.nn.L1Loss()
                
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer)
            if self.opt.GAN_loss or self.opt.act_GAN_loss:
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_A)
            if self.opt.Temporal_GAN_loss:
                self.optimizer_D_T_A = torch.optim.Adam(self.netD_T_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_T_A)
                self.loss_D_T = 0.0
            
        self.eval_dataloader = create_eval_dataloader(self.opt, "val")
        

    def setup_with_G(self, opt, model, verbose=True):
        if len(self.opt.gpu_ids) > 1:
            self.modelG = model.netG_A.module
        else:
            self.modelG = model.netG_A
        if not opt.isTrain or opt.continue_train:
            print("load %s network" % opt.which_epoch)
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
        b = self.img1.size(0)
        activations = self.modelG.model[:self.opt.layer_idx](torch.cat((self.img1, self.img2), 0)).detach()  # TODO: hyper-parameter tuning
        self.real_act = activations[b:] # [B, 256, 64, 128]
        self.ref_act = activations[:b]
        self.real_diff = self.real_act - self.ref_act
        self.img1_resized = F.interpolate(self.img1, size=activations.shape[2:], mode='bicubic')
        self.img2_resized = F.interpolate(self.img2, size=activations.shape[2:], mode='bicubic')
        # if shifted, self.ref_act is warped and self.flow is not None
        self.fake_diff, self.ref_act, self.flow = self.netR(self.img1_resized, self.img2_resized, self.ref_act, 0) 
            
        if self.opt.lambda_feature > 0:
            self.hook_acts(self.main_acts)
            
        self.real_im = self.modelG.model[self.opt.layer_idx:](self.real_act).detach()
        
        if self.opt.lambda_feature > 0:
            self.unregister_forward_hook()
        
        self.fake_act = self.ref_act + self.fake_diff
        
        if self.opt.lambda_feature > 0:
            self.hook_acts(self.RED_acts)
        self.fake_im = self.modelG.model[self.opt.layer_idx:](self.fake_act)
        self.ref_output = self.modelG.model[self.opt.layer_idx:](self.ref_act)

    def backward(self):
        lambda_l1 = self.opt.lambda_L1
        lambda_l1_out = self.opt.lambda_L1_out
        if self.opt.actl1:
            self.loss_l1 = self.criterionL1(self.fake_act, self.real_act)
        else:
            self.loss_l1 = self.criterionL1(self.fake_diff, self.real_diff)
        self.loss_l1_out = self.criterionL1_out(self.fake_im, self.real_im)
        
        self.loss = self.loss_l1 * lambda_l1 + self.loss_l1_out * lambda_l1_out
        
        # GAN loss
        if self.opt.GAN_loss:
            pred_fake = self.netD_A(self.fake_im)
            self.loss_G_A = self.criterionGAN(pred_fake, True, for_discriminator=False)
            self.loss += self.loss_G_A
            
        if self.opt.act_GAN_loss:
            pred_fake = self.netD_A(torch.cat((self.img1_resized, self.img2_resized, self.ref_act, self.fake_diff), 1))
            self.loss_G_A = self.criterionGAN(pred_fake, True, for_discriminator=False)
            self.loss += self.loss_G_A
            
        if self.opt.lpips:
            self.loss_lpips = self.netLPIPS(self.fake_im, self.real_im).mean()
            self.loss += self.opt.lambda_lpips * self.loss_lpips
        
        if self.opt.lambda_feature > 0:
            self.loss_feature = self.calc_feature_loss() * self.opt.lambda_feature
            self.loss += self.loss_feature
            
        if self.opt.shift_param:
            warped_img1 = warp(self.img1_resized, self.flow)
            self.loss_shift = self.opt.lambda_shift * self.criterionL1_shift(warped_img1, self.img2_resized)
            self.loss += self.loss_shift
        
        self.loss.backward()

    
        
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        if self.opt.GAN_loss:
            fake_B = self.fake_B_pool.query(self.fake_im)
            real = self.real_im
        elif self.opt.act_GAN_loss:
            real = torch.cat((self.img1_resized, self.img2_resized, self.ref_act, self.real_diff), 1)
            fake = torch.cat((self.img1_resized, self.img2_resized, self.ref_act, self.fake_diff), 1)
            fake_B = self.fake_B_pool.query(fake)
        self.loss_D_A = self.backward_D_basic(self.netD_A, real, fake_B)
        #self.loss_D_A = self.loss_D_A.data
    
    def calc_D_T_loss(self, past_frame, real_frame, fake_frame):
        real_B = torch.cat((past_frame, real_frame), 1)
        fake_B = torch.cat((past_frame, fake_frame), 1)
        pred_real = self.netD_T_A(real_B)
        loss_D_T_real = self.criterionGAN(pred_real, True)
        
        pred_fake = self.netD_T_A(fake_B.detach())
        loss_D_T_fake = self.criterionGAN(pred_fake, False)
        loss_D_T = (loss_D_T_real + loss_D_T_fake) * 0.5
        return loss_D_T

    def backward_D_T_A(self):
        self.loss_D_T = self.opt.lambda_D_T * self.calc_D_T_loss(self.ref_output, self.real_im, self.fake_im)
        self.loss_D_T.backward()
        
    def optimize_parameters(self, steps):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
            
        if self.opt.GAN_loss or self.opt.act_GAN_loss:
            # D_A
            self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            self.optimizer_D_A.step()
        
        if self.opt.Temporal_GAN_loss:
            self.optimizer_D_T_A.zero_grad()
            self.backward_D_T_A()
            self.optimizer_D_T_A.step()
        
        
            
            
        
    def save(self, label):
        self.save_network(self.netR, 'RED', label, self.gpu_ids)
        if self.opt.GAN_loss or self.opt.act_GAN_loss:
            self.save_network(self.netD_A, "D_A", label, self.gpu_ids)
        if self.opt.Temporal_GAN_loss:
            self.save_network(self.netD_A, "D_T_A", label, self.gpu_ids)


    def evaluate_model(self, step):
        self.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        save_dir = os.path.join(self.save_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netR.eval()
        
        
        with torch.no_grad(): 
            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
                self.set_input(data_i) # load random video and random index image
                activations = self.modelG.model[:self.opt.layer_idx](self.img1) # randomly chosen image : self.img1
                
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
                    img1_resized = F.interpolate(self.img1, size=activations.shape[2:], mode='bicubic')
                    nextimg_resized = F.interpolate(self.next_img, size=activations.shape[2:], mode='bicubic')
                    fake_diff, warped_activations, _ = self.netR(img1_resized, nextimg_resized, activations, 0)
                    real_im = self.modelG.model(self.next_img)
                    fake_im = self.modelG.model[self.opt.layer_idx:](warped_activations + fake_diff)
            
                    for k in range(len(self.img1_paths)):
                        img1_name, ext = os.path.splitext(self.img1_paths[k])
                        name = f"{img1_name}_{j}{ext}" # interval_originalname
                        input1_im = tensor2im(self.img1, idx=k)
                        input2_im = tensor2im(self.next_img, idx=k)
                        real = tensor2im(real_im, idx=k)
                        fake = tensor2im(fake_im, idx=k)
                        save_image(input1_im, os.path.join(save_dir, 'input1', '%s' % self.img1_paths[k]), create_dir=True)
                        save_image(input2_im, os.path.join(save_dir, 'input2', '%s' % name), create_dir=True)
                        save_image(real, os.path.join(save_dir, 'real', '%s' % name), create_dir=True)
                        save_image(fake, os.path.join(save_dir, 'fake', '%s' % name), create_dir=True)

        self.netR.train()
    
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
                #video = cv2.VideoWriter(os.path.join(result_path, vid_name+'.mp4'), fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=10, frameSize=(self.opt.loadSizeW*3, self.opt.loadSizeH))

                video = cv2.VideoWriter(os.path.join(result_path, vid_name+'.mp4'), fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=10, frameSize=(self.opt.fineSizeW*3, self.opt.fineSizeH))
                img_list = sorted(os.listdir(seq_i['seq_path'][0]))[:200] # 300:  limit length of test video
                for i, data_i in enumerate(img_list):
                    data_path = os.path.join(seq_i['seq_path'][0], data_i)
                    data_path = [data_path] # one batch setting
                    self.set_test_input(data_path)
                    # self.next_img : current frame
                    # reference_img : past frame(picked every interval)
                    if i % self.opt.max_interval == 0:
                        reference_img = self.next_img
                        fake_im = self.modelG(reference_img)
                        #real_act = self.modelG.model[:self.opt.layer_idx](self.next_img)
                        #fake_im = self.modelG.model[self.opt.layer_idx:](real_act)
                        real_im = fake_im
                        activations = self.modelG.model[:self.opt.layer_idx](reference_img)
                        ref_resized = F.interpolate(reference_img, size=activations.shape[2:], mode='bicubic')
                        # hm = make_heatmap(activations)
                        # hm1 = make_heatmap(real_act)
                        # hm2 = np.zeros((256, 512, 3), np.uint8)
                    else:
                        nextimg_resized = F.interpolate(self.next_img, size=activations.shape[2:], mode='bicubic')
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
                    #print(input_im.shape, real_im.shape, fake_im.shape)
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
        ref_resized = F.interpolate(self.next_img, size=activations.shape[2:], mode='bicubic')
        img2 = img_list[1]
        data_path = os.path.join(self.seq['seq_path'][0], img2)
        self.set_test_input([data_path])
        nextimg_resized = F.interpolate(self.next_img, size=activations.shape[2:], mode='bicubic')
        
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
    
