from torch import nn
import torch
from models.modules.shift_modules import Learnable_Shift
from util.util import warp
from cnngeometric_pytorch.model.cnn_geometric_model import CNNGeometric
from collections import OrderedDict
from cnngeometric_pytorch.geotnf.transformation import GeometricTnf
from cnngeometric_pytorch.image.normalization import normalize_image

class REDBlock(nn.Module):
    def __init__(self, input_nc, ngf=64, n_convs=8, norm_layer=nn.BatchNorm2d):
        super(REDBlock, self).__init__()
        self.model = []
        for i in range(n_convs-1):
            if i == 0:
                self.model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, padding=1, groups=input_nc), norm_layer(input_nc, eps=1e-04), nn.Conv2d(input_nc, ngf, kernel_size=1),
                            nn.ReLU(True), norm_layer(ngf, eps=1e-04),]
            else:
                self.model += [nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, groups=ngf), norm_layer(ngf, eps=1e-04), nn.Conv2d(ngf, ngf, kernel_size=1),
                            nn.ReLU(True), norm_layer(ngf, eps=1e-04)]
        
        self.model += [nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, groups=ngf), norm_layer(ngf, eps=1e-04), nn.Conv2d(ngf, ngf, kernel_size=1)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class REDNet(nn.Module):
    def __init__(self, input_nc, ngfs, n_convs=5, shift_param=False):
        super(REDNet, self).__init__()
        self.shift_param = shift_param
        layers = []
        if self.shift_param:
            # self.learnable_shift = Learnable_Shift(input_nc=6, n_channel=64, n_convs=2)
            model_aff_path = "saved_models/best_checkpoint_adam_full_affine_grid_lossvgg.pth.tar"
            use_cuda = torch.cuda.is_available()
            self.model_aff = CNNGeometric(output_dim=6,use_cuda=use_cuda,feature_extraction_cnn='vgg')
            self.resizeCNN = GeometricTnf(out_h=64, out_w=128, use_cuda = True)
            checkpoint = torch.load(model_aff_path, map_location=lambda storage, loc: storage)
            checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
            self.model_aff.load_state_dict(checkpoint['state_dict'])
            self.affTnf = GeometricTnf(geometric_model='affine', out_h=64, out_w=128, use_cuda=use_cuda)
            self.model_aff.eval()
            for param in self.model_aff.parameters():
                param.requires_grad = False
        for ngf in ngfs:
            layers += [REDBlock(input_nc, ngf, n_convs)]

        self.model = nn.Sequential(*layers)

    def forward(self, img1, img2, ref_act, idx=0):
        theta_aff = None
        if self.shift_param:
            # shift_params = self.learnable_shift(torch.cat((img1, img2), 1))
            #print(shift_params)
            # B, C, H, W = img1.shape
            # shift_lr = shift_params[:, 0].expand(H, W, B).permute(2,0,1) # shift right(+) or left(-)
            # shift_ud = shift_params[:, 1].expand(H, W, B).permute(2,0,1) # shift up(+) or down(-)
            # flow = torch.ones([B, 2, H, W])
            # flow[:, 0, :, :] = shift_lr
            # flow[:, 1, :, :] = shift_ud
            # flow = flow.float().cuda()
            # warped_ref_act = warp(ref_act, flow)
            # warped_img1 = warp(img1, flow)
            
            norm_img1 = normalize_image(img1)
            norm_img2 = normalize_image(img2)
            batch = {'source_image': norm_img1, "target_image": norm_img2}
            
            theta_aff = self.model_aff(batch)
            warped_img1 = self.affTnf(batch['source_image'], theta_aff.view(-1,2,3))
            warped_img1 = normalize_image(warped_img1, forward=False)
            warped_ref_act = self.affTnf(ref_act, theta_aff.view(-1,2,3))
            
            x = torch.cat((warped_img1, img2, warped_ref_act), 1)
            return self.model[idx](x), warped_ref_act, theta_aff
        
        x = torch.cat((img1, img2, ref_act), 1)
        return self.model[idx](x), ref_act, theta_aff
        