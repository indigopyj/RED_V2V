from torchprofile import profile_macs
from options.test_options import TestOptions
from models.models import create_model, create_model_RED
import torch
from torch import nn
from data.data_loader import CreateDataLoader


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.batch_size = 1
    
    opt.RED = False
    if opt.RED:
        opt.model = "unsup_single"
        mainG_path = opt.main_G_path
        model = create_model_RED(opt)
        mainG = create_model(opt)
        model.setup_with_G(opt, mainG)
        model.eval()
    else:
        model = create_model(opt)
        model.test()
        
    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    
    model.profile()
    
        
    