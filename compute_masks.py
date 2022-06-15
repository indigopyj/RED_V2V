from util.eval import *
import os
import numpy
import glob

fw_path = "../flownet2-pytorch/fw_flow"
bw_path = "../flownet2-pytorch/bw_flow"
mask_path = "../flownet2-pytorch/occlusion_masks"


video_list = os.listdir(fw_path)

for v in video_list:
    fw_flow_path = os.path.join(fw_path, v)
    bw_flow_path = os.path.join(bw_path, v)
    fw_list = sorted(glob.glob(os.path.join(fw_flow_path, "*.flo")))
    bw_list = sorted(glob.glob(os.path.join(bw_flow_path, "*.flo")))
    
    mask_folder = os.path.join(mask_path, v)
    for (fw, bw) in zip(fw_list, bw_list):
        fw_flow = read_flo(fw)
        bw_flow = read_flo(bw)
        file_name = os.path.splitext(os.path.basename(fw))[0] + '.npy'
        occ_mask = detect_occlusion(fw_flow, bw_flow)
        os.makedirs(mask_folder, exist_ok=True)
        np.save(os.path.join(mask_folder, file_name), occ_mask)
        print("saved %s", os.path.join(mask_folder, file_name))
    print("")
        
        
