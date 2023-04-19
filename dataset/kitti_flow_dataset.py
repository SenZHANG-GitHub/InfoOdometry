import pdb
import numpy as np 
import torch 
import torch.nn.functional as F

import os
import math 
import random 
from glob import glob 
from tqdm import tqdm
from scipy.misc import imread
import copy
import re

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

## Utility functions
def read_gen(file_name):
    ext = os.path.splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        # if im.shape[2] > 3:
        #     return im[:,:,:3]
        # else:
        #     return im
        return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    else:
        raise ValueError()
    return []

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        if len(img.shape) == 2:
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2]
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]


def readFlowKITTI(filename):
    """
    => For HPC capacity: pip install opencv-python==4.2.0.34 to avoid wheel compilation
    """
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow,valid


## Dataset classes
class KITTIFlowDataset(torch.utils.data.Dataset):
    def __init__(self, split, root="data/kitti/optical_flow"):
        """
        NOTE: Both 2012/ and 2015/
        => Input: flowfeat rather than [img1, img2]
        => Output: flow to be reconstructed from latent representation from flowfeat
        """
        assert split in ["2012", "2015"]
        self.root = root
        self.split = split
        self.render_size = [320, 1216] # corresponds to [1024, 5, 19] flowfeat

        self.img0_list, self.img1_list, self.flowfeat_list, self.flow_list = [], [], [], []
        with open(os.path.join(self.root, "training-{}.txt".format(self.split)), "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                self.img0_list.append(line[0])
                self.img1_list.append(line[1])
                self.flowfeat_list.append(line[2])
                self.flow_list.append(line[3])
        
        # load flowfeat and flow into memory to speed up
        self.preloaded_flowfeat, self.preloaded_flow, self.preloaded_valid = {}, {}, {}
        for flowfeat, flow in zip(self.flowfeat_list, self.flow_list):
            tagff = flowfeat.split("/")[-1].split("+")[0]
            tagf  = flow.split("/")[-1].split(".")[0]
            assert tagff == tagf 
            self.preloaded_flowfeat[flowfeat] = torch.load(flowfeat).float() # [1024, 5, 19]

            flow_, valid_ = readFlowKITTI(flow) # [375, 1241, 2], [375, 1241]
            flow_size = flow_.shape[:2]
            cropper = StaticCenterCrop(flow_size, self.render_size)
            flow_ = cropper(flow_) # [320, 1216, 2]
            flow_ = np.array(flow_).astype(np.float32)
            valid_ = cropper(valid_)
            self.preloaded_flow[flow] = flow_
            self.preloaded_valid[flow] = valid_
    
    def __getitem__(self, index):
        # index = index % len(self.flowfeat_list)
        flowfeat = self.preloaded_flowfeat[self.flowfeat_list[index]] # [1024, 5, 19]

        flow = self.preloaded_flow[self.flow_list[index]] # [320, 1216, 2]
        valid = self.preloaded_valid[self.flow_list[index]] # [320, 1216]

        flow = torch.from_numpy(flow).permute(2, 0, 1).float() # [2, 320, 1216]
        valid = torch.from_numpy(valid).float() # [320, 1216]

        return flowfeat, flow, valid


    def __len__(self):
        return len(self.flowfeat_list)


class KITTIFlowImgDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root = "data/kitti/optical_flow"
        self.render_size = [320, 1216] # After flownet2S out_conv6_1 -> [5, 19]

        self.all_training_list = []
        for year in ["2012", "2015"]:
            imgs_0 = sorted(glob(os.path.join(self.root, year, "training/image_2/*_10.png")))
            imgs_1 = sorted(glob(os.path.join(self.root, year, "training/image_2/*_11.png")))
            for im0, im1 in zip(imgs_0, imgs_1):
                tag0 = im0.split("/")[-1].split(".")[0]
                tag1 = im1.split("/")[-1].split(".")[0]
                flowfeat = os.path.join(self.root, year, "training/flowfeat_image_2/{}+{}.pt".format(tag0, tag1))
                flow = os.path.join(self.root, year, "training/flow_occ/{}.png".format(tag0))
                self.all_training_list.append([im0, im1, flowfeat, flow])
    
    def split(self):
        all_data = copy.deepcopy(self.all_training_list)
        val_percent = 0.05
        random.shuffle(all_data)
        val_num = int(len(all_data) * val_percent)
        val_data = all_data[:val_num]
        train_data = all_data[val_num:]
        print("============================")
        print("=> Spliting in total {} image pairs".format(len(all_data)))
        print("=> Train: {} image pairs v.s. Val: {} image pairs".format(len(train_data), len(val_data)))

        with open(os.path.join(self.root, "train.txt"), "w") as f:
            for datum in train_data:
                f.write("{}\n".format(" ".join(datum)))
        
        with open(os.path.join(self.root, "val.txt"), "w") as f:
            for datum in val_data:
                f.write("{}\n".format(" ".join(datum)))
    
    def prepare_flowfeat(self):
        from flownet_model import FlowNet2S
        # NOTE: FlowNet2S will perform runtime normalization
        rgb_max = 255.
        flow_model = FlowNet2S(rgb_max).to(device='cuda:0')
        resume_ckp = torch.load("data/flownet_models/FlowNet2S_checkpoint.pth.tar")
        flow_model.load_state_dict(resume_ckp['state_dict'])
        flow_model.eval()
        from tqdm import tqdm
        for datum in tqdm(self.all_training_list):
            img0_path, img1_path, flowfeat_path = datum[0], datum[1], datum[2]

            # torch tensor [3, 2, 320, 1216]
            x_img_pair = self.read_images(img0_path, img1_path).type(torch.FloatTensor).to(device="cuda:0")
            x_img_pair = x_img_pair.unsqueeze(0) # [1, 3, 2, 320, 1216]

            with torch.no_grad():
                obs = flow_model(x_img_pair) # [1, out_conv6_1] e.g. [1, 1024, 5, 19]
            
            torch.save(obs[0].data.clone().cpu(), flowfeat_path)
    
    def read_images(self, img0_path, img1_path):
        img0 = read_gen(img0_path)
        img1 = read_gen(img1_path)
        assert img0.shape == img1.shape
        img_size = img0.shape[:2]
        cropper = StaticCenterCrop(img_size, self.render_size)
        imgs = [img0, img1]
        imgs = list(map(cropper, imgs))
        imgs = np.array(imgs).transpose(3,0,1,2)
        imgs = torch.from_numpy(imgs) # [3, 2, 320, 1216]
        return imgs
    
    def __getitem__(self, index):
        img0_path = self.all_training_list[index][0]
        img1_path = self.all_training_list[index][1]
        return self.read_images(img0_path, img1_path)
    
    def __len__(self):
        return len(self.all_training_list)



if __name__ == "__main__":
    tmp = KITTIFlowDataset("train")
    tmp.test()
    # tmp.prepare_flowfeat()

    
    







