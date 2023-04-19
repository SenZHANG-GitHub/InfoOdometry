import os
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
from dataset.vkitti2_dataset import VKitti2Dataset
from flownet_model import FlowNet2S
from utils.file_io import read_kitti_img
import PIL
from PIL import Image
import numpy as np
from flownet_utils.frame_utils import StaticCenterCrop

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="vkitti2", choices=["vkitti2", "kitti", "euroc"], type=str, required=True)
parser.add_argument("--ftype", choices=["downsample", "raw_freq"], type=str, help="only used for --dataset euroc")


def save_vkitti2_flownet_features():
    total_sequences = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
    subscenes = ["clone", "fog", "overcast", "morning", "rain", "sunset", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]

    rgb_max = 255.
    flow_model = FlowNet2S(rgb_max).to(device='cuda:0')
    resume_ckp = torch.load("data/flownet_models/FlowNet2S_checkpoint.pth.tar")
    flow_model.load_state_dict(resume_ckp['state_dict'])
    flow_model.eval()

    for seq in total_sequences:
        for subscene in subscenes:
            print("==============================")
            print("=> processing seq {}-{}".format(seq, subscene))
            path_seq = "data/virtual-kitti/vKitti2/prepare/{}/{}/flownet_features".format(seq, subscene)
            if os.path.isdir(path_seq):
                shutil.rmtree(path_seq)
            os.mkdir(path_seq)

            vkitti2_data = VKitti2Dataset(seq=seq, subscene=subscene)

            print('total image pair: {}'.format(len(vkitti2_data)))
            for _idx, datum in tqdm(enumerate(vkitti2_data)):
                img_features, last_timestamp, curr_timestamp = datum

                outname = '{}/{}-{}.pt'.format(path_seq, last_timestamp, curr_timestamp)
                if os.path.isfile(outname):
                    raise ValueError('filename {} already exists'.format(outname))

                img_features = img_features.unsqueeze(0) # [1, 3, 2, 320, 1216] and range (0, 255.)
                img_features = img_features.type(torch.FloatTensor).to(device='cuda:0')

                ## flow_model.forward(x) will do running average to transfer (0, 255.) to (0, 1.)
                flownet_features = flow_model(img_features) 
                flownet_features = flownet_features.squeeze(0)
                flownet_features = flownet_features.data.clone().cpu()
                torch.save(flownet_features, outname)
    

def save_euroc_flownet_features(opt):
    total_sequences = [
        'MH_01_easy', 
        'MH_02_easy', 
        'MH_03_medium', 
        'MH_04_difficult',
        'MH_05_difficult', 
        'V1_01_easy', 
        'V1_02_medium', 
        'V1_03_difficult', 
        'V2_01_easy', 
        'V2_02_medium', 
        'V2_03_difficult'
    ]

    rgb_max = 255.
    flow_model = FlowNet2S(rgb_max).to(device='cuda:0')
    resume_ckp = torch.load("data/flownet_models/FlowNet2S_checkpoint.pth.tar")
    flow_model.load_state_dict(resume_ckp['state_dict'])
    flow_model.eval()

    for seq in total_sequences:
        print("==============================")
        print("=> processing seq: {}".format(seq))
        path_seq = "data/euroc/{}/{}/flownet_features".format(seq, opt.ftype)
        if os.path.isdir(path_seq):
            shutil.rmtree(path_seq)
        os.mkdir(path_seq)

        # arg:debug in EuRocDataset determines whether self.data is enabled for processing
        if opt.ftype == "downsample":
            downsample = True 
        if opt.ftype == "raw_freq":
            downsample = False
        euroc_data = EuRocDataset(seq=seq, base_dir="data/euroc", downsample=downsample)

        print('total image pair: {}'.format(len(euroc_data)))
        for _idx, datum in tqdm(enumerate(euroc_data)):
            img_features, last_timestamp, curr_timestamp = datum

            outname = '{}/{}-{}.pt'.format(path_seq, last_timestamp, curr_timestamp)
            if os.path.isfile(outname):
                raise ValueError('filename {} already exists'.format(outname))

            img_features = img_features.unsqueeze(0) # [1, 3, 2, 480, 752] and range (0, 255.)
            img_features = img_features.type(torch.FloatTensor).to(device='cuda:0')

            ## flow_model.forward(x) will do running average to transfer (0, 255.) to (0, 1.)
            flownet_features = flow_model(img_features) # [1, 1024, 8, 12]
            flownet_features = flownet_features.squeeze(0)
            flownet_features = flownet_features.data.clone().cpu()
            torch.save(flownet_features, outname)


def save_kitti_flownet_features():
    # total_sequences = ['00', '01', '02', '04', '05', '06', '07', '08', '09', '10']
    total_sequences = ["03"]
    
    rgb_max = 255.
    flow_model = FlowNet2S(rgb_max).to(device='cuda:0')
    resume_ckp = torch.load("data/flownet_models/FlowNet2S_checkpoint.pth.tar")
    flow_model.load_state_dict(resume_ckp['state_dict'])
    flow_model.eval()
    
    for seq in total_sequences:
        print('----------------------------')
        print('processing seq: {}'.format(seq))
        assert seq == "03"
        path_seq = 'data/kitti/old_flownet_features/clip_length_5/{}/'.format(seq)
        base_dir = "data/kitti/odometry/dataset"
        render_size = [320, 1216]
        imgs = read_kitti_img(base_dir, seq)

        for _idx, _ in tqdm(enumerate(imgs)):
            if _idx == 0: continue 
            # outname = '{}/{}-{}.pt'.format(path_seq, last_ts, curr_ts)
            outname = "{}/{}_{}.pt".format(path_seq, imgs[_idx-1][0], imgs[_idx][0])
            if os.path.isfile(outname):
                # raise ValueError('filename {} already exists'.format(outname))
                continue
                
            img_0 = "{}/sequences/{}/image_2/{}.jpg".format(base_dir, seq, imgs[_idx-1][0].split("-")[1])
            img_1 = "{}/sequences/{}/image_2/{}.jpg".format(base_dir, seq, imgs[_idx][0].split("-")[1])
            
            # (1242, 375)
            last_img = np.array(PIL.Image.open(img_0).convert("RGB")) # [375, 1242, 3]
            curr_img = np.array(PIL.Image.open(img_1).convert("RGB")) # [375, 1242, 3]
            img_size = curr_img.shape[:2]
            cropper = StaticCenterCrop(img_size, render_size)
            last_img = cropper(last_img) # [320, 1216, 3]
            curr_img = cropper(curr_img) # [320, 1216, 3]
            
            last_img = torch.from_numpy(last_img).permute(2,0,1)  # [3, 320, 1216]
            curr_img = torch.from_numpy(curr_img).permute(2,0,1)  # [3, 320, 1216]

            img_features = torch.stack((last_img, curr_img), dim=1).type(torch.FloatTensor) # [3, 2, 320, 1216], max=255.

            img_features = img_features.unsqueeze(0) # [1, 3, 2, 320, 1216] and range (0, 255.)
            img_features = img_features.type(torch.FloatTensor).to(device='cuda:0')

            ## flow_model.forward(x) will do running average to transfer (0, 255.) to (0, 1.)
            flownet_features = flow_model(img_features) # [1, 1024, 5, 19]
            flownet_features = flownet_features.squeeze(0)
            flownet_features = flownet_features.data.clone().cpu()
            torch.save(flownet_features, outname)
            

if __name__ == "__main__":
    opt = parser.parse_args()
    if opt.dataset == "euroc":
        save_euroc_flownet_features(opt)
    elif opt.dataset == "vkitti2":
        save_vkitti2_flownet_features()
    elif opt.dataset == "kitti":
        save_kitti_flownet_features()
    else:
        raise NotImplementedError()
