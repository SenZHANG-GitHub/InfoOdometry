import os
import csv
import pdb
import torch
import random
import numpy as np
import sophus as sp
import PIL
from PIL import Image
from tqdm import tqdm
import flownet_utils.frame_utils as frame_utils
from flownet_utils.frame_utils import StaticCenterCrop


def read_kitti_pose(base_dir, sequence):
    """
    -> base_dir: data/kitti/odometry/dataset
    -> read the ground-truth pose for each image
    -> list of poses (np.array: (12,))
    """
    poses = []
    with open('{}/poses/{}.txt'.format(base_dir, sequence), mode='r') as f:
        for _i, _line in enumerate(f.readlines()):
            pose_label = '{}-{:06d}'.format(sequence, _i)
            if _line[-1] == '\n':
                _line = _line[:-1]
            poses.append([pose_label, np.array([float(x) for x in _line.split(' ')])])
    return poses


def get_kitti_imgpair(last_img, curr_img, base_dir, img_transforms=None, render_size=None):
    """
    (1) last_img: '00-000000'
    (2) curr_img: '00-000001'
    """
    if img_transforms is None and render_size is None: 
        raise ValueError('one and only on of img_transforms and render_size should be given')
    if img_transforms is not None and render_size is not None: 
        raise ValueError('one and only on of img_transforms and render_size should be given')

    last_seq = last_img.split('-')[0]
    last_idx = last_img.split('-')[1]
    curr_seq = curr_img.split('-')[0]
    curr_idx = curr_img.split('-')[1]
    assert last_seq == curr_seq
    assert int(last_idx) + 1 == int(curr_idx)
    last_img_path = '{}/sequences/{}/image_2/{}.jpg'.format(base_dir, last_seq, last_idx)
    curr_img_path = '{}/sequences/{}/image_2/{}.jpg'.format(base_dir, curr_seq, curr_idx)
    
    if img_transforms is None:
        # use FlowNet2/C/S
        assert render_size is not None
        img1 = frame_utils.read_gen(last_img_path)
        img2 = frame_utils.read_gen(curr_img_path)
        images = [img1, img2]
        image_size = img1.shape[:2]
        cropper = StaticCenterCrop(image_size, render_size)
        images = list(map(cropper, images))
        images = np.array(images).transpose(3,0,1,2)

        # tmp_time = timer()
        # images = images.astype(np.float32)
        # tout = timer() - tmp_time
        # if tout > 1: print('np_float32: {} s'.format(tout))

        images = torch.from_numpy(images)
        return images
    else:
        # if train_img_from_scrach: ToTensor will transform (H,W,C) PIL image in [0,255] to (C,H,W) in [0.0,1.0]
        r_last_img = img_transforms(PIL.Image.open(last_img_path)) # [3, 192, 640] for kitti
        r_curr_img = img_transforms(PIL.Image.open(curr_img_path)) # [3, 192, 640] for kitti
        return torch.stack((r_last_img, r_curr_img), dim=1).type(torch.FloatTensor)


def get_kitti_depthpair(last_img, curr_img, base_dir, render_size=None):
    """
    (1) last_img: '00-000000'
    (2) curr_img: '00-000001'
    """
    last_seq = last_img.split('-')[0]
    last_idx = int(last_img.split('-')[1])
    curr_seq = curr_img.split('-')[0]
    curr_idx = int(curr_img.split('-')[1])
    assert last_seq == curr_seq
    assert int(last_idx) + 1 == int(curr_idx)
    last_depth_path = '{}/depths/{}/{:010d}.png'.format(base_dir, last_seq, last_idx)
    curr_depth_path = '{}/depths/{}/{:010d}.png'.format(base_dir, curr_seq, curr_idx)
    
    assert render_size is not None
    depth1 = Image.open(last_depth_path)
    depth2 = Image.open(curr_depth_path)
    depth1 = np.array(depth1, dtype=np.float32) / 256.0
    depth2 = np.array(depth2, dtype=np.float32) / 256.0
    depths = [depth1, depth2]
    depth_size = depth1.shape[:2]
    cropper = StaticCenterCrop(depth_size, render_size)
    depths = list(map(cropper, depths))

    # [2, 320, 1216] (No need to transpose)
    depths = np.array(depths)
    depths = torch.from_numpy(depths)
    return depths
    

def read_kitti_img(base_dir, sequence):
    """
    -> base_dir: data/kitti/odometry/dataset
    -> Return: list of image filenames
    """
    imgs = []
    path_img = '{}/sequences/{}/image_2/'.format(base_dir, sequence)
    num_img = len([x for x in os.listdir(path_img) if x[-4:] == '.jpg'])
    print('reading {} images of sequence {}'.format(num_img, sequence))
    for _i in tqdm(range(num_img)):
        img_label = '{}-{:06d}'.format(sequence, _i)
        imgs.append([img_label])
    return imgs


def read_kitti_imu(base_dir, sequence):
    """
    -> base_dir: data/kitti/odometry/dataset
    -> Return: list of [imu_label (00-000000), np.array(11, 6)]
        -> the inner list is the imu data between two images (len: 11,12,14 -> trim to 11)
    -> format of raw imu data ([0] ~ [29])
        * [11] ax: acceleration in x (m/s^2)
        * [12] ay: acceleration in y (m/s^2)
        * [13] az: acceleration in z (m/s^2)
        * [17] wx: angular rate around x (rad/s)
        * [18] wy: angular rate around y (rad/s)
        * [19] wz: angular rate around z (rad/s)
    """
    imus = []
    # wx, wy, wz, ax, ay, az
    imu_ind = [17, 18, 19, 11, 12, 13]
    path_imu = '{}/sequences/{}/oxts_100hz/oxts_data.txt'.format(base_dir, sequence)
    with open(path_imu, mode='r') as f:
        for _cnt, _line in enumerate(f.readlines()):
            imu_label = '{}-{:06d}'.format(sequence, _cnt)
            if _line[-1] == '\n':
                _line = _line[:-1]
            if _line == 'nan':
                imus.append([imu_label, _line])
            else:
                _line = _line.split('|')
                # each comp in _line -> a list with length-30
                _line = [[float(x) for x in _imu.split(' ') if x != ''] for _imu in _line]
                # each comp in _line -> a list with length-6
                _line_imu = [[_imu[_ind] for _ind in imu_ind] for _imu in _line]
                # trim to 11 imu records between two image frames
                _line_imu = np.array(_line_imu)[:11,:] # [11, 6] 
                _line = np.array(_line)[:11,:] # [11, 30]
                imus.append([imu_label, _line_imu, _line]) 
    return imus
