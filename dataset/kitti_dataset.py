import PIL
from PIL import Image
import numpy as np

import torch
import pdb
import os
import shutil
from tqdm import tqdm
import shutil
from shutil import rmtree
import random

from torch._six import int_classes as _int_classes
from torchvision import transforms

from utils.file_io import read_kitti_pose
from utils.file_io import read_kitti_img
from utils.file_io import read_kitti_imu
from utils.file_io import get_kitti_imgpair
from utils.file_io import get_kitti_depthpair

from utils.tools import get_relative_pose 
from utils.tools import get_zero_se3
from utils.tools import euler_to_quaternion
from utils.tools import rotationMatrixToEulerAngles
import flownet_utils.frame_utils as frame_utils
from flownet_utils.frame_utils import StaticCenterCrop


def get_img_transforms(args, img_mean=None, img_std=None):
    """
    -> use args.resize_mode and args.new_img_size
    -> if not --train_img_from_scratch: Do normalization and 255->1 in FlowNetS
    -> use the img_mean and img_std of each sequence
    """
    transform_ops = []
    if args.resize_mode == 'crop':
        transform_ops.append(transforms.CenterCrop((args.new_img_size[0], args.new_img_size[1])))
    elif args.resize_mode == 'rescale':
        transform_ops.append(transforms.Resize((args.new_img_size[0], args.new_img_size[1])))
    transform_ops.append(transforms.ToTensor())
    if img_mean is not None and img_std is not None:
        transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
    return transforms.Compose(transform_ops)
                

class KittiClipDataset(torch.utils.data.Dataset):
    def __init__(self, args, load_kitti_rgb=False, load_kitti_depth=False, clip_length=None, overlap=None, on_the_fly=None, img_prefeat=None, base_dir=None, train_img_from_scratch=None):
        """
        -> cut the consecutive frames into clips with clip_length=5 by default
        -> save each clip as a datapoint
        """
        base_dir = base_dir if base_dir is not None else args.base_dir
        if base_dir[-1] == '/': base_dir = base_dir[:-1]
        self.base_dir = base_dir # 'data/kitti/odometry/dataset
        
        self.overlap = overlap if overlap is not None else args.clip_overlap
        self.on_the_fly = on_the_fly if on_the_fly is not None else args.on_the_fly
        self.clip_length = clip_length if clip_length is not None else args.clip_length
        self.img_prefeat = img_prefeat if img_prefeat is not None else args.img_prefeat
        self.use_img_prefeat = False if self.img_prefeat == 'none' else True
        self.train_img_from_scratch = train_img_from_scratch if train_img_from_scratch is not None else args.train_img_from_scratch
        self.load_kitti_rgb = load_kitti_rgb
        self.load_kitti_depth = load_kitti_depth
        
        if self.train_img_from_scratch:
            raise NotImplementedError()
            self.img_transforms = get_img_transforms(args)
        else:
            if 'img' in args.sensors and args.flownet_model == 'none':
                raise ValueError('--flownet_model must be given if not using --train_img_from_scratch')
            self.render_size = [320, 1216] # (kitti_size // 64) * 64
            
        self.args = args
        assert self.args.clip_length
        self.clips = [] 
        self.loaded_imgs = dict() # used when not using --on_the_fly and --img_prefeat none or self.load_kitti_rgb
        self.loaded_depths = dict()
        self.loaded_prefeats = dict() # used when not using --on_the_fly and --img_prefeat flownet/resnet
        
    
    def trim_samples(self, sample_size_ratio):
        # randomly reduce the number of clips to sample_size_ratio 
        self.clips = random.sample(self.clips, int(sample_size_ratio * len(self.clips)))

    
    def get_clip_pose_scale(self):
        """
        calculate averaged clip pose scales from self.clips
        """
        clip_poses = []
        for clip in self.clips:
            frame_all_poses = []
            for frame in clip:
                rel_pose = frame['curr_relative_pose'][1]
                # all_norm = np.linalg.norm(rel_pose)
                # trans_norm = np.linalg.norm(rel_pose[:3])
                # rot_norm = np.linalg.norm(rel_pose[3:])
                frame_all_poses.append(rel_pose)
            clip_poses.append(frame_all_poses)
        return clip_poses
                

    def extend(self, kitti_data):
        """
        -> cut the consecutive sub_sequences into clips with clip_length=5 by default
        -> save each clip as a datapoint
        -> kitti_data.sub_seqs: A dict with keys {'imus', 'imgs', 'rel_poses', 'global_poses'}
            -> Correspondence: imus[i] ~ (imgs[i], imgs[i+1])
                * last_global_pose: global_poses[i], curr_global_pose: global_poses[i+1]
                * last_rel_pose: rel_pose[i], curr_rel_pose: rel_poses[i+1] 
        """
        print('getting clips of sequence {} ...'.format(kitti_data.sequence))
        # self.img_transforms = kitti_data.img_transforms
        clips = []
        for i_seq, sub_seq in enumerate(kitti_data.sub_seqs):
            if len(sub_seq['imus']) < self.clip_length:
                # discard sub_sequences that are shorter than clip_length
                continue
            if (not self.on_the_fly and self.load_kitti_rgb) or (not self.on_the_fly and not self.use_img_prefeat):
                self.update_loaded_imgs(sub_seq)
            if not self.on_the_fly and self.load_kitti_depth:
                self.update_loaded_depths(sub_seq) 
            if self.overlap:
                for j_seq in range(len(sub_seq['imus'])-1-self.clip_length):
                    clips.append(self.get_clip(sub_seq, j_seq))
            else:
                for j_seq in range(0, len(sub_seq['imus'])-1, self.clip_length):
                    if j_seq + self.clip_length > len(sub_seq['imus']) - 1:
                        j_seq = len(sub_seq['imus']) - 1 - self.clip_length
                    clips.append(self.get_clip(sub_seq, j_seq))
        if not self.on_the_fly and self.use_img_prefeat:
            self.update_loaded_prefeats(clips)
        self.clips.extend(clips)
    
    
    def update_loaded_prefeats(self, clips):
        """
        load the images in sub_seq into self.loaded_prefeats
        """
        for _clip in clips:
            for _frame in _clip:
                # 1 channel to 3 channels -> RGB or an init_conv layer from 1 -> 3
                r_last_timestamp, r_curr_timestamp = _frame['img_pair']
                assert r_last_timestamp.split('-')[0] == r_curr_timestamp.split('-')[0]
                fseq = r_last_timestamp.split('-')[0]
                flabel = '{}_{}'.format(r_last_timestamp, r_curr_timestamp)
                fpath = 'data/kitti/old_flownet_features/clip_length_{}/{}/{}.pt'.format(self.args.clip_length, fseq, flabel)
                self.loaded_prefeats[flabel] = torch.load(fpath) # [1024, 5, 19]
                                
                
    def update_loaded_imgs(self, sub_seq):
        """
        load the images in sub_seq into self.loaded_imgs
        """
        for _img in sub_seq['imgs']:
            if _img[0] not in self.loaded_imgs.keys(): # '00-000000'
                tmp_seq = _img[0].split('-')[0] # '00'
                tmp_idx = _img[0].split('-')[1] # '000000'
                img_path = '{}/sequences/{}/image_2/{}.jpg'.format(self.base_dir, tmp_seq, tmp_idx)
                if self.train_img_from_scratch:
                    # if train_img_from_scratch: ToTensor will transform (H,W,C) PIL image in [0,255] to (C,H,W) in [0.0,1.0] # [3, 192, 640] for kitti
                    raise NotImplementedError()
                    img_data = self.img_transforms(PIL.Image.open(img_path))
                else: 
                    # use FlowNet2/C/S pretrained models
                    tmp_img = frame_utils.read_gen(img_path)
                    image_size = tmp_img.shape[:2]
                    cropper = StaticCenterCrop(image_size, self.render_size)
                    img_data = cropper(tmp_img)
                self.loaded_imgs[_img[0]] = img_data
    

    def update_loaded_depths(self, sub_seq):
        """
        load the depthsdepth_size in sub_seq into self.loaded_depths
        """
        for _img in sub_seq['imgs']:
            if _img[0] not in self.loaded_depths.keys(): # '00-000000'
                tmp_seq = _img[0].split('-')[0] # '00'
                tmp_idx = int(_img[0].split('-')[1]) # 0
                depth_path = '{}/depths/{}/{:010d}.png'.format(self.base_dir, tmp_seq, tmp_idx)
                # Read the depth png
                tmp_depth = Image.open(depth_path)
                tmp_depth = np.array(tmp_depth, dtype=np.float32) / 256.0
                depth_size = tmp_depth.shape[:2]
                cropper = StaticCenterCrop(depth_size, self.render_size)

                # [370, 1226] -> [320, 1216]
                depth_data = cropper(tmp_depth)
                self.loaded_depths[_img[0]] = depth_data

    
    def get_clip(self, sub_seq, j_seq):
        """
        get a clip data from sub_seq starting from j_seq (imus)
        -> each sub_seq is a dict: {
                'imus':         list of [imu_label,  np(11, 6), np(11, 30)], # len: k
                'imgs':         list of [img_label],                         # len: k + 1
                'rel_poses':    list of [pose_label, np(6,)],                # len: k + 1
                'global_poses': list of [pose_label, np(7,)],                # len: k + 1
            }
        -> for the first transition in the clip (e.g.)
            * imu:              sub_seq['imus'][j_seq]
            * img_pair:         [sub_seq['imgs'][j_seq], sub_seq['imgs'][j_seq + 1]]
            * last_rel_pose:    sub_seq['rel_poses'][j_seq]
            * curr_rel_pose:    sub_seq['rel_poses'][j_seq + 1]
            * last_global_pose: sub_seq['global_poses'][j_seq]
            * curr_global_pose: sub_seq['global_poses'][j_seq + 1]
        """
        tmpclip = []
        for k_seq in range(self.clip_length):
            last_img = sub_seq['imgs'][j_seq+k_seq][0]     # '00-000000'
            curr_img = sub_seq['imgs'][j_seq+1+k_seq][0]   # '00-000001'
            assert int(last_img.split('-')[0]) == int(curr_img.split('-')[0]) 
            assert int(last_img.split('-')[1]) == int(curr_img.split('-')[1]) - 1
            img_pair = [last_img, curr_img]   # [time_0, time_1]  

            tmpclip.append({
                'img_pair':           img_pair,                               # [time_0, time_1]
                'imus':               sub_seq['imus'][j_seq+k_seq],           # [time_0, np(11, 6), np(11, 30)]
                'last_relative_pose': sub_seq['rel_poses'][j_seq+k_seq],      # [time_0, np(6,)]
                'curr_relative_pose': sub_seq['rel_poses'][j_seq+1+k_seq],    # [time_1, np(6,)]
                'last_global_pose':   sub_seq['global_poses'][j_seq+k_seq],   # [time_0, np(7,)]
                'curr_global_pose':   sub_seq['global_poses'][j_seq+1+k_seq], # [time_1, np(7,)]
            })
        return tmpclip

    
    def _get_img_prefeat(self, last_ts, curr_ts):
        """
        get the pretrained flownet features
        """
        assert last_ts.split('-')[0] == curr_ts.split('-')[0]
        fseq = last_ts.split('-')[0]
        flabel = '{}_{}'.format(last_ts, curr_ts)
        # [1024, 5, 19]
        if self.on_the_fly:
            return torch.load('data/kitti/old_flownet_features/clip_length_{}/{}/{}.pt'.format(self.args.clip_length, fseq, flabel))
        else:
            return self.loaded_prefeats[flabel] 
    
    
    def _get_img_pair(self, last_ts, curr_ts):
        """
        get image pair data if not self.use_img_prefeat
        """
        if self.on_the_fly:
            if self.train_img_from_scratch: # [3, 2, 376, 1241] 
                raise NotImplementedError()
                r_img_features = get_kitti_imgpair(last_ts, curr_ts, self.base_dir, img_transforms=self.img_transforms)
            else: # use FlowNet2/C/S pretrained models
                r_img_features = get_kitti_imgpair(last_ts, curr_ts, self.base_dir, render_size=self.render_size)
        else: # use image data pre-loaded in self.loaded_imgs
            if self.train_img_from_scratch:
                # if train_img_from_scratch: ToTensor will transform (H,W,C) PIL image in [0,255] to (C,H,W) in [0.0,1.0] # [3, 192, 640] for kitti
                raise NotImplementedError()
                r_last_img = self.loaded_imgs[last_ts]
                r_curr_img = self.loaded_imgs[curr_ts]
                r_img_features = torch.stack((r_last_img, r_curr_img), dim=1).type(torch.FloatTensor)
            else: 
                # use FlowNet2/C/S pretrained models
                r_last_img = self.loaded_imgs[last_ts]
                r_curr_img = self.loaded_imgs[curr_ts]
                r_img_features = [r_last_img, r_curr_img]
                r_img_features = np.array(r_img_features).transpose(3,0,1,2)
                r_img_features = torch.from_numpy(r_img_features) # [3, 2, 320, 1216]
        return r_img_features


    def _get_depth_pair(self, last_ts, curr_ts):
        """
        get image pair data if not self.use_img_prefeat
        """
        if self.on_the_fly:
            r_depth_features = get_kitti_depthpair(last_ts, curr_ts, self.base_dir, render_size=self.render_size)
        else: # use depth data pre-loaded in self.loaded_depths
            
            r_last_depth = self.loaded_depths[last_ts]
            r_curr_depth = self.loaded_depths[curr_ts]
            r_depth_features = [r_last_depth, r_curr_depth]

            # NOTE: [2, 320, 1216] (No need to transpose)
            r_depth_features = np.array(r_depth_features)
            r_depth_features = torch.from_numpy(r_depth_features) # [2, 320, 1216]
        return r_depth_features
        

    def __getitem__(self, idx):
        """
        returns a list of 
        (1) image pair (last and present)
        (2) imu data in between
        (3) last_relative_pose (from last_last to last) -> idx 
        (4) curr_relative_pose (from last to present)   -> idx + 1
        (5) last_global_pose w.r.t world frame -> idx
        (6) curr_global_pose w.r.t world frame -> idx + 1
        (7) timestamp -> idx + 1
        """
        img_features = []
        imu_features = []
        last_relative_pose = []
        curr_relative_pose = []
        last_global_pose = []
        curr_global_pose = []
        last_timestamp = []
        curr_timestamp = []
        rgb_features = []
        depth_features = []
        # imu_labels = []
        # imu_features_all = []
        # getitem_time = timer()
        for _frame in self.clips[idx]:
            # 1 channel to 3 channels -> RGB or an init_conv layer from 1 -> 3
            r_last_timestamp, r_curr_timestamp = _frame['img_pair']
            
            if self.use_img_prefeat:
                r_img_features = self._get_img_prefeat(r_last_timestamp, r_curr_timestamp)
            else:
                r_img_features = self._get_img_pair(r_last_timestamp, r_curr_timestamp)
            img_features.append(r_img_features) 

            if self.load_kitti_rgb:
                r_rgb_features = self._get_img_pair(r_last_timestamp, r_curr_timestamp)
                rgb_features.append(r_rgb_features)
            
            if self.load_kitti_depth:
                r_depth_features = self._get_depth_pair(r_last_timestamp, r_curr_timestamp)
                depth_features.append(r_depth_features)

            r_imu_features       = _frame['imus'][1]              # np.array(11, 6)
            r_last_relative_pose = _frame['last_relative_pose'][1]  # np.array(6, )
            r_curr_relative_pose = _frame['curr_relative_pose'][1]  # np.array(6, )
            r_last_global_pose   = _frame['last_global_pose'][1]    # np.array(7, )
            r_curr_global_pose   = _frame['curr_global_pose'][1]    # np.array(7, )
            # r_imu_labels         = _frame['imus'][0]                # 00-000000
            # r_imu_features_all   = _frame['imus'][2]             # np.array(11, 30)
            
            imu_features.append(torch.from_numpy(r_imu_features))             # [11, 6]
            last_relative_pose.append(torch.from_numpy(r_last_relative_pose)) # [6] se3 or t_euler
            curr_relative_pose.append(torch.from_numpy(r_curr_relative_pose)) # [6] se3 or t_euler
            last_global_pose.append(torch.from_numpy(r_last_global_pose))     # [7] t_and_q
            curr_global_pose.append(torch.from_numpy(r_curr_global_pose))     # [7] t_and_q
            last_timestamp.append(r_last_timestamp)
            curr_timestamp.append(r_curr_timestamp)
            # imu_labels.append(r_imu_labels)
            # imu_features_all.append(r_imu_features_all)                       # [11, 30]

        # img_features:         length-5 list with component [3, 2, 376, 1241] if not use_img_prefeat
        #                       length-5 list with component [1024, 6, 20] if use_img_prefeat
        # imu_features:         length-5 list with component [11, 6]
        # last_relative_pose:   length-5 list with component [6]
        # curr_relative_pose:   length-5 list with component [6]
        # last_global_pose:     length-5 list with component [7]
        # curr_global_pose:     length-5 list with component [7]
        # last_timestamp:       length-5 list with component [1]
        # curr_timestamp:       length-5 list with component [1]
        # imu_labels:           length-5 list with component [1]
        # imu_features_all:     length-5 list with component [11, 30]
        
        # print('getitem time: {}'.format(timer() - getitem_time))

        if self.load_kitti_rgb:
            return img_features, imu_features, last_relative_pose, curr_relative_pose, last_global_pose, curr_global_pose, last_timestamp, curr_timestamp, rgb_features
        elif self.load_kitti_depth:
            return img_features, imu_features, last_relative_pose, curr_relative_pose, last_global_pose, curr_global_pose, last_timestamp, curr_timestamp, depth_features
        else:
            return img_features, imu_features, last_relative_pose, curr_relative_pose, last_global_pose, curr_global_pose, last_timestamp, curr_timestamp


    def __len__(self):
        return len(self.clips)


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, sequence, args, base_dir=None, t_euler_loss=None, debug=None, train_img_from_scratch=None):
        """ 
        base_dir: data/kitti/odometry/dataset
        sequence: e.g. 00

        Data structure, e.g. (seq 00)
        (1) data/kitti/odometry/dataset/poses/00.txt 
            -> should be absolute poses and requires extra calculation for relative poses between two images
        (2) data/kitti/odometry/sequences/00/image_2/000000.jpg (left-camera; image_3/: right-camera) 
            -> original size: 
        (3) data/kitti/odometry/sequences/00/oxts_100hz/oxts_data.txt (right-camera)
            -> the number of oxts = the number of images - 1
            -> because oxts are the imu sequence between a image-pair (two-images)
            -> imu data for image(0:1), image(1:2), ...., image(last-1:last)
            -> exist 'nan' items that should not be used (since the raw imu data are not strictly continuous w.r.t. 100 hz)
            -> length of imu data between two images: 11, 12 or 14 (only one 14 datum) -> should trim to 11 when used 
        """
        # raise NotImplementedError('check the rgb_max of kitti images')
        base_dir = base_dir if base_dir is not None else args.base_dir
        if base_dir[-1] == '/':
            base_dir = base_dir[:-1]
        self.base_dir               = base_dir
        self.sequence               = sequence
        self.train_img_from_scratch = train_img_from_scratch if train_img_from_scratch is not None else args.train_img_from_scratch
        self.debug                  = debug if debug is not None else args.debug
        self.t_euler_loss           = t_euler_loss if t_euler_loss is not None else args.t_euler_loss

        # if not --train_img_from_scratch -> use FlowNetS which will do image normalization itself
        args_read_data = {
            'base_dir'       : self.base_dir, 
            'seq'            : self.sequence
        }
        self.sub_seqs = self.read_data(**args_read_data)
        # self.imgs = read_kitti_img(self.base_dir, self.sequence)
    

    def get_pose_scale(self, return_norm=False, return_pose=False):
        """
        get the relative pose scales of current sequence
        (1) overall as se3 (2) translation as se3 (3) rotation as se3
        """
        se3_norm = []
        trans_norm = []
        rot_norm = []
        rel_poses = []
        for sub_seq in self.sub_seqs:
            for _rel_pose in sub_seq['rel_poses'][1:]:
                if return_pose:
                    rel_poses.append(_rel_pose[1])
                else:
                    se3_norm.append(np.linalg.norm(_rel_pose[1]))
                    trans_norm.append(np.linalg.norm(_rel_pose[1][:3]))
                    rot_norm.append(np.linalg.norm(_rel_pose[1][3:]))
        if return_pose:
            return rel_poses
        else:
            print('************************************')
            print('the mean se3 norm of {}: {}'.format(self.sequence, np.mean(se3_norm)))
            print('the mean translation norm of {}: {}'.format(self.sequence, np.mean(trans_norm)))
            print('the mean rotation norm of {}: {}'.format(self.sequence, np.mean(rot_norm)))
            print('************************************')
            if return_norm:
                return se3_norm, trans_norm, rot_norm


    def read_data(self, base_dir, seq):
        """
        Return: A list of sub_sequences that are continous without nan 
        -> each sub_sequence is a dict: {
            'imus':         list of [imu_label,  np(11, 6)],         # len: k
            'imgs':         list of [img_label],                     # len: k + 1
            'rel_poses':    list of [pose_label, np(6,)],            # len: k + 1
            'global_poses': list of [pose_label, np(7,)],            # len: k + 1
        }
        -> transform the rel/global poses to the same format as euroc dataset
            * rel_poses: se3.log() with length 6
            * global_poses: t_q with length 7
        """
        
        imgs = read_kitti_img(base_dir, seq) # 271
        poses = read_kitti_pose(base_dir, seq)
        if seq == "03":
            imus = imgs[:-1]
        else:
            imus = read_kitti_imu(base_dir, seq) # 270
            
        # Each imu item is for a pair (img[0], img[1]): the imu records between the two images
        #     -> len(imgs) = len(poses) = len(imus) + 1
        #     -> Each pose is T_0i for i-th image
        #     -> Correspondence: imu[i] ~ (img[i], img[i+1])
        #         * last_global_pose: global_poses[i], curr_global_pose: global_poses[i+1]
        #         * last_rel_pose: rel_pose[i], curr_rel_pose: rel_poses[i+1] 

        assert len(imgs) == len(poses)
        assert len(imus) == len(imgs) - 1 

        sub_seqs = []
        tmp_seq = {
            'imus': [], 'imgs': [], 'rel_poses': [], 'global_poses': []
        }
        print('----------------------------')
        print('loading sequence {} ...'.format(seq))
        for _i, _imu in tqdm(enumerate(imus)):
            if (not seq == "03") and len(_imu[1]) == 3 and _imu[1] == 'nan':
                if len(tmp_seq['imus']) > 0:
                    sub_seqs.append(tmp_seq)
                    tmp_seq = {
                        'imus': [], 'imgs': [], 'rel_poses': [], 'global_poses': []
                    }
            else:
                # label matching: imus[_i][0] = imgs[_i] = poses[_i][0]
                assert imus[_i][0] == imgs[_i][0]
                assert imus[_i][0] == poses[_i][0]

                # get the translation vector of _i and _i+1
                trans_i   = poses[_i][1].reshape(3,4)[:,-1]
                trans_ip1 = poses[_i+1][1].reshape(3,4)[:,-1]
                # get the euler angles (rads) from rotation matrix of _i and _i+1
                rot_i_euler   = rotationMatrixToEulerAngles(poses[_i][1].reshape(3,4)[:3,:3])
                rot_ip1_euler = rotationMatrixToEulerAngles(poses[_i+1][1].reshape(3,4)[:3,:3])
                rot_i_quat    = euler_to_quaternion(rot_i_euler, isRad=True)
                rot_ip1_quat  = euler_to_quaternion(rot_ip1_euler, isRad=True)
                # tq_R0 ~ tq_i, tq_R1 ~ tq_ip1
                tq_i   = np.concatenate([trans_i, rot_i_quat])
                tq_ip1 = np.concatenate([trans_ip1, rot_ip1_quat])

                if seq == "03": 
                    tmp_seq['imus'].append(np.zeros((11, 6)))
                else:
                    tmp_seq['imus'].append(_imu)
                if len(tmp_seq['imgs']) == 0:
                    assert len(tmp_seq['rel_poses']) == 0
                    assert len(tmp_seq['global_poses']) == 0
                    tmp_seq['imgs'].append(imgs[_i])
                    # rel_poses[0] and global_poses[0] are the first last_rel/global_pose for imu[0]  
                    # the last relative pose for seq beginning is set as no movement 
                    if self.t_euler_loss:
                        tmp_seq['rel_poses'].append([poses[_i][0], np.zeros(6)])
                    else:
                        tmp_seq['rel_poses'].append([poses[_i][0], get_zero_se3()])
                    tmp_seq['global_poses'].append([poses[_i][0], tq_i])

                tmp_seq['imgs'].append(imgs[_i+1])
                tmp_seq['global_poses'].append([poses[_i+1][0], tq_ip1])
                tmp_seq['rel_poses'].append([poses[_i+1][0], get_relative_pose(tq_i, tq_ip1, self.t_euler_loss)])
                
        # append the last sub_seq
        if len(tmp_seq['imus']) > 0:
            sub_seqs.append(tmp_seq)        
                
        return sub_seqs
    


def load_kitti_clips(seqs=None, batch_size=None, shuffle=None, overlap=None, args=None, sample_size_ratio=None, load_kitti_rgb=False, load_kitti_depth=False):
    """
    -> initialize datasets of clips with specified clip_length 
    -> return: torch.utils.data.DataLoader
    """
    tmp_clips = KittiClipDataset(
        args=args, overlap=overlap, 
        load_kitti_rgb=load_kitti_rgb,
        load_kitti_depth=load_kitti_depth
    )
    for seq in seqs:
        tmp_data = KittiDataset(args=args, sequence=seq)
        tmp_clips.extend(tmp_data)
    
    # randomly reduce the sample_size to sample_size_ratio
    if 0 < sample_size_ratio < 1:
        tmp_clips.trim_samples(sample_size_ratio)
    
    
    print("============================")
    print("=> Important: Number of used clips: {} (sample_size ratio: {})".format(len(tmp_clips), sample_size_ratio))
        
    data_loader = torch.utils.data.DataLoader(
        dataset = tmp_clips,
        batch_size = batch_size,
        shuffle = shuffle
    )
    return data_loader


if __name__ == "__main__":
    pass

    
    







