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
from glob import glob
import random

from torch._six import int_classes as _int_classes
from torchvision import transforms

from utils.tools import get_relative_pose 
from utils.tools import get_zero_se3
from utils.tools import euler_to_quaternion
from utils.tools import rotationMatrixToEulerAngles
import flownet_utils.frame_utils as frame_utils
from flownet_utils.frame_utils import StaticCenterCrop


class VKitti2ClipDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None,  clip_len=None, overlap=None, on_the_fly=None, vkitti2_clone_only=None, seqs=None, t_euler_loss=None, subscene=None):
        """
        => Now use the pretrained optical flow features by default
        => ds_type: "downsample" or "raw_freq" or "both"
        """
        self.base_dir = base_dir # data/virtual-kitti/vKitti2
        # NOTE: --clip_length: number of image pairs in a clip, while self.clip_len: number ofi images in a clip
        self.clip_len = clip_len + 1
        self.overlap = overlap
        self.on_the_fly = on_the_fly
        self.seqs = seqs
        self.t_euler_loss = t_euler_loss

        if subscene is None:
            if vkitti2_clone_only:
                self.subscenes = ["clone"]
            else:
                self.subscenes = ["clone", "fog", "overcast", "morning", "rain", "sunset", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]
        else:
            self.subscenes = subscene

        # See self.load_clips() for the data formats
        self.clips, self.prefeats, self.global_trajs, self.rel_poses = self.load_clips()
    

    def load_clips(self):
        """
        Output
            => clips: list of [img_root, [img_ids]] e.g. ["Scene01/clone/frames/rgb/Camera_0", ["rgb_00000.jpg", "rgb_00001.jpg", ...]]
            => global_trajs: 
                => key: "{}_{}_{}".format(seq, subscene, img_id_0) e.g. Scene01_15-deg-left_0
                => val:  t_q with length 7
            => rel_poses:
                => key: "{}+{}".format(label_0, label_1) e.g. Scene01_fog_0+Scene01_fog_1
                => val: np.array(6,): results of get_relative_pose()
            => prefeats: (used when not --on_the_fly)
                => key: "{}+{}".format(label_0, label_1) e.g. Scene01_fog_0+Scene01_fog_1
                => val: results of torch.load()
        """
        clips, prefeats, global_trajs, rel_poses = [], dict(), dict(), dict()
        for seq in self.seqs:
            for subscene in self.subscenes:
                print("=> Loading seq {}-{}...".format(seq, subscene))
                with open("{}/prepare/{}/{}/gt_traj0.txt".format(self.base_dir, seq, subscene), "r") as f:
                    for line in f.readlines():
                        line = line.strip().split()
                        img_idx = int(line[0]) # 0
                        pose_mat = [float(x) for x in line[1:13]] # len 12 (3x4)
                        pose_mat = np.array(pose_mat).reshape(3, 4)
                        trans = pose_mat[:,-1]
                        rot_euler = rotationMatrixToEulerAngles(pose_mat[:3,:3])
                        rot_quat = euler_to_quaternion(rot_euler, isRad=True)
                        pose_tq = np.concatenate([trans, rot_quat])

                        label = "{}_{}_{}".format(seq, subscene, img_idx) # Scene01_15-deg-left_0
                        global_trajs[label] = pose_tq
                
                clip_file = "overlap.txt" if self.overlap else "non_overlap.txt"
                with open("{}/prepare/{}/{}/clip_len_{}/{}".format(self.base_dir, seq, subscene, self.clip_len, clip_file), "r") as f:
                    for line in f.readlines():
                        line = line.strip().split()
                        img_root = line[0] # "Scene01/clone/frames/rgb/Camera_0"
                        img_ids = line[1:] # ["rgb_00000.jpg", "rgb_00001.jpg", ...]
                        assert len(img_ids) == self.clip_len  
                        clips.append([img_root, img_ids]) # [img_root, [img_ids]]
                        for idx, _ in enumerate(img_ids): # e.g. rgb_00000.jpg
                            if idx == 0: continue
                            img_id_0 = int(img_ids[idx-1].split(".")[0].split("_")[-1]) # 0 
                            img_id_1 = int(img_ids[idx].split(".")[0].split("_")[-1]) #1
                            label_0 = "{}_{}_{}".format(seq, subscene, img_id_0) # Scene01_15-deg-left_0
                            label_1 = "{}_{}_{}".format(seq, subscene, img_id_1) # Scene01_15-deg-left_1

                            rel_label = "{}+{}".format(label_0, label_1)
                            if rel_label not in rel_poses.keys():
                                rel_poses[rel_label] = get_relative_pose(global_trajs[label_0], global_trajs[label_1], self.t_euler_loss)

                            if not self.on_the_fly and rel_label not in prefeats.keys():
                                pt_path = "{}/prepare/{}/{}/flownet_features".format(self.base_dir, seq, subscene)
                                pt_file = "rgb_{:05d}-rgb_{:05d}.pt".format(img_id_0, img_id_1)
                                assert os.path.isfile("{}/{}".format(pt_path, pt_file))
                                # shape of .pt feature: [1024, 5, 19]
                                prefeats[rel_label] = torch.load("{}/{}".format(pt_path, pt_file))

        return clips, prefeats, global_trajs, rel_poses
    

    def __len__(self):
        return len(self.clips)


    def __getitem__(self, clip_idx):
        """
        returns a list of 
        (1) image pair (last and present)
        (2) imu data in between
        (3) last_relative_pose (from last_last to last) -> idx 
        (4) curr_relative_pose (from last to present) -> idx + 1
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


        img_root = self.clips[clip_idx][0] # e.g. "Scene01/clone/frames/rgb/Camera_0"
        clip = self.clips[clip_idx][1] # e.g. ["rgb_00000.jpg", "rgb_00001.jpg", ...]

        seq = img_root.split("/")[0]
        subscene = img_root.split("/")[1]
        
        for idx, _ in enumerate(clip):
            if idx == 0: continue 
            img_0, img_1 = clip[idx-1], clip[idx] # "rgb_00000.jpg", "rgb_00001.jpg"
            img_id_0 = int(img_0.split(".")[0].split("_")[-1]) # 0
            img_id_1 = int(img_1.split(".")[0].split("_")[-1]) # 1
            label_0 = "{}_{}_{}".format(seq, subscene, img_id_0)
            label_1 = "{}_{}_{}".format(seq, subscene, img_id_1)
            rel_label = "{}+{}".format(label_0, label_1)
            
            # Load optical flow features
            if self.on_the_fly:
                feat_path = "{}/{}/{}/flownet_features/gb_{:05d}-rgb_{:05d}.pt".format(self.base_dir, seq, subscene, img_id_0, img_id_1)
                r_img_features = torch.load(feat_path)
            else:
                r_img_features = self.prefeats[rel_label]
            img_features.append(r_img_features)

            # Set IMU features to all be 0 (Will not be used anyway)
            r_imu_features = np.zeros((11, 6))
            imu_features.append(torch.from_numpy(r_imu_features)) # [11, 6]

            # Load relative pose in se3 or t_euler
            if idx == 1:
                #  The last_last to last for the first pair should be zero
                rel_pose_0 = np.zeros(6) if self.t_euler_loss else get_zero_se3()
            else:
                img_before_0 = clip[idx-2]
                img_id_before_0 = int(img_before_0.split(".")[0].split("_")[-1]) 
                label_before_0 = "{}_{}_{}".format(seq, subscene, img_id_before_0)
                rel_label_last = "{}+{}".format(label_before_0, label_0)
                rel_pose_0 = self.rel_poses[rel_label_last]
            rel_pose_1 = self.rel_poses[rel_label]
            last_relative_pose.append(torch.from_numpy(rel_pose_0)) # [6] se3 or t_euler
            curr_relative_pose.append(torch.from_numpy(rel_pose_1)) #[6] se3 or t_euler

            # Load absolute global pose in t_and_q
            r_last_global_pose = self.global_trajs[label_0]
            r_curr_global_pose = self.global_trajs[label_1]
            last_global_pose.append(torch.from_numpy(r_last_global_pose)) # [7] t_and_q
            curr_global_pose.append(torch.from_numpy(r_curr_global_pose)) # [7] t_and_q

            last_timestamp.append(label_0)
            curr_timestamp.append(label_1)

        # img_features:         length-5 list with component [3, 2, 480, 752] if not use_img_prefeat
        #                       length-5 list with component [1024, 8, 12] if use_img_prefeat
        # imu_features:         length-5 list with component [11, 6]
        # last_relative_pose:   length-5 list with component [6]
        # curr_relative_pose:   length-5 list with component [6]
        # last_global_pose:     length-5 list with component [7]
        # curr_global_pose:     length-5 list with component [7]
        # curr_timestamp:       length-5 list with component [1]
        return img_features, imu_features, last_relative_pose, curr_relative_pose, last_global_pose, curr_global_pose, last_timestamp, curr_timestamp


class VKitti2Dataset(torch.utils.data.Dataset):
    def __init__(self, seq=None, subscene=None):
        """
        NOTE: Be aware that virtual KITTI 2 don't release the frequency (Hz)
        => Used for preparing flownet features
        """
        self.seq = seq
        self.base_dir = "data/virtual-kitti/vKitti2"
        self.subscene = subscene
        # self.subscenes = ["clone", "fog", "morning", "rain", "overcast", "sunset", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]

        # Used for: [375, 1242] -> [320, 1216]
        self.render_size = [320, 1216] # (kitti_size // 64) * 64
        self.img_pairs = self.read_data(self.base_dir, self.seq, self.subscene)


    def read_data(self, base_dir, seq, subscene):
        """
        Load image pairs from data/virtual-kitti/vKitti2/prepare/Scene01/clone/cam0.txt (e.g.)
        """
        pairs = []
        with open("{}/prepare/{}/{}/cam0.txt".format(base_dir, seq, subscene), "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx == 0: continue 
                img_0 = lines[idx-1].strip()
                img_1 = lines[idx].strip()
                pairs.append([img_0, img_1])
        return pairs
    
    def __len__(self):
        return len(self.img_pairs)
    

    def __getitem__(self, idx):
        # e.g. Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg
        img_0, img_1 = self.img_pairs[idx]
        img_0 = "{}/{}".format(self.base_dir, img_0)
        img_1 = "{}/{}".format(self.base_dir, img_1)
        ts_0 = img_0.split("/")[-1].split(".")[0] # rgb_00000
        ts_1 = img_1.split("/")[-1].split(".")[0] # rgb_00001

        # (1242, 375)
        last_img = np.array(PIL.Image.open(img_0).convert("RGB")) # [375, 1242, 3]
        curr_img = np.array(PIL.Image.open(img_1).convert("RGB")) # [375, 1242, 3]

        img_size = curr_img.shape[:2]
        cropper = StaticCenterCrop(img_size, self.render_size)
        last_img = cropper(last_img) # [320, 1216, 3]
        curr_img = cropper(curr_img) # [320, 1216, 3]

        last_img = torch.from_numpy(last_img).permute(2,0,1)  # [3, 320, 1216]
        curr_img = torch.from_numpy(curr_img).permute(2,0,1)  # [3, 320, 1216]

        img_features = torch.stack((last_img, curr_img), dim=1).type(torch.FloatTensor) # [3, 2, 320, 1216], max=255.
        return img_features, ts_0, ts_1



def load_vkitti2_clips(seqs=None, batch_size=None, shuffle=None, overlap=None, clip_len=None, on_the_fly=None, vkitti2_clone_only=None, t_euler_loss=None, subscene=None):
    """
    -> initialize datasets of clips with specified clip_length 
    -> return: torch.utils.data.DataLoader
    """
    base_dir = "data/virtual-kitti/vKitti2"
    vkitti2_clips = VKitti2ClipDataset(
        base_dir=base_dir,  
        clip_len=clip_len, 
        overlap=overlap, 
        on_the_fly=on_the_fly, 
        vkitti2_clone_only=vkitti2_clone_only, 
        seqs=seqs,
        t_euler_loss=t_euler_loss,
        subscene=subscene
    )
   
    data_loader = torch.utils.data.DataLoader(
        dataset = vkitti2_clips,
        batch_size = batch_size,
        shuffle = shuffle
    )
    return data_loader



if __name__ == "__main__":
    pass

    
    







