import PIL
from PIL import Image
import numpy as np
import torch
import pdb
import os
import shutil
from tqdm import tqdm
from shutil import rmtree
import random

from torch._six import int_classes as _int_classes
from utils.tools import get_relative_pose
from utils.tools import get_zero_se3


class EuRocClipDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None,  clip_len=None, overlap=None, on_the_fly=None, ds_type=None, seqs=None, t_euler_loss=None):
        """
        => Now use the pretrained optical flow features by default
        => ds_type: "downsample" or "raw_freq" or "both"
        """
        self.base_dir = base_dir # data/euroc
        # NOTE: --clip_length: number of image pairs in a clip, while self.clip_len: number ofi images in a clip
        self.clip_len = clip_len + 1
        self.overlap = overlap
        self.on_the_fly = on_the_fly
        self.seqs = seqs
        self.t_euler_loss = t_euler_loss

        assert ds_type in ["downsample", "raw_freq", "both"]
        if ds_type == "both":
            ds_types = ["downsample", "raw_freq"]
        else:
            ds_types = [ds_type]

        # imus =>  key: seq+ms; val: str
        # trajs => key: seq+ms; val: list of float [x, y, z, qw, qx, qy, qz]
        self.imus, self.trajs = self.load_imu_traj(ds_types)
        self.clips, self.prefeats = self.load_clip(ds_types)
    

    def load_imu_traj(self, ds_types):
        """
        """
        # NOTE: flag: "downsample" or "raw"
        imus = dict() # key: flag+seq+ms; val: str
        trajs = dict() # key: flag+seq+ms; val: list of float [x, y, z, qw, qx, qy, qz]
        for seq in self.seqs:
            for ds_type_ in ds_types:
                subseqs = []
                flag =ds_type_
                with open("{}/{}/{}/subseq.txt".format(self.base_dir, seq, ds_type_), "r") as f:
                    for line in f.readlines():
                        subseqs.append(line.strip())
                
                for suff in subseqs:
                    with open("{}/{}/{}/imu0_{}.txt".format(self.base_dir, seq, ds_type_, suff), "r") as f:
                        for line in f.readlines():
                            line = line.strip()
                            imu_ms = int(line.split(",")[0])
                            imu_ms = "{}+{}+{}".format(flag, seq, imu_ms) # "downsample+MH_01_easy+12312321"
                            if line.split(",")[1] == "none":
                                imus[imu_ms] = "none"
                            else:
                                imus[imu_ms] = line

                    with open("{}/{}/{}/gt_traj_{}.txt".format(self.base_dir, seq, ds_type_, suff), "r") as f:
                        for line in f.readlines():
                            line = line.strip()
                            traj_ms = int(line.split(",")[0])
                            traj_ms = "{}+{}+{}".format(flag, seq, traj_ms) # "downsample+MH_01_easy+12312321"
                            if line.split(",")[1] == "none":
                                trajs[traj_ms] = "none"
                            else:
                                trajs[traj_ms] = [float(x) for x in line.split(",")[1:]]
        return imus, trajs


    def load_clip(self, ds_types):
        """
        """
        clips = [] # each component: [flag, "MH_01_easy", [img0, img1, ...]]
        prefeats = dict() # preloaded flownet features
        for seq in self.seqs:
            print("=> Loading sequence: {}".format(seq))
            for ds_type_ in ds_types:
                subseqs = []
                flag = ds_type_
                with open("{}/{}/{}/subseq.txt".format(self.base_dir, seq, ds_type_), "r") as f:
                    for line in f.readlines():
                        subseqs.append(line.strip())
                
                clip_file = "overlap.txt" if self.overlap else "non_overlap.txt" 
                with open("{}/{}/{}/clip_len_{}/{}".format(self.base_dir, seq, ds_type_, self.clip_len, clip_file), "r") as f:
                    for line in f.readlines():
                        line = [int(x) for x in line.strip().split(",")]
                        assert len(line) == self.clip_len

                        # Valid check: Filter out clips that have unmatched imu and gt_traj
                        ok_flag = True
                        for img in line:
                            if ok_flag and self.imus["{}+{}+{}".format(flag, seq, img)] == "none": ok_flag = False 
                            if ok_flag and self.trajs["{}+{}+{}".format(flag, seq, img)] == "none": ok_flag = False 
                        if not ok_flag: continue 

                        clips.append([flag, seq, line]) # ["MH_01_easy", [img0, img1, ...]]
                        if not self.on_the_fly:
                            for idx, img in enumerate(line):
                                if idx == 0: continue
                                feat_path = "{}/{}/{}/flownet_features/{}-{}.pt".format(self.base_dir, seq, ds_type_, line[idx-1], line[idx])
                                feat_label = "{}+{}+{}-{}".format(flag, seq, line[idx-1], line[idx])
                                prefeats[feat_label] = torch.load(feat_path)

        return clips, prefeats



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
        imu_labels = [] # not used for euroc
        imu_features_all = [] # not used for euroc

        flag = self.clips[clip_idx][0] # "downsample" or "raw_freq"
        seq = self.clips[clip_idx][1]
        clip = self.clips[clip_idx][2]
        for idx, img in enumerate(clip):
            if idx == 0: continue 
            img_0, img_1 = clip[idx-1], clip[idx]
            ms_0, ms_1 = "{}+{}+{}".format(flag, seq, img_0), "{}+{}+{}".format(flag, seq, img_1)
            
            # Load optical flow features
            if self.on_the_fly:
                feat_path = "{}/{}/{}/flownet_features/{}-{}.pt".format(self.base_dir, seq, flag, img_0, img_1)
                r_img_features = torch.load(feat_path)
            else:
                r_img_features = self.prefeats["{}+{}+{}-{}".format(flag, seq, img_0, img_1)]
            img_features.append(r_img_features)

            # Load imu features
            imu = self.imus[ms_0].split("|")
            assert int(imu[0].split(",")[0]) == img_0
            assert int(imu[-1].split(",")[0]) == img_1
            imu = [[float(y) for y in x.split(",")[1:]] for x in imu]
            r_imu_features = np.array(imu)
            assert r_imu_features.shape == (11, 6)
            imu_features.append(torch.from_numpy(r_imu_features)) # [11, 6]

            # Load relative pose in se3 or t_euler
            traj_0 = self.trajs[ms_0]
            traj_1 = self.trajs[ms_1]
            if idx == 1:
                #  The last_last to last for the first pair should be zero
                rel_pose_0 = np.zeros(6) if self.t_euler_loss else get_zero_se3()
            else:
                ms_last_last = "{}+{}+{}".format(flag, seq, clip[idx-2])
                traj_last_last = self.trajs[ms_last_last]
                rel_pose_0 = get_relative_pose(traj_last_last, traj_0, self.t_euler_loss)
            rel_pose_1 = get_relative_pose(traj_0, traj_1, self.t_euler_loss)
            last_relative_pose.append(torch.from_numpy(rel_pose_0)) # [6] se3 or t_euler
            curr_relative_pose.append(torch.from_numpy(rel_pose_1)) #[6] se3 or t_euler

            # Load absolute global pose in t_and_q
            r_last_global_pose = np.array(traj_0)
            r_curr_global_pose = np.array(traj_1)
            last_global_pose.append(torch.from_numpy(r_last_global_pose)) # [7] t_and_q
            curr_global_pose.append(torch.from_numpy(r_curr_global_pose)) # [7] t_and_q

            last_timestamp.append(ms_0)
            curr_timestamp.append(ms_1)

        # img_features:         length-5 list with component [3, 2, 480, 752] if not use_img_prefeat
        #                       length-5 list with component [1024, 8, 12] if use_img_prefeat
        # imu_features:         length-5 list with component [11, 6]
        # last_relative_pose:   length-5 list with component [6]
        # curr_relative_pose:   length-5 list with component [6]
        # last_global_pose:     length-5 list with component [7]
        # curr_global_pose:     length-5 list with component [7]
        # curr_timestamp:       length-5 list with component [1]
        return img_features, imu_features, last_relative_pose, curr_relative_pose, last_global_pose, curr_global_pose, last_timestamp, curr_timestamp


    def __len__(self):
        return len(self.clips)


class EuRocDataset(torch.utils.data.Dataset):
    def __init__(self, seq=None, base_dir=None, t_euler_loss=None, train_img_from_scratch=None, ds_type=None):
        """ e.g.
        args.base_dir: data/euroc/
        args.sequences: [V1_01_easy, V1_02_medium] -> sequence 
        downsample: if True: 10/100Hz; if False 20/200Hz
        """
        # global camera pose -> [0:3] translation, [3:] quaternion
        self.seq = seq
        self.base_dir = base_dir
        self.ds_type = ds_type
        assert ds_type in ["downsample", "raw_freq"]
        self.t_euler_loss = t_euler_loss 
        self.on_the_fly = True
        self.train_img_from_scratch = train_img_from_scratch 
        self.data = self.read_data(seq, base_dir, ds_type)
        
        # # self.ds_imgs: list of sub_seqs, each sub_seq with length K
        # # self.ds_imus: list of sub_seqs, each sub_seq with length K-1
        # # self.ds_relative_pose: list of sub_seqs, each sub_seq with length K, [ts, np(6,)]
        # # self.ds_global_poses: list of sub_seqs, each sub_seq with length K
        # self.ds_imgs, self.ds_imus, self.ds_relative_poses, self.ds_global_poses = self.downsample_img_imu() 


    def read_data(self, seq, base_dir, ds_type):
        """Read data from data/euroc/seq/raw_freq or downsample
        => Check scripts/README.md for format details
        """
        freq = "10/100Hz" if ds_type=="downsample" else "20/200Hz"
        dpath = ds_type

        print("=====================")
        print("=> Reading {} data for {}...".format(freq, seq))

        subseqs = [] # "A, B, ..."
        with open("{}/{}/{}/subseq.txt".format(base_dir, seq, dpath), "r") as f:
            for line in f.readlines():
                subseqs.append(line.strip())
        print("=> Total number of subseqs: {} ({})".format(len(subseqs), subseqs))
        all_data = []
        for suff in subseqs:
            cam0, imu0, traj0 = [], [], []
            with open("{}/{}/{}/cam0_{}.txt".format(base_dir, seq, dpath, suff), "r") as f:
                for line in f.readlines():
                    line = line.strip().split(",")
                    cam0.append([int(line[0]), line[-1]]) # [ms, img_path]
            with open("{}/{}/{}/imu0_{}.txt".format(base_dir, seq, dpath, suff), "r") as f:
                for line in f.readlines(): 
                    tmp = line.strip().split(",")
                    if tmp[1] == "none":
                        imu0.append([int(tmp[0]), "none"]) 
                    else:
                        imu0.append([int(tmp[0]), line.strip()]) # [ms, imu_records_str]
            with open("{}/{}/{}/gt_traj_{}.txt".format(base_dir, seq, dpath, suff), "r") as f:
                for line in f.readlines():
                    line = line.strip().split(",")
                    if line[1] == "none":
                        traj0.append([int(line[0]), "none"])
                    else:
                        traj0.append([int(line[0]), ",".join(line[1:])]) # [ms, traj_str]
            
            ## Check vadility
            assert len(cam0) == len(imu0) == len(traj0)
            for img, imu, traj in zip(cam0, imu0, traj0):
                assert img[0] == imu[0] == traj[0]
            
            ## Load data into self.data if "none" is not present in both imu and traj 
            for idx, img in enumerate(cam0):
                if idx == 0: continue 
                if imu0[idx-1][1] == "none" or imu0[idx][1] == "none":
                    continue 
                if traj0[idx-1][1] == "none" or traj0[idx][1] == "none":
                    continue
                all_data.append([cam0[idx-1], cam0[idx]])
        
        print("=> Total number of valid image pair: {}".format(len(all_data)))
        return all_data

        # # the last relative pose for seq beginning is set as no movement 
        #     if self.t_euler_loss:
        #         tmp_relative_pose.append(np.array([self.trajectory_abs[_indices[0]][0], np.zeros(6)]))
        #     else:
        #         tmp_relative_pose.append(np.array([self.trajectory_abs[_indices[0]][0], get_zero_se3()]))
        #     tmp_global_pose.append(self.trajectory_abs[_indices[0]])

        #     for _i in range(len(_seq) - 1):
        #         tmp_relative_pose.append(
        #             get_relative_pose(self.trajectory_abs[_indices[_i]], self.trajectory_abs[_indices[_i+1]], self.t_euler_loss)
        #         )
        #         tmp_global_pose.append(self.trajectory_abs[_indices[_i+1]])


    def __len__(self):
        """
        an image pair forms a data point
        """
        return len(self.data)


    def __getitem__(self, idx):
        """
        returns
        (1) image pair (last and present)
        (2) imu data in between
        (3) last_relative_pose (from last_last to last) -> idx 
        (4) curr_relative_pose (from last to present) -> idx + 1
        (5) last_global_pose w.r.t world frame -> idx
        (6) curr_global_pose w.r.t world frame -> idx + 1
        (7) timestamp -> idx + 1 (We use ms rather than ns)
        """
        # 1 channel to 3 channels -> RGB or an init_conv layer from 1 -> 3
        # NOTE: img_0/1: [ms, data_path]
        img_0, img_1 = self.data[idx]
        timestamp_0 = img_0[0]
        timestamp_1 = img_1[0]

        if self.on_the_fly:
            last_img = np.array(PIL.Image.open(img_0[1]).convert('RGB'))
            curr_img = np.array(PIL.Image.open(img_1[1]).convert('RGB'))
        else:
            last_img = img_0[1] # [480, 752, 3]
            curr_img = img_1[1]
        last_img = torch.from_numpy(last_img).permute(2,0,1)  # [3, 480, 752]
        curr_img = torch.from_numpy(curr_img).permute(2,0,1)  # [3, 480, 752]

        img_features = torch.stack((last_img, curr_img), dim=1).type(torch.FloatTensor) # [3, 2, 480, 752], max=255.

        return img_features, timestamp_0, timestamp_1


def load_euroc_clips(seqs=None, batch_size=None, shuffle=None, overlap=None, clip_len=None, on_the_fly=None, ds_type=None, t_euler_loss=None):
    """
    -> initialize datasets of clips with specified clip_length 
    -> return: torch.utils.data.DataLoader
    """
    base_dir = "data/euroc"
    euroc_clips = EuRocClipDataset(
        base_dir=base_dir,  
        clip_len=clip_len, 
        overlap=overlap, 
        on_the_fly=on_the_fly, 
        ds_type=ds_type, 
        seqs=seqs,
        t_euler_loss=t_euler_loss
    )
   
    data_loader = torch.utils.data.DataLoader(
        dataset = euroc_clips,
        batch_size = batch_size,
        shuffle = shuffle
    )
    return data_loader


if __name__ == "__main__":
    pass

    
    







