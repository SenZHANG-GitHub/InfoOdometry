"""
Usage: Put this script in the data folder data/virtual-kitti/vKitti2/ and run
"""


import os 
import pdb 
from glob import glob



def read_poses(scene, subscene):
    poses = [] # "frameid pose(16x1)"
    with open("Scene{}/{}/extrinsic.txt".format(scene, subscene), "r") as f: 
        for line in f.readlines():
            line = line.strip().split()
            if line[1] != "0": 
                continue 
            out = "{} {}".format(line[0], " ".join(line[2:]))
            poses.append(out)
    return poses


def process(scene, subscene):
    print("=> processing {}: {}".format(scene, subscene))
    outpath = "prepare/Scene{}/{}".format(scene, subscene)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    
    # read image names: e.g. rgb_00000.jpg
    imgs = sorted(glob("Scene{}/{}/frames/rgb/Camera_0/*.jpg".format(scene, subscene)))
    with open("{}/cam0.txt".format(outpath), "w") as f:
        for img in imgs:
            f.write("{}\n".format(img))
    with open("{}/gt_traj0.txt".format(outpath), "w") as f:
        for line in read_poses(scene, subscene):
            # format: frameID r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3 0 0 0 1
            f.write("{}\n".format(line))
    
    img_root = "/".join(imgs[0].split("/")[:-1])
    for clip_len in [5, 6, 7, 8, 9, 10, 11]:
        clip_path = "{}/clip_len_{}".format(outpath, clip_len)
        if not os.path.isdir(clip_path):
            os.makedirs(clip_path)
        
        clips_ov, clips_non_ov = [], []
        # Get non-overlapped clips
        for sidx in list(range(len(imgs))[::clip_len]):
            if sidx + clip_len > len(imgs): continue
            clip = [img_root]
            for j in range(clip_len):
                clip.append(imgs[sidx+j].split("/")[-1])
            # ["Scene01/clone/frame/Camera_0", "rgb_00000.jpg", ...]
            clips_non_ov.append(clip)
        # Get overlapped clips
        for sidx, img in enumerate(imgs):
            if sidx + clip_len > len(imgs): continue
            clip = [img_root]
            for j in range(clip_len):
                clip.append(imgs[sidx+j].split("/")[-1])
            # ["Scene01/clone/frame/Camera_0", "rgb_00000.jpg", ...]
            clips_ov.append(clip)
        
        with open("{}/overlap.txt".format(clip_path), "w") as f:
            for clip in clips_ov:
                f.write("{}\n".format(" ".join(clip)))
        with open("{}/non_overlap.txt".format(clip_path), "w") as f:
            for clip in clips_non_ov:
                f.write("{}\n".format(" ".join(clip)))


def main():
    scenes = ["01", "02", "06", "18", "20"]
    subscenes = ["clone", "fog", "morning", "overcast", "rain", "sunset", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]
    for scene in scenes:
        for subscene in subscenes:
            process(scene, subscene)

if __name__ == "__main__":
    main()