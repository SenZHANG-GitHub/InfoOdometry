"""
Downsample EuRoC to /home/szha2609/data/euroc/MH_01_easy/downsample/: cam0_A, cam0_B, imu0_A, imu0_B, pose0_A, pose0_B 
NOTE: 
    => First downsample images from 20Hz to 10Hz (2 subsequences usually)
    => Then match the imu and pose to downsampled images
"""

import os
import pdb
from shutil import rmtree
from collections import Counter
from tqdm import tqdm


def downsample_img_imu():
    """
    => V2_03_difficult need extra concerns, we do it for other sequences first
    => We save the timestamp in ms rather than ns to avoid numerical trivial errors
    """
    sequences = [
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
        # 'V2_03_difficult'
    ]
    for seq in sequences:
        print("===============================")
        print("=> processing {}...".format(seq))
        base_dir = "/home/szha2609/data/euroc/{}".format(seq)

        imus = []
        imu_data = {}
        imu_interval = 10 # 5ms for 200Hz, 10ms for 100Hz
        with open("{}/mav0/imu0/data.csv".format(base_dir), "r") as f:
            for line in f.readlines():
                if line[0] == "#": continue
                line = line.strip() # remove "\n"
                imu_ts = round(int(line.split(",")[0]) / 10**6)
                imus.append(imu_ts) # ms
                imu_data[imu_ts] = ",".join(line.split(",")[1:])

        imgs = [] # ms
        with open("/home/szha2609/data/euroc/{}/mav0/cam0/data.csv".format(seq), "r") as f:
            for line in f.readlines():
                if line[0] == "#": continue
                curr_ts = int(line.split(",")[0])
                imgs.append(curr_ts) # ns for now
        imgs_A = imgs[0::2]
        imgs_B = imgs[1::2]

        corr_poses = {}
        with open("{}/mav0/preprocessing/corrected_pose.csv".format(base_dir), "r") as f:
            for line in f.readlines():
                if line[0] == "#": continue 
                line = line.strip()
                curr_ts = int(line.split()[0])
                corr_poses[curr_ts] = " ".join(line.split()[1:])

        gt_traj = []
        with open("{}/mav0/preprocessing/trajectory_gt.csv".format(base_dir), "r") as f:
            for line in f.readlines():
                if line[0] == "#": continue 
                line = line.strip()
                curr_ts = int(line.split(",")[0])

                # assert curr_ts in imgs 

                # NOTE: offset means the timeoffset between state_gt_traj and images
                # NOTE: The format of gt_traj: p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z 
                # NOTE: Others or latter ones may be of interest: v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2], time_offset[ns]. Though we omit these metrics for this task

                offset = int(line.split(",")[-1])
                # curr_ts = round(curr_ts / 10**6) # ns -> ms
                # if offset > 256: 
                #     gt_traj[curr_ts] = "none"
                # else:
                #     gt_traj[curr_ts] = ",".join(line.split(",")[1:8])
                
                gt_traj.append("{},{},{}".format(curr_ts, ",".join(line.split(",")[1:8]), offset))
        
        assert len(gt_traj) == len(imgs)
        gt_traj_A = gt_traj[0::2]
        gt_traj_B = gt_traj[1::2]
        
        if not os.path.isdir("{}/downsample".format(base_dir)):
            os.mkdir("{}/downsample".format(base_dir))

        # if os.path.isfile("{}/downsample/timestamp_cam0_A.txt".format(base_dir)):
        #     os.remove("{}/downsample/timestamp_cam0_A.txt".format(base_dir))
        # if os.path.isfile("{}/downsample/timestamp_cam0_B.txt".format(base_dir)):
        #     os.remove("{}/downsample/timestamp_cam0_B.txt".format(base_dir))

        with open("{}/downsample/subseq.txt".format(base_dir), "w") as f:
            f.write("A\n")
            f.write("B\n")

        ## Save the cam0 files: format for each line: ms, ns
        with open("{}/downsample/cam0_A.txt".format(base_dir), "w") as f:
            with open("{}/downsample/gt_traj_A.txt".format(base_dir), "w") as fg:
                with open("{}/downsample/corr_pose_A.txt".format(base_dir), "w") as ft:
                    with open("{}/downsample/imu0_A.txt".format(base_dir), "w") as fi:
                        for idx, img_ns in enumerate(imgs_A):
                            img_ms = round(img_ns / 10**6)
                            img_path = "data/euroc/{}/mav0/cam0/data/{}.png".format(seq, img_ns)
                            f.write("{},{},{}\n".format(img_ms, img_ns, img_path))

                            gt_ns = int(gt_traj_A[idx].split(",")[0])
                            gt_ms = round(gt_ns / 10**6)

                            offset = int(gt_traj_A[idx].split(",")[-1])
                            assert abs(img_ns - gt_ns) == abs(offset)
                            if abs(offset) > 256:
                                fg.write("{},none\n".format(img_ms))
                                ft.write("{} none\n".format(img_ms))
                            else:
                                assert img_ms == gt_ms
                                # NOTE: old format: ms, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z 
                                fg.write("{},{}\n".format(img_ms, ",".join(gt_traj_A[idx].split(",")[1:-1])))

                                # NOTE: new format (the same as KITTI): ms row-aligned 3x4 transformation matrix
                                assert img_ns in corr_poses.keys()
                                ft.write("{} {}\n".format(img_ms, corr_poses[img_ns]))
                            
                            none_flag = False 
                            for j in range(11):
                                if none_flag is False and img_ms + j * imu_interval not in imus:
                                    none_flag = True
                            if none_flag:
                                fi.write("{},none\n".format(img_ms))
                                continue 

                            line = []
                            for j in range(11):
                                imu_idx = img_ms + j * imu_interval
                                line.append("{},{}".format(imu_idx, imu_data[imu_idx]))
                            line = "|".join(line)
                            fi.write("{}\n".format(line))


        with open("{}/downsample/cam0_B.txt".format(base_dir), "w") as f:
            with open("{}/downsample/gt_traj_B.txt".format(base_dir), "w") as fg:
                with open("{}/downsample/corr_pose_B.txt".format(base_dir), "w") as ft:
                    with open("{}/downsample/imu0_B.txt".format(base_dir), "w") as fi:
                        for idx, img_ns in enumerate(imgs_B):
                            img_ms = round(img_ns / 10**6)
                            img_path = "data/euroc/{}/mav0/cam0/data/{}.png".format(seq, img_ns)
                            f.write("{},{},{}\n".format(img_ms, img_ns, img_path))

                            gt_ns = int(gt_traj_B[idx].split(",")[0])
                            gt_ms = round(gt_ns / 10**6)

                            offset = int(gt_traj_B[idx].split(",")[-1])
                            assert abs(img_ns - gt_ns) == abs(offset)
                            if abs(offset) > 256:
                                fg.write("{},none\n".format(img_ms))
                                ft.write("{} none\n".format(img_ms))
                            else:
                                assert img_ms == gt_ms
                                # NOTE: oldformat: ms, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z 
                                fg.write("{},{}\n".format(img_ms, ",".join(gt_traj_B[idx].split(",")[1:-1])))

                                # NOTE: new format (the same as KITTI): ms row-aligned 3x4 transformation matrix
                                assert img_ns in corr_poses.keys()
                                ft.write("{} {}\n".format(img_ms, corr_poses[img_ns]))
                            
                            none_flag = False 
                            for j in range(11):
                                if none_flag is False and img_ms + j * imu_interval not in imus:
                                    none_flag = True
                            if none_flag:
                                fi.write("{},none\n".format(img_ms))
                                continue 

                            line = []
                            for j in range(11):
                                imu_idx = img_ms + j * imu_interval
                                line.append("{},{}".format(imu_idx, imu_data[imu_idx]))
                            line = "|".join(line)
                            fi.write("{}\n".format(line))

def downsample_img_imu_outlier():
    """
    => V2_03_difficult need extra concerns, we do it for other sequences first
    => We save the timestamp in ms rather than ns to avoid numerical trivial errors
    """
    seq = "V2_03_difficult"
    print("===============================")
    print("=> processing {}...".format(seq))
    base_dir = "/home/szha2609/data/euroc/{}".format(seq)

    imus = []
    imu_data = {}
    imu_interval = 10 # 5ms for 200Hz, 10ms for 100Hz
    with open("{}/mav0/imu0/data.csv".format(base_dir), "r") as f:
        for line in f.readlines():
            if line[0] == "#": continue
            line = line.strip() # remove "\n"
            imu_ts = round(int(line.split(",")[0]) / 10**6)
            imus.append(imu_ts) # ms
            imu_data[imu_ts] = ",".join(line.split(",")[1:])

    imgs = [] # ms
    with open("/home/szha2609/data/euroc/{}/mav0/cam0/data.csv".format(seq), "r") as f:
        for line in f.readlines():
            if line[0] == "#": continue
            curr_ts = int(line.split(",")[0])
            imgs.append(curr_ts) # ns for now
    
    corr_poses = {}
    with open("{}/mav0/preprocessing/corrected_pose.csv".format(base_dir), "r") as f:
        for line in f.readlines():
            if line[0] == "#": continue 
            line = line.strip()
            curr_ts = int(line.split()[0])
            corr_poses[curr_ts] = " ".join(line.split()[1:])

    gt_traj = []
    with open("{}/mav0/preprocessing/trajectory_gt.csv".format(base_dir), "r") as f:
        for line in f.readlines():
            if line[0] == "#": continue 
            line = line.strip()
            curr_ts = int(line.split(",")[0])

            # assert curr_ts in imgs 

            # NOTE: offset means the timeoffset between state_gt_traj and images
            # NOTE: The format of gt_traj: p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z 
            # NOTE: Others or latter ones may be of interest: v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2], time_offset[ns]. Though we omit these metrics for this task

            offset = int(line.split(",")[-1])
            # curr_ts = round(curr_ts / 10**6) # ns -> ms
            # if offset > 256: 
            #     gt_traj[curr_ts] = "none"
            # else:
            #     gt_traj[curr_ts] = ",".join(line.split(",")[1:8])
            
            gt_traj.append("{},{},{}".format(curr_ts, ",".join(line.split(",")[1:8]), offset))
    
    assert len(gt_traj) == len(imgs)

    if not os.path.isdir("{}/raw_freq".format(base_dir)):
        os.mkdir("{}/raw_freq".format(base_dir))
    
    # ## NOTE: check the img interval in ms
    # gaps = []
    # for idx, img in enumerate(imgs):
    #     if idx > 0:
    #         gap = round(imgs[idx]/10**6) - round(imgs[idx-1]/10**6)
    #         gaps.append(gap)
    # # print("=> gaps in ms: {}".format(Counter(gaps)))
    # # NOTE: The output is {50: 1507, 100: 414}

    ## Get all 10 Hz subsequences
    all_seqs = []
    all_gt_traj = []
    tmp_seq = []
    tmp_gt_traj = []
    img_interval = 100 # 50 for 20Hz, 100 for 10Hz 

    # We here use two loops to get all the sub_seqs
    def is_included(img, all_seqs):
        for seq in all_seqs:
            if img in seq:
                return True 
        return False

    for idx, img in enumerate(imgs):
        # img: ns
        if is_included(img, all_seqs):
            continue

        tmp_seq = []
        tmp_seq.append(img)
        tmp_gt_traj = []
        tmp_gt_traj.append(gt_traj[idx])
        
        for j in range(idx+1, len(imgs)):
            if round(imgs[j]/10**6) - round(tmp_seq[-1]/10**6) > img_interval:
                break
            if is_included(img, all_seqs):
                continue
            if round(imgs[j]/10**6) - round(tmp_seq[-1]/10**6) == img_interval:
                tmp_seq.append(imgs[j])
                tmp_gt_traj.append(gt_traj[j])
        
        if len(tmp_seq) >= 5:
            all_seqs.append(tmp_seq)
            all_gt_traj.append(tmp_gt_traj)


    # NOTE: The length of seqs in all_seqs with length >= 5:
    #       => [546, 214, 132, 125, 39, 51, 143, 132, 199, 27, 138, 48, 41, 80] 
    # pdb.set_trace()
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

    if not os.path.isdir("{}/downsample".format(base_dir)):
        os.mkdir("{}/downsample".format(base_dir))
    
    with open("{}/downsample/subseq.txt".format(base_dir), "w") as f:
        for l in letters:
            f.write("{}\n".format(l))

    ## Save the timestamps: format for each line: ms, ns
    for idx_seq, subseq in enumerate(all_seqs):
        suff = letters[idx_seq]
        with open("{}/downsample/cam0_{}.txt".format(base_dir, suff), "w") as f:
            with open("{}/downsample/gt_traj_{}.txt".format(base_dir, suff), "w") as fg:
                with open("{}/downsample/corr_pose_{}.txt".format(base_dir, suff), "w") as ft:
                    with open("{}/downsample/imu0_{}.txt".format(base_dir, suff), "w") as fi:
                        for idx, img_ns in enumerate(subseq):
                            img_ms = round(img_ns / 10**6)
                            img_path = "data/euroc/{}/mav0/cam0/data/{}.png".format(seq, img_ns)
                            f.write("{},{},{}\n".format(img_ms, img_ns, img_path))

                            gt_ns = int(all_gt_traj[idx_seq][idx].split(",")[0])
                            gt_ms = round(gt_ns / 10**6)

                            offset = int(all_gt_traj[idx_seq][idx].split(",")[-1])
                            assert abs(img_ns - gt_ns) == abs(offset)
                            if abs(offset) > 256:
                                fg.write("{},none\n".format(img_ms))
                                ft.write("{} none\n".format(img_ms))
                            else:
                                assert img_ms == gt_ms
                                # NOTE: oldformat: ms, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z 
                                fg.write("{},{}\n".format(img_ms, ",".join(all_gt_traj[idx_seq][idx].split(",")[1:-1])))

                                # NOTE: new format (the same as KITTI): ms row-aligned 3x4 transformation matrix
                                assert img_ns in corr_poses.keys()
                                ft.write("{} {}\n".format(img_ms, corr_poses[img_ns]))
                            
                            none_flag = False 
                            for j in range(11):
                                if none_flag is False and img_ms + j * imu_interval not in imus:
                                    none_flag = True
                            if none_flag:
                                fi.write("{},none\n".format(img_ms))
                                continue 

                            line = []
                            for j in range(11):
                                imu_idx = img_ms + j * imu_interval
                                line.append("{},{}".format(imu_idx, imu_data[imu_idx]))
                            line = "|".join(line)
                            fi.write("{}\n".format(line))


def get_downsample_clip(clip_len):
    sequences = [
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
    for seq in sequences:
        print("===============================")
        print("=> processing {}...".format(seq))
        base_dir = "/home/szha2609/data/euroc/{}".format(seq)
        subseqs = []
        with open("{}/downsample/subseq.txt".format(base_dir), "r") as f:
            for line in f.readlines():
                subseqs.append(line.strip())
        print("=> Total number of subseqs: {} ({})".format(len(subseqs), subseqs))
        
        clips_ov = [] # overlapped clips
        clips_non_ov = [] # non-overlapped clips
        for suff in subseqs:
            imgs = []
            with open("{}/downsample/cam0_{}.txt".format(base_dir, suff), "r") as f:
                for line in f.readlines():
                    imgs.append(int(line.strip().split(",")[0])) # ms (interval: 100)
            # Get non-overlapped clips
            for sidx in list(range(len(imgs))[::clip_len]):
                if sidx + clip_len > len(imgs): continue 
                clip = []
                for j in range(clip_len):
                    clip.append(imgs[sidx+j])
                clips_non_ov.append(clip)
            
            # Get overlapped clips
            for sidx, img in enumerate(imgs):
                if sidx + clip_len > len(imgs): continue 
                clip = []
                for j in range(clip_len):
                    clip.append(imgs[sidx+j])
                clips_ov.append(clip)

        opath = "{}/downsample/clip_len_{}".format(base_dir, clip_len)
        if os.path.isdir(opath):
            rmtree(opath)
        os.mkdir(opath)

        with open("{}/overlap.txt".format(opath), "w") as f:
            for clip in clips_ov:
                f.write("{}\n".format(",".join([str(x) for x in clip])))

        with open("{}/non_overlap.txt".format(opath), "w") as f:
            for clip in clips_non_ov:
                f.write("{}\n".format(",".join([str(x) for x in clip])))

if __name__=="__main__":
    # downsample_img_imu()
    # downsample_img_imu_outlier()

    for clip_len in [7, 9, 11]:
        get_downsample_clip(clip_len)














