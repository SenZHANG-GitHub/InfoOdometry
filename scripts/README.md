## Updated on 2021/08/24 for EuRoC 1.1
* Now we generate 
    * **data/euroc/seq/downsample**: 10/100Hz sequences
    * **data/euroc/seq/raw_freq**: 20/200Hz sequences
* Inside each folder we have
    * **subseq.txt**: The suffix of all subsequences 
        * e.g. A, B, C, D, ..
    * **cam0_A(/B/...).txt**: The image sequences 
        * Format: ms,ns,data_path
    * **imu0_A(/B/...).txt**: The corresponding imu
        * Each line contains 10 imu records separated by |
        * For each imu record within the | separator
            * Format: ms,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RSi_S_z
        * The imu line will be "img_ms,none" if any of the 100Hz timestamps between two image frames cannot be found in imu raw data
    * **gt_traj_A(/B/...).txt**: The corresponding ground truth poses in translation and quaternion
        * Format: img_ms,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z 
    * **corr_pose_A(/B/...).txt**: The corresponding ground truth poses in KITTI format
        * Format: img_ms "row-aligned 3x4 transformation matrix separated by ' '"

===============================================
## Updated on 2021/08/24 for EuRoC 1.0
* Now we generate downsample/ under each sequence (10/100 Hz for images and imus)
* We save the images based on the ms unit rather than ns unit
* Format for cam0_A/B.txt 
    * ms,ns,data_path
* Format for imu0_A/B.txt
    * Each line contains 10 imu records separated by |
    * The 10 imu records correspond to the interval between two adjacent images
    * For each imu record within the | separator
        * ms,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RSi_S_z
    * The imu line can be "img_ms,none" if any of the 100Hz timestamps between two image frames cannot be found in imu raw data
* Format for gt_traj_A/B.txt
    * img_ms,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z 
* Format for corr_pose_A/B.txt (Consistent with KITTI)
    * img_ms "row-aligned 3x4 transformation matrix separated by ' '"

* The above are processed for sequences other than V2_03_difficult which has different image frequencies inside the sequence and we will do it later

* We do the same thing for raw_freq/, except that 
    * We remove the _A/B suffix
    * Important: For raw_freq, the images are still named by ns rather than ms!!! Different from downsample data => The reason: We don't want to keep an extra copy
    * Now we only save the image names in cam0.txt
        * Format: timestamp(ms),timestamp(ns),image_path(e.g. data/euroc/MH_01_easy/mav0/cam0/data/ns.png)
        * Format in short: ms,ns,data_path





=================================================
## Early version of vinet preprocessing:
* trajectory_gt.csv is directly read from state_groundtruth_estimate0/data.csv
    * Add an extra last item -> The offset between this timestamp with the closest timestamp in img
* corrected_pose.csv: From trajectory_gt.csv
    * The first item: The timestamp in img.csv!
    * The following items: The pose for this img/imu timestamp 
    * Pose format: The same as KITTI: row-aligned 3x4 transformation matrix
    * We remove the timestamps with large offsets (Usually at the begining and the end)
