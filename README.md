# InfoOdometry

This is the official PyTorch implementation for **[Information-Theoretic Odometry Learning], IJCV 2022**

If you find this work useful in your research, please consider citing our paper:
```
@article{zhang2022information,
  title={Information-theoretic odometry learning},
  author={Zhang, Sen and Zhang, Jing and Tao, Dacheng},
  journal={International Journal of Computer Vision},
  volume={130},
  number={11},
  pages={2553--2570},
  year={2022},
  publisher={Springer}
}
```
This repo also contains **our re-implementations of DeepVO and VINet**:

[1] Wang, Sen, et al. "Deepvo: Towards end-to-end visual odometry with deep recurrent convolutional neural networks." 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017.

[2] Clark, Ronald, et al. "Vinet: Visual-inertial odometry as a sequence-to-sequence learning problem." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.


## Installation Guide
1. Install necessary python packages
    + ```pip install -r requirements.txt```

2. Setup cuda environment 
    + set CUDA_HOME
        + e.g. ```CUDA_HOME="/path/to/cuda-10.0```
    + set LD_LIBRARY_PATH
        + e.g. ```/path/to/cuda/cuda-10.0/lib64```

3. Install correlation_package 
    + if only use ```--use_img_prefeat```
        + Step 3 can be skipped 
        + But need to download pre-saved features generated by ```python scripts/prepare_flownet_features``` 
    + set PYTHONPATH
        + e.g. ```PYTHONPATH="$PYTHONPATH:/path/to/correlation_package"```
        + e.g. ```PYTHONPATH="$PYTHONPATH:/path/to/channelnorm_package"```
        + e.g. ```PYTHONPATH="$PYTHONPATH:/path/to/resample2d_package"```
    + ```bash flownet_install.sh```

4. Setup necessary folders and files
    + ```mkdir -p ckp/tmp/src/```
    + ```mkdir -p ckp/pretrained_flownet/```
        + download FlowNet2-C_checkpoint.pth.tar into this folder
    + ```mkdir tb_dir```
    + ```mkdir eval```
    + ```mkdir data```
        + ```cd data```
        + ```ln -s ~/data/euroc euroc```
    + ```python scripts/preprocessing.py```
    + ```python scripts/prepare_flownet_features.py```

5. Setup kitti dataset for visual-inertial odometry
    + download odometry dataset from kitti odometry leaderboard: ```data/kitti/odometry/dataset```
    + download sync datasets for sequences ```00,01,02,04,05,06,07,08,09,10```
        + move the folders ```image_02``` and ```oxts``` into (e.g. ```data/kitti/odometry/dataset/sync/00/```)
        + contains folders ```image_02``` and ```oxts```
    + download unsync datasets for sequences ```00,01,02,04,05,06,07,08,09,10```
        + move ```oxts``` into (e.g. ```data/kitti/odometry/dataset/raw_oxts/00/```)
        + contains folder ```data``` and ```dataformat.txt``` and ```timestamps.txt```
    + ```python scripts/match_kitti_imu.py```
    + ```python scripts/prepare_flownet_features --dataset kitti```


## Usage 
1. training
    + ```python main.py --gpu 0 --dataset kitti --batch_size 8 --epoch 300 --lr_schedule 150,250 --use_img_prefeat --on_the_fly --exp_name XXX --model vinet```
    + if not ```--use_img_prefeat```: train img_encoder from scratch, otherwise load optical flownet features
    + ```python main.py --gpu 0 --use_flownet FlowNet2S --prefeat_type out_conv6_1```
2. evaluation
    + ```python main.py --eval --exp_name XXX --gpu 0 --on_the_fly --use_img_feat --corrupt --eval_gt_last_pose```

## Possible args for training
1. See param.py for detailed explanations and default values 
    + for args without default value, the default is actually False and we should just use "--arg" directly to make it True 

    | useful args for training      | default   |  
    | :----                         | :----:    |
    | --gpu                         | 0         |
    | --batch_size                  | 8         |
    | --epoch                       | 300       |
    | --lr_schedule                 | 150,250   |
    | --exp_name                    | tmp       |
    | --dataset                     | euroc     |
    | --lr                          | 1e-4      |
    | --model                       | vinet     |
    | --rotation_weight             | 100       |
    | --img_hidden_size             | 128       |
    | --imu_lstm_hidden_size        | 128       |
    | --fused_lstm_hidden_size      | 1024      |
    | --last_pose_tiles             | 8         |
    | --last_pose_hidden_size       | 32        |
    | --zero_first_last_pose_train  | True      |
    | --use_img_prefeat             |           |
    | --on_the_fly                  |           |


## Arguments for DeepVO, InfoVO, VINet and InfoVIO
* **deepvo**
    * ``` --transition_model deepvo --epoch 300 --batch_size 16 --img_prefeat flownet --t_euler_loss --dataset kitti --clip_length 5 -- rec_loss mean --observation_beta 100 --observation_imu_beta 10 --belief_rnn gru -- imu_rnn gru --embedding_size 1024 --hidden_size 256 --belief_size 256 --state_size 128 --translation_weight 1 --rotation_weight 100```
* **infovo**
    * ```--transition_model double --epoch 300 --batch_size 16 --img_prefeat flownet --t_euler_loss --dataset kitti --clip_length 5 --rec_loss mean --belief_rnn gru --imu_rnn gru --embedding_size 1024 --hidden_size 256 --belief_size 256 --state_size 128 --rec_type posterior --world_kl_beta 0.1 --kl_free_nats max --observation_beta 0 --observation_imu_beta 0 --translation_weight 1 --rotation_weight 100```
* **vinet**
    * ```--transition_model deepvio --epoch 300 --batch_size 16 --img_prefeat flownet --t_euler_loss --dataset kitti --clip_length 5 -- rec_loss mean --observation_beta 100 --observation_imu_beta 10 --belief_rnn gru -- imu_rnn gru --embedding_size 1024 --hidden_size 256 --belief_size 256 --state_size 128 --translation_weight 1 --rotation_weight 100```
* **infovio** 
    * ```--transition_model double-vinet --epoch 300 --batch_size 16 --img_prefeat flownet --t_euler_loss --dataset kitti --clip_length 5 --rec_loss mean --belief_rnn gru --imu_rnn gru --embedding_size 1024 --hidden_size 256 --belief_size 256 --state_size 128 --rec_type posterior --world_kl_beta 0.1 --kl_free_nats max --observation_beta 0 --observation_imu_beta 0 --translation_weight 1 --rotation_weight 100```


* **NOTE**: observation_model and observation_imu_model are now disabled for --transition_model "deepvo" and "deepvio"


## Acknowledgment
This repo is built upon the excellent works of [sophus](https://github.com/strasdat/Sophus), [flownet2](https://github.com/NVIDIA/flownet2-pytorch), and a third-party pytorch implementation of [dreamer](https://github.com/zhaoyi11/dreamer-pytorch). The borrowed codes are licensed under the original license respectively.