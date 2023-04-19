import argparse
import os
import pdb
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='vinetParser')
        
        # args for failure case uncertainty analysis
        self.parser.add_argument('--eval_failure', action='store_const', default=False, const=True, help='need to be used with --eval')
        self.parser.add_argument('--noise_std_factor', type=float, default=0.1)
        self.parser.add_argument('--img_prefeat', type=str, default='flownet', help='none, flownet or resnet (not implemented yet)')
        self.parser.add_argument('--failure_type', type=str, default='noise', help='noise, missing, mixed or none')
        self.parser.add_argument('--sample_size_ratio', type=float, default=1., help='the ratio of total non-overlapped clips (1) only take effect in (0,1) (2) only used in training')
        self.parser.add_argument('--imu_only', action='store_const', default=False, const=True, help='need to be used with --transition_model double')
        
        # args to put world_kl_beta out of torch.max/min
        self.parser.add_argument('--world_kl_out', action='store_const', default=False, const=True)
        
        # args for uncertainty measurement
        self.parser.add_argument('--transition_model', type=str, default='single', help='single, double, single-vinet, multi-vinet, deepvo, deepvio')
        self.parser.add_argument('--rec_type', type=str, default='posterior', help='posterior or prior')
        self.parser.add_argument('--imu_rnn', type=str, default='gru', help='gru or lstm')
        self.parser.add_argument('--eval_uncertainty', action='store_const', default=False, const=True)
        self.parser.add_argument('--uncertainty_groups', type=int, default=1, help='1, 2')
        self.parser.add_argument('--kl_free_nats', type=str, default='max', help='none, min, max')
        self.parser.add_argument('--free_nats', type=float, default=3, help='free nats')
        self.parser.add_argument('--world_kl_beta', type=float, default=0.1, help='kl weight for posterior and prior states in the world model')
        self.parser.add_argument('--global_kl_beta', type=float, default=0, help='global kl weight (0 to disable)')
        self.parser.add_argument('--eval_ckp', type=str, default='best', help='best, last')
        self.parser.add_argument('--translation_weight', type=float, default=1, help='weight for translation_loss')
        self.parser.add_argument('--rotation_weight', type=float, default=100, help='weight for rotation_loss')
        
        # for soft / hard deepvio baselines
        self.parser.add_argument('--soft', action='store_const', default=False, const=True)
        self.parser.add_argument('--hard', action='store_const', default=False, const=True)
        self.parser.add_argument('--gumbel_tau_start', type=float, default=1.0)
        self.parser.add_argument('--gumbel_tau_final', type=float, default=0.5, help="0.5 is also the default for evaluation")
        self.parser.add_argument('--gumbel_tau_epoch_ratio', type=float, default=0.5, help="the ratio of epochs used to anneal to gumbel_tau_final")
        self.parser.add_argument('--hard_mode', type=str, default='onehot', help='onehot or gumbel_softmax')
        
        # for stochastic-only (double-stochastic / double-vinet-stochastic)
        self.parser.add_argument('--stochastic_mode', type=str, default='none', help='v1, v2 or v3: details in notability-sen')
        
        # args for training information models 
        self.parser.add_argument('--seed', type=int, default=666, help='random seed')
        self.parser.add_argument('--activation_function', type=str, default='relu', choices=dir(F), help='model activation function')
        self.parser.add_argument('--embedding_size', type=int, default=1024, help='observation embedding size')
        self.parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
        self.parser.add_argument('--belief_size', type=int, default=256, help='belief/hidden size')
        self.parser.add_argument('--belief_rnn', type=str, default='gru', help='lstm or gru')
        self.parser.add_argument('--state_size', type=int, default=128, help='state/latent size')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        self.parser.add_argument('--overshooting_distance', type=int, default=10, help='latent overshooting distance/latent overshooting weight for t=1')
        self.parser.add_argument('--overshooting_kl_beta', type=float, default=0, help='latent overshooting kl weight for t > 1 (o to disable)')
        self.parser.add_argument('--overshooting_pose_scale', type=float, default=0, help='latent overshooting pose prediction weight for t > 1 (0 to disable)')
        self.parser.add_argument('--observation_beta', type=float, default=0, help='observation loss weight; 0 to disable')
        self.parser.add_argument('--observation_imu_beta', type=float, default=0, help='observation imu loss weight; 0 to disable')
        self.parser.add_argument('--bit_depth', type=int, default=5, help='image bit depth (quantisation)')
        self.parser.add_argument('--adam_epsilon', type=float, default=1e-4, help='adam optimizer epsilon value')
        self.parser.add_argument('--grad_clip_norm', type=float, default=1000, help='gradient clipping norm')
        self.parser.add_argument('--rec_loss', type=str, default='mean', choices=["sum", "mean"], help='observation reconstruction loss type: sum or mean')
        self.parser.add_argument('--load_model', type=str, default='none', help='path for pre-saved models (.pt file) to load')
        
        # args for using FlowNet2/C/S pretrained features
        self.parser.add_argument('--train_img_from_scratch', action='store_const', default=False, const=True)
        self.parser.add_argument('--img_batch_norm', type=bool, default=False, help='can only be true when --train_img_from_scratch')
        self.parser.add_argument('--direct_img', action='store_const', default=False, const=True, help='remove the image fc before the lstm')
        self.parser.add_argument('--fp16', action='store_const', default=False, const=True)
        self.parser.add_argument('--prefeat_type', type=str, default='out_conv6_1', help='prefeat layer name in flownet models')
        self.parser.add_argument('--flownet_model', type=str, default='FlowNet2S', help='none, FlowNet2, FlowNet2S, FlowNet2C')
        self.parser.add_argument('--imgfeat_mode', type=str, default='flatten', help='flatten or pooling')
        self.parser.add_argument('--prepare_flownet_features', action='store_const', default=False, const=True)
        

        # args for training parameters
        self.parser.add_argument('--gpu', type=str, default='0', help='specify the list of gpus separated by , : e.g. 0,1,2,3')
        self.parser.add_argument('--epoch', type=int, default=300)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--lr_warmup', action='store_const', default=False, const=True)
        self.parser.add_argument('--n_warmup_steps', type=int, default=12800)
        self.parser.add_argument('--lr_schedule', type=str, default='150,250', help='epoch to reduce lr to intial_lr times the corresponding lr_factor, separated by , ')
        self.parser.add_argument('--lr_factor', type=str, default='0.1,0.05', help='used together with --lr_schedule, separated by , ')
        self.parser.add_argument('--eval_batch_size', type=int, default=1, help='if --train: equal to batch_size; if --eval: 1 by default')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='should be either adam or sgd')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='only for sgd optimizer')
        self.parser.add_argument('--train_discard_num', type=int, default=0, help='the number of beginning frames to be discarded in training')
        self.parser.add_argument('--eval_discard_num', type=int, default=0, help='the number of beginning frames to be discarded in eval')
        self.parser.add_argument('--sensors', type=str, default='img', help='img,imu,pose: whether use images, imu and last pose inputs => deprecated')
        self.parser.add_argument('--imu_lstm_hidden_size', type=int, default=128)
        self.parser.add_argument('--pose_tiles', type=int, default=10, help='the tile number of previous poses of last timestamp')
        self.parser.add_argument('--rgb_max', type=float, default=255., help='the max value of image data')
        self.parser.add_argument('--eval_interval', type=int, default=5, help='the frequency (by epoch) to eval and save the ckp')

        # args for dataset
        self.parser.add_argument('--dataset', type=str, default='euroc', choices=["kitti", "euroc", "vkitti2"], help='euroc, kitti (determine base_dir, train/eval_sequences')
        self.parser.add_argument('--base_dir', type=str, default='none', help='should not be specified')
        self.parser.add_argument('--train_sequences', type=str, default='none', help='separated by , ')
        self.parser.add_argument('--eval_sequences', type=str, default='none', help='separated by , ')
        self.parser.add_argument('--clip_length', type=int, default=5)
        self.parser.add_argument('--clip_overlap', action='store_const', default=False, const=True)
        self.parser.add_argument('--euroc_ds_train', type=str, default="both", choices=["downsample", "raw_freq", "both"])
        self.parser.add_argument('--euroc_ds_eval', type=str, default="both", choices=["downsample", "raw_freq", "both"])
        self.parser.add_argument('--euroc_ds_type', type=str, default=None, help="Deprecated => Remain here for code compacity in eval")
        self.parser.add_argument('--resize_mode', type=str, default='rescale', help='crop or rescale')
        self.parser.add_argument('--new_img_size', type=str, default='192,640', help='two int separated by , ')
        
        # args for evaluating euroc
        self.parser.add_argument('--eval_euroc_interp', action='store_const', default=False, const=True, help="Load 10/100Hz and interpolate the predicted pose to 20/200Hz")

        # args for directories of ckp and tb
        self.parser.add_argument('--exp_name', type=str, default='tmp')
        self.parser.add_argument('--ckp_dir', type=str, default='ckp/')
        self.parser.add_argument('--tb_dir', type=str, default='tb_dir/')

        # args with store_const type
        self.parser.add_argument('--eval', action='store_const', default=False, const=True, help='eval')
        self.parser.add_argument('--debug', action='store_const', default=False, const=True, help='fast debug')
        self.parser.add_argument('--on_the_fly', action='store_const', default=False, const=True)
        self.parser.add_argument('--eval_global', action='store_const', default=False, const=True, help='whether evaluate global error')
        self.parser.add_argument('--t_euler_loss', action='store_const', default=False, const=True, help='otherwise use se3 loss')
        
        # transformer parameters
        self.parser.add_argument('--tfm_enc_last', action='store_const', default=False, const=True, help='whether use the last element or the mean of transformer encoder output seq for imu encoding, mean by default')
        self.parser.add_argument('--tfm_clip_last', action='store_const', default=False, const=True, help='only predict the last frame-pair pose')

        # ablation parameters
        self.parser.add_argument("--rec_target", type=str, default="flowfeat", choices=["flowfeat", "flow", "depth", "rgb"])
        self.parser.add_argument("--rec_weight", type=float, default=1.0)
        self.parser.add_argument('--rec_embedding_size', type=int, default=1024, help='embedding size for flownet feature reconstruction')
        self.parser.add_argument('--rec_activation_function', type=str, default='relu', choices=dir(F), help='model activation function')
        self.parser.add_argument('--rec_from_noise', action='store_const', default=False, const=True, help='use N(0,1) noise as posterior_states input for training')
        self.parser.add_argument("--rec_flow_split", type=str, default="2012-2015", help="train-val, e.g. 2012-2015 or 2015-2012")
        
        ## NOTE: Parameters for generalization ablation studies
        self.parser.add_argument("--vkitti2_clone_only", action='store_const', default=False, const=True, help="for --dataset vkitti2: only use the clone subscene")
        self.parser.add_argument("--vkitti2_eval_subscene", type=str, nargs='+', help="will only load --vkitti2_subscene and disable --vkitti2_clone_only")
        self.parser.add_argument("--flowfeat_size_dataset", type=str, default="kitti", help="kitti(1024x5x19) or euroc (1024x8x12): The flowfeat size of current experiment")
        self.parser.add_argument("--eval_outname", type=str, default="tmp", help="output name to be saved in eval/ when --eval")
        self.parser.add_argument("--finetune_only_decoder", action='store_const', default=False, const=True, help="fix encoder and transition_model and only finetune the decoder")
        self.parser.add_argument("--finetune", action='store_const', default=False, const=True, help="tell the system that we are finetuning")
        # self.parser.add_argument("--", help="for --dataset vkitti2")
        # self.parser.add_argument("--", help="for --dataset vkitti2")
        # self.parser.add_argument("--", help="for --dataset vkitti2")
        
        self.args = self.parser.parse_args()
        if self.args.eval:
            # if --eval: load the args of --exp_name except for bk_args
            # NOTE: must specify --ckp_dir --exp_name
            bk_args = ['gpu', 'on_the_fly', 'eval_discard_num', 'eval_batch_size', 'eval', 'eval_uncertainty', 'uncertainty_groups', 'eval_ckp', 'eval_failure', 'noise_std_factor', 'img_prefeat', 'failure_type', 'eval_euroc_interp', 'euroc_ds_eval', 'rec_flow_split', 'flowfeat_size_dataset', 'ckp_dir', 'eval_outname', 'exp_name', 'vkitti2_clone_only', 'vkitti2_eval_subscene']
            bk_vals = dict()
            for _arg in bk_args:
                bk_vals[_arg] = getattr(self.args, _arg)
            curr_dataset = self.args.dataset
            curr_flowfeat_size_dataset = self.args.flowfeat_size_dataset
            curr_eval_sequences = self.args.eval_sequences

            # curr_eval_sequences = self.args.eval_sequences
            prev_arg_file = '{}{}/src/args.txt'.format(self.args.ckp_dir, self.args.exp_name)
            prev_args = read_args(prev_arg_file)
            self.args = self.parser.parse_args(prev_args)
            for _arg in bk_args:
                setattr(self.args, _arg, bk_vals[_arg])

            self.args.flowfeat_size_dataset = curr_flowfeat_size_dataset

            print("============================================")
            print("=> The dataset used by the loaded pretrain model: {}".format(self.args.dataset))
            print("=> The dataset evaluated currently: {}".format(curr_dataset))
            self.args.dataset = curr_dataset

            self.args.sensors = self.args.sensors.split(',')
            # self.args.train_sequences = "none"
            if curr_eval_sequences == "none":
                raise ValueError("--eval_sequences must be specified when --eval")
            else:
                self.args.eval_sequences = curr_eval_sequences.split(',')
            print("=> eval_sequences: {}".format(self.args.eval_sequences))

            self.check_eligibility()
        else:
            # training mode
            self.args.eval_batch_size = self.args.batch_size
            if self.args.dataset == 'euroc':
                self.args.base_dir = 'data/euroc/'
                if self.args.train_sequences == 'none':
                    # otherwise use the train_sequences specified by the user
                    self.args.train_sequences = 'V1_01_easy,V2_01_easy,MH_01_easy,MH_02_easy,V1_02_medium,V2_02_medium,MH_03_medium,V1_03_difficult,V2_03_difficult,MH_05_difficult'
                if self.args.eval_sequences == 'none':
                    self.args.eval_sequences = 'MH_04_difficult'
            elif self.args.dataset == 'kitti':
                self.args.base_dir = 'data/kitti/odometry/dataset'
                if self.args.train_sequences == 'none': 
                    # otherwise use the train_sequences specified by the user
                    self.args.train_sequences = '00,01,02,04,06,08,09'
                if self.args.eval_sequences == 'none':
                    self.args.eval_sequences = '05,07,10'
            else:
                if self.args.train_sequences == "none":
                    raise NotImplementedError("--train_sequences has not default values for --dataset {}".format(self.args.dataset))
                if self.args.eval_sequences == "none":
                    raise NotImplementedError("--eval_sequences has not default values for --dataset {}".format(self.args.dataset))
            
            self.args.sensors         = self.args.sensors.split(',')
            self.args.train_sequences = self.args.train_sequences.split(',')
            self.args.eval_sequences  = self.args.eval_sequences.split(',')
            self.check_eligibility()
            exp_folder = '{}{}'.format(self.args.ckp_dir, self.args.exp_name)
            if os.path.isdir(exp_folder):
                if self.args.exp_name == 'tmp': 
                    shutil.rmtree(exp_folder)
                else:
                    # raise ValueError('residues exist for the specified experiment {}'.format(exp_folder))
                    shutil.rmtree(exp_folder)
            os.mkdir(exp_folder)

            tb_folder = '{}{}'.format(self.args.tb_dir, self.args.exp_name)
            if os.path.isdir(tb_folder):
                if self.args.exp_name == 'tmp': 
                    shutil.rmtree(tb_folder)
                else:
                    # raise ValueError('residues exist for the specified experiment {}'.format(tb_folder))
                    shutil.rmtree(tb_folder)
            os.mkdir(tb_folder)

            # backup codes
            pyfiles = [
                '*.py', 
                'dataset/*.py', 
                'scripts/*.py', 
                'utils/*.py', 
            ]
            os.mkdir('{}{}/src'.format(self.args.ckp_dir, self.args.exp_name))
            for pyfile in pyfiles:
                os.system('cp {} {}{}/src/'.format(pyfile, self.args.ckp_dir, self.args.exp_name))
            
            # write args before any changes
            with open('{}{}/src/args.txt'.format(self.args.ckp_dir, self.args.exp_name), mode='w') as f:
                arg_dict = vars(self.args)
                sorted_keys = sorted(arg_dict.keys())
                max_len = max([len(x) for x in sorted_keys])
                for _key in sorted_keys:
                    f.write('--{}\t{}\n'.format(_key, arg_dict[_key]))
        
        if not self.args.eval:
            self.args.lr_schedule = [int(x) for x in self.args.lr_schedule.split(',')]
            self.args.lr_factor   = [float(x) for x in self.args.lr_factor.split(',')]
            if len(self.args.lr_schedule) > 1:
                for _i in range(len(self.args.lr_schedule) - 1):
                    if self.args.lr_schedule[_i] >= self.args.lr_schedule[_i+1]:
                        raise ValueError('--lr_schedule must be in increasing order, e.g. 150,250')
            if len(self.args.lr_schedule) != len(self.args.lr_factor):
                raise ValueError('the length of --lr_schedule should be the same as --lr_factor')

        self.args.gpu = [int(x) for x in self.args.gpu.split(',')]
        if len(self.args.gpu) == 1:
            device_index = self.args.gpu[0]
            if device_index > torch.cuda.device_count() - 1:
                raise ValueError('cannot find gpu {} in this machine'.format(device_index))
            self.args.device = torch.device('cuda', index=device_index)
        else:
            raise NotImplementedError('currently only support single gpu training')


    def check_eligibility(self):
        # check eligibility
        if self.args.optimizer not in {'adam', 'sgd'}:
            raise ValueError('optimizer {} is illegal. Currently only adam and sgd are allowed.'.format(self.args.optimizer))
        
        if self.args.dataset == 'euroc':
            for seq in self.args.train_sequences:
                if not os.path.isdir('{}{}'.format(self.args.base_dir, seq)):
                    raise ValueError('the specified train_sequence folder {}{} does not exist'.format(self.args.base_dir, seq))
            for seq in self.args.eval_sequences:
                if not os.path.isdir('{}{}'.format(self.args.base_dir, seq)):
                    raise ValueError('the specified eval_sequence folder {}{} does not exist'.format(self.args.base_dir, seq))
        elif self.args.dataset == 'kitti':
            # import pdb; pdb.set_trace()
            if type(self.args.train_sequences) == str: self.args.train_sequences = self.args.train_sequences.split(",")
            if type(self.args.eval_sequences) == str: self.args.eval_sequences = self.args.eval_sequences.split(",")
            for seq in self.args.train_sequences:
                if not os.path.isdir('{}/sequences/{}'.format(self.args.base_dir, seq)):
                    raise ValueError('the specified train_sequence folder {}{} does not exist'.format(self.args.base_dir, seq))
            for seq in self.args.eval_sequences:
                if not os.path.isdir('{}/sequences/{}'.format(self.args.base_dir, seq)):
                    raise ValueError('the specified eval_sequence folder {}{} does not exist'.format(self.args.base_dir, seq))

        if not os.path.isdir(self.args.ckp_dir):
            raise ValueError('the specified checkpoint folder {} does not exist'.format(self.args.ckp_dir))
            
        if not os.path.isdir(self.args.tb_dir):
            raise ValueError('the specified tensorboard folder {} does not exist'.format(self.args.tb_dir))
        
        if len(self.args.sensors) == 0:
            raise ValueError('at least one sensor should be specified as input')


    def get_args(self):
        return self.args


exception_args = ['--load_model', '--device']

def read_args(arg_file):
    prev_args = []
    with open(arg_file, mode='r') as f:
        # compitable with early versions
        for _line in f.readlines():
            _line = _line.split()
            if _line[0] in exception_args:
                continue
            if _line[1] == 'False':
                continue
            if _line[1] == 'True':
                _line = _line[:1] # action_const type args 
            if len(_line) > 2:
                _line[1] = ''.join(_line[1:])
                _line = _line[:2]
            for _comp in _line:
                for _char in ["[", "]", "'"," "]:
                    _comp = _comp.replace(_char, "")
                prev_args.append(_comp)
    return prev_args


















