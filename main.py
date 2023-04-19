import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch.utils.data
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
from tqdm import tqdm

import os
import pdb
import numpy as np
import quaternion
from transforms3d import euler

from utils.tools import SequenceTimer
from utils.tools import euler_to_quaternion
from utils.tools import RunningAverager
from utils.tools import save_model
from utils.tools import eval_rel_error
from utils.tools import get_lr
from utils.tools import factor_lr_schedule
from utils.tools import ScheduledOptim
from utils.tools import construct_models

from param import Param
from dataset.euroc_dataset import load_euroc_clips
from dataset.kitti_dataset import load_kitti_clips
from dataset.vkitti2_dataset import load_vkitti2_clips
from info_model import bottle


def train(args):
    """
    args: see param.py for details
    """
    # torch.cuda.manual_seed(args.seed)
    epoch = args.epoch
    batch = args.batch_size
    writer = SummaryWriter(log_dir='{}{}/'.format(args.tb_dir, args.exp_name))
    
    # use_imu: denote whether img and imu are used at the same time
    # args.imu_only: denote only imu is used
    flownet_model, transition_model, use_imu, use_info, observation_model, observation_imu_model, pose_model, encoder = construct_models(args)
    
    if args.finetune_only_decoder:
        assert args.finetune
        print("=> only finetune pose_model, while fixing encoder and transition_model")
    if args.finetune:
        assert args.load_model is not "none"

    if args.finetune_only_decoder:
        param_list = list(pose_model.parameters())
    else:
        param_list = list(transition_model.parameters()) + list(pose_model.parameters()) + list(encoder.parameters())
        if observation_model: param_list += list(observation_model.parameters())
        if observation_imu_model: param_list += list(observation_imu_model.parameters())
    
    if args.lr_warmup:
        optimizer = ScheduledOptim(
            optimizer=optim.Adam(param_list, betas=(0.9, 0.98), eps=1e-09),
            init_lr=args.lr,
            d_model=args.belief_size,
            n_warmup_steps=args.n_warmup_steps
        )
    else:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(param_list, lr=args.lr, eps=args.adam_epsilon)
        else:
            raise ValueError('optimizer {} is currently not supported'.format(args.optimizer))
    
    # NOTE: Load prev ckp after we have specified optimizer 
    if args.load_model is not "none":
        assert os.path.exists(args.load_model)
        print("=> loading previous trained model: {}".format(args.load_model))
        model_dicts = torch.load(args.load_model, map_location="cuda:0")
        transition_model.load_state_dict(model_dicts["transition_model"], strict=True)
        if observation_model: observation_model.load_state_dict(model_dicts["observation_model"], strict=True)
        if observation_imu_model: observation_imu_model.load_state_dict(model_dicts["observation_imu_model"], strict=True)
        pose_model.load_state_dict(model_dicts["pose_model"], strict=True)
        encoder.load_state_dict(model_dicts["encoder"], strict=True)
        
        if not args.finetune:
            # for vkitti2 we are finetuning v.s. fot kitti and euroc we are resuming
            # NOTE: for --finetune we will our own optimizer setting
            optimizer.load_state_dict(model_dicts["optimizer"], strict=True)
    
    if use_info:
        # global prior N(0, I)
        global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device)) 
        free_nats = torch.full((1, ), args.free_nats, device=args.device) # allowed deviation in kl divergence
    
    if not args.lr_warmup:
        lmbda = lambda epoch: factor_lr_schedule(epoch, divide_epochs=args.lr_schedule, lr_factors=args.lr_factor)
        scheduler = LambdaLR(optimizer, lr_lambda=lmbda)

    # initialize datasets and data loaders

    if args.dataset == "euroc":
        train_clips = load_euroc_clips(
            seqs = args.train_sequences,
            batch_size = args.batch_size, 
            shuffle = True,
            overlap = args.clip_overlap,
            clip_len = args.clip_length,
            on_the_fly = args.on_the_fly,
            ds_type = args.euroc_ds_train,
            t_euler_loss=args.t_euler_loss
        )

        eval_clips = load_euroc_clips(
            seqs = args.eval_sequences,
            batch_size = args.eval_batch_size, 
            shuffle = False,
            overlap = False,
            clip_len = args.clip_length,
            on_the_fly = args.on_the_fly,
            ds_type = args.euroc_ds_eval,
            t_euler_loss=args.t_euler_loss
        )

    elif args.dataset == "kitti":
        train_clips = load_kitti_clips(
            seqs = args.train_sequences,
            batch_size = args.batch_size,
            shuffle = True,
            overlap = args.clip_overlap,
            args = args,
            sample_size_ratio = args.sample_size_ratio
        )

        eval_clips = load_kitti_clips(
            seqs = args.eval_sequences,
            batch_size = args.eval_batch_size,
            shuffle = False,
            overlap = False,
            args = args,
            sample_size_ratio = 1.
        )
    elif args.dataset == "vkitti2":
        train_clips = load_vkitti2_clips(
            seqs = args.train_sequences,
            batch_size = args.batch_size, 
            shuffle = True,
            overlap = args.clip_overlap,
            clip_len = args.clip_length,
            on_the_fly = args.on_the_fly,
            vkitti2_clone_only = args.vkitti2_clone_only,
            t_euler_loss=args.t_euler_loss
        )

        eval_clips = load_vkitti2_clips(
            seqs = args.eval_sequences,
            batch_size = args.eval_batch_size, 
            shuffle = False,
            overlap = False,
            clip_len = args.clip_length,
            on_the_fly = args.on_the_fly,
            vkitti2_clone_only = args.vkitti2_clone_only,
            t_euler_loss=args.t_euler_loss,
            subscene = args.vkitti2_eval_subscene
        )

    else:
        raise NotImplementedError()

    best_epoch = {'sqrt_then_avg': 0} # 'avg_then_sqrt': 0
    best_rpe_all = {'sqrt_then_avg': 1.0} # 'avg_then_sqrt': 1.0
    best_metrics = {'sqrt_then_avg': None} # 'avg_then_sqrt': None
    eval_msgs = []
    
    # gumbel temperature for hard deepvio
    if args.hard:
        start_tau = args.gumbel_tau_start # default 1.0
        final_tau = args.gumbel_tau_final # default 0.5
        anneal_epochs = int(epoch * args.gumbel_tau_epoch_ratio)
        step_tau = (start_tau - final_tau) / (anneal_epochs - 1)
        gumbel_tau = start_tau + step_tau
    
    # starting training (the same epochs for each sequence)
    curr_iter = 0
    for epoch_idx in range(epoch):    
        print('-----------------------------------------')
        print('starting epoch {}...'.format(epoch_idx))
        print('learning rate: {:.6f}'.format(get_lr(optimizer)))
        pose_model.train()
        if args.finetune_only_decoder:
            encoder.eval()
            transition_model.eval()
            if observation_model: observation_model.eval()
            if observation_imu_model: observation_imu_model.eval()
        else:
            encoder.train()
            transition_model.train()
            if observation_model: observation_model.train()
            if observation_imu_model: observation_imu_model.train()
        
        # update gumbel temperature if using hard deepvio
        if args.hard:
            if epoch_idx < anneal_epochs: gumbel_tau -= step_tau
            print('-> gumbel temperature: {}'.format(gumbel_tau))
        
        batch_timer = SequenceTimer()
        last_batch_index = len(train_clips) - 1
        for batch_idx, batch_data in enumerate(train_clips):
            if args.debug and batch_idx >= 10: break
            # x_img_list:                length-5 list with component [batch, 3, 2, H, W]
            # x_imu_list:                length-5 list with component [batch, 11, 6]
            # x_last_rel_pose_list:      length-5 list with component [batch, 6]    # se3 or t_euler (--t_euler_loss)
            # y_rel_pose_list:           length-5 list with component [batch, 6]    # se3 or t_euler (--t_euler_loss)
            # y_last_global_pose_list:   length-5 list with component [batch, 7]    # t_quaternion
            # y_global_pose_list:        length-5 list with component [batch, 7]    # t_quaternion
            x_img_list, x_imu_list, x_last_rel_pose_list, y_rel_pose_list, y_last_global_pose_list, y_global_pose_list, _, _ = batch_data
            
            x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [time, batch, 3, 2, H, W]
            y_rel_poses = torch.stack(y_rel_pose_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [time, batch, 6]      
            if use_imu or args.imu_only:
                x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [time, batch, 11, 6]      
            running_batch_size = x_img_pairs.size()[1] # might be different for the last batch
            
            # transitions start at time t = 0 
            # create initial belief and state for time t = 0
            init_state = torch.zeros(running_batch_size, args.state_size, device=args.device)
            init_belief = torch.zeros(running_batch_size, args.belief_size, device=args.device)
               
            # if we use flownet_feature as reconstructed observations -> no need for dequantization
            with torch.no_grad():
                # [time, batch, out_conv6_1] e.g. [5, 16, 1024, 5, 19]
                observations = x_img_pairs if args.img_prefeat == 'flownet' else bottle(flownet_model, (x_img_pairs, ))
            obs_size = observations.size()
            observations = observations.view(obs_size[0], obs_size[1], -1)
                
            # update belief/state using posterior from previous belief/state, previous pose and current observation (over entire sequence at once)
            # output: [time, ] with init states already removed
            if args.finetune_only_decoder:
                with torch.no_grad():
                    if use_imu:
                        encode_observations = (bottle(encoder, (observations, )), x_imu_seqs)
                    elif args.imu_only:
                        encode_observations = x_imu_seqs
                    else:
                        encode_observations = bottle(encoder, (observations, ))
                    
                    args_transition = {
                        'prev_state': init_state, # not used if not use_info
                        'poses': y_rel_poses, # not used if not use_info during training
                        'prev_belief': init_belief,
                        'observations': encode_observations
                    }
                    
                    if args.hard: args_transition['gumbel_temperature'] = gumbel_tau
                    
                    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(**args_transition)
            
            else:
                if use_imu:
                    encode_observations = (bottle(encoder, (observations, )), x_imu_seqs)
                elif args.imu_only:
                    encode_observations = x_imu_seqs
                else:
                    encode_observations = bottle(encoder, (observations, ))
                
                args_transition = {
                    'prev_state': init_state, # not used if not use_info
                    'poses': y_rel_poses, # not used if not use_info during training
                    'prev_belief': init_belief,
                    'observations': encode_observations
                }
                
                if args.hard: args_transition['gumbel_temperature'] = gumbel_tau
                
                beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(**args_transition)
            
            if use_info:
                # observation reconstruction for images
                if args.observation_beta != 0:
                    beliefs_visual = beliefs[0] if use_imu else beliefs
                    if args.finetune_only_decoder:
                        with torch.no_grad():
                            if args.rec_type == 'posterior':
                                pred_observations = bottle(observation_model, (beliefs_visual, posterior_states, )) 
                            elif args.rec_type == 'prior':
                                pred_observations = bottle(observation_model, (beliefs_visual, prior_states, )) 
                    else:
                        if args.rec_type == 'posterior':
                            pred_observations = bottle(observation_model, (beliefs_visual, posterior_states, )) 
                        elif args.rec_type == 'prior':
                            pred_observations = bottle(observation_model, (beliefs_visual, prior_states, )) 
                    if args.rec_loss == 'sum':
                        observation_loss = F.mse_loss(pred_observations, observations, reduction='none').sum(dim=2).mean(dim=(0,1)) # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
                    elif args.rec_loss == 'mean':
                        observation_loss = F.mse_loss(pred_observations, observations, reduction='none').mean(dim=2).mean(dim=(0,1)) # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
                    observation_loss = args.observation_beta * observation_loss
            
                # observation reconstruction for imus
                if use_imu and args.observation_imu_beta != 0:
                    if args.finetune_only_decoder:
                        with torch.no_grad():
                            if args.rec_type == 'posterior':
                                pred_imu_observations = bottle(observation_imu_model, (beliefs[1], posterior_states, )) 
                            elif args.rec_type == 'prior':
                                pred_imu_observations = bottle(observation_imu_model, (beliefs[1], prior_states, )) 
                    else:
                        if args.rec_type == 'posterior':
                            pred_imu_observations = bottle(observation_imu_model, (beliefs[1], posterior_states, )) 
                        elif args.rec_type == 'prior':
                            pred_imu_observations = bottle(observation_imu_model, (beliefs[1], prior_states, )) 
                    if args.rec_loss == 'sum':
                        observation_imu_loss = F.mse_loss(pred_imu_observations, x_imu_seqs.view(pred_imu_observations.size()), reduction='none').sum(dim=2).mean(dim=(0,1)) # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
                    elif args.rec_loss == 'mean':
                        observation_imu_loss = F.mse_loss(pred_imu_observations, x_imu_seqs.view(pred_imu_observations.size()), reduction='none').mean(dim=2).mean(dim=(0,1)) # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
                    observation_imu_loss = args.observation_imu_beta * observation_imu_loss
                    
                if args.kl_free_nats == 'none':
                    kl_loss = args.world_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2).mean(dim=(0,1))
                elif args.kl_free_nats == 'min':
                    if args.world_kl_out:
                        kl_loss = args.world_kl_beta * torch.min(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0,1))
                    else:
                        kl_loss = torch.min(args.world_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0,1))
                elif args.kl_free_nats == 'max':
                    if args.world_kl_out:
                        kl_loss = args.world_kl_beta * torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0,1))
                    else:
                        kl_loss = torch.max(args.world_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0,1))
                if args.global_kl_beta != 0:
                    if running_batch_size == args.batch_size:
                        kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0,1))
                    else:
                        tmp_global_prior = Normal(torch.zeros(running_batch_size, args.state_size, device=args.device), torch.ones(running_batch_size, args.state_size, device=args.device))
                        kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), tmp_global_prior).sum(dim=2).mean(dim=(0,1)) 
            
            
            pred_rel_poses = bottle(pose_model, (posterior_states, ))
            pose_trans_loss = args.translation_weight * F.mse_loss(pred_rel_poses[:,:,:3], y_rel_poses[:,:,:3], reduction='none').sum(dim=2).mean(dim=(0,1)) 
            pose_rot_loss = args.rotation_weight * F.mse_loss(pred_rel_poses[:,:,3:], y_rel_poses[:,:,3:], reduction='none').sum(dim=2).mean(dim=(0,1)) 
            
            total_loss = pose_trans_loss + pose_rot_loss
            if use_info:
                total_loss += kl_loss
                if args.observation_beta != 0: total_loss += observation_loss
                if use_imu and args.observation_imu_beta != 0: total_loss += observation_imu_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(param_list, args.grad_clip_norm, norm_type=2)
            optimizer.step() # if using ScheduledOptim -> will also update learning rate

            writer.add_scalar('train/total_loss', total_loss.item(), curr_iter)
            if use_info:
                if args.observation_beta != 0: writer.add_scalar('train/observation_visual_loss', observation_loss.item(), curr_iter)
                if use_imu and args.observation_imu_beta != 0: writer.add_scalar('train/observation_imu_loss', observation_imu_loss.item(), curr_iter)
                writer.add_scalar('train/kl_loss', kl_loss.item(), curr_iter)
            writer.add_scalar('train/pose_trans_loss', pose_trans_loss.item(), curr_iter)
            writer.add_scalar('train/pose_rot_loss', pose_rot_loss.item(), curr_iter)
            writer.add_scalar('train/learning_rate', get_lr(optimizer), curr_iter)
            curr_iter += 1

            batch_timer.tictoc()
            remain_time = batch_timer.get_remaining_time(batch_idx, last_batch_index)
            remain_time = '{:.0f}h:{:2.0f}m:{:2.0f}s'.format(remain_time//3600, (remain_time%3600)//60, (remain_time%60))
            
            loss_str = '{:.5f}+{:.5f}'.format(pose_trans_loss.item(), pose_rot_loss.item())
            if use_info:
                loss_str = '{:.5f}+{}'.format(kl_loss.item(), loss_str)
                if use_imu and args.observation_imu_beta != 0: loss_str = '{:.5f}+{}'.format(observation_imu_loss.item(), loss_str)
                if args.observation_beta != 0: loss_str = '{:.5f}+{}'.format(observation_loss.item(), loss_str)
            print('epoch: {:3d} | {:4d}/{} | loss: {:.5f} ({}) | time: {:.3f}s | remaining: {}'.format(epoch_idx, batch_idx, last_batch_index, total_loss.item(), loss_str, batch_timer.get_last_time_elapsed(), remain_time))

        # update learning rate for next epoch
        if not args.lr_warmup: scheduler.step()

        # evaluate the model after training each sequence
        # if gt_last_pose is False, then zero_first must be True
        if epoch_idx % args.eval_interval== 0:
            pose_model.eval()
            encoder.eval()
            transition_model.eval()
            if observation_model:observation_model.eval()
            if observation_imu_model: observation_imu_model.eval()
            
            # move eval directly here
            with torch.no_grad():
                print('----------------------------------------')
                batch_timer = SequenceTimer()
                last_batch_index = len(eval_clips) - 1
                loss_avg  = dict()
                loss_list = ['total_loss', 'pose_trans_loss', 'pose_rot_loss']
                if use_info:
                    loss_list += ['kl_loss']
                    if args.observation_beta != 0: loss_list += ['observation_visual_loss']
                    if use_imu and args.observation_imu_beta != 0: loss_list += ['observation_imu_loss']
                for _met in loss_list:
                    loss_avg[_met] = RunningAverager()
                list_eval = dict()
                for _met in ['rpe', 'ate']:
                    for _suf in ['_all', '_trans', '_rot_axis', '_rot_euler']:
                        list_eval['{}{}'.format(_met, _suf)] = []
                for batch_idx, batch_data in enumerate(eval_clips):
                    if args.debug and batch_idx >= 10: break
                    x_img_list, x_imu_list, x_last_rel_pose_list, y_rel_pose_list, y_last_global_pose_list, y_global_pose_list, _, _ = batch_data
                    
                    x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [time, batch, 3, 2, H, W]
                    y_rel_poses = torch.stack(y_rel_pose_list, dim=0).type(torch.FloatTensor).to(device=args.device)
                    if use_imu or args.imu_only: 
                        x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=args.device)
                    running_eval_batch_size = x_img_pairs.size()[1] # might be different at the last batch
                    init_state = torch.zeros(running_eval_batch_size, args.state_size, device=args.device)
                    init_belief = torch.zeros(running_eval_batch_size, args.belief_size, device=args.device)
                    
                    observations = x_img_pairs if args.img_prefeat == 'flownet' else bottle(flownet_model, (x_img_pairs, ))
                    obs_size = observations.size()
                    observations = observations.view(obs_size[0], obs_size[1], -1)
                    
                    if use_imu:
                        encode_observations = (bottle(encoder, (observations, )), x_imu_seqs)
                    elif args.imu_only:
                        encode_observations = x_imu_seqs
                    else:
                        encode_observations = bottle(encoder, (observations, ))
                        
                    # with one more returns: poses
                    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, pred_rel_poses = transition_model(
                        prev_state=init_state, # not used if not use_info
                        poses=pose_model, 
                        prev_belief=init_belief,
                        observations=encode_observations
                    )
                    
                    if use_info:
                        if args.observation_beta != 0:
                            beliefs_visual = beliefs[0] if use_imu else beliefs
                            if args.rec_type == 'posterior':
                                pred_observations = bottle(observation_model, (beliefs_visual, posterior_states, ))
                            elif args.rec_type == 'prior':
                                pred_observations = bottle(observation_model, (beliefs_visual, prior_states, ))
                            if args.rec_loss == 'sum': 
                                observation_loss = F.mse_loss(pred_observations, observations, reduction='none').sum(dim=2).mean(dim=(0,1)) 
                            elif args.rec_loss == 'mean':
                                observation_loss = F.mse_loss(pred_observations, observations, reduction='none').mean(dim=2).mean(dim=(0,1)) 
                            # observation_loss = args.observation_beta * observation_loss
                        
                        if use_imu and args.observation_imu_beta != 0:
                            if args.rec_type == 'posterior':
                                pred_imu_observations = bottle(observation_imu_model, (beliefs[1], posterior_states, ))
                            elif args.rec_type == 'prior':
                                pred_imu_observations = bottle(observation_imu_model, (beliefs[1], prior_states, ))
                            if args.rec_loss == 'sum':
                                observation_imu_loss = F.mse_loss(pred_imu_observations, x_imu_seqs.view(pred_imu_observations.size()), reduction='none').sum(dim=2).mean(dim=(0,1)) 
                            elif args.rec_loss == 'mean':
                                observation_imu_loss = F.mse_loss(pred_imu_observations, x_imu_seqs.view(pred_imu_observations.size()), reduction='none').mean(dim=2).mean(dim=(0,1)) 
                            # observation_imu_loss = args.observation_imu_beta * observation_imu_loss
                        
                        kl_loss = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2).mean(dim=(0,1))
                        # kl_loss = args.world_kl_beta * kl_loss
                        if args.global_kl_beta != 0:
                            if running_eval_batch_size == args.eval_batch_size:
                                # kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0,1))
                                kl_loss += kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0,1))
                            else:
                                tmp_global_prior = Normal(torch.zeros(running_eval_batch_size, args.state_size, device=args.device), torch.ones(running_eval_batch_size, args.state_size, device=args.device))
                                # kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), tmp_global_prior).sum(dim=2).mean(dim=(0,1))
                                kl_loss += kl_divergence(Normal(posterior_means, posterior_std_devs), tmp_global_prior).sum(dim=2).mean(dim=(0,1))
                    
                    pose_trans_loss = args.translation_weight * F.mse_loss(pred_rel_poses[:,:,:3], y_rel_poses[:,:,:3], reduction='none').sum(dim=2).mean(dim=(0,1))
                    pose_rot_loss = args.rotation_weight * F.mse_loss(pred_rel_poses[:,:,3:], y_rel_poses[:,:,3:], reduction='none').sum(dim=2).mean(dim=(0,1))
                    
                    total_loss = pose_trans_loss + pose_rot_loss
                    if use_info:
                        total_loss += kl_loss
                        if args.observation_beta != 0: total_loss += observation_loss
                        if use_imu and args.observation_imu_beta != 0: total_loss += observation_imu_loss
        
                    loss_avg['total_loss'].append(total_loss)
                    if use_info:
                        if args.observation_beta != 0: loss_avg['observation_visual_loss'].append(observation_loss)
                        if use_imu and args.observation_imu_beta != 0: loss_avg['observation_imu_loss'].append(observation_imu_loss)
                        loss_avg['kl_loss'].append(kl_loss)
                    loss_avg['pose_trans_loss'].append(pose_trans_loss)
                    loss_avg['pose_rot_loss'].append(pose_rot_loss)
                    
                    for _fidx in range(args.clip_length):
                        # (1) evaluate relative pose error (2) no discard_num is used
                        eval_rel = eval_rel_error(pred_rel_poses[_fidx], y_rel_poses[_fidx], t_euler_loss=args.t_euler_loss)
                        for _met in ['rpe_all', 'rpe_trans', 'rpe_rot_axis', 'rpe_rot_euler']:
                            list_eval[_met].extend(eval_rel[_met])
                    
                    batch_timer.tictoc()
                    remain_time = batch_timer.get_remaining_time(batch_idx, last_batch_index)
                    remain_time = '{:.0f}h:{:2.0f}m:{:2.0f}s'.format(remain_time//3600, (remain_time%3600)//60, (remain_time%60))

                    loss_str = '{:.5f}+{:.5f}'.format(pose_trans_loss.item(), pose_rot_loss.item())
                    if use_info:
                        loss_str = '{:.5f}+{}'.format( kl_loss.item(), loss_str)
                        if use_imu and args.observation_imu_beta != 0: loss_str = '{:.5f}+{}'.format(observation_imu_loss.item(), loss_str)
                        if args.observation_beta != 0: loss_str = '{:.5f}+{}'.format(observation_loss.item(), loss_str)
                    print('eval: {:4d}/{} | loss: {:.5f} ({}) | time: {:.3f}s | remaining: {}'.format(batch_idx, last_batch_index, total_loss.item(), loss_str, batch_timer.get_last_time_elapsed(), remain_time))

            out_eval = dict()
            for _loss in loss_list:
                out_eval[_loss] = loss_avg[_loss].item()
                writer.add_scalar('eval/{}'.format(_loss), out_eval[_loss], curr_iter)
            out_eval['rpe_rot_axis'] = np.mean(np.array(list_eval['rpe_rot_axis']))
            writer.add_scalar('eval_sqrt_then_avg/rpe_rot_axis', out_eval['rpe_rot_axis'], curr_iter)
            for _met in ['rpe_all', 'rpe_trans', 'rpe_rot_euler']:
                out_eval[_met] = dict()
                # out_eval[_met]['avg_then_sqrt'] = np.sqrt(np.mean(np.array(list_eval[_met])))
                out_eval[_met]['sqrt_then_avg'] = np.mean(np.sqrt(np.array(list_eval[_met])))
                writer.add_scalar('eval_sqrt_then_avg/{}'.format(_met), out_eval[_met]['sqrt_then_avg'], curr_iter)

            check_str = '{}{}/ckp_latest.pt'.format(args.ckp_dir, args.exp_name)
            save_args = {
                'transition_model': transition_model, 
                'observation_model': observation_model, # None if not used
                'observation_imu_model': observation_imu_model, # None if not used
                'pose_model': pose_model, 
                'encoder': encoder,
                'optimizer': optimizer,
                'epoch': epoch_idx,
                'metrics': out_eval
            }
            save_model(path=check_str, **save_args)
            if epoch_idx > 198 and epoch_idx % 100 == 0:
                check_str = '{}{}/ckp_epoch_{}.pt'.format(args.ckp_dir, args.exp_name, epoch_idx + 1)
                save_model(path=check_str, **save_args)

            if out_eval['rpe_all']['sqrt_then_avg'] < best_rpe_all['sqrt_then_avg'] or best_metrics['sqrt_then_avg'] is None:
                best_rpe_all['sqrt_then_avg'] = out_eval['rpe_all']['sqrt_then_avg']
                best_epoch['sqrt_then_avg'] = epoch_idx
                best_metrics['sqrt_then_avg'] = out_eval
                check_str = '{}{}/ckp_best-rpe-all_sqrt_then_avg.pt'.format(args.ckp_dir, args.exp_name)
                save_model(path=check_str, **save_args)
            
            print('====================================')
            print('current epoch for sqrt_then_avg')
            print('====================================')
            loss_str = 'pose_trans_loss: {:.5f} | pose_rot_loss: {:.5f}'.format(out_eval['pose_trans_loss'], out_eval['pose_rot_loss'])
            if use_info:
                loss_str = 'kl_loss: {:.5f} \n{}'.format( out_eval['kl_loss'], loss_str)
                if use_imu and args.observation_imu_beta != 0: loss_str = 'observation_imu_loss: {:.5f} | {}'.format(out_eval['observation_imu_loss'], loss_str)
                if args.observation_beta != 0: loss_str = 'observation_visual_loss: {:.5f} | {}'.format(out_eval['observation_visual_loss'], loss_str)
            print('eval epoch: {} | total_loss: {:.5f} | {}'.format(epoch_idx, out_eval['total_loss'], loss_str))
            print('rpe_all: {:.5f} | rpe_trans: {:.5f} | rpe_rot_axis: {:.5f} | rpe_rot_euler: {:.5f}'.format(out_eval['rpe_all']['sqrt_then_avg'], out_eval['rpe_trans']['sqrt_then_avg'], out_eval['rpe_rot_axis'], out_eval['rpe_rot_euler']['sqrt_then_avg']))
            
            print('====================================')
            print('best epoch for sqrt_then_avg')
            print('====================================')
            loss_str = 'pose_trans_loss: {:.5f} | pose_rot_loss: {:.5f}'.format(best_metrics['sqrt_then_avg']['pose_trans_loss'], best_metrics['sqrt_then_avg']['pose_rot_loss'])
            if use_info:
                loss_str = 'kl_loss: {:.5f} \n{}'.format(best_metrics['sqrt_then_avg']['kl_loss'], loss_str)
                if use_imu and args.observation_imu_beta != 0: loss_str = 'observation_imu_loss: {:.5f} | {}'.format(best_metrics['sqrt_then_avg']['observation_imu_loss'], loss_str)
                if args.observation_beta != 0: loss_str = 'observation_visual_loss: {:.5f} | {}'.format(best_metrics['sqrt_then_avg']['observation_visual_loss'], loss_str)
            print('best epoch: {} | total_loss: {:.5f} | {}'.format(best_epoch['sqrt_then_avg'], best_metrics['sqrt_then_avg']['total_loss'], loss_str))
            print('rpe_all: {:.5f} | rpe_trans: {:.5f} | rpe_rot_axis: {:.5f} | rpe_rot_euler: {:.5f}'.format(best_metrics['sqrt_then_avg']['rpe_all']['sqrt_then_avg'], best_metrics['sqrt_then_avg']['rpe_trans']['sqrt_then_avg'], best_metrics['sqrt_then_avg']['rpe_rot_axis'], best_metrics['sqrt_then_avg']['rpe_rot_euler']['sqrt_then_avg']))

    writer.export_scalars_to_json('{}{}/writer_scalars.json'.format(args.ckp_dir, args.exp_name))
    writer.close()


def evaluate(args):
    """
    args: see param.py for details
    -> args now loaded from args.txt saved during training except for --gpu and --on_the_fly
    """
    if args.eval_ckp == 'best':
        to_load_ckps = ['ckp_best-rpe-all_sqrt_then_avg.pt'] 
    elif args.eval_ckp == 'last':
        to_load_ckps = ['ckp_latest.pt'] 
    msg_list = []
    # fix eval_batch_size to 1 in evaluating with overlapping clips

    if args.dataset == "euroc":
        # eval_clips = load_euroc_clips(
        #     seqs = args.eval_sequences,
        #     batch_size = args.eval_batch_size, 
        #     shuffle = False,
        #     overlap = False,
        #     clip_len = args.clip_length,
        #     on_the_fly = args.on_the_fly,
        #     ds_type = args.euroc_ds_eval,
        #     t_euler_loss=args.t_euler_loss
        # )

        eval_overlap_clips = load_euroc_clips(
            seqs = args.eval_sequences,
            batch_size = args.eval_batch_size, 
            shuffle = False,
            overlap = True,
            clip_len = args.clip_length,
            on_the_fly = args.on_the_fly,
            ds_type = args.euroc_ds_eval,
            t_euler_loss=args.t_euler_loss
        )
    elif args.dataset == "vkitti2":
        # eval_clips = load_vkitti2_clips(
        #     seqs = args.eval_sequences,
        #     batch_size = args.eval_batch_size, 
        #     shuffle = False,
        #     overlap = False,
        #     clip_len = args.clip_length,
        #     on_the_fly = args.on_the_fly,
        #     vkitti2_clone_only = args.vkitti2_clone_only,
        #     t_euler_loss=args.t_euler_loss,
        #     subscene = args.vkitti2_eval_subscene
        # )

        eval_overlap_clips = load_vkitti2_clips(
            seqs = args.eval_sequences,
            batch_size = args.eval_batch_size, 
            shuffle = False,
            overlap = True,
            clip_len = args.clip_length,
            on_the_fly = args.on_the_fly,
            vkitti2_clone_only = args.vkitti2_clone_only,
            t_euler_loss=args.t_euler_loss,
            subscene = args.vkitti2_eval_subscene
        )
    elif args.dataset == "kitti":
        # eval_clips = load_kitti_clips(
        #     seqs = args.eval_sequences,
        #     batch_size = args.eval_batch_size, 
        #     shuffle = False, 
        #     overlap = False, 
        #     args = args,
        #     sample_size_ratio= 1.
        # )

        eval_overlap_clips = load_kitti_clips(
            seqs = args.eval_sequences,
            batch_size = args.eval_batch_size,
            shuffle = False,
            overlap = True,
            args = args,
            sample_size_ratio=1.
        )
    else:
        raise NotImplementedError()
        

    for _ckp in to_load_ckps:
        msg_list.append('\n========================================')
        msg_list.append("evaluating {} for {}".format(_ckp, args.exp_name))
        msg_list.append('rpe is evaluated over all possible transitions')
        msg_list.append('========================================')
        flownet_model, transition_model, use_imu, use_info, observation_model, observation_imu_model, pose_model, encoder = construct_models(args)

        ckp_path = '{}{}/{}'.format(args.ckp_dir, args.exp_name, _ckp)
        print('loading {}'.format(ckp_path))
        if os.path.isfile(ckp_path):
            # keys: 'model', 'optimizer', 'epoch', 'metrics'
            loaded_dict = torch.load(ckp_path, map_location=torch.device('cpu'))
            transition_model.load_state_dict(loaded_dict['transition_model'], strict=True)
            if observation_model: observation_model.load_state_dict(loaded_dict['observation_model'], strict=True)
            if observation_imu_model: observation_imu_model.load_state_dict(loaded_dict['observation_imu_model'], strict=True)
            pose_model.load_state_dict(loaded_dict['pose_model'], strict=True)
            encoder.load_state_dict(loaded_dict['encoder'], strict=True)
            loaded_metrics = loaded_dict['metrics']
            msg_list.append('loaded epoch: {}'.format(loaded_dict['epoch']))
        else:
            raise ValueError('cannot find the checkpoint {}'.format(ckp_path))

        transition_model = transition_model.to(device=args.device)
        if observation_model: observation_model = observation_model.to(device=args.device)
        if observation_imu_model: observation_imu_model = observation_imu_model.to(device=args.device)
        pose_model = pose_model.to(device=args.device)
        encoder = encoder.to(device=args.device)
        transition_model.eval()
        if observation_model: observation_model.eval()
        if observation_imu_model: observation_imu_model.eval()
        pose_model.eval()
        encoder.eval()
        
        args_eval = {
            'args' : args,
            'flownet_model': flownet_model,
            'transition_model': transition_model, 
            'use_imu': use_imu,
            'use_info': use_info,
            'observation_model': observation_model, 
            'observation_imu_model': observation_imu_model,
            'pose_model': pose_model,
            'encoder': encoder
        }
        # msg_list.append('========================================')
        # msg_list.append('evaluation with non-overlap clips')
        # msg_list.append('========================================')
        # msg_eval = eval_with_clips(eval_clips=eval_clips, **args_eval)
        # msg_list.extend(msg_eval)
        
        msg_list.append('========================================')
        msg_list.append('evaluation with overlap clips and mean predictions over multiple positions in a clip')
        msg_list.append('========================================')
        msg_eval = eval_with_overlap_clips(eval_clips=eval_overlap_clips, **args_eval)
        msg_list.extend(msg_eval)

    with open('eval/eval_{}.log'.format(args.exp_name), mode='w') as f:
        for _msg in msg_list:
            print(_msg)
            f.write('{}\n'.format(_msg))


def evaluate_euroc_interp(args):
    """
    NOTE:
    (1) Load downsample (10/100Hz) data and obtain the corresponding relative poses
    (2) Interpolate 10/100Hz rel_poses to 20/200Hz rel_poses for evaluation
    => We don't use --euroc_ds_eval anymore
    """
    if args.eval_ckp == 'best':
        to_load_ckps = ['ckp_best-rpe-all_sqrt_then_avg.pt'] 
    elif args.eval_ckp == 'last':
        to_load_ckps = ['ckp_latest.pt'] 
    msg_list = []
    # fix eval_batch_size to 1 in evaluating with overlapping clips

    if args.dataset == "euroc":
        eval_overlap_clips = load_euroc_clips(
            seqs = args.eval_sequences,
            batch_size = args.eval_batch_size, 
            shuffle = False,
            overlap = True,
            clip_len = args.clip_length,
            on_the_fly = args.on_the_fly,
            ds_type = "downsample",
            t_euler_loss=args.t_euler_loss
        )
    else:
        raise NotImplementedError()
        

    for _ckp in to_load_ckps:
        msg_list.append('\n========================================')
        msg_list.append("evaluating {} for {}".format(_ckp, args.exp_name))
        msg_list.append('rpe is evaluated over all possible transitions')
        msg_list.append('========================================')
        flownet_model, transition_model, use_imu, use_info, observation_model, observation_imu_model, pose_model, encoder = construct_models(args)

        ckp_path = '{}{}/{}'.format(args.ckp_dir, args.exp_name, _ckp)
        print('loading {}'.format(ckp_path))
        if os.path.isfile(ckp_path):
            # keys: 'model', 'optimizer', 'epoch', 'metrics'
            loaded_dict = torch.load(ckp_path, map_location=torch.device('cpu'))
            transition_model.load_state_dict(loaded_dict['transition_model'], strict=True)
            if observation_model: observation_model.load_state_dict(loaded_dict['observation_model'], strict=True)
            if observation_imu_model: observation_imu_model.load_state_dict(loaded_dict['observation_imu_model'], strict=True)
            pose_model.load_state_dict(loaded_dict['pose_model'], strict=True)
            encoder.load_state_dict(loaded_dict['encoder'], strict=True)
            loaded_metrics = loaded_dict['metrics']
            msg_list.append('loaded epoch: {}'.format(loaded_dict['epoch']))
        else:
            raise ValueError('cannot find the checkpoint {}'.format(ckp_path))

        transition_model = transition_model.to(device=args.device)
        if observation_model: observation_model = observation_model.to(device=args.device)
        if observation_imu_model: observation_imu_model = observation_imu_model.to(device=args.device)
        pose_model = pose_model.to(device=args.device)
        encoder = encoder.to(device=args.device)
        transition_model.eval()
        if observation_model: observation_model.eval()
        if observation_imu_model: observation_imu_model.eval()
        pose_model.eval()
        encoder.eval()
        
        args_eval = {
            'args' : args,
            'flownet_model': flownet_model,
            'transition_model': transition_model, 
            'use_imu': use_imu,
            'use_info': use_info,
            'observation_model': observation_model, 
            'observation_imu_model': observation_imu_model,
            'pose_model': pose_model,
            'encoder': encoder
        }
        # msg_list.append('========================================')
        # msg_list.append('evaluation with non-overlap clips')
        # msg_list.append('========================================')
        # msg_eval = eval_with_clips(eval_clips=eval_clips, **args_eval)
        # msg_list.extend(msg_eval)
        
        msg_list.append('========================================')
        msg_list.append('evaluation with overlap clips and mean predictions over multiple positions in a clip')
        msg_list.append('========================================')
        msg_eval = eval_with_overlap_clips_interp(eval_clips=eval_overlap_clips, **args_eval)
        msg_list.extend(msg_eval)

    with open('eval/{}.log'.format(args.eval_outname), mode='w') as f:
        for _msg in msg_list:
            print(_msg)
            f.write('{}\n'.format(_msg))




def eval_with_clips(args, eval_clips, flownet_model, transition_model, use_imu, use_info, observation_model, observation_imu_model, pose_model, encoder):
    """
    evaluate the model on clips in overlapped eval_clips
    -> each transition might be visited multiple times
    """
    msgs = []
    batch_timer = SequenceTimer()
    last_batch_index = len(eval_clips) - 1
    
    if use_info: global_prior = Normal(torch.zeros(args.eval_batch_size, args.state_size, device=args.device), torch.ones(args.eval_batch_size, args.state_size, device=args.device))
    
    with torch.no_grad():
        list_eval = dict()
        for _met in ['rpe', 'ate']:
            for _suf in ['_all', '_trans', '_rot_axis', '_rot_euler']:
                list_eval['{}{}'.format(_met, _suf)] = []
        for batch_idx, batch_data in enumerate(eval_clips):
            if args.debug and batch_idx >= 10: break
            
            x_img_list, x_imu_list, x_last_rel_pose_list, y_rel_pose_list, y_last_global_pose_list, y_global_pose_list, _, _ = batch_data    

            x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [t, batch, 3, 2, H, W]
            y_rel_poses = torch.stack(y_rel_pose_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [t, batch, 6]
            if use_imu or args.imu_only: 
                x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=args.device)
            running_eval_batch_size = x_img_pairs.size()[1] # might be different at the last batch
            init_state = torch.zeros(running_eval_batch_size, args.state_size, device=args.device)
            init_belief = torch.zeros(running_eval_batch_size, args.belief_size, device=args.device)
            
            observations = x_img_pairs if args.img_prefeat == 'flownet' else bottle(flownet_model, (x_img_pairs, ))
            obs_size = observations.size()
            observations = observations.view(obs_size[0], obs_size[1], -1)

            if use_imu:
                encode_observations = (bottle(encoder, (observations, )), x_imu_seqs)
            elif args.imu_only:
                encode_observations = x_imu_seqs
            else:
                encode_observations = bottle(encoder, (observations, ))
            # with one more return: poses
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, pred_rel_poses = transition_model(
                prev_state=init_state,
                poses=pose_model,
                prev_belief=init_belief,
                observations=encode_observations
            )
            
            for _fidx in range(args.clip_length):
                # (1) evaluate relative pose error (2) no discard_num is used
                eval_rel = eval_rel_error(pred_rel_poses[_fidx], y_rel_poses[_fidx], t_euler_loss=args.t_euler_loss)
                for _met in ['rpe_all', 'rpe_trans', 'rpe_rot_axis', 'rpe_rot_euler']:
                    list_eval[_met].extend(eval_rel[_met])
    
            batch_timer.tictoc()
            remain_time = batch_timer.get_remaining_time(batch_idx, last_batch_index)
            remain_time = '{:.0f}h:{:2.0f}m:{:2.0f}s'.format(remain_time//3600, (remain_time%3600)//60, (remain_time%60))
            
            print('eval: {:4d}/{} | time: {:.3f}s | remaining: {}'.format(batch_idx, last_batch_index, batch_timer.get_last_time_elapsed(), remain_time))
    
    out_eval = dict()
    out_eval['rpe_rot_axis'] = np.mean(np.array(list_eval['rpe_rot_axis']))
    for _met in ['rpe_all', 'rpe_trans', 'rpe_rot_euler']:
        out_eval[_met] = dict()
        out_eval[_met]['sqrt_then_avg'] = np.mean(np.sqrt(np.array(list_eval[_met])))
    msgs.append('rpe_all: {:.5f} | rpe_trans: {:.5f} | rpe_rot_axis: {:.5f} | rpe_rot_euler: {:.5f}'.format(out_eval['rpe_all']['sqrt_then_avg'], out_eval['rpe_trans']['sqrt_then_avg'], out_eval['rpe_rot_axis'], out_eval['rpe_rot_euler']['sqrt_then_avg']))
    return msgs


def eval_with_overlap_clips_interp(args, eval_clips, flownet_model, transition_model, use_imu, use_info, observation_model, observation_imu_model, pose_model, encoder):
    """
    evaluate the model on clips in overlapped eval_clips
    -> each transition might be visited multiple times
    """
    msgs = []
    overlap_pred_rel_poses = dict()
    overlap_gt_rel_poses  = dict()
    loss_avg = dict()
    loss_list = ['total_loss', 'pose_trans_loss', 'pose_rot_loss']
    if use_info: 
        loss_list += ['kl_loss']
        if args.observation_beta != 0: loss_list += ['observation_visual_loss']
        if use_imu and args.observation_imu_beta != 0: loss_list += ['observation_imu_loss']
    for _met in loss_list:
        loss_avg[_met] = RunningAverager()

    batch_timer = SequenceTimer()
    last_batch_index = len(eval_clips) - 1
    
    if use_info: global_prior = Normal(torch.zeros(args.eval_batch_size, args.state_size, device=args.device), torch.ones(args.eval_batch_size, args.state_size, device=args.device))
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_clips):
            if args.debug and batch_idx >= 10: break
            
            x_img_list, x_imu_list, x_last_rel_pose_list, y_rel_pose_list, y_last_global_pose_list, y_global_pose_list, last_time_stamp_list, curr_time_stamp_list = batch_data    

            x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [t, batch, 3, 2, H, W]
            y_rel_poses = torch.stack(y_rel_pose_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [t, batch, 6]
            if use_imu or args.imu_only: 
                x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=args.device)
            running_eval_batch_size = x_img_pairs.size()[1] # might be different at the last batch
            init_state = torch.zeros(running_eval_batch_size, args.state_size, device=args.device)
            init_belief = torch.zeros(running_eval_batch_size, args.belief_size, device=args.device)
            
            observations = x_img_pairs if args.img_prefeat == 'flownet' else bottle(flownet_model, (x_img_pairs, ))
            obs_size = observations.size()
            observations = observations.view(obs_size[0], obs_size[1], -1)

            if use_imu:
                encode_observations = (bottle(encoder, (observations, )), x_imu_seqs)
            elif args.imu_only:
                encode_observations = x_imu_seqs
            else:
                encode_observations = bottle(encoder, (observations, ))
            # with one more return: poses
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, pred_rel_poses = transition_model(
                prev_state=init_state,
                poses=pose_model,
                prev_belief=init_belief,
                observations=encode_observations
            )
            
            if use_info:
                if args.observation_beta != 0:
                    beliefs_visual = beliefs[0] if use_imu else beliefs
                    if args.rec_type == 'posterior':
                        pred_observations = bottle(observation_model, (beliefs_visual, posterior_states, ))
                    elif args.rec_type == 'prior':
                        pred_observations = bottle(observation_model, (beliefs_visual, prior_states, ))
                    if args.rec_loss == 'sum':
                        observation_loss = F.mse_loss(pred_observations, observations, reduction='none').sum(dim=2).mean(dim=(0,1))
                    elif args.rec_loss == 'mean':
                        observation_loss = F.mse_loss(pred_observations, observations, reduction='none').mean(dim=2).mean(dim=(0,1))
                    # observation_loss = args.observation_beta * observation_loss
                
                if use_imu and args.observation_imu_beta != 0:
                    if args.rec_type == 'posterior':
                        pred_imu_observations = bottle(observation_imu_model, (beliefs[1], posterior_states, ))
                    elif args.rec_type == 'prior':
                        pred_imu_observations = bottle(observation_imu_model, (beliefs[1], prior_states, ))
                    if args.rec_loss == 'sum':
                        observation_imu_loss = F.mse_loss(pred_imu_observations, x_imu_seqs.view(pred_imu_observations.size()), reduction='none').sum(dim=2).mean(dim=(0,1)) 
                    elif args.rec_loss == 'mean':
                        observation_imu_loss = F.mse_loss(pred_imu_observations, x_imu_seqs.view(pred_imu_observations.size()), reduction='none').mean(dim=2).mean(dim=(0,1))
                    # observation_imu_loss = args.observation_imu_beta * observation_imu_loss
                    
                kl_loss = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2).mean(dim=(0,1))
                # kl_loss = args.world_kl_beta * kl_loss
                if args.global_kl_beta != 0:
                    if running_eval_batch_size == args.eval_batch_size:
                        # kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0,1))
                        kl_loss += kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0,1))
                    else:
                        tmp_global_prior = Normal(torch.zeros(running_eval_batch_size, args.state_size, device=args.device), torch.ones(running_eval_batch_size, args.state_size, device=args.device))
                        # kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), tmp_global_prior).sum(dim=2).mean(dim=(0,1))
                        kl_loss += kl_divergence(Normal(posterior_means, posterior_std_devs), tmp_global_prior).sum(dim=2).mean(dim=(0,1))
            
            pose_trans_loss = args.translation_weight * F.mse_loss(pred_rel_poses[:,:,:3], y_rel_poses[:,:,:3], reduction='none').sum(dim=2).mean(dim=(0,1))
            pose_rot_loss = args.rotation_weight * F.mse_loss(pred_rel_poses[:,:,3:], y_rel_poses[:,:,3:], reduction='none').sum(dim=2).mean(dim=(0,1))
            
            total_loss = pose_trans_loss + pose_rot_loss
            if use_info: 
                total_loss += kl_loss
                if args.observation_beta != 0: total_loss += observation_loss
                if use_imu and args.observation_imu_beta != 0: total_loss += observation_imu_loss
        
            loss_avg['total_loss'].append(total_loss)
            if use_info:
                if args.observation_beta !=0: loss_avg['observation_visual_loss'].append(observation_loss)
                if use_imu and args.observation_imu_beta != 0: loss_avg['observation_imu_loss'].append(observation_imu_loss)
                loss_avg['kl_loss'].append(kl_loss)
            loss_avg['pose_trans_loss'].append(pose_trans_loss)
            loss_avg['pose_rot_loss'].append(pose_rot_loss)
                    
            for _fidx in range(args.clip_length):
                # ts_key is used for saving multiple-results for one frame-pair due to its different locations in a clip
                # last_time_stamp_list: [time, batch, value]
                for _b in range(running_eval_batch_size):
                    ts_key = '{}-{}'.format(last_time_stamp_list[_fidx][_b], curr_time_stamp_list[_fidx][_b])
                    
                    if ts_key not in overlap_pred_rel_poses.keys():
                        overlap_pred_rel_poses[ts_key] = dict()
                        overlap_gt_rel_poses[ts_key]   = y_rel_poses[_fidx][_b].unsqueeze(0)        # [1, 6]
                    overlap_pred_rel_poses[ts_key][_fidx] = pred_rel_poses[_fidx][_b].unsqueeze(0) # [ [1, 6] ]

            batch_timer.tictoc()
            remain_time = batch_timer.get_remaining_time(batch_idx, last_batch_index)
            remain_time = '{:.0f}h:{:2.0f}m:{:2.0f}s'.format(remain_time//3600, (remain_time%3600)//60, (remain_time%60))
            
            loss_str = '{:.5f}+{:.5f}'.format(pose_trans_loss, pose_rot_loss)
            if use_info: 
                loss_str = '{:.5f}+{}'.format(kl_loss, loss_str)
                if use_imu and args.observation_imu_beta != 0: loss_str = '{:.5f}+{}'.format(observation_imu_loss, loss_str)
                if args.observation_beta !=0: loss_str = '{:.5f}+{}'.format(observation_loss, loss_str)
            print('eval: {:4d}/{} | loss: {:.5f} ({}) | time: {:.3f}s | remaining: {}'.format(batch_idx, last_batch_index, total_loss, loss_str, batch_timer.get_last_time_elapsed(), remain_time))

    ts_keys = overlap_pred_rel_poses.keys()

    # eval_discard_num = args.eval_discard_num
    
    for eval_discard_num in range(2):
        eval_metrics = dict()
        for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_axis', 'rpe_rot_euler']:
            eval_metrics[_metric] = []
        num_start_end = 0

        ## Interpolate the predicted relative poses at 10/100Hz to 20/200Hz
        # assert args.euroc_ds_eval == "downsample"
        interp_rel_poses = {}
        for _key in tqdm(ts_keys):
            # _key: 'last_time_stamp-curr_time_stamp'
            _cnt = len(overlap_pred_rel_poses[_key])
            if _cnt < args.clip_length: 
                num_start_end += 1
                continue
            tmp_gt_rel_pose   = overlap_gt_rel_poses[_key] # [1, 6]
            tmp_pred_rel_pose = torch.cat([v for fidx, v in overlap_pred_rel_poses[_key].items() if fidx >= eval_discard_num], dim=0) # [clip_length - eval_discard_num, 6]
            # discard from being the first in the clip 
            discard_pred_rel_pose = torch.mean(tmp_pred_rel_pose, dim=0)
            discard_pred_rel_pose = discard_pred_rel_pose.unsqueeze(0)  # [1, 6]

            # NOTE: There we go!!! discard_pred_rel_pose is the relative pose for _key: '{}-{}'.format(last_time_stamp_list[_fidx][_b], curr_time_stamp_list[_fidx][_b])
            # NOTE: Let's start from here and doßßß interpolation
            # e.g. _key = 'downsample+MH_04_difficult+1403638129345-downsample+MH_04_difficult+1403638129445'
            _ts0 = int(_key.split("-")[0].split("+")[-1])
            _ts1 = int(_key.split("-")[1].split("+")[-1])
            assert _ts1 - _ts0 == 100
            _ts_middle = _ts0 + 50

            flag = "raw_freq+{}".format(_key.split("-")[0].split("+")[1])
            _key0 = "{}+{}-{}+{}".format(flag, _ts0, flag, _ts_middle)
            _key1 = "{}+{}-{}+{}".format(flag, _ts_middle, flag, _ts1)
            for _k in [_key0, _key1]:
                if _k not in interp_rel_poses.keys():
                    interp_rel_poses[_k] = []

            #NOTE: Get the interpolation of rel_pose (Rotation needs special concern (quaternion usually))
            assert args.t_euler_loss
            assert discard_pred_rel_pose.shape[0] == 1 

            rel_trans = discard_pred_rel_pose[0][:3].cpu().numpy()
            rel_euler = discard_pred_rel_pose[0][3:].cpu().numpy()
            
            #NOTE: r = r * np.pi / 180 (by default rel_rot_euler are angles in degrees rather than radians)
            rel_q1 = euler_to_quaternion(rel_euler, isRad=False)
            rel_q1 = np.quaternion(*rel_q1)
            rel_q0 = np.quaternion(1,0,0,0)
            
            # NOTE: Caveats for euler angle and quaternion conversion
            # (1) np.quaternion.from_euler_angles() does not use [roll, pitch, yaw]!! (See wolfram MathWorld Euler Parameters'definition -> axangles actually!)
            # (2) our utils.tools.euler_to_quaternion() and transforms3d.euler.euler2quat() use [roll, pitch, yaw] and produce the same results
            # (3) rel_euler are angles in degrees!!! While our euler_to_quaternion() will do degree-to-radian conversion inside the function, for correctly running euler.euler2quat() we need: rel_euler / 180 * np.pi manually
            # (4) euler.quat2euler() also results in radian, and the factor *180/np.pi is needed to obtain degrees
            # NOTE: e.g. recover rel_euler in degree: np.array(euler.quat2euler(quaternion.as_float_array(rel_q1))) / np.pi * 180

            half_trans = rel_trans / 2.0
            half_quat = np.slerp_vectorized(rel_q0, rel_q1, 0.5)
            half_euler = np.array(euler.quat2euler(quaternion.as_float_array(half_quat))) / np.pi * 180
            half_rel_pose = np.concatenate([half_trans, half_euler]) # [6,]
            for _k in [_key0, _key1]:
                interp_rel_poses[_k].append(half_rel_pose)
        
        ## Get the ground-truth relative poses at 20/200Hz
        gt_rel_poses = euroc_get_raw_freq_gt(args)
        

        ## Compute the evaluation metrics at 20/200Hz
        print("=> Total number of interpolated relative pose predictions: {}".format(len(interp_rel_poses.keys())))
        for _key in tqdm(interp_rel_poses.keys()):
            assert _key in gt_rel_poses.keys()
            pred_rel = np.stack(interp_rel_poses[_key], axis=0) # [num, 6]
            pred_rel = np.expand_dims(np.mean(pred_rel, axis=0), axis=0) # [1, 6]
            # if pred_rel.shape[0] == 1:
            #     pred_rel = np.expand_dims(pred_rel[0], axis=0) # [1, 6]
            # else:
            #     pred_rel = np.expand_dims(pred_rel[1], axis=0) # [1, 6]
                
            gt_rel = np.expand_dims(gt_rel_poses[_key], 0) # [1, 6]

            # rpe_all, rpe_trans, rpe_axis: np.sum(array ** 2) -> not sqrt yet; rpe_rot: anxis-angle (mode of So3.log())
            assert args.t_euler_loss
            eval_rel = eval_rel_error(pred_rel, gt_rel, t_euler_loss=args.t_euler_loss)
            for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_euler', 'rpe_rot_axis']:
                eval_metrics[_metric].append(np.array(eval_rel[_metric]))
            

        # msgs.append('total number of pairs: {}'.format(len(ts_keys)))
        # msgs.append('invalid number of pairs: {}'.format(num_start_end))
        msgs.append('discard number (from being the first of clips): {}'.format(eval_discard_num))

        tmp_metrics = dict()
        tmp_metrics['rpe_rot_axis'] = np.mean(eval_metrics['rpe_rot_axis'])
        for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_euler']:
            tmp_metrics[_metric] = dict()
            tmp_metrics[_metric]['sqrt_then_avg'] = np.mean(np.sqrt(eval_metrics[_metric]))

        msgs.append('rpe_all: {:.5f} | rpe_trans: {:.5f} | rpe_rot_axis: {:.5f} | rpe_rot_euler: {:.5f}'.format(tmp_metrics['rpe_all']['sqrt_then_avg'], tmp_metrics['rpe_trans']['sqrt_then_avg'], tmp_metrics['rpe_rot_axis'], tmp_metrics['rpe_rot_euler']['sqrt_then_avg']))
    
    return msgs


def euroc_get_raw_freq_gt(args):
    """Get the ground-truth relative poses of euroc at raw_freq 20/200Hz
    """
    # NOTE: If we don't use overlap, the pair between the end and the beginning of adjacent non-overlap clips will be missing
    eval_clips = load_euroc_clips(
            seqs = args.eval_sequences,
            batch_size = args.eval_batch_size, 
            shuffle = False,
            overlap = True,
            clip_len = args.clip_length,
            on_the_fly = args.on_the_fly,
            ds_type = "raw_freq",
            t_euler_loss=args.t_euler_loss
        )
    
    gt_rel_poses  = dict()
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(eval_clips)):
            if args.debug and batch_idx >= 10: break
            
            x_img_list, x_imu_list, x_last_rel_pose_list, y_rel_pose_list, y_last_global_pose_list, y_global_pose_list, last_time_stamp_list, curr_time_stamp_list = batch_data    

            x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [t, batch, 3, 2, H, W]
            y_rel_poses = torch.stack(y_rel_pose_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [t, batch, 6]
            running_eval_batch_size = x_img_pairs.size()[1] # might be different at the last batch

            for _fidx in range(args.clip_length):
                # ts_key is used for saving multiple-results for one frame-pair due to its different locations in a clip
                # last_time_stamp_list: [time, batch, value]
                for _b in range(running_eval_batch_size):
                    ts_key = '{}-{}'.format(last_time_stamp_list[_fidx][_b], curr_time_stamp_list[_fidx][_b])
                    
                    if ts_key not in gt_rel_poses.keys():
                        gt_rel_poses[ts_key] = y_rel_poses[_fidx][_b].cpu().numpy() # [6,]
    return gt_rel_poses


def eval_with_overlap_clips(args, eval_clips, flownet_model, transition_model, use_imu, use_info, observation_model, observation_imu_model, pose_model, encoder):
    """
    evaluate the model on clips in overlapped eval_clips
    -> each transition might be visited multiple times
    """
    msgs = []
    overlap_pred_rel_poses = dict()
    overlap_gt_rel_poses  = dict()
    loss_avg = dict()
    loss_list = ['total_loss', 'pose_trans_loss', 'pose_rot_loss']
    if use_info: 
        loss_list += ['kl_loss']
        if args.observation_beta != 0: loss_list += ['observation_visual_loss']
        if use_imu and args.observation_imu_beta != 0: loss_list += ['observation_imu_loss']
    for _met in loss_list:
        loss_avg[_met] = RunningAverager()

    batch_timer = SequenceTimer()
    last_batch_index = len(eval_clips) - 1
    
    if use_info: global_prior = Normal(torch.zeros(args.eval_batch_size, args.state_size, device=args.device), torch.ones(args.eval_batch_size, args.state_size, device=args.device))
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_clips):
            if args.debug and batch_idx >= 10: break
            
            x_img_list, x_imu_list, x_last_rel_pose_list, y_rel_pose_list, y_last_global_pose_list, y_global_pose_list, last_time_stamp_list, curr_time_stamp_list = batch_data    

            x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [t, batch, 3, 2, H, W]
            y_rel_poses = torch.stack(y_rel_pose_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [t, batch, 6]
            if use_imu or args.imu_only: 
                x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=args.device)
            running_eval_batch_size = x_img_pairs.size()[1] # might be different at the last batch
            init_state = torch.zeros(running_eval_batch_size, args.state_size, device=args.device)
            init_belief = torch.zeros(running_eval_batch_size, args.belief_size, device=args.device)
            
            observations = x_img_pairs if args.img_prefeat == 'flownet' else bottle(flownet_model, (x_img_pairs, ))
            obs_size = observations.size()
            observations = observations.view(obs_size[0], obs_size[1], -1)

            if use_imu:
                encode_observations = (bottle(encoder, (observations, )), x_imu_seqs)
            elif args.imu_only:
                encode_observations = x_imu_seqs
            else:
                encode_observations = bottle(encoder, (observations, ))
            # with one more return: poses
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, pred_rel_poses = transition_model(
                prev_state=init_state,
                poses=pose_model,
                prev_belief=init_belief,
                observations=encode_observations
            )
            
            if use_info:
                if args.observation_beta != 0:
                    beliefs_visual = beliefs[0] if use_imu else beliefs
                    if args.rec_type == 'posterior':
                        pred_observations = bottle(observation_model, (beliefs_visual, posterior_states, ))
                    elif args.rec_type == 'prior':
                        pred_observations = bottle(observation_model, (beliefs_visual, prior_states, ))
                    if args.rec_loss == 'sum':
                        observation_loss = F.mse_loss(pred_observations, observations, reduction='none').sum(dim=2).mean(dim=(0,1))
                    elif args.rec_loss == 'mean':
                        observation_loss = F.mse_loss(pred_observations, observations, reduction='none').mean(dim=2).mean(dim=(0,1))
                    # observation_loss = args.observation_beta * observation_loss
                
                if use_imu and args.observation_imu_beta != 0:
                    if args.rec_type == 'posterior':
                        pred_imu_observations = bottle(observation_imu_model, (beliefs[1], posterior_states, ))
                    elif args.rec_type == 'prior':
                        pred_imu_observations = bottle(observation_imu_model, (beliefs[1], prior_states, ))
                    if args.rec_loss == 'sum':
                        observation_imu_loss = F.mse_loss(pred_imu_observations, x_imu_seqs.view(pred_imu_observations.size()), reduction='none').sum(dim=2).mean(dim=(0,1)) 
                    elif args.rec_loss == 'mean':
                        observation_imu_loss = F.mse_loss(pred_imu_observations, x_imu_seqs.view(pred_imu_observations.size()), reduction='none').mean(dim=2).mean(dim=(0,1))
                    # observation_imu_loss = args.observation_imu_beta * observation_imu_loss
                    
                kl_loss = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2).mean(dim=(0,1))
                # kl_loss = args.world_kl_beta * kl_loss
                if args.global_kl_beta != 0:
                    if running_eval_batch_size == args.eval_batch_size:
                        # kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0,1))
                        kl_loss += kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0,1))
                    else:
                        tmp_global_prior = Normal(torch.zeros(running_eval_batch_size, args.state_size, device=args.device), torch.ones(running_eval_batch_size, args.state_size, device=args.device))
                        # kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), tmp_global_prior).sum(dim=2).mean(dim=(0,1))
                        kl_loss += kl_divergence(Normal(posterior_means, posterior_std_devs), tmp_global_prior).sum(dim=2).mean(dim=(0,1))
            
            pose_trans_loss = args.translation_weight * F.mse_loss(pred_rel_poses[:,:,:3], y_rel_poses[:,:,:3], reduction='none').sum(dim=2).mean(dim=(0,1))
            pose_rot_loss = args.rotation_weight * F.mse_loss(pred_rel_poses[:,:,3:], y_rel_poses[:,:,3:], reduction='none').sum(dim=2).mean(dim=(0,1))
            
            total_loss = pose_trans_loss + pose_rot_loss
            if use_info: 
                total_loss += kl_loss
                if args.observation_beta != 0: total_loss += observation_loss
                if use_imu and args.observation_imu_beta != 0: total_loss += observation_imu_loss
        
            loss_avg['total_loss'].append(total_loss)
            if use_info:
                if args.observation_beta !=0: loss_avg['observation_visual_loss'].append(observation_loss)
                if use_imu and args.observation_imu_beta != 0: loss_avg['observation_imu_loss'].append(observation_imu_loss)
                loss_avg['kl_loss'].append(kl_loss)
            loss_avg['pose_trans_loss'].append(pose_trans_loss)
            loss_avg['pose_rot_loss'].append(pose_rot_loss)
                    
            for _fidx in range(args.clip_length):
                # ts_key is used for saving multiple-results for one frame-pair due to its different locations in a clip
                # last_time_stamp_list: [time, batch, value]
                for _b in range(running_eval_batch_size):
                    ts_key = '{}-{}'.format(last_time_stamp_list[_fidx][_b], curr_time_stamp_list[_fidx][_b])
                    
                    if ts_key not in overlap_pred_rel_poses.keys():
                        overlap_pred_rel_poses[ts_key] = dict()
                        overlap_gt_rel_poses[ts_key]   = y_rel_poses[_fidx][_b].unsqueeze(0)        # [1, 6]
                    overlap_pred_rel_poses[ts_key][_fidx] = pred_rel_poses[_fidx][_b].unsqueeze(0) # [ [1, 6] ]

            batch_timer.tictoc()
            remain_time = batch_timer.get_remaining_time(batch_idx, last_batch_index)
            remain_time = '{:.0f}h:{:2.0f}m:{:2.0f}s'.format(remain_time//3600, (remain_time%3600)//60, (remain_time%60))
            
            loss_str = '{:.5f}+{:.5f}'.format(pose_trans_loss, pose_rot_loss)
            if use_info: 
                loss_str = '{:.5f}+{}'.format(kl_loss, loss_str)
                if use_imu and args.observation_imu_beta != 0: loss_str = '{:.5f}+{}'.format(observation_imu_loss, loss_str)
                if args.observation_beta !=0: loss_str = '{:.5f}+{}'.format(observation_loss, loss_str)
            print('eval: {:4d}/{} | loss: {:.5f} ({}) | time: {:.3f}s | remaining: {}'.format(batch_idx, last_batch_index, total_loss, loss_str, batch_timer.get_last_time_elapsed(), remain_time))

    ts_keys = overlap_pred_rel_poses.keys()

    # eval_discard_num = args.eval_discard_num
    for eval_discard_num in range(2):
        eval_metrics = dict()
        for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_axis', 'rpe_rot_euler']:
            eval_metrics[_metric] = []
        num_start_end = 0
        
        final_pred_rel = dict()
        for _key in tqdm(ts_keys):
            # _key: 'last_time_stamp-curr_time_stamp'
            _cnt = len(overlap_pred_rel_poses[_key])
            if _cnt < args.clip_length: 
                num_start_end += 1
                continue
            tmp_gt_rel_pose   = overlap_gt_rel_poses[_key] # [1, 6]
            tmp_pred_rel_pose = torch.cat([v for fidx, v in overlap_pred_rel_poses[_key].items() if fidx >= eval_discard_num], dim=0) # [clip_length - eval_discard_num, 6]
            # discard from being the first in the clip 
            discard_pred_rel_pose    = torch.mean(tmp_pred_rel_pose, dim=0)
            discard_pred_rel_pose    = discard_pred_rel_pose.unsqueeze(0)                # [1, 6]
            
            final_pred_rel[_key] = discard_pred_rel_pose[0].cpu().numpy()

            # NOTE: There we go!!! discard_pred_rel_pose is the relative pose for _key: '{}-{}'.format(last_time_stamp_list[_fidx][_b], curr_time_stamp_list[_fidx][_b])
            # NOTE: Let's start from here and do interpolation

            # rpe_all, rpe_trans, rpe_axis: np.sum(array ** 2) -> not sqrt yet; rpe_rot: anxis-angle (mode of So3.log())
            eval_rel = eval_rel_error(discard_pred_rel_pose, tmp_gt_rel_pose, t_euler_loss=args.t_euler_loss)
            for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_euler', 'rpe_rot_axis']:
                eval_metrics[_metric].append(np.array(eval_rel[_metric]))
        
        if eval_discard_num == 1:
            # seq 07: 1100 pairs, seq 10: 1200 pairs
            pose_path = "pred_pose/{}/rel".format(args.exp_name)
            if not os.path.isdir(pose_path): os.makedirs(pose_path)
            all_keys = list(final_pred_rel.keys())
            for seq in args.eval_sequences:
                seq_keys = [x for x in all_keys if "-{}-".format(seq) in x]
                seq_keys.sort()
                with open("{}/{}.txt".format(pose_path, seq), "w") as f:
                    for k_ in seq_keys:
                        img1 = int(k_.split("-")[1])
                        img2 = int(k_.split("-")[3])
                        rel_pose = " ".join([str(x) for x in list(final_pred_rel[k_])]) # np.array(6) to str 
                        f.write("{} {} {}\n".format(img1, img2, rel_pose))
                        
                

        # msgs.append('total number of pairs: {}'.format(len(ts_keys)))
        # msgs.append('invalid number of pairs: {}'.format(num_start_end))
        msgs.append('discard number (from being the first of clips): {}'.format(eval_discard_num))

        tmp_metrics = dict()
        tmp_metrics['rpe_rot_axis'] = np.mean(eval_metrics['rpe_rot_axis'])
        for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_euler']:
            tmp_metrics[_metric] = dict()
            tmp_metrics[_metric]['sqrt_then_avg'] = np.mean(np.sqrt(eval_metrics[_metric]))

        msgs.append('rpe_all: {:.5f} | rpe_trans: {:.5f} | rpe_rot_axis: {:.5f} | rpe_rot_euler: {:.5f}'.format(tmp_metrics['rpe_all']['sqrt_then_avg'], tmp_metrics['rpe_trans']['sqrt_then_avg'], tmp_metrics['rpe_rot_axis'], tmp_metrics['rpe_rot_euler']['sqrt_then_avg']))
    

    ## NOTE: Now we disable the evaluation of each position
    # msgs.append('===========================================')
    # for _pos in range(args.clip_length):
    #     eval_metrics = dict()
    #     for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_axis', 'rpe_rot_euler']:
    #         eval_metrics[_metric] = []
    #     num_start_end = 0
    #     for _key in tqdm(ts_keys):
    #         # _key: 'last_time_stamp-curr_time_stamp'
    #         _cnt = len(overlap_pred_rel_poses[_key])
    #         if _cnt < args.clip_length: 
    #             num_start_end += 1
    #             continue
    #         tmp_gt_rel_pose   = overlap_gt_rel_poses[_key] # [1, 6]
    #         pos_pred_rel_pose = overlap_pred_rel_poses[_key][_pos] # [1, 6]

    #         # rpe_all, rpe_trans, rpe_axis: np.sum(array ** 2) -> not sqrt yet; rpe_rot: anxis-angle (mode of So3.log())
    #         eval_rel = eval_rel_error(pos_pred_rel_pose, tmp_gt_rel_pose, t_euler_loss=args.t_euler_loss)
    #         for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_euler', 'rpe_rot_axis']:
    #             eval_metrics[_metric].append(np.array(eval_rel[_metric]))

    #     # msgs.append('total number of pairs: {}'.format(len(ts_keys)))
    #     # msgs.append('invalid number of pairs: {}'.format(num_start_end))
    #     msgs.append('position of the frame-pair in the clip: {}'.format(_pos))

    #     tmp_metrics = dict()
    #     tmp_metrics['rpe_rot_axis'] = np.mean(eval_metrics['rpe_rot_axis'])
    #     for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_euler']:
    #         tmp_metrics[_metric] = dict()
    #         tmp_metrics[_metric]['sqrt_then_avg'] = np.mean(np.sqrt(eval_metrics[_metric]))

    #     msgs.append('rpe_all: {:.5f} | rpe_trans: {:.5f} | rpe_rot_axis: {:.5f} | rpe_rot_euler: {:.5f}'.format(tmp_metrics['rpe_all']['sqrt_then_avg'], tmp_metrics['rpe_trans']['sqrt_then_avg'], tmp_metrics['rpe_rot_axis'], tmp_metrics['rpe_rot_euler']['sqrt_then_avg']))
    
    return msgs




def main():
    param = Param()
    args = param.get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not args.eval and not args.eval_euroc_interp:
        train(args)
    
    if args.eval_euroc_interp:
        assert not args.eval
        evaluate_euroc_interp(args)
    
    if args.eval:
        assert not args.eval_euroc_interp
        evaluate(args)
        

if __name__ == '__main__':
    start_time = timer()
    main()
    running_time = timer() - start_time
    print('==============================5==========')
    print('total running time: {:.0f}h:{:2.0f}m:{:2.0f}s'.format(running_time//3600, (running_time%3600)//60, (running_time%60)))
    print('========================================')



