import os
import pdb
import time
import math
from os.path import *
import numpy as np
import quaternion
import sophus as sp
import torch
from timeit import default_timer as timer

from seq_model import SeqVINet
from info_model import SingleHiddenTransitionModel
from info_model import DoubleHiddenTransitionModel
from info_model import SingleHiddenVITransitionModel
from info_model import DoubleHiddenVITransitionModel
from info_model import MultiHiddenVITransitionModel
from info_model import DoubleStochasticTransitionModel
from info_model import DoubleStochasticVITransitionModel
from info_model import ObservationModel
from info_model import PoseModel
from info_model import Encoder
from flownet_model import FlowNet2
from flownet_model import FlowNet2C
from flownet_model import FlowNet2S


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps=12800):
        self._optimizer = optimizer
        self.param_groups = self._optimizer.param_groups
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def state_dict(self):
        return self._optimizer.state_dict()

    
    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)


    def step(self):
        "Step with the inner optimizer and update learning_rate"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class SequenceTimer:
    def __init__(self):
        self.curr_time = timer()
        self.avg_time = 0 # average running time for one step
        self.cnt_step = 0
        self.last_time_elapsed = 0
    
    def tictoc(self):
        curr_time = timer()
        self.last_time_elapsed = curr_time - self.curr_time
        self.curr_time = curr_time
        self.cnt_step += 1
        self.avg_time += (self.last_time_elapsed - self.avg_time) / self.cnt_step
    
    def get_last_time_elapsed(self):
        return self.last_time_elapsed
    
    def get_remaining_time(self, curr_step, total_step):
        return (total_step - curr_step) * self.avg_time
    

class RunningAverager:
    def __init__(self):
        self.avg = 0
        self.cnt = 0
    
    def append(self, value):
        self.cnt += 1
        self.avg += (value - self.avg) / self.cnt
    
    def item(self):
        return self.avg
    
    def cnt(self):
        return self.cnt


def get_relative_pose(tq_R0, tq_R1, t_euler_loss=False):
    """
    input: T_R0, T_R1: (translation, quaternion): np.array with length 7
    output: T_01: se3: np.array with length 6
    """
    # p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []
    t_R0 = tq_R0[0:3]
    t_R1 = tq_R1[0:3]
    ww0,wx0,wy0,wz0 = tq_R0[3:]
    ww1,wx1,wy1,wz1 = tq_R1[3:]

    q_R0 = np.quaternion(ww0,wx0,wy0,wz0)
    q_R0 = q_R0.normalized()
    q_R1 = np.quaternion(ww1,wx1,wy1,wz1)
    q_R1 = q_R1.normalized()
    t_R0_q = np.quaternion(0, t_R0[0], t_R0[1], t_R0[2])
    t_R1_q = np.quaternion(0, t_R1[0], t_R1[1], t_R1[2])

    # q_10 = q_R1.inverse() * q_R0 
    # t_10_q = q_R1.inverse() * (t_R0_q - t_R1_q) * q_R1
    # t_10 = t_10_q.imag
    # tq_10 = [*t_10, *quaternion.as_float_array(q_10)]
    # se_10 = pose_tq_to_se(tq_10)

    q_01 = q_R0.inverse() * q_R1 
    t_01_q = q_R0.inverse() * (t_R1_q - t_R0_q) * q_R0
    t_01 = t_01_q.imag
    tq_01 = [*t_01, *quaternion.as_float_array(q_01)]
    if t_euler_loss:
        se_01_obj = pose_tq_to_se(tq_01, return_obj=True)
        out_t = np.array(se_01_obj.t, dtype=float).squeeze()
        out_euler = rotationMatrixToEulerAngles(np.array(se_01_obj.so3.matrix(), dtype=float))
        out_euler = out_euler / np.pi * 180 # in degrees
        se_01 = np.concatenate((out_t, out_euler))
    else:
        se_01 = pose_tq_to_se(tq_01)

    return np.array(se_01)


def rotationMatrixToQuaternion(R) :
    # assert(isRotationMatrix(R))
    euler_radian = rotationMatrixToEulerAngles(R)
    return euler_to_quaternion(euler_radian, isRad=True)



# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-3
 
 
# Calculates rotation matrix to euler angles
# The result is for ZYX euler angles
def rotationMatrixToEulerAngles(R) :
    # assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


def euler_to_quaternion(r, isRad=False):
    if not isRad:
        # By default, isRad is False => r is euler angles in degrees!
        r = r * np.pi / 180
    (yaw, pitch, roll) = (r[2], r[1], r[0])
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw, qx, qy, qz = makeFirstPositive(qw, qx, qy, qz)
    return np.array([qw, qx, qy, qz])


def normalize_imgfeat(img_pair, rgb_max):
    # input -> img_pair: [batch, 3, 2, h, w]
    # output -> img_feature: [batch, 6, 480, 752]
    rgb_mean = img_pair.contiguous().view(img_pair.size()[:2]+(-1,)).mean(dim=-1).view(img_pair.size()[:2] + (1,1,1,)) # [batch, 3, 1, 1, 1]
    img_pair = (img_pair - rgb_mean) / rgb_max # [batch, 3, 2, 480, 752], normalized
    x1 = img_pair[:,:,0,:,:]
    x2 = img_pair[:,:,1,:,:]
    img_features = torch.cat((x1,x2), dim = 1) # [batch, 6, 480, 752]
    return img_features


def makeFirstPositive(ww, wx, wy, wz):
    """
    make the first non-zero element in q positive
    q_array = [ww, wx, wy, wz]
    """
    q_array = [ww, wx, wy, wz]
    for q_ele in q_array:
        if q_ele == 0:
            continue
        if q_ele < 0:
            q_array = [-x for x in q_array]
        break
    return q_array[0], q_array[1], q_array[2], q_array[3]


def pose_tq_to_se(pose_tq, return_obj=False):
    """
    transform pose representation from (translation, quaternion) to se3
    pose_tq: [x, y, z, ww, wx, wy, wz]
    """
    trans = sp.Vector3(*pose_tq[0:3])
    ww, wx, wy, wz = makeFirstPositive(*pose_tq[3:])
    
    q_real = ww
    q_img = sp.Vector3(wx, wy, wz)
    q = sp.Quaternion(q_real,q_img)
    R = sp.So3(q)
    
    # SO3 to SE3 to se3
    Se3Obj = sp.Se3(R, trans)
    numpy_vec = np.array(Se3Obj.log()).squeeze().astype(float) # squeeze(): [6, 1] -> [6,]
    if return_obj:
        return Se3Obj
    return numpy_vec


def globalToTEuler(tq_R):
    """
    input: T_R0, T_R1: [timestamp, (translation, quaternion): np.array with length 7]
    output: T_01: [timestamp, t and euler]
    """
    timestamp = tq_R[0] 
    se_obj = pose_tq_to_se(tq_R[1], return_obj=True)
    out_t = np.array(se_obj.t, dtype=float).squeeze()
    out_euler = rotationMatrixToEulerAngles(np.array(se_obj.so3.matrix(), dtype=float))
    out_euler = out_euler / np.pi * 180 # in degrees
    return np.array([timestamp, np.concatenate((out_t, out_euler))])


def get_zero_se3():
    """
    output: se3 vector6 for zero movement
    """
    zero_q = sp.Quaternion(1, sp.Vector3(0,0,0))
    RT = sp.Se3(sp.So3(zero_q), sp.Vector3(0,0,0))
    numpy_vec = np.array(RT.log()).astype(float)
    return np.concatenate(numpy_vec)


def mean_Se3(list_Se3):
    """
    input: a list of Se3 object
    return: a Se3 object averaged over Se3.log() 
    """
    tmp_sum = list_Se3[0].log()
    for _comp in list_Se3[1:]:
        tmp_sum += _comp.log()
    tmp_sum /= len(list_Se3)
    return sp.Se3.exp(tmp_sum)


def noise_add_(x_data, noise_std_factor, device):
    # default for args.noise_std_factor: 0.1
    std = x_data.std()
    if abs(std - 0.) < 1e-5:
        std = 1.
    # std = 1.
    noise = torch.empty_like(x_data).normal_(mean=x_data.mean(), std=noise_std_factor * std)
    noise = noise.type(torch.FloatTensor).to(device)
    tmp = x_data + noise
    tmp = tmp.to(device)
    return tmp


def noise_zero_(x_data, noise_std_factor, device):
    # default for args.noise_std_factor: 0.1
    std = x_data.std()
    if abs(std - 0.) < 1e-5:
        std = 1.
    # std = 1.
    noise = torch.empty_like(x_data).normal_(mean=x_data.mean(), std=noise_std_factor * std)
    noise = noise.type(torch.FloatTensor).to(device)
    return noise


def save_model(path, transition_model, pose_model, encoder, optimizer, epoch, metrics, observation_model=None, observation_imu_model=None):
    states = {
        'transition_model': transition_model.state_dict(),
        'pose_model': pose_model.state_dict(),
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch, 
        'metrics': metrics
    }
    if observation_model:
        states['observation_model'] = observation_model.state_dict()
    if observation_imu_model: 
        states['observation_imu_model'] = observation_imu_model.state_dict()
    torch.save(states, path)


def construct_models(args):
    # check the validity of soft / hard deepvio
    if args.soft or args.hard:
        if args.transition_model not in  ['deepvio', 'double-vinet']: 
            raise ValueError('--soft and --hard must be used with deepvio or double-vinet')
    
    # construct flownet_model
    flownet_model = None
    if args.flownet_model != 'none' and args.img_prefeat == 'none':
        if args.train_img_from_scratch:
            raise ValueError('if --flownet_model -> --train_img_from_scratch should not be used')
        if args.flownet_model == 'FlowNet2':
            flownet_model = FlowNet2(args).to(device=args.device)
        elif args.flownet_model == 'FlowNet2S':
            flownet_model = FlowNet2S(args).to(device=args.device)
        elif args.flownet_model == 'FlowNet2C':
            flownet_model = FlowNet2C(args).to(device=args.device)
        else:
            raise ValueError('--flownet_model: {} is not supported'.format(args.flownet_model))
        resume_ckp = 'flownet/pretrained_models/{}_checkpoint.pth.tar'.format(args.flownet_model)
        flow_ckp = torch.load(resume_ckp)
        flownet_model.load_state_dict(flow_ckp['state_dict'])
        flownet_model.eval()
    
    base_args = {
        'args': args,
        'belief_size': args.belief_size,
        'state_size': args.state_size,
        'hidden_size': args.hidden_size,
        'embedding_size': args.embedding_size,
        'activation_function': args.activation_function
    }
    use_imu = False
    use_info = False
    if args.transition_model == 'deepvo':
        transition_model = SeqVINet(use_imu=False, **base_args).to(device=args.device)
    elif args.transition_model == 'deepvio':
        transition_model = SeqVINet(use_imu=True, **base_args).to(device=args.device)
        use_imu = True
    elif args.transition_model == 'single':
        transition_model = SingleHiddenTransitionModel(**base_args).to(device=args.device)
        use_info = True
    elif args.transition_model == 'double':
        transition_model = DoubleHiddenTransitionModel(**base_args).to(device=args.device)
        use_info = True
    elif args.transition_model == 'double-stochastic':
        transition_model = DoubleStochasticTransitionModel(**base_args).to(device=args.device)
        use_info = True
    elif args.transition_model == 'single-vinet':
        transition_model = SingleHiddenVITransitionModel(**base_args).to(device=args.device)
        use_imu = True
        use_info = True
    elif args.transition_model == 'double-vinet':
        transition_model = DoubleHiddenVITransitionModel(**base_args).to(device=args.device)
        use_imu = True
        use_info = True
    elif args.transition_model == 'double-vinet-stochastic':
        transition_model = DoubleStochasticVITransitionModel(**base_args).to(device=args.device)
        use_imu = True
        use_info =True
    elif args.transition_model == 'multi-vinet':
        transition_model = MultiHiddenVITransitionModel(**base_args).to(device=args.device)
        use_imu = True
        use_info = True
    observation_model = None
    observation_imu_model = None
    if args.transition_model not in ['deepvo', 'deepvio']:
        if args.observation_beta !=0: observation_model = ObservationModel(symbolic=True, observation_type='visual', **base_args).to(device=args.device)
        if use_imu and args.observation_imu_beta != 0:
            observation_imu_model = ObservationModel(symbolic=True, observation_type='imu', **base_args).to(device=args.device)
    pose_model = PoseModel(**base_args).to(device=args.device)
    encoder = Encoder(symbolic=True, **base_args).to(device=args.device)
    return flownet_model, transition_model, use_imu, use_info, observation_model, observation_imu_model, pose_model, encoder
    

def eval_rel_error(pred_rel_pose, gt_rel_pose, t_euler_loss):
    """
    pred_rel_pose: predicted se3R6_01 [batch, 6] -> se3 or t_euler
    gt_rel_pose: from se3R6_01 [batch, 6] -> se3 or t_euler
    return: 
    -> rpe_all, rpe_trans: [batch] np.sum(array ** 2) -> not sqrt yet
    -> rpe_rot_axis: [batch] anxis-angle (mode of So3.log())
    -> rpe_rot_euler: [batch] np.sum(array ** 2) -> not sqrt yet

    -> v1: from TT'
    -> v2: from ||t-t'||, ||r-r'|| (disabled)
    """
    assert pred_rel_pose.shape[0] == gt_rel_pose.shape[0]
    batch_size = pred_rel_pose.shape[0]
    eval_rel = dict()
    for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_axis', 'rpe_rot_euler']:
        eval_rel[_metric] = [] 

    for _i in range(batch_size):
        if type(gt_rel_pose) == np.ndarray:
            tmp_pred_rel_pose = pred_rel_pose[_i]
            tmp_gt_rel_pose   = gt_rel_pose[_i]
        else:
            tmp_pred_rel_pose = pred_rel_pose[_i].cpu().numpy()
            tmp_gt_rel_pose   = gt_rel_pose[_i].cpu().numpy()

        if t_euler_loss:
            # transform t_euler to se3
            tmp_pred_t          = tmp_pred_rel_pose[0:3]
            tmp_pred_euler      = tmp_pred_rel_pose[3:]
            tmp_pred_quat       = euler_to_quaternion(tmp_pred_euler)
            tmp_pred_rel_pose   = pose_tq_to_se(np.concatenate((tmp_pred_t, tmp_pred_quat)))

            tmp_gt_t            = tmp_gt_rel_pose[0:3]
            tmp_gt_euler        = tmp_gt_rel_pose[3:]
            tmp_gt_quat         = euler_to_quaternion(tmp_gt_euler)
            tmp_gt_rel_pose     = pose_tq_to_se(np.concatenate((tmp_gt_t, tmp_gt_quat)))

        se_pred = sp.Vector6(*list(tmp_pred_rel_pose))
        se_gt = sp.Vector6(*list(tmp_gt_rel_pose))
        T_01_pred = sp.Se3.exp(se_pred)
        T_01_gt = sp.Se3.exp(se_gt)

        ## calculate eval_rel for v1 (from TT')
        T_01_rel = T_01_gt.inverse() * T_01_pred # Se3 object
        tmp_rpe_all = np.sum(np.array(T_01_rel.log(), dtype=float)**2)  # (4.46) in SLAM12
        tmp_rpe_trans = np.sum(np.array(T_01_rel.t, dtype=float)**2)    # (4.47) in SLAM12
        eval_rel['rpe_all'].append(np.array(tmp_rpe_all))
        eval_rel['rpe_trans'].append(np.array(tmp_rpe_trans))

        axis_01_rel = np.linalg.norm(np.array(T_01_rel.so3.log(), dtype=float)) 
        axis_01_rel = axis_01_rel / np.pi * 180                         # transform to degrees
        eval_rel['rpe_rot_axis'].append(np.array(axis_01_rel))

        euler_01_rel = np.array(T_01_rel.so3.matrix(), dtype=float)
        euler_01_rel = rotationMatrixToEulerAngles(euler_01_rel)
        euler_01_rel = euler_01_rel / np.pi * 180                       # transform to degrees
        tmp_euler = np.sum(euler_01_rel**2)
        eval_rel['rpe_rot_euler'].append(tmp_euler)

    # each value in eval_rel: a list  with length eval_batch_size
    return eval_rel


def eval_global_error(accu_global_pose, gt_global_pose):
    """
    input: (list -> batch)
    -> accu_global_pose: list of sp.Se3 Object
    -> gt_global_pose: list of (translation, quaternion) with length 7
    return: 
    -> ate_all, ate_trans [batch] -> np.sum(array ** 2)
    -> ate_rot_axis: [batch] axis-angle
    -> ate_rot_euler: [batch] -> np.sum(array ** 2)

    -> v1: from TT'
    -> v2: from ||t-t'||, ||r-r'|| (disabled)
    """
    assert len(accu_global_pose) == len(gt_global_pose)
    batch_size = len(accu_global_pose)

    eval_global = dict()
    for _metric in ['ate_all', 'ate_trans', 'ate_rot_axis', 'ate_rot_euler']:
        eval_global[_metric] = []
            
    for _i in range(batch_size):
        T_R1_pred = accu_global_pose[_i]
        T_R1_gt = pose_tq_to_se(gt_global_pose[_i], return_obj=True)

        ## calculate v1: from TT'
        T_R1_rel        = T_R1_gt.inverse() * T_R1_pred
        tmp_ate_all     = np.sum(np.array(T_R1_rel.log(), dtype=float)**2)  # (4.46) in SLAM12
        tmp_ate_trans   = np.sum(np.array(T_R1_rel.t, dtype=float)**2)      # (4.47) in SLAM12
        eval_global['ate_all'].append(np.array(tmp_ate_all))
        eval_global['ate_trans'].append(np.array(tmp_ate_trans))

        axis_R1_rel     = np.linalg.norm(np.array(T_R1_rel.so3.log(), dtype=float))
        axis_R1_rel     = axis_R1_rel / np.pi * 180                         # transform to degrees
        eval_global['ate_rot_axis'].append(np.array(axis_R1_rel))

        euler_R1_rel    = np.array(T_R1_rel.so3.matrix(), dtype=float)
        euler_R1_rel    = rotationMatrixToEulerAngles(euler_R1_rel)
        euler_R1_rel    = euler_R1_rel / np.pi * 180                        # transform to degrees
        tmp_euler       = np.sum(euler_R1_rel**2)
        eval_global['ate_rot_euler'].append(tmp_euler)

    # each value in eval_global: a list with length eval_batch_size
    return eval_global


def get_lr(optimizer):
    """
    currently only support optimizer with one param group
    -> please use multiple optimizers separately for multiple param groups
    """
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    assert len(lr_list) == 1
    return lr_list[0]


def factor_lr_schedule(epoch, divide_epochs=[], lr_factors=[]):
    """
    -> divide_epochs need to be sorted
    -> divide_epochs and lr_factors should be one-to-one matched
    """
    assert len(divide_epochs) == len(lr_factors)
    tmp_lr_factors = [1.] + lr_factors
    for _i, divide_epoch in enumerate(divide_epochs):
        idx = _i 
        if epoch < divide_epoch: 
            break
        idx += 1
    return tmp_lr_factors[idx]


