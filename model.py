import os
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import normalize_imgfeat
from subnets import ImgEncoder
from subnets import img_conv

def get_prefeat_size(dataset, flow_model):
    """
    for input image size rescaled to [320, 1216] for kitti
    """
    prefeat_size = dict()
    if dataset == 'kitti': 
        if flow_model in ['FlowNet2C', 'FlowNet2S']:
            prefeat_size['out_conv6']             = [1024, 5,  19]
            prefeat_size['out_conv6_1']           = [1024, 5,  19]
            prefeat_size['flow6']                 = [2,    5,  19]
            prefeat_size['flow6_up']              = [2,    10, 38]
            prefeat_size['concat2']               = [194,  80, 304]
            prefeat_size['flow2']                 = [2,    80, 304]
        elif flow_model == 'FlowNet2':
            prefeat_size['flownetc_flow2']        = [2,    80,  304]
            prefeat_size['concat1']               = [12,   320, 1216]
            prefeat_size['flownets1_flow2']       = [2,    80,  304]
            prefeat_size['concat2']               = [12,   320, 1216]
            prefeat_size['flownets2_flow']        = [2,    80,  304]
            prefeat_size['concat3']               = [11,   320, 1216]
            prefeat_size['fusion_out_conv2']      = [128,  80,  304]
            prefeat_size['fusion_flow2']          = [2,    80,  304]
            prefeat_size['fusion_concat0']        = [82,   320, 1216]
            prefeat_size['fusion_out_interconv0'] = [16,   320, 1216]
    elif dataset == 'euroc':
        if flow_model in ['FlowNet2C', 'FlowNet2S']:
            prefeat_size['out_conv6']             = [1024, 7,   11]
            prefeat_size['out_conv6_1']           = [1024, 7,   11]
            prefeat_size['flow6']                 = [2,    7,   11]
            prefeat_size['flow6_up']              = [2,    14,  22]
            prefeat_size['concat2']               = [194,  112, 176]
            prefeat_size['flow2']                 = [2,    112, 176]
        elif flow_model == 'FlowNet2':
            prefeat_size['flownetc_flow2']        = [2,    112, 176]
            prefeat_size['concat1']               = [12,   448, 704]
            prefeat_size['flownets1_flow2']       = [2,    112, 176]
            prefeat_size['concat2']               = [12,   448, 704]
            prefeat_size['flownets2_flow']        = [2,    112, 176]
            prefeat_size['concat3']               = [11,   448, 704]
            prefeat_size['fusion_out_conv2']      = [128,  112, 176]
            prefeat_size['fusion_flow2']          = [2,    112, 176]
            prefeat_size['fusion_concat0']        = [82,   448, 704]
            prefeat_size['fusion_out_interconv0'] = [16,   448, 704]
    else:
        raise ValueError('dataset {} is currently not supported'.format(dataset))

    return prefeat_size


class MLPEnvModel(nn.Module):
    def __init__(self, args, batch_norm=True):
        """
        args: see param.py for details
        Predict the next state (img, imu, etc.) from previous states and actions/poses
        s'_{t+1} = f(s_t, p_t, p_{t+1})
        """
        super(MLPEnvModel, self).__init__()
        
        # get all args and specify which sensor to use in the model
        self.args = args

        self.fused_feat_size = 0
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        
        self.batch_norm = batch_norm
        self.img_feat_size = args.env_imgfeat_size # 1024
        
        if self.args.flownet_model in ['FlowNet2', 'FlowNet2C', 'FlowNet2S']: 
            # use pretrained features
            prefeat_size      = get_prefeat_size(self.args.dataset, self.args.flownet_model)[self.args.prefeat_type]
            self.img_enc_size = prefeat_size[0]  # e.g. 1024
            self.feat_mapsize = {self.args.dataset: prefeat_size[1] * prefeat_size[2]} # e.g. 5 x 19 or 7 x 11
        elif self.args.flownet_model == 'none':
            raise NotImplementedError('use train-from-scratch image encoder for env model')
        else:
            raise ValueError('one and only one of --train_img_from_scratch and --flownet_model should be given')

        # use flattened flownet features
        self.img_conv1 = img_conv(self.batch_norm, self.img_enc_size, self.img_enc_size//2, kernel_size=1)
        self.img_conv2 = img_conv(self.batch_norm, self.img_enc_size//2, self.img_enc_size//2, kernel_size=3)
        self.img_conv3 = img_conv(self.batch_norm, self.img_enc_size//2, self.img_enc_size, kernel_size=1)
        
        self.img_prefeat_size = self.img_enc_size * self.feat_mapsize[self.args.dataset]
        self.img_fc = nn.Linear(self.img_prefeat_size, self.img_feat_size)
        self.fused_feat_size += self.img_feat_size

        # use tiled pose or use a fc layer to encode pose
        # both last_pose and curr_pose share this part
        self.use_pose_fc = self.args.last_pose_fc
        self.pose_tiles  = self.args.last_pose_tiles
        self.pose_hidden_size = self.args.last_pose_hidden_size
        
        if self.use_pose_fc:
            self.pose_fc    = nn.Linear(6 * self.pose_tiles, self.pose_hidden_size)
        self.pose_feat_size = self.pose_hidden_size if self.use_pose_fc else 6 * self.pose_tiles
        self.fused_feat_size += 2 * self.pose_feat_size
        
        self.rnn_fusion = nn.LSTM(
            input_size  = self.fused_feat_size,
            hidden_size = self.args.fused_lstm_hidden_size, # 1024
            num_layers  = 2,
            batch_first = True
        )

        self.env_fc_1 = nn.Linear(self.args.fused_lstm_hidden_size, self.img_enc_size//2)
        self.env_fc_2 = nn.Linear(self.img_enc_size//2, self.img_prefeat_size)

        # initialize module weights
        self.init_weights()


    def forward(self, image_pair=None, last_pose=None, curr_pose=None,fused_lstm_hidden=None):
        """
        image_pair (s_t): [batch, feat_channels, fH, fW] if use --flownet_model
        last_pose (p_t):  [batch, 6] 
        curr_pose (p_{t+1})
        """
        fused_feat = []
        
        # [batch, 3, 2, H, W]
        if image_pair is None: raise ValueError('image_pair must be given')
        batch_size = image_pair.size()[0]
        img_feat = image_pair 
        img_feat = self.img_conv1(img_feat)
        img_feat = self.img_conv2(img_feat)
        img_feat = self.img_conv3(img_feat)

        # # use flattened flownet features to extract image features
        img_feat = img_feat.contiguous().view(batch_size, -1)
        img_feat = self.relu(self.img_fc(img_feat))
        fused_feat.append(img_feat)
    
        # get last pose embedding by repeat or an embedding layer
        # [batch, 6 * pose_embedding_tiles] # -> [batch, last_pose_hidden_size]
        if last_pose is None: raise ValueError('last_pose must be given')
        last_pose_feat = last_pose.repeat(1, self.pose_tiles) 
        if self.use_pose_fc:
            last_pose_feat = self.pose_fc(last_pose_feat)
            last_pose_feat = self.relu(last_pose_feat)
        fused_feat.append(last_pose_feat)
        
        if curr_pose is None: raise ValueError('curr_pose must be given')
        curr_pose_feat = curr_pose.repeat(1, self.pose_tiles) 
        if self.use_pose_fc:
            curr_pose_feat = self.pose_fc(curr_pose_feat)
            curr_pose_feat = self.relu(curr_pose_feat)
        fused_feat.append(curr_pose_feat)
        
        # [batch, 1, (img_feat_size) + (imu_feat_size) + (last_pose_feat_size)]
        fused_feat = torch.cat(fused_feat, 1) 
        fused_feat = fused_feat.unsqueeze(1) 
        
        # r_out: [batch, 1, fused_lstm_hidden_size]
        # self.rnn_fusion_state = (h_n, h_c)
        # h_n: [2, batch, fused_lstm_hidden_size]
        # h_c: [2, batch, fused_lstm_hidden_size]
        # note r_out[:, -1, :] is equal to h_n[-1, :, :]
        fused_feat, fused_lstm_hidden = self.rnn_fusion(fused_feat, fused_lstm_hidden)
        fused_hidden_states_out = fused_lstm_hidden

        pred_feat = self.relu(self.env_fc_1(fused_feat[:, -1, :]))
        pred_feat = self.env_fc_2(pred_feat)

        return pred_feat, fused_hidden_states_out


    def init_weights(self):
        """
        follow a deepvo pytorch implementation gitub repo
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                nn.init.xavier_normal_(m.weight.data)

            elif isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                #     m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTMCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        # nn.init.orthogonal(param)
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        # Forget gate bias trick: Initially during training, it is often helpful
                        # to initialize the forget gate bias to a large value, to help information
                        # flow over longer time steps.
                        # In a PyTorch LSTM, the biases are stored in the following order:
                        # [ b_ig | b_fg | b_gg | b_og ]
                        # where, b_ig is the bias for the input gate, 
                        # b_fg is the bias for the forget gate, 
                        # b_gg (see LSTM docs, Variables section), 
                        # b_og is the bias for the output gate.
                        # So, we compute the location of the forget gate bias terms as the 
                        # middle one-fourth of the bias vector, and initialize them.
                        
                        # First initialize all biases to zero
                        # nn.init.uniform_(param)
                        nn.init.constant_(param, 0.)
                        bias = getattr(m, name)
                        n = bias.size(0)
                        start, end = n // 4, n // 2
                        bias.data[start:end].fill_(10.)



class VINet(nn.Module):
    def __init__(self, args):
        """
        args: see param.py for details
        """
        super(VINet, self).__init__()
        
        # get all args and specify which sensor to use in the model
        self.args = args
        self.use_img  = 'img' in args.sensors
        self.use_imu  = 'imu' in args.sensors
        self.use_pose = 'pose' in args.sensors

        self.fused_feat_size = 0
        self.relu = nn.LeakyReLU(0.1,inplace=True)

        if self.use_img:
            self.img_feat_size = args.img_hidden_size
            
            if self.args.train_img_from_scratch:
                print('-> train image encoder from scratch')
                self.img_encoder  = ImgEncoder(self.args, batch_norm=self.args.img_batch_norm) 
                self.img_enc_size = 256
                self.feat_mapsize = {'kitti': 2 * 5, 'euroc': 4 * 6} # might need to be updated
            elif self.args.flownet_model in ['FlowNet2', 'FlowNet2C', 'FlowNet2S']: 
                # use pretrained features
                prefeat_size = get_prefeat_size(self.args.dataset, self.args.flownet_model)[self.args.prefeat_type]
                self.img_enc_size = prefeat_size[0]  # e.g. 1024
                self.feat_mapsize = {self.args.dataset: prefeat_size[1] * prefeat_size[2]} # e.g. 5 x 19 or 7 x 11
            else:
                raise ValueError('one and only one of --train_img_from_scratch and --flownet_model should be given')

            if self.args.imgfeat_mode == 'flatten':
                # use flattened flownet features
                self.img_prefeat_size = self.img_enc_size * self.feat_mapsize[self.args.dataset] 
                if self.args.direct_img:
                    self.img_feat_size = self.img_prefeat_size
                else:
                    self.img_fc = nn.Linear(self.img_prefeat_size, self.img_feat_size)
            elif self.args.imgfeat_mode == 'pooling':
                # use channel/spatial pooling
                base_filter = 128
                self.channel_lower_fc_1 = nn.Linear(self.img_enc_size, base_filter * 2)
                self.channel_lower_fc_2 = nn.Linear(base_filter * 2, base_filter)
                self.channel_upper_fc_1 = nn.Linear(self.img_enc_size + base_filter, base_filter * 2)
                self.channel_upper_fc_2 = nn.Linear(base_filter * 2, base_filter)
                self.spatial_fc_1 = nn.Linear(2*self.feat_mapsize[self.args.dataset]+base_filter, base_filter * 2) 
                self.spatial_fc_2 = nn.Linear(base_filter * 2, base_filter) 
                self.img_fc = nn.Linear(base_filter, self.img_feat_size)     
            else:
                raise ValueError('--imgfeat_mode {} is not supported'.format(self.args.imgfeat_mode))

            self.fused_feat_size    += self.img_feat_size         

        if self.use_imu:
            self.imu_feat_size      = self.args.imu_lstm_hidden_size
            self.fused_feat_size   += self.imu_feat_size
            self.rnn_imu = nn.LSTM(
                input_size  = 6, 
                hidden_size = self.imu_feat_size,
                num_layers  = 2,
                batch_first = True
            )

        if self.use_pose:
            if self.args.last_pose_fc:
                self.last_pose_fc = nn.Linear(6 * self.args.last_pose_tiles, self.args.last_pose_hidden_size)
            self.last_pose_feat_size = self.args.last_pose_hidden_size if self.args.last_pose_fc else 6 * self.args.last_pose_tiles
            self.fused_feat_size    += self.last_pose_feat_size

        self.rnn_fusion = nn.LSTM(
            input_size  = self.fused_feat_size,
            hidden_size = self.args.fused_lstm_hidden_size,
            num_layers  = 2,
            batch_first = True
        )

        self.pose_fc_1 = nn.Linear(self.args.fused_lstm_hidden_size, 128)
        self.pose_fc_2 = nn.Linear(128, 6)

        # initialize module weights
        self.init_weights()


    def forward(self, image_pair=None, imu_seq=None, last_pose=None, imu_lstm_hidden=None, fused_lstm_hidden=None):
        """
        image_pair: [batch, 3, 2, H, W] if use --train_img_from_scratch
                    [batch, feat_channels, fH, fW] if use --flownet_model
        imu_seq:    [batch, seq=11, 6]
        last_pose:  [batch, 6] # could be relateive: [batch, 6] or global: [batch, 7] 
        """
        fused_feat = []
        hidden_states_out = dict()
        if self.use_img:
            # [batch, 3, 2, H, W]
            if image_pair is None: raise ValueError('image_pair must be given when sensors include img')
            batch_size = image_pair.size()[0]
            img_feat = image_pair 
            if self.args.train_img_from_scratch:
                # [batch, 3, 2, H, W] -> [batch, 6, H, W]
                # img_feat = normalize_imgfeat(img_feat, rgb_max=self.args.rgb_max)
                img_feat = torch.cat((img_feat[:,:,0,:,:], img_feat[:,:,1,:,:]), dim = 1) # [batch, 6, 480, 752]
                img_feat = self.img_encoder(img_feat) # [batch, 256, 2, 5] for resized kitti [192, 640]

            if self.args.imgfeat_mode == 'flatten':
                # # use flattened flownet features to extract image features
                img_feat = img_feat.contiguous().view(batch_size, -1)
                if not self.args.direct_img:
                    img_feat = self.relu(self.img_fc(img_feat))   
            elif self.args.imgfeat_mode == 'pooling':
                # # use channel/spatial pooling to extract image features
                img_spatial_avg = img_feat.mean(1).contiguous().view(batch_size, -1)                # [batch, 6x20]
                img_spatial_max = img_feat.max(1)[0].contiguous().view(batch_size, -1)              # [batch, 6x20]
                img_channel_avg_upper = img_feat.mean(3).mean(2)                                    # [batch, enc_size]
                img_channel_avg_lower = img_feat.mean(3).mean(2)                                    # [batch, enc_size]
                img_channel_avg_lower = self.relu(self.channel_lower_fc_1(img_channel_avg_lower))
                img_channel_avg_lower = self.relu(self.channel_lower_fc_2(img_channel_avg_lower))
                img_channel_avg_upper = torch.cat((img_channel_avg_upper,img_channel_avg_lower), 1) # [batch, enc_size+enc_size//4]
                img_channel_avg_upper = self.relu(self.channel_upper_fc_1(img_channel_avg_upper))
                img_channel_avg_upper = self.relu(self.channel_upper_fc_2(img_channel_avg_upper))   # [batch, enc_size//4]
                img_feat = torch.cat((img_spatial_avg, img_spatial_max, img_channel_avg_upper), 1)  
                img_feat = self.relu(self.spatial_fc_1(img_feat))                                   # [batch, enc_size//2]
                img_feat = self.relu(self.spatial_fc_2(img_feat))                                   # [batch, enc_size//4]
                img_feat = self.relu(self.img_fc(img_feat))                                         # [batch, 128]

            fused_feat.append(img_feat)
        
        if self.use_imu:
            # get imu embedding by rnnIMU
            # imu_feat: [batch, 11, imu_lstm_hidden_size] -> [batch, imu_lstm_hidden_size]
            # self.rnn_imu_state = (imu_n, imu_c)
            # imu_n: [2, batch, imu_lstm_hidden_size]
            # imu_c: [2, batch, imu_lstm_hidden_size]
            if imu_seq is None: raise ValueError('imu_seq must be given when sensors include imu')
            if imu_lstm_hidden is None: raise ValueError('imu_lstm_hidden must be giben when sensors include imu')
            imu_feat, imu_lstm_hidden = self.rnn_imu(imu_seq, imu_lstm_hidden)
            imu_feat = imu_feat[:, -1, :]
            hidden_states_out['imu_lstm_hidden'] = imu_lstm_hidden 
            fused_feat.append(imu_feat)

        if self.use_pose:
            # get last pose embedding by repeat or an embedding layer
            # [batch, 6 * pose_embedding_tiles] # -> [batch, last_pose_hidden_size]
            if last_pose is None: raise ValueError('last_pose must be given when sensors include pose')
            last_pose_feat = last_pose.repeat(1, self.args.last_pose_tiles) 
            if self.args.last_pose_fc:
                last_pose_feat = self.last_pose_fc(last_pose_feat)
                last_pose_feat = self.relu(last_pose_feat)
            fused_feat.append(last_pose_feat)
        
        # [batch, 1, (img_feat_size) + (imu_feat_size) + (last_pose_feat_size)]
        fused_feat = torch.cat(fused_feat, 1) 
        fused_feat = fused_feat.unsqueeze(1) 
        
        # r_out: [batch, 1, fused_lstm_hidden_size]
        # self.rnn_fusion_state = (h_n, h_c)
        # h_n: [2, batch, fused_lstm_hidden_size]
        # h_c: [2, batch, fused_lstm_hidden_size]
        # note r_out[:, -1, :] is equal to h_n[-1, :, :]
        fused_feat, fused_lstm_hidden = self.rnn_fusion(fused_feat, fused_lstm_hidden)
        hidden_states_out['fused_lstm_hidden'] = fused_lstm_hidden

        pose_out = self.relu(self.pose_fc_1(fused_feat[:, -1, :]))
        pose_out = 0.01 * self.pose_fc_2(pose_out)

        return pose_out, hidden_states_out


    def init_weights(self):
        """
        follow a deepvo pytorch implementation gitub repo
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                nn.init.xavier_normal_(m.weight.data)

            elif isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                #     m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTMCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        # nn.init.orthogonal(param)
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        # Forget gate bias trick: Initially during training, it is often helpful
                        # to initialize the forget gate bias to a large value, to help information
                        # flow over longer time steps.
                        # In a PyTorch LSTM, the biases are stored in the following order:
                        # [ b_ig | b_fg | b_gg | b_og ]
                        # where, b_ig is the bias for the input gate, 
                        # b_fg is the bias for the forget gate, 
                        # b_gg (see LSTM docs, Variables section), 
                        # b_og is the bias for the output gate.
                        # So, we compute the location of the forget gate bias terms as the 
                        # middle one-fourth of the bias vector, and initialize them.
                        
                        # First initialize all biases to zero
                        # nn.init.uniform_(param)
                        nn.init.constant_(param, 0.)
                        bias = getattr(m, name)
                        n = bias.size(0)
                        start, end = n // 4, n // 2
                        bias.data[start:end].fill_(10.)


def flatten_imgfeat(img_feat):
    """
    Input: [batch, 1024, 6, 8]
    Output: [batch, 1024]
    """
    batch_size = img_feat.shape[0]
    img_pool = nn.AdaptiveAvgPool2d(1)
    flat_img_feat = img_pool(img_feat) 
    flat_img_feat = flat_img_feat.squeeze()
    if batch_size == 1:
        flat_img_feat = flat_img_feat.unsqueeze(0)
    return flat_img_feat


def init_hidden_rnn(batch_size, lstm_hidden_size, device, mode='zero'):
    """
    reset rnn state
    mode: zero or random
    """
    # Variable has been deprecated -> torch.zeros / randn etc. just do the same thing
    # init_state with requires_grad=True/False should make no differences 
    if mode == 'zero':
        init_rnn_h0 = torch.zeros((2, batch_size, lstm_hidden_size), requires_grad=False)
        init_rnn_c0 = torch.zeros((2, batch_size, lstm_hidden_size), requires_grad=False)

    elif mode == 'random':
        init_rnn_h0 = torch.randn((2, batch_size, lstm_hidden_size), requires_grad=False)
        init_rnn_c0 = torch.randn((2, batch_size, lstm_hidden_size), requires_grad=False)
    else:
        raise ValueError('mode should be either zero or random, but provided with {}'.format(mode))

    init_rnn_h0 = init_rnn_h0.to(device=device)
    init_rnn_c0 = init_rnn_c0.to(device=device)
    return (init_rnn_h0, init_rnn_c0)


if __name__ == '__main__':
    probs = 0.7 * torch.ones(4, 6)
    gumbel_sigmoid(probs=probs,tau=1,hard=True)
