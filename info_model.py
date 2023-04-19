import pdb
from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F

flownet_featsize = {
    'kitti': 1024 * 5 * 19,
    'euroc': 1024 * 8 * 12
}

# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    # f: flownet
    # e.g. x_tuple: (x_img_list, ) -> x_img_list: [time, batch, 3, 2, H, W]
    x_sizes = tuple(map(lambda x: x.size(), x_tuple)) # ([time, batch, 3, 2, H, W], )
    # process the size reshape for each tensor in x_tuple
    # the new batch_size = time x the old batch_size 
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes))) # f([time x batch, 3, 2, H, W])
    if type(y) == tuple:
        y_size = y[0].size()
        return [_y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:]) for _y in y]
    else:
        y_size = y.size()
        # reshape the output into time x the old batch_size x features
        return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])

def img_conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(0.1, inplace=True)
        )

def img_upconv(batch_norm, in_planes, out_planes):
    if batch_norm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )


class SingleHiddenTransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev 
        self.fc_embed_state = nn.Linear(2 * state_size, belief_size)
        if args.belief_rnn == 'lstm':
            self.rnn = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
        elif args.belief_rnn == 'gru':
            self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size + 6 * args.pose_tiles, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        use_pose_model = True if type(poses) == PoseModel else False # type(poses) == torch.Tensor # [time, batch, 6]
        T = self.args.clip_length + 1 # number of time steps # self.args.clip_length = poses.size(0)
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        if self.args.belief_rnn == 'lstm':
            lstm_hiddens = [(torch.empty(0), torch.empty(0))] * T
            lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))
        if use_pose_model: 
            pred_poses, pred_stds = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update beliefs which is shared for both posterior_states and prior_states
            hidden = self.act_fn(self.fc_embed_state(torch.cat([posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, lstm_hiddens[t + 1] = self.rnn(hidden, lstm_hiddens[t])
                beliefs[t + 1] = belief_rnn.squeeze(1)
            
            # Update posterior_states with beliefs and observations
            hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
            posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
            posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
            posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            
            # Get poses: use pose model in evaluation where no gt poses are available
            if use_pose_model: 
                with torch.no_grad(): 
                    _pose = poses(posterior_means[t + 1])
                    if self.args.eval_uncertainty:
                        _plist = [] 
                        for k in range(100):
                            _plist.append(poses(posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])))
                        _plist = torch.stack(_plist, dim=0) # [k, batch, 6]
                        # _pose = _plist.mean(dim=0) 
                        pred_stds[t_ + 1] = torch.std(torch.norm(_plist, p=2, dim=2), dim=0)
                    pred_poses[t_ + 1] = _pose
            else:
                _pose = poses[t_ + 1]
            _pose = _pose.repeat(1, self.args.pose_tiles)
            
            # Update state_priors with beliefs and poses
            hidden = self.act_fn(self.fc_embed_belief_prior(torch.cat([beliefs[t + 1], _pose], dim=1)))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
            
        # Return new hidden states (init states are removed)
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        if use_pose_model: 
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty or self.args.eval_failure: 
                hidden += [torch.stack(pred_stds, dim=0)]
        return hidden


class DoubleHiddenTransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.imu_only = args.imu_only
        self.embedding_size = embedding_size
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev # 6 * args.pose_tiles + embedding_size
        
        self.fc_embed_state_posterior = nn.Linear(2 * state_size + embedding_size, belief_size)
        self.fc_embed_state_prior = nn.Linear(2 * state_size + 6 * args.pose_tiles, belief_size)
        if args.belief_rnn == 'lstm':
            self.rnn_posterior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
            self.rnn_prior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
        elif args.belief_rnn == 'gru':
            self.rnn_posterior = nn.GRUCell(belief_size, belief_size)
            self.rnn_prior = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)

        if self.imu_only:
            if args.imu_rnn == 'lstm':
                self.rnn_embed_imu = nn.LSTM(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
            elif args.imu_rnn == 'gru':
                self.rnn_embed_imu = nn.GRU(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)


    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        use_pose_model = True if type(poses) == PoseModel else False # type(poses) == torch.Tensor # [time, batch, 6]
        T = self.args.clip_length + 1 # number of time steps # self.args.clip_length = poses.size(0)
        prior_beliefs, posterior_beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        prior_beliefs[0], posterior_beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_belief, prev_state, prev_state
        if self.args.belief_rnn == 'lstm':
            prior_lstm_hiddens, posterior_lstm_hiddens = [(torch.empty(0), torch.empty(0))] * T, [(torch.empty(0), torch.empty(0))] * T
            prior_lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))
            posterior_lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))
        if use_pose_model: 
            pred_poses, pred_stds = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            
        if self.imu_only:
            running_batch_size = prev_belief.size()[0]
            rnn_embed_imu_hiddens = [torch.empty(0)] * T
            prev_rnn_embed_imu_hidden = torch.zeros(2, running_batch_size, self.args.embedding_size, device=self.args.device)
            if self.args.imu_rnn == 'lstm':
                rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
            elif self.args.imu_rnn == 'gru':
                rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden
        
        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update posterior_states
            if self.imu_only:
                hidden, rnn_embed_imu_hiddens[t + 1] = self.rnn_embed_imu(observations[t_ + 1], rnn_embed_imu_hiddens[t])
                hidden = self.act_fn(self.fc_embed_state_posterior(torch.cat([hidden[:,-1,:], posterior_states[t], prior_states[t]], dim=1)))
            else: # image features
                hidden = self.act_fn(self.fc_embed_state_posterior(torch.cat([observations[t_ + 1], posterior_states[t], prior_states[t]], dim=1)))
                
            if self.args.belief_rnn == 'gru':
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, posterior_beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, posterior_lstm_hiddens[t + 1] = self.rnn_posterior(hidden, posterior_lstm_hiddens[t])
                posterior_beliefs[t + 1] = belief_rnn.squeeze(1)
                
            hidden = self.act_fn(self.fc_embed_belief_posterior(posterior_beliefs[t + 1]))
            posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
            posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
            posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            
            # Get poses: use pose model in evaluation where no gt poses are available
            if use_pose_model: 
                with torch.no_grad(): 
                    _pose = poses(posterior_means[t + 1])
                    if self.args.eval_uncertainty:
                        _plist = [] 
                        for k in range(100):
                            _plist.append(poses(posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])))
                        _plist = torch.stack(_plist, dim=0) # [k, batch, 6]
                        # _pose = _plist.mean(dim=0) 
                        pred_stds[t_ + 1] = torch.std(torch.norm(_plist, p=2, dim=2), dim=0)
                    pred_poses[t_ + 1] = _pose
            else:
                _pose = poses[t_ + 1]
            _pose = _pose.repeat(1, self.args.pose_tiles)
            
            # Update state_priors 
            hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                prior_beliefs[t + 1] = self.rnn_prior(hidden, prior_beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, prior_lstm_hiddens[t + 1] = self.rnn_prior(hidden, prior_lstm_hiddens[t])
                prior_beliefs[t + 1] = belief_rnn.squeeze(1)
                
            hidden = self.act_fn(self.fc_embed_belief_prior(prior_beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
            
        # Return new hidden states (init states are removed)
        if self.args.rec_type == 'posterior':
            hidden = [torch.stack(posterior_beliefs[1:], dim=0)]
        elif self.args.rec_type == 'prior':
            hidden = [torch.stack(prior_beliefs[1:], dim=0)]
        hidden += [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        if use_pose_model: 
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty or self.args.eval_failure: 
                hidden += [torch.stack(pred_stds, dim=0)]
        return hidden


class DoubleStochasticTransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev # 6 * args.pose_tiles + embedding_size
        self.stochastic_mode = args.stochastic_mode
        if args.belief_rnn == 'lstm': raise NotImplementedError('do not support lstm for double-stochastic')
        
        if self.stochastic_mode == 'v1':
            self.fc_embed_state_posterior = nn.Linear(embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, 2 * state_size)
            self.rnn_prior = nn.GRUCell(belief_size, 2 * state_size)
        elif self.stochastic_mode == 'v2':
            self.fc_embed_state_posterior = nn.Linear(state_size + embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(state_size + 6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, 2 * state_size)
            self.rnn_prior = nn.GRUCell(belief_size, 2 * state_size)
        elif self.stochastic_mode == 'v3':
            self.fc_embed_state_posterior = nn.Linear(state_size + embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(state_size + 6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, state_size)
            self.rnn_prior = nn.GRUCell(belief_size, state_size)
            self.fc_embed_rnn_posterior = nn.Linear(state_size, hidden_size)
            self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
            self.fc_embed_rnn_prior = nn.Linear(state_size, hidden_size)
            self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)


    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        use_pose_model = True if type(poses) == PoseModel else False # type(poses) == torch.Tensor # [time, batch, 6]
        T = self.args.clip_length + 1 # number of time steps # self.args.clip_length = poses.size(0)
        prior_beliefs, posterior_beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        prior_beliefs[0], posterior_beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_belief, prev_state, prev_state
        if use_pose_model: 
            pred_poses, pred_stds = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update posterior_states
            if self.stochastic_mode == 'v1':
                hidden = self.act_fn(self.fc_embed_state_posterior(observations[t_ + 1]))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, torch.cat([posterior_states[t], prior_states[t]], dim=1))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(posterior_beliefs[t + 1], 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            elif self.stochastic_mode == 'v2':
                hidden = self.act_fn(self.fc_embed_state_posterior(torch.cat([observations[t_ + 1], prior_states[t]], dim=1)))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, torch.cat([posterior_states[t], posterior_states[t]], dim=1))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(posterior_beliefs[t + 1], 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            elif self.stochastic_mode == 'v3':
                hidden = self.act_fn(self.fc_embed_state_posterior(torch.cat([observations[t_ + 1], prior_states[t]], dim=1)))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, posterior_states[t])
                hidden = self.act_fn(self.fc_embed_rnn_posterior(posterior_beliefs[t + 1]))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
                
            # Get poses: use pose model in evaluation where no gt poses are available
            if use_pose_model: 
                with torch.no_grad(): 
                    _pose = poses(posterior_means[t + 1])
                    if self.args.eval_uncertainty:
                        _plist = [] 
                        for k in range(100):
                            _plist.append(poses(posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])))
                        _plist = torch.stack(_plist, dim=0) # [k, batch, 6]
                        # _pose = _plist.mean(dim=0) 
                        pred_stds[t_ + 1] = torch.std(torch.norm(_plist, p=2, dim=2), dim=0)
                    pred_poses[t_ + 1] = _pose
            else:
                _pose = poses[t_ + 1]
            _pose = _pose.repeat(1, self.args.pose_tiles)
            
            # Update state_priors 
            if self.stochastic_mode == 'v1':
                hidden = self.act_fn(self.fc_embed_state_prior(_pose))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, torch.cat([posterior_states[t], prior_states[t]], dim=1))
                prior_means[t + 1], _prior_std_dev = torch.chunk(prior_beliefs[t + 1], 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])  
            elif self.stochastic_mode == 'v2':
                hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t]], dim=1)))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, torch.cat([prior_states[t], prior_states[t]], dim=1))
                prior_means[t + 1], _prior_std_dev = torch.chunk(prior_beliefs[t + 1], 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])  
            elif self.stochastic_mode == 'v3':
                hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t]], dim=1)))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, prior_states[t])
                hidden = self.act_fn(self.fc_embed_rnn_prior(prior_beliefs[t + 1]))
                prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
            
        # Return new hidden states (init states are removed)
        if self.args.rec_type == 'posterior':
            hidden = [torch.stack(posterior_beliefs[1:], dim=0)]
        elif self.args.rec_type == 'prior':
            hidden = [torch.stack(prior_beliefs[1:], dim=0)]
        hidden += [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        if use_pose_model: 
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty or self.args.eval_failure: 
                hidden += [torch.stack(pred_stds, dim=0)]
        return hidden


class SingleHiddenVITransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev # 6 * args.pose_tiles + embedding_size
        
        self.fc_embed_state = nn.Linear(2 * state_size, belief_size)
        if args.belief_rnn == 'lstm':
            self.rnn = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
        elif args.belief_rnn == 'gru':
            self.rnn = nn.GRUCell(belief_size, belief_size)
        if args.imu_rnn == 'lstm':
            self.rnn_embed_imu = nn.LSTM(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        elif args.imu_rnn == 'gru':
            self.rnn_embed_imu = nn.GRU(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + 2 * embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size + 6 * args.pose_tiles, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)


    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        observations_visual = observations[0]
        observations_imu = observations[1]
        use_pose_model = True if type(poses) == PoseModel else False # type(poses) == torch.Tensor # [time, batch, 6]
        T = self.args.clip_length + 1 # number of time steps # self.args.clip_length = poses.size(0)
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        if self.args.belief_rnn == 'lstm':
            lstm_hiddens = [(torch.empty(0), torch.empty(0))] * T
            lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))

        running_batch_size = prev_belief.size()[0]
        rnn_embed_imu_hiddens = [torch.empty(0)] * T
        prev_rnn_embed_imu_hidden = torch.zeros(2, running_batch_size, self.args.embedding_size, device=self.args.device)
        if self.args.imu_rnn == 'lstm':
            rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
        elif self.args.imu_rnn == 'gru':
            rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden
            
        if use_pose_model: 
            pred_poses, pred_stds = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update beliefs which is shared for both posterior_states and prior_states
            hidden = self.act_fn(self.fc_embed_state(torch.cat([posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, lstm_hiddens[t + 1] = self.rnn(hidden, lstm_hiddens[t])
                beliefs[t + 1] = belief_rnn.squeeze(1)
                
            hidden, rnn_embed_imu_hiddens[t + 1] = self.rnn_embed_imu(observations_imu[t_ + 1], rnn_embed_imu_hiddens[t])
            
            # Update posterior_states with beliefs and observations
            hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations_visual[t_ + 1], hidden[:,-1,:]], dim=1)))
            posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
            posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
            posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            
            # Get poses: use pose model in evaluation where no gt poses are available
            if use_pose_model: 
                with torch.no_grad(): 
                    _pose = poses(posterior_means[t + 1])
                    if self.args.eval_uncertainty:
                        _plist = [] 
                        for k in range(100):
                            _plist.append(poses(posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])))
                        _plist = torch.stack(_plist, dim=0) # [k, batch, 6]
                        # _pose = _plist.mean(dim=0) 
                        pred_stds[t_ + 1] = torch.std(torch.norm(_plist, p=2, dim=2), dim=0)
                    pred_poses[t_ + 1] = _pose
            else:
                _pose = poses[t_ + 1]
            _pose = _pose.repeat(1, self.args.pose_tiles)
            
            # Update state_priors with beliefs and poses
            hidden = self.act_fn(self.fc_embed_belief_prior(torch.cat([beliefs[t + 1], _pose], dim=1)))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
            
        # Return new hidden states (init states are removed)
        hidden = [(torch.stack(beliefs[1:], dim=0), torch.stack(beliefs[1:], dim=0))]
        hidden += [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        if use_pose_model: 
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty or self.args.eval_failure: 
                hidden += [torch.stack(pred_stds, dim=0)]
        return hidden


class DoubleHiddenVITransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.use_soft = args.soft
        self.use_hard = args.hard
        self.embedding_size = embedding_size
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_posterior = nn.Linear(2 * state_size + 2 * embedding_size, belief_size)
        self.fc_embed_state_prior = nn.Linear(2 * state_size + 6 * args.pose_tiles, belief_size)
        if args.belief_rnn == 'lstm':
            self.rnn_posterior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
            self.rnn_prior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
        elif args.belief_rnn == 'gru':
            self.rnn_posterior = nn.GRUCell(belief_size, belief_size)
            self.rnn_prior = nn.GRUCell(belief_size, belief_size)
        if args.imu_rnn == 'lstm':
            self.rnn_embed_imu = nn.LSTM(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        elif args.imu_rnn == 'gru':
            self.rnn_embed_imu = nn.GRU(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        self.fc_embed_belief_posterior = nn.Linear(belief_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        
        if self.use_soft:
            self.sigmoid = nn.Sigmoid()
            self.soft_fc_img = nn.Linear(2 * embedding_size, embedding_size)
            self.soft_fc_imu = nn.Linear(2 * embedding_size, embedding_size)
            
        if self.use_hard:
            self.sigmoid = nn.Sigmoid()
            self.hard_fc_img = nn.Linear(2 * embedding_size, embedding_size)
            self.hard_fc_imu = nn.Linear(2 * embedding_size, embedding_size)
            if args.hard_mode == 'onehot':
                self.onehot_hard = True
            elif args.hard_mode == 'gumbel_soft':
                self.onehot_hard = False
            self.eps = 1e-10

    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations, gumbel_temperature=0.5):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        observations_visual = observations[0]
        observations_imu = observations[1]
        use_pose_model = True if type(poses) == PoseModel else False # type(poses) == torch.Tensor # [time, batch, 6]
        T = self.args.clip_length + 1 # number of time steps # self.args.clip_length = poses.size(0)
        prior_beliefs, posterior_beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        prior_beliefs[0], posterior_beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_belief, prev_state, prev_state
        if self.args.belief_rnn == 'lstm':
            prior_lstm_hiddens, posterior_lstm_hiddens = [(torch.empty(0), torch.empty(0))] * T, [(torch.empty(0), torch.empty(0))] * T
            prior_lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))
            posterior_lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))
        if use_pose_model: 
            pred_poses, pred_stds = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            
        running_batch_size = prev_belief.size()[0]
        rnn_embed_imu_hiddens = [torch.empty(0)] * T
        prev_rnn_embed_imu_hidden = torch.zeros(2, running_batch_size, self.args.embedding_size, device=self.args.device)
        if self.args.imu_rnn == 'lstm':
            rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
        elif self.args.imu_rnn == 'gru':
            rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden
            
        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update posterior_states
            hidden, rnn_embed_imu_hiddens[t + 1] = self.rnn_embed_imu(observations_imu[t_ + 1], rnn_embed_imu_hiddens[t])
            fused_feat = torch.cat([observations_visual[t_ + 1], hidden[:,-1,:]], dim=1)
            if self.use_soft:
                soft_mask_img = self.sigmoid(self.soft_fc_img(fused_feat))
                soft_mask_imu = self.sigmoid(self.soft_fc_imu(fused_feat))
                soft_mask = torch.ones_like(fused_feat).to(device=self.args.device)
                soft_mask[:, :self.embedding_size] = soft_mask_img
                soft_mask[:, self.embedding_size:] = soft_mask_imu
                fused_feat = fused_feat * soft_mask
            if self.use_hard:
                prob_img = self.sigmoid(self.hard_fc_img(fused_feat))
                prob_imu = self.sigmoid(self.hard_fc_imu(fused_feat))
                hard_mask_img = self.gumbel_sigmoid(prob_img, gumbel_temperature)
                hard_mask_imu = self.gumbel_sigmoid(prob_imu, gumbel_temperature)
                hard_mask_img = hard_mask_img[:, :, 0]
                hard_mask_imu = hard_mask_imu[:, :, 0]
                hard_mask = torch.ones_like(fused_feat).to(device=self.args.device)
                hard_mask[:, :self.embedding_size] = hard_mask_img
                hard_mask[:, self.embedding_size:] = hard_mask_imu
                fused_feat = fused_feat * hard_mask
            
            hidden = self.act_fn(self.fc_embed_state_posterior(torch.cat([fused_feat, posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, posterior_beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, posterior_lstm_hiddens[t + 1] = self.rnn_posterior(hidden, posterior_lstm_hiddens[t])
                posterior_beliefs[t + 1] = belief_rnn.squeeze(1)
                
            hidden = self.act_fn(self.fc_embed_belief_posterior(posterior_beliefs[t + 1]))
            posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
            posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
            posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            
            # Get poses: use pose model in evaluation where no gt poses are available
            if use_pose_model: 
                with torch.no_grad(): 
                    _pose = poses(posterior_means[t + 1])
                    if self.args.eval_uncertainty:
                        _plist = [] 
                        for k in range(100):
                            _plist.append(poses(posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])))
                        _plist = torch.stack(_plist, dim=0) # [k, batch, 6]
                        # _pose = _plist.mean(dim=0) 
                        pred_stds[t_ + 1] = torch.std(torch.norm(_plist, p=2, dim=2), dim=0)
                    pred_poses[t_ + 1] = _pose
            else:
                _pose = poses[t_ + 1]
            _pose = _pose.repeat(1, self.args.pose_tiles)
            
            # Update state_priors 
            hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                prior_beliefs[t + 1] = self.rnn_prior(hidden, prior_beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, prior_lstm_hiddens[t + 1] = self.rnn_prior(hidden, prior_lstm_hiddens[t])
                prior_beliefs[t + 1] = belief_rnn.squeeze(1)
                
            hidden = self.act_fn(self.fc_embed_belief_prior(prior_beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
            
        # Return new hidden states (init states are removed)
        if self.args.rec_type == 'posterior':
            hidden = [(torch.stack(posterior_beliefs[1:], dim=0), torch.stack(posterior_beliefs[1:], dim=0))]
        elif self.args.rec_type == 'prior':
            hidden = [torch.stack(prior_beliefs[1:], dim=0)]
        hidden += [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        if use_pose_model: 
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty or self.args.eval_failure: 
                hidden += [torch.stack(pred_stds, dim=0)]
        return hidden


    def gumbel_sigmoid(self, probs, tau):
        """
        input: 
        -> probs: [batch, feat_size]: each element is the probability to be 1
        return: 
        -> gumbel_dist: [batch, feat_size, 2]
            -> if self.onehot_hard == True:  one_hot vector (as in SelectiveFusion)
            -> if self.onehot_hard == False: gumbel softmax approx
        """
        log_probs = torch.stack((torch.log(probs + self.eps), torch.log(1 - probs + self.eps)), dim=-1) # [batch, feat_size, 2]
        gumbel = torch.rand_like(log_probs).to(device=self.args.device)
        gumbel = -torch.log(-torch.log(gumbel + self.eps) + self.eps)
        log_probs = log_probs + gumbel # [batch, feat_size, 2]
        gumbel_dist = F.softmax(log_probs / tau, dim=-1) # [batch, feat_size, 2]
        if self.onehot_hard:
            _shape = gumbel_dist.shape
            _, ind = gumbel_dist.max(dim=-1)
            gumbel_hard = torch.zeros_like(gumbel_dist).view(-1, _shape[-1])
            gumbel_hard.scatter_(dim=-1, index=ind.view(-1,1), value=1.0)
            gumbel_hard = gumbel_hard.view(*_shape)
            gumbel_dist = (gumbel_hard - gumbel_dist).detach() + gumbel_dist
        return gumbel_dist
    

class DoubleStochasticVITransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev # 6 * args.pose_tiles + embedding_size
        self.stochastic_mode = args.stochastic_mode
        if args.belief_rnn == 'lstm': raise NotImplementedError('do not support belief_rnn lstm for double-vinet-stochastic')
        
        if args.imu_rnn == 'lstm':
            self.rnn_embed_imu = nn.LSTM(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        elif args.imu_rnn == 'gru':
            self.rnn_embed_imu = nn.GRU(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        
        if self.stochastic_mode == 'v1':
            self.fc_embed_state_posterior = nn.Linear(2 * embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, 2 * state_size)
            self.rnn_prior = nn.GRUCell(belief_size, 2 * state_size)
        elif self.stochastic_mode == 'v2':
            self.fc_embed_state_posterior = nn.Linear(state_size + 2 * embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(state_size + 6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, 2 * state_size)
            self.rnn_prior = nn.GRUCell(belief_size, 2 * state_size)
        elif self.stochastic_mode == 'v3':
            self.fc_embed_state_posterior = nn.Linear(state_size + 2 * embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(state_size + 6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, state_size)
            self.rnn_prior = nn.GRUCell(belief_size, state_size)
            self.fc_embed_rnn_posterior = nn.Linear(state_size, hidden_size)
            self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
            self.fc_embed_rnn_prior = nn.Linear(state_size, hidden_size)
            self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)


    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        observations_visual = observations[0]
        observations_imu = observations[1]
        use_pose_model = True if type(poses) == PoseModel else False # type(poses) == torch.Tensor # [time, batch, 6]
        T = self.args.clip_length + 1 # number of time steps # self.args.clip_length = poses.size(0)
        prior_beliefs, posterior_beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        prior_beliefs[0], posterior_beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_belief, prev_state, prev_state
        
        if use_pose_model: 
            pred_poses, pred_stds = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            
        running_batch_size = prev_belief.size()[0]
        rnn_embed_imu_hiddens = [torch.empty(0)] * T
        prev_rnn_embed_imu_hidden = torch.zeros(2, running_batch_size, self.args.embedding_size, device=self.args.device)
        if self.args.imu_rnn == 'lstm':
            rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
        elif self.args.imu_rnn == 'gru':
            rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden
            
        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update posterior_states
            hidden, rnn_embed_imu_hiddens[t + 1] = self.rnn_embed_imu(observations_imu[t_ + 1], rnn_embed_imu_hiddens[t])
            imu_feat = hidden[:,-1,:]
            if self.stochastic_mode == 'v1':
                hidden = self.act_fn(self.fc_embed_state_posterior(torch.cat([observations_visual[t_ + 1], imu_feat], dim=1)))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, torch.cat([posterior_states[t], prior_states[t]], dim=1))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(posterior_beliefs[t + 1], 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            elif self.stochastic_mode == 'v2':
                hidden = self.act_fn(self.fc_embed_state_posterior(torch.cat([observations_visual[t_ + 1], imu_feat, prior_states[t]], dim=1)))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, torch.cat([posterior_states[t], posterior_states[t]], dim=1))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(posterior_beliefs[t + 1], 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            elif self.stochastic_mode == 'v3':
                hidden = self.act_fn(self.fc_embed_state_posterior(torch.cat([observations_visual[t_ + 1], imu_feat, prior_states[t]], dim=1)))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, posterior_states[t])
                hidden = self.act_fn(self.fc_embed_rnn_posterior(posterior_beliefs[t + 1]))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            
            # Get poses: use pose model in evaluation where no gt poses are available
            if use_pose_model: 
                with torch.no_grad(): 
                    _pose = poses(posterior_means[t + 1])
                    if self.args.eval_uncertainty:
                        _plist = [] 
                        for k in range(100):
                            _plist.append(poses(posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])))
                        _plist = torch.stack(_plist, dim=0) # [k, batch, 6]
                        # _pose = _plist.mean(dim=0) 
                        pred_stds[t_ + 1] = torch.std(torch.norm(_plist, p=2, dim=2), dim=0)
                    pred_poses[t_ + 1] = _pose
            else:
                _pose = poses[t_ + 1]
            _pose = _pose.repeat(1, self.args.pose_tiles)
            
            # Update state_priors 
            if self.stochastic_mode == 'v1':
                hidden = self.act_fn(self.fc_embed_state_prior(_pose))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, torch.cat([posterior_states[t], prior_states[t]], dim=1))
                prior_means[t + 1], _prior_std_dev = torch.chunk(prior_beliefs[t + 1], 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])   
            elif self.stochastic_mode == 'v2':
                hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t]], dim=1)))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, torch.cat([prior_states[t], prior_states[t]], dim=1))
                prior_means[t + 1], _prior_std_dev = torch.chunk(prior_beliefs[t + 1], 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])   
            elif self.stochastic_mode == 'v3':
                hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t]], dim=1)))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, prior_states[t])
                hidden = self.act_fn(self.fc_embed_rnn_prior(prior_beliefs[t + 1]))
                prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1]) 
            
        # Return new hidden states (init states are removed)
        if self.args.rec_type == 'posterior':
            hidden = [(torch.stack(posterior_beliefs[1:], dim=0), torch.stack(posterior_beliefs[1:], dim=0))]
        elif self.args.rec_type == 'prior':
            hidden = [torch.stack(prior_beliefs[1:], dim=0)]
        hidden += [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        if use_pose_model: 
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty or self.args.eval_failure: 
                hidden += [torch.stack(pred_stds, dim=0)]
        return hidden


class MultiHiddenVITransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev # 6 * args.pose_tiles + embedding_size
        
        self.fc_embed_visual_state_posterior = nn.Linear(2 * state_size + embedding_size, belief_size)
        self.fc_embed_imu_state_posterior = nn.Linear(2 * state_size + embedding_size, belief_size)
        self.fc_embed_state_prior = nn.Linear(2 * state_size + 6 * args.pose_tiles, belief_size)
        if args.belief_rnn == 'lstm':
            self.rnn_visual_posterior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
            self.rnn_imu_posterior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
            self.rnn_prior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
        elif args.belief_rnn == 'gru':
            self.rnn_visual_posterior = nn.GRUCell(belief_size, belief_size)
            self.rnn_imu_posterior = nn.GRUCell(belief_size, belief_size)
            self.rnn_prior = nn.GRUCell(belief_size, belief_size)
        if args.imu_rnn == 'lstm':
            self.rnn_embed_imu = nn.LSTM(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        elif args.imu_rnn == 'gru':
            self.rnn_embed_imu = nn.GRU(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        self.fc_embed_belief_posterior = nn.Linear(2 * belief_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)


    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        observations_visual = observations[0]
        observations_imu = observations[1]
        use_pose_model = True if type(poses) == PoseModel else False # type(poses) == torch.Tensor # [time, batch, 6]
        T = self.args.clip_length + 1 # number of time steps # self.args.clip_length = poses.size(0)
        prior_beliefs, posterior_beliefs_visual, posterior_beliefs_imu, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        prior_beliefs[0], posterior_beliefs_visual[0], posterior_beliefs_imu[0] = prev_belief, prev_belief, prev_belief
        prior_states[0], posterior_states[0] = prev_state, prev_state
        
        if self.args.belief_rnn == 'lstm':
            prior_lstm_hiddens, posterior_lstm_hiddens_visual, posterior_lstm_hiddens_imu = [(torch.empty(0), torch.empty(0))] * T, [(torch.empty(0), torch.empty(0))] * T, [(torch.empty(0), torch.empty(0))] * T
            prior_lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))
            posterior_lstm_hiddens_visual[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))
            posterior_lstm_hiddens_imu[0] = (prev_belief.unsqueeze(0).repeat(2,1,1), prev_belief.unsqueeze(0).repeat(2,1,1))
        
        running_batch_size = prev_belief.size()[0]
        rnn_embed_imu_hiddens = [torch.empty(0)] * T
        prev_rnn_embed_imu_hidden = torch.zeros(2, running_batch_size, self.args.embedding_size, device=self.args.device)
        if self.args.imu_rnn == 'lstm':
            rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
        elif self.args.imu_rnn == 'gru':
            rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden
            
        if use_pose_model: 
            pred_poses, pred_stds = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update posterior_states
            hidden = self.act_fn(self.fc_embed_visual_state_posterior(torch.cat([observations_visual[t_ + 1], posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                posterior_beliefs_visual[t + 1] = self.rnn_visual_posterior(hidden, posterior_beliefs_visual[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                hidden, posterior_lstm_hiddens_visual[t + 1] = self.rnn_visual_posterior(hidden, posterior_lstm_hiddens_visual[t])
                posterior_beliefs_visual[t + 1] = hidden[:,-1,:]
            
            hidden, rnn_embed_imu_hiddens[t + 1] = self.rnn_embed_imu(observations_imu[t_ + 1], rnn_embed_imu_hiddens[t])
            hidden = self.act_fn(self.fc_embed_imu_state_posterior(torch.cat([hidden[:,-1,:], posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                posterior_beliefs_imu[t + 1] = self.rnn_imu_posterior(hidden, posterior_beliefs_imu[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                hidden, posterior_lstm_hiddens_imu[t + 1] = self.rnn_imu_posterior(hidden, posterior_lstm_hiddens_imu[t])
                posterior_beliefs_imu[t + 1] = hidden[:,-1,:]
                
            hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([posterior_beliefs_visual[t + 1], posterior_beliefs_imu[t + 1]], dim=1)))
            posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
            posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
            posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
            
            # Get poses: use pose model in evaluation where no gt poses are available
            if use_pose_model: 
                with torch.no_grad(): 
                    _pose = poses(posterior_means[t + 1])
                    if self.args.eval_uncertainty:
                        _plist = [] 
                        for k in range(100):
                            _plist.append(poses(posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])))
                        _plist = torch.stack(_plist, dim=0) # [k, batch, 6]
                        # _pose = _plist.mean(dim=0) 
                        pred_stds[t_ + 1] = torch.std(torch.norm(_plist, p=2, dim=2), dim=0)
                    pred_poses[t_ + 1] = _pose
            else:
                _pose = poses[t_ + 1]
            _pose = _pose.repeat(1, self.args.pose_tiles)
            
            # Update state_priors 
            hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                prior_beliefs[t + 1] = self.rnn_prior(hidden, prior_beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, prior_lstm_hiddens[t + 1] = self.rnn_prior(hidden, prior_lstm_hiddens[t])
                prior_beliefs[t + 1] = belief_rnn.squeeze(1)
                
            hidden = self.act_fn(self.fc_embed_belief_prior(prior_beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
            
        # Return new hidden states (init states are removed)
        if self.args.rec_type == 'posterior':
            hidden = [(torch.stack(posterior_beliefs_visual[1:], dim=0), torch.stack(posterior_beliefs_imu[1:], dim=0))]
        elif self.args.rec_type == 'prior':
            hidden = [(torch.stack(prior_beliefs[1:], dim=0), torch.stack(prior_beliefs[1:], dim=0))]
        hidden += [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        if use_pose_model: 
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty or self.args.eval_failure: 
                hidden += [torch.stack(pred_stds, dim=0)]
        return hidden


class PoseModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu'):
        """
        embedding_size: not used (for code consistency in main.py)
        """
        # use posterior_states for pose prediction (since prior_states already contains pose information)
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_trans = nn.Linear(hidden_size, 3)
        self.fc3_rot = nn.Linear(hidden_size, 3)
    
    # @jit.script_method
    # def forward(self, state, state_std):
    def forward(self, state):
        hidden = self.act_fn(self.fc1(state))
        hidden = self.act_fn(self.fc2(hidden))
        trans = self.fc3_trans(hidden)
        rot = self.fc3_rot(hidden)
        return torch.cat([trans, rot], dim=1)


class SymbolicObservationModel(nn.Module):
    def __init__(self, args, belief_size, state_size, embedding_size, activation_function='relu', observation_type='visual'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        if observation_type =='visual':
            self.fc3 = nn.Linear(embedding_size, flownet_featsize[args.flowfeat_size_dataset])
        elif observation_type == 'imu':
            self.fc3 = nn.Linear(embedding_size, 6 * 11) # for each frame-pair

    # @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(nn.Module):
    def __init__(self, belief_size, state_size, embedding_size, activation_function='relu', batch_norm=False):
        raise NotImplementedError('need to check the final output image size')
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.upconv6 = img_upconv(batch_norm, 256, 256)
        self.upconv5 = img_upconv(batch_norm, 256, 128)
        self.upconv4 = img_upconv(batch_norm, 128, 64)
        self.upconv3 = img_upconv(batch_norm, 64, 32)
        self.upconv2 = img_upconv(batch_norm, 32, 16)
        self.upconv1 = img_upconv(batch_norm, 16, 6)
        # self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2) 
        # self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        # self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        # self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)  
        
    # @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.upconv6(hidden)
        hidden = self.upconv5(hidden)
        hidden = self.upconv4(hidden)
        hidden = self.upconv3(hidden)
        hidden = self.upconv2(hidden)
        observation = self.upconv1(hidden)
        return observation


def ObservationModel(symbolic, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu', observation_type='visual'):
    """
    hidden_size: not used (for code consistency in main.py)
    """
    if symbolic: # use Flownet2S features
        return SymbolicObservationModel(args, belief_size, state_size, embedding_size, activation_function, observation_type) 
    else: # train from scratch
        if observation_type != 'visual': raise ValueError('error: observation must be visual for symbolic being False')
        return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class SymbolicEncoder(nn.Module):
    def __init__(self, args, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(flownet_featsize[args.flowfeat_size_dataset], embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    # @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden
    
    
class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, activation_function='relu', batch_norm=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = img_conv(batch_norm, 6, 16, kernel_size=7, stride=2)
        self.conv2 = img_conv(batch_norm, 16, 32, kernel_size=5, stride=2)
        self.conv3 = img_conv(batch_norm, 32, 64, kernel_size=3, stride=2)
        self.conv4 = img_conv(batch_norm, 64, 128, kernel_size=3, stride=2)
        self.conv5 = img_conv(batch_norm, 128, 256, kernel_size=3, stride=2)
        self.conv6 = img_conv(batch_norm, 256, 256, kernel_size=3, stride=2)
        self.conv6_1 = img_conv(batch_norm, 256, 256, kernel_size=3, stride=2)
        
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    # @jit.script_method
    def forward(self, observation):
        # observation: [batch, 6, H, W]
        hidden = self.conv1(observation)
        hidden = self.conv2(hidden)
        hidden = self.conv3(hidden)
        hidden = self.conv4(hidden)
        hidden = self.conv5(hidden)
        hidden = self.conv6(hidden)
        hidden = self.conv6_1(hidden)
        pdb.set_trace()
        hidden = hidden.view(hidden.size()[0], -1)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, args, belief_size, state_size, hidden_size, embedding_size,activation_function='relu'):
    """
    hidden_size: not used (for code consistency in main.py)
    """
    if symbolic: # use FlowNet2S features
        return SymbolicEncoder(args, embedding_size, activation_function)
    else: # train from scratch
        return VisualEncoder(embedding_size, activation_function)

# To do: Separate imu encoder here as well