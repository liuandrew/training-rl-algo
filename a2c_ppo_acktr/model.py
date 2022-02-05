import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        elif type(base) == str:
            #Attempt to find a base class with the given string
            base = globals()[base]

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs






#Andy: Add FlexBase currently for allowing shared layers between
# actor and critic
class FlexBase(NNBase):
    '''
    NN module that allows for shared actor and critic layers as well
    as varying number of output heads
    
    num_layers: how many hidden MLP layers from input (or from GRU) to output heads
    num_shared_layers: how many of these MLP layers should be shared between actor and critic 
        -1 means all layers should be shared
    '''
    def __init__(self, num_inputs, recurrent=True, hidden_size=64,
                num_layers=2, num_shared_layers=-1):
        super(FlexBase, self).__init__(recurrent, num_inputs, hidden_size)
        
        print('Using FlexBase')
        print('num shared layers is ' + str(num_shared_layers))

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        shared_layers = []
        critic_layers = []
        actor_layers = []
        
        # generate all the shared layers
        cur_shared_layers = 0
        in_dim = num_inputs
        for i in range(num_layers):
            if num_shared_layers == -1 or cur_shared_layers < num_shared_layers:
                shared_layers.append(init_(nn.Linear(in_dim, hidden_size)))
                shared_layers.append(nn.Tanh())
                in_dim = hidden_size # only first layer with have input size num_inputs
                cur_shared_layers += 1
        
        # generate the non-shared layers
        if num_shared_layers != -1:
            remaining_layers = num_layers - num_shared_layers
        else:
            remaining_layers = 0
        
        for i in range(remaining_layers):
            critic_layers.append(init_(nn.Linear(in_dim, hidden_size)))
            critic_layers.append(nn.Tanh())
            actor_layers.append(init_(nn.Linear(in_dim, hidden_size)))
            actor_layers.append(nn.Tanh())
            in_dim = hidden_size # only first layer with have input size num_inputs
            
        # finally create the critic linear output
        critic_layers.append(init_(nn.Linear(in_dim, 1)))
        
        if len(shared_layers) > 0:
            self.shared_layers = nn.Sequential(*shared_layers)
        else:
            self.shared_layers = None
        
        if len(actor_layers) > 0:
            self.actor_layers = nn.Sequential(*actor_layers)
        else:
            self.actor_layers = None
            
        if len(critic_layers) > 0:
            self.critic_layers = nn.Sequential(*critic_layers)
        else:
            self.critic_layers = None
            
        self.train()
            
    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        
        if self.shared_layers:
            x = self.shared_layers(x)
            
        if self.actor_layers:
            hidden_actor = self.actor_layers(x)
        else:
            # if all layers are shared between actor and critic,
            # the last output of shared layers will be x
            # which will be used by the dist function in Policy (model.py)
            hidden_actor = x
            
        if self.critic_layers:
            # this should always run since we will output the critic evaluation here
            critic_val = self.critic_layers(x)
        else:
            raise Exception('Something mysterious happened... there was no final critic head')
        
        return critic_val, hidden_actor, rnn_hxs