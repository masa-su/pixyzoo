from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
# from torch.distributions.normal import Normal
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np

from pixyz.distributions import Normal


# Wraps the input tuple for a function to process a (time, batch, features sequence in batch, features) (assumes one output)
def bottle(f, x_dict):
    """
    x_dict: {"Name of the variable": variable}
    """
    assert len(x_dict) >= 1
    # x_tuple: (T, B, features...)
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    # apply neural network
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1]
                                   [1], *x[1][2:]), zip(x_tuple, x_sizes)))
    # (T * B, features...)にしてからネットワークに食わせる(x.view(x_size[0] * x_size[1], * x_size[2:]))
    # x_tupleの中身はconcatしていない
    feed_dict = {}
    for name, tensor in x_dict.items():
        T, B, feature_dims = tensor.size()[0], tensor.size[1], tensor.size[2:]
        feed_dict[name] = tensor.view(T*B, *feature_dims)
    y = f(feed_dict)
    y_size = y.size()
    output = y.view(T, B, *y_size[1:])
    return output

class TransitionModel(jit.ScriptModule):    # corresponds to RSSM?
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(
            state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        """
        # p(s_t | h_t)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        """

        """
        # p(o_t | h_t, s_t)
        self.fc_embed_belief_posterior = nn.Linear(
            belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        """

        # pixyz dists
        self.stochastic_state_model = StochasticStateModel(
            h_size=belief_size, hidden_size=hidden_size, activation=self.act_fn, min_std_dev=self.min_std_dev)

        self.obs_encoder = ObsEncoder(
            h_size=belief_size, s_size=state_size, activation=self.act_fn, embedding_size=embedding_size, hidden_size=hidden_size, min_std_dev=self.min_std_dev)

        self.modules = [self.fc_embed_state_action, self.stochastic_state_model, self.obs_encoder]



        # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
        # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
        # t :  0  1  2  3  4  5
        # o :    -X--X--X--X--X-
        # a : -X--X--X--X--X-
        # n : -X--X--X--X--X-
        # pb: -X-
        # ps: -X-
        # b : -x--X--X--X--X--X-
        # s : -x--X--X--X--X--X-


    @jit.script_method
    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor, observations: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        '''
        generate a sequence of data

        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                    torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = \
                        [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if observations is None else posterior_states[t]
            _state = _state if nonterminals is None else _state * \
                nonterminals[t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(
                torch.cat([_state, actions[t]], dim=1)))
            # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            # s_t ~ p(s_t | h_t) (Stochastic State Model)
            """
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(
                self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t +1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + \
                prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            """
            prior_states[t + 1] = self.stochastic_state_model.sample({'h_t': beliefs[t + 1]})["s_t"]
            loc_and_scale = self.stochastic_state_model({'h_t': beliefs[t + 1]})
            prior_means[t + 1], prior_std_devs[t + 1] = loc_and_scale['loc'], loc_and_scale['scale']

            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                # s_t ~ q(s_t | h_t, o_t) (Observation Model)
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                """
                hidden = self.act_fn(self.fc_embed_belief_posterior(
                    torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(
                    self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t +
                                    1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + \
                    posterior_std_devs[t + 1] * \
                    torch.randn_like(posterior_means[t + 1])
                """
                posterior_means[t + 1] = self.obs_encoder.sample({'h_t': beliefs[t + 1], 'o_t': observations[t_ + 1]})['s_t']
                loc_and_scale = self.obs_encoder({'h_t': beliefs[t + 1], 'o_t': observations[t_ + 1]})
                posterior_means[t + 1] = loc_and_scale['means']
                posterior_std_devs[t + 1] = loc_and_scale['scale']

        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(
            prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(
                posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


# class SymbolicObservationModel(jit.ScriptModule):
class DenseDecoder(nn.Module):
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation

class ObsEncoder(Normal):
    """o_t ~ p(o_t | h_t, s_t)"""
    def __init__(self, h_size, s_size, activation, embedding_size, hidden_size, min_std_dev):
        super().__init__(var=["s_t"], cond_var=["h_t", "o_t"])
        self.activation = activation
        self.fc1 = nn.Linear(
            h_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * s_size)
        self.min_std_dev = min_std_dev

    def forward(self, h_t, o_t):
        hidden = self.activation(self.fc1(torch.cat([h_t, o_t], dim=1)))
        loc, scale = torch.chunk(self.fc2(hidden), 2, dim=1)
        scale = F.softplus(scale) + self.min_std_dev
        return {"loc": loc, "scale": scale}

class StochasticStateModel(Normal):
    """p(s_t | h_t)"""
    def __init__(self, h_size, hidden_size, activation, s_size, min_std_dev):
        super().__init__(var=['s_t'], cond_var=['h_t'], name="StochasticStateModel")
        self.fc1 = nn.Linear(h_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * s_size)
        self.activation = activation
        self.min_std_dev = min_std_dev

    def forward(self, h_t):
        hidden = self.activation(self.fc1(h_t))
        loc, scale = torch.chunk(
            self.fc2(hidden), 2, dim=1)
        scale = F.softplus(scale) + self.min_std_dev
        return {"loc": loc, "scale": scale}


class ConvDecoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.fc1, self.conv1,
                        self.conv2, self.conv3, self.conv4]

    def forward(self, belief, state):
        # No nonlinearity here
        hidden = self.fc1(torch.cat([belief, state], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    if symbolic:
        return DenseDecoder(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return ConvDecoder(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class ValueModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ActorModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, action_size, dist='tanh_normal',
                 activation_function='elu', min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2*action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    @jit.script_method
    def forward(self, belief, state):
        raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * \
            torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return action_mean, action_std

    def get_action(self, belief, state, det=False):
        action_mean, action_std = self.forward(belief, state)
        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = torch.distributions.Independent(dist, 1)
        dist = SampleDist(dist)
        if det:
            return dist.mode()
        else:
            return dist.rsample()


class SymbolicEncoder(jit.ScriptModule):
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class VisualEncoder(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        hidden = self.fc(hidden)
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(
            1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()
