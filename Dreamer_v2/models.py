from typing import Any, Dict, Optional, List, Union, Tuple
from pixyz import distributions
from pixyz.distributions.exponential_distributions import Categorical
import torch
from torch import jit, nn, var
from torch.nn import functional as F
import torch.distributions
from torch.distributions.kl import kl_divergence
import numpy as np
from pixyz.distributions import Normal

from schedulers import init_scheduler
# Wraps the input tuple for a function to process a (time, batch, features sequence in batch, features) (assumes one output)


def bottle_tuple(f, x_tuple, var_name: str = '', kwargs={}):
    # x_tuple: (T, B, features...)
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1],
                                   *x[1][2:]), zip(x_tuple, x_sizes)), **kwargs)
    if var_name != '':
        y = y[var_name]
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class ModifiedCategorical(torch.distributions.OneHotCategorical):
    def __init__(self, probs=None, logits=None):
        """OneHotCategorical dist with raparameterization method
            called Straight-Through Grads with Auto-Diff"""
        self.dist = super().__init__(logits=logits, probs=probs)

    def mode(self) -> torch.Tensor:
        """return values which is most likely to be selected"""
        num_classes = self.dist.probs.size()[-1]
        return torch.nn.functional.one_hot(torch.argmax(self.dist.probs, dim=-1), num_classes=num_classes)

    def sample(self, sample_shape=()) -> torch.Tensor:
        sample = self.dist.sample(sample_shape=sample_shape)
        while len(probs.size()) < len(sample.size()):
            probs = probs[None]  # add dimension to the top
        sample += probs - probs.detach()  # Straight-Through Gradients with Auto-Diff
        return sample


class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size: int, state_size: int, num_action: int, hidden_size: int, embedding_size: int, kl_free: str, kl_scale: str, kl_balance: str, activation_function: str = 'relu', min_std_dev: float = 0.1, disable_gru_norm: bool = False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(
            state_size + num_action, belief_size)
        self.rnn = NormGRUCell(
            belief_size=belief_size, norm=not disable_gru_norm)
        # pixyz dists
        self.stochastic_state_model = StochasticStateModel(
            h_size=belief_size, s_size=state_size, hidden_size=hidden_size, min_std_dev=self.min_std_dev)

        self.obs_encoder = ObsEncoder(
            h_size=belief_size, s_size=state_size, activation=self.act_fn, embedding_size=embedding_size, hidden_size=hidden_size, min_std_dev=self.min_std_dev)

        # initialize schedulers of constants related with kld calculation
        self.kl_balance_sched = init_scheduler(config=kl_balance)
        self.kl_free_sched = init_scheduler(config=kl_free)
        self.kl_scale_sched = init_scheduler(config=kl_scale)

        self.modules = [self.fc_embed_state_action,
                        self.stochastic_state_model, self.obs_encoder, self.rnn]
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

    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor, observations: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        '''
        generate a sequence of data

        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_logits, posterior_states, posterior_logits
        '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_logits, posterior_states, posterior_logits = \
            [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(
                0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], posterior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

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
            prior_states[t + 1] = self.stochastic_state_model.sample(
                {'h_t': beliefs[t + 1]}, reparam=True)["s_t"]
            prior_logits[t +
                         1] = self.stochastic_state_model(h_t=beliefs[t + 1])["probs"]

            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                # s_t ~ q(s_t | h_t, o_t) (Observation Model)
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                posterior_states[t + 1] = self.obs_encoder.sample(
                    {'h_t': beliefs[t + 1], 'o_t': observations[t_ + 1]}, reparam=True)['s_t']
                posterior_logits[t + 1] = self.obs_encoder(
                    h_t=beliefs[t + 1], o_t=observations[t_ + 1])['probs']

        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(
            prior_logits[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(
                posterior_logits[1:], dim=0)]
        return hidden

    def calc_kld(self, current_step: int, posterior_logits: torch.Tensor, prior_logits: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """calculate KL Divergence for the given priors and posteriors"""
        # get constants using schedulers
        kl_free = self.kl_free_sched(current_step)
        kl_balance = self.kl_balance_sched(current_step)
        kl_scale = self.kl_scale_sched(current_step)
        def dist(probs): return torch.distributions.Independent(
            ModifiedCategorical(probs=probs), 1)
        if kl_balance == 0.5:
            value = kl_divergence(dist(posterior_logits), dist(prior_logits))
            loss = value.mean()  # this value should be (1, )
        else:
            value_lhs = value = kl_divergence(
                dist(posterior_logits), dist(prior_logits.detach()))
            loss_lhs = torch.max(value_lhs.mean(), torch.tensor(kl_free))
            value_rhs = kl_divergence(
                dist(posterior_logits.detach()), dist(prior_logits))
            loss_lhs = torch.max(value_rhs.mean(), torch.tensor(kl_free))
            loss = (1 - kl_balance) * loss_lhs + kl_balance * kl_balance

        loss *= kl_scale
        return loss, value


class DenseDecoder(Normal):
    def __init__(self, observation_size: torch.Tensor, belief_size: torch.Tensor, state_size: int, embedding_size: int, activation_function: str = 'relu'):
        super().__init__(var=['o_t'], cond_var=['h_t', 's_t'])
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, h_t, s_t) -> Dict:
        # reshape inputs
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.view(T*B, *features_shape)
        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.view(T*B, *features_shape)

        hidden = self.act_fn(self.fc1(torch.cat([h_t, s_t], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        features_shape = observation.size()[1:]
        observation = observation.view(T, B, *features_shape)
        return {'loc': observation, 'scale': 1.0}


class ObsEncoder(Categorical):
    """s_t ~ p(s_t | h_t, o_t)"""

    def __init__(self, h_size: int, s_size: int, activation: nn.Module, embedding_size: int, hidden_size: int, min_std_dev: float):

        super().__init__(var=["s_t"], cond_var=["h_t", "o_t"])
        self.activation = activation
        self.fc1 = nn.Linear(
            h_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, s_size)
        self.min_std_dev = min_std_dev
        self.softmax = nn.Softmax(dim=-1)
        self.modules = [self.fc1, self.fc2]

    def forward(self, h_t: torch.Tensor, o_t: torch.Tensor) -> Dict:
        hidden = self.activation(self.fc1(torch.cat([h_t, o_t], dim=1)))
        hidden = self.softmax(self.activation(self.fc2(hidden)))
        return {"probs": hidden}


class StochasticStateModel(distributions.Categorical):
    """p(s_t | h_t)"""

    def __init__(self, h_size: int, hidden_size: int, s_size: int, min_std_dev: float, activation: callable = nn.ELU):

        super().__init__(var=['s_t'], cond_var=[
            'h_t'], name="StochasticStateModel")
        self.fcs = nn.Sequential(
            nn.Linear(h_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, s_size),
            activation(),
            nn.Softmax(dim=-1)
        )
        self.activation = activation
        self.min_std_dev = min_std_dev

    def forward(self, h_t) -> Dict:
        hidden = self.fcs(h_t)
        return {"probs": hidden}


class ConvDecoder(Normal):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size: int, state_size: int, embedding_size: int, activation_function: str = 'relu'):
        super().__init__(var=['o_t'], cond_var=['h_t', 's_t'])
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.fc1, self.conv1,
                        self.conv2, self.conv3, self.conv4]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.view(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.view(T*B, *features_shape)

        # No nonlinearity here
        hidden = self.fc1(torch.cat([h_t, s_t], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        features_shape = observation.size()[1:]
        observation = observation.view(T, B, *features_shape)
        return {'loc': observation, 'scale': 1.0}


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu') -> nn.Module:
    if symbolic:
        return DenseDecoder(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return ConvDecoder(belief_size, state_size, embedding_size, activation_function)


class RewardModel(Normal):
    def __init__(self, h_size: int, s_size: int, hidden_size: int, activation='relu'):
        # p(o_t | h_t, s_t)
        super().__init__(cond_var=['h_t', 's_t'], var=['r_t'])
        self.act_fn = getattr(F, activation)
        self.fc1 = nn.Linear(s_size + h_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.view(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.view(T*B, *features_shape)

        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        features_shape = reward.size()[1:]
        reward = reward.view(T, B, *features_shape)
        scale = torch.ones_like(reward)
        return {'loc': reward, 'scale': scale}


class ValueModel(Normal):
    def __init__(self, belief_size: int, state_size: int, hidden_size: int, activation_function: str = 'relu'):
        super().__init__(cond_var=['h_t', 's_t'], var=['r_t'])
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.view(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.view(T*B, *features_shape)

        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        loc = self.fc4(hidden).squeeze(dim=1)
        features_shape = loc.size()[1:]
        loc = loc.view(T, B, *features_shape)
        scale = torch.ones_like(loc)
        return {'loc': loc, 'scale': scale}


class Pie(Normal):
    def __init__(self, belief_size: int, state_size: int, hidden_size: int, num_action: int, dist: str = 'tanh_normal',
                 activation_function: str = 'elu', min_std: float = 1e-4, init_std: float = 5, mean_scale: float = 5):
        super().__init__(cond_var=['h_t', 's_t'], var=['a_t'])
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2*num_action)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        raw_init_std = torch.log(torch.exp(torch.tensor(
            self._init_std, dtype=torch.float32)) - 1)
        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * \
            torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return {'loc': action_mean, 'scale': action_std}

class CategoricalActorModel(distributions.Categorical):
    def __init__(self, num_actions: int, h_size: int, s_size: int, num_layers: int, num_units: int, activation: Any = nn.ELU, ):
        super().__init__(cond_var=['h_t', 's_t'], var=['a_t'])
        assert num_layers >= 2, f'This model requires at least 2 layers, but {num_layers} are given'
        layers = [nn.Linear(in_features=h_size + s_size,
                            out_features=num_units), activation()]
        for _ in range(num_layers - 2):
            layers.append(
                nn.Linear(in_features=num_units, out_features=num_units))
            layers.append(activation())
        layers.append(nn.Linear(in_features=num_units, out_features=num_actions))
        layers.append(nn.Softmax(dim=-1))
        self.fc = nn.Sequential(*layers)

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        inputs = torch.cat([h_t, s_t], axis=-1)
        out = self.fc(inputs)
        return {'probs': out}

    def get_action(self, h_t: torch.Tensor, s_t: torch.Tensor, det: bool = False) -> torch.Tensor:
        if det:
            probs = self.forward(h_t=h_t, s_t=s_t)["probs"]
            action = np.argmax(probs, axis=-1)
            return action
        else:
            return self.sample({"h_t": h_t, "s_t": s_t}, reparam=True)["a_t"]


class SymbolicEncoder(jit.ScriptModule):
    def __init__(self, observation_size: int, embedding_size: int, activation_function: str = 'relu'):
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

    def __init__(self, embedding_size: int, activation_function: str = 'relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2,
                        self.conv3, self.conv4, self.fc]

    @jit.script_method
    def forward(self, observation: torch.Tensor):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        hidden = self.fc(hidden)
        return hidden


def Encoder(symbolic: bool, observation_size: int, embedding_size: int, activation_function: str = 'relu') -> Union[SymbolicEncoder, VisualEncoder]:
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)


class NormGRUCell(nn.Module):
    def __init__(self, belief_size, norm=False, act=torch.tanh, update_bias=-1, **kwargs):
        super().__init__()
        self._size = belief_size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(
            2 * belief_size, 3 * belief_size, bias=norm is not None, **kwargs)
        if norm:
            self._norm = nn.LayerNorm(normalized_shape=3*belief_size, eps=1e-3)

    def forward(self, inputs, state):
        # state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.chunk(parts, 3, -1)
        # print(reset.size(), cand.size(), update.size()) # (50, 400)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        # print(cand.size()) (50, 400)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output
