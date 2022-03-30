import os
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, Optional, List
import torch
from torch import jit, nn, var, optim
from torch.nn import functional as F
import torch.distributions
from torch.distributions.kl import kl_divergence
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions import Normal

from pixyz.distributions import Normal as pixyz_Normal

# Wraps the input tuple for a function to process a (time, batch, features sequence in batch, features) (assumes one output)


def bottle_tupele_multimodal(f, x_tuples, var_name: str = "", kwargs={}):
    xs_size = []
    xs = dict()
    for name in x_tuples.keys():
        x_size = x_tuples[name].size()
        x = x_tuples[name].reshape(x_size[0] * x_size[1], *x_size[2:])

        xs_size.append(x_size)
        xs[name] = x

    y = f(xs, **kwargs)
    if var_name != "":
        y = y[var_name]
    y_size = y.size()
    output = y.reshape(xs_size[0][0], xs_size[0][1], *y_size[1:])
    return output


class TransitionModel(nn.Module):
    __constants__ = ["min_std_dev"]

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function="relu", min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)

        # pixyz dists
        self.stochastic_state_model = StochasticStateModel(h_size=belief_size, s_size=state_size, hidden_size=hidden_size, activation=self.act_fn, min_std_dev=self.min_std_dev)

        self.obs_encoder = ObsEncoder(
            h_size=belief_size,
            s_size=state_size,
            activation=self.act_fn,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            min_std_dev=self.min_std_dev,
        )

        self.modules = [self.fc_embed_state_action, self.stochastic_state_model, self.obs_encoder, self.rnn]
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

    def forward(
        self,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_belief: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        generate a sequence of data

        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                    torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = (
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
        )
        beliefs[0], posterior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if observations is None else posterior_states[t]
            _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            # s_t ~ p(s_t | h_t) (Stochastic State Model)
            prior_states[t + 1] = self.stochastic_state_model.sample({"h_t": beliefs[t + 1]}, reparam=True)["s_t"]
            loc_and_scale = self.stochastic_state_model(h_t=beliefs[t + 1])
            prior_means[t + 1], prior_std_devs[t + 1] = loc_and_scale["loc"], loc_and_scale["scale"]

            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                # s_t ~ q(s_t | h_t, o_t) (Observation Model)
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                posterior_states[t + 1] = self.obs_encoder.sample({"h_t": beliefs[t + 1], "o_t": observations[t_ + 1]}, reparam=True)["s_t"]
                loc_and_scale = self.obs_encoder(h_t=beliefs[t + 1], o_t=observations[t_ + 1])
                posterior_means[t + 1] = loc_and_scale["loc"]
                posterior_std_devs[t + 1] = loc_and_scale["scale"]

        # Return new hidden states
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


class ObsEncoder(pixyz_Normal):
    """s_t ~ p(s_t | h_t, o_t)"""

    def __init__(self, h_size: int, s_size: int, activation: nn.Module, embedding_size: int, hidden_size: int, min_std_dev: float):

        super().__init__(var=["s_t"], cond_var=["h_t", "o_t"])
        self.activation = activation
        self.fc1 = nn.Linear(h_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * s_size)
        self.min_std_dev = min_std_dev
        self.modules = [self.fc1, self.fc2]

    def forward(self, h_t: torch.Tensor, o_t: torch.Tensor) -> Dict:
        hidden = self.activation(self.fc1(torch.cat([h_t, o_t], dim=1)))
        loc, scale = torch.chunk(self.fc2(hidden), 2, dim=1)
        scale = F.softplus(scale) + self.min_std_dev
        return {"loc": loc, "scale": scale}


class StochasticStateModel(pixyz_Normal):
    """p(s_t | h_t)"""

    def __init__(self, h_size: int, hidden_size: int, activation: nn.Module, s_size: int, min_std_dev: float):

        super().__init__(var=["s_t"], cond_var=["h_t"], name="StochasticStateModel")
        self.fc1 = nn.Linear(h_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * s_size)
        self.activation = activation
        self.min_std_dev = min_std_dev

    def forward(self, h_t) -> Dict:
        hidden = self.activation(self.fc1(h_t))
        loc, scale = torch.chunk(self.fc2(hidden), 2, dim=1)
        scale = F.softplus(scale) + self.min_std_dev
        return {"loc": loc, "scale": scale}


class ValueModel(pixyz_Normal):
    def __init__(self, belief_size: int, state_size: int, hidden_size: int, activation_function: str = "relu"):
        super().__init__(cond_var=["h_t", "s_t"], var=["r_t"])
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T * B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T * B, *features_shape)

        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        loc = self.fc4(hidden).squeeze(dim=1)
        features_shape = loc.size()[1:]
        loc = loc.reshape(T, B, *features_shape)
        scale = torch.ones_like(loc)
        return {"loc": loc, "scale": scale}


class Pie(pixyz_Normal):
    def __init__(
        self,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        action_size: int,
        dist: str = "tanh_normal",
        activation_function: str = "elu",
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
    ):
        super().__init__(cond_var=["h_t", "s_t"], var=["a_t"])
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2 * action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        raw_init_std = torch.log(torch.exp(torch.tensor(self._init_std, dtype=torch.float32)) - 1)
        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return {"loc": action_mean, "scale": action_std}


class ActorModel(nn.Module):
    def __init__(
        self,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        action_size: int,
        dist: str = "tanh_normal",
        activation_function: str = "elu",
        min_std: float = 1e-4,
        init_std: float = 5,
        mean_scale: float = 5,
    ):
        super().__init__()
        self.pie = Pie(
            belief_size,
            state_size,
            hidden_size,
            action_size,
            dist=dist,
            activation_function=activation_function,
            min_std=min_std,
            init_std=init_std,
            mean_scale=mean_scale,
        )

    def get_action(self, belief: torch.Tensor, state: torch.Tensor, det: bool = False) -> torch.Tensor:
        if det:
            # get mode
            actions = self.pie.sample({"h_t": belief, "s_t": state}, sample_shape=[100], reparam=True)["a_t"]  # (100, 2450, 6)
            actions = torch.tanh(actions)
            batch_size = actions.size(1)
            feature_size = actions.size(2)
            logprob = self.pie.get_log_prob({"h_t": belief, "s_t": state, "a_t": actions}, sum_features=False)  # (100, 2450, 6)
            logprob -= torch.log(1 - actions.pow(2) + 1e-6)
            logprob = logprob.sum(dim=-1)
            indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
            return torch.gather(actions, 0, indices).squeeze(0)

        else:
            return torch.tanh(self.pie.sample({"h_t": belief, "s_t": state}, reparam=True)["a_t"])


class RewardModel(pixyz_Normal):
    def __init__(self, h_size: int, s_size: int, hidden_size: int, activation="relu"):
        # p(r_t | h_t, s_t)
        super().__init__(cond_var=["h_t", "s_t"], var=["r_t"])
        self.act_fn = getattr(F, activation)
        self.fc1 = nn.Linear(s_size + h_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T * B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T * B, *features_shape)

        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        features_shape = reward.size()[1:]
        reward = reward.reshape(T, B, *features_shape)
        scale = torch.ones_like(reward)
        return {"loc": reward, "scale": scale}


class DenseDecoder(pixyz_Normal):
    def __init__(self, observation_size: torch.Tensor, belief_size: torch.Tensor, state_size: int, embedding_size: int, activation_function: str = "relu"):
        super().__init__(var=["o_t"], cond_var=["h_t", "s_t"])
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, h_t, s_t) -> Dict:
        # reshape inputs
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T * B, *features_shape)
        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T * B, *features_shape)

        hidden = self.act_fn(self.fc1(torch.cat([h_t, s_t], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        features_shape = observation.size()[1:]
        observation = observation.reshape(T, B, *features_shape)
        return {"loc": observation, "scale": 1.0}


class ConvDecoder(pixyz_Normal):
    __constants__ = ["embedding_size"]

    def __init__(self, belief_size: int, state_size: int, embedding_size: int, activation_function: str = "relu"):
        super().__init__(var=["o_t"], cond_var=["h_t", "s_t"])
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T * B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T * B, *features_shape)

        # No nonlinearity here
        hidden = self.fc1(torch.cat([h_t, s_t], dim=1))
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        features_shape = observation.size()[1:]
        observation = observation.reshape(T, B, *features_shape)
        return {"loc": observation, "scale": 1.0}


class SoundDecoder(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(SoundDecoder, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Sequential(nn.Linear(self.state_size + self.hidden_size, 250), nn.Tanh(), nn.Linear(250, 250))

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(5, 64, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2), bias=False),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 2), bias=False),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(64, 64, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(32, 32, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4), bias=False),
        )

        self.modules = [self.conv1, self.fc1]

    def forward(self, s_t: torch.Tensor, h_t: torch.Tensor):
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T * B, *features_shape)
        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T * B, *features_shape)
        x = torch.cat([h_t, s_t], dim=1)
        recon = self.fc1(x.reshape(T * B, -1))
        recon = self.conv1(recon.reshape(T * B, 5, 10, 5))
        recon = recon.squeeze(1)
        features_shape = recon.size()[1:]
        recon = recon.reshape(T, B, *features_shape)
        # return recon
        return {"loc": recon, "scale": 1.0}


class MultimodalObservationModel(nn.Module):
    __constants__ = ["embedding_size"]

    def __init__(
        self,
        observation_names,
        observation_shapes,
        visual_embedding_size: int,
        symbolic_embedding_size: int,
        belief_size: int,
        state_size: int,
        cnn_activation_function: str = "relu",
        dense_activation_function: str = "relu",
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.observation_names = observation_names
        self.observation_models = dict()
        self.modules = []
        for name in self.observation_names:
            observation_shape = observation_shapes[name]
            if "image" in name:
                self.observation_models[name] = ConvDecoder(belief_size, state_size, visual_embedding_size, cnn_activation_function).to(device)
            elif name == "sound":
                self.observation_models[name] = SoundDecoder(state_size=state_size, hidden_size=belief_size).to(device)
            else:
                self.observation_models[name] = DenseDecoder(observation_shape[0], belief_size, state_size, symbolic_embedding_size, dense_activation_function).to(device)
            self.modules += self.observation_models[name].modules

    # @jit.script_method
    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor):
        preds = dict()
        for name in self.observation_models.keys():
            pred = self.observation_models[name](h_t, s_t)
            preds[name] = pred
        return preds

    def get_log_prob(self, inputs, sum_features=False):
        observation_log_prob = dict()
        h_t = inputs["h_t"]
        s_t = inputs["s_t"]
        o_t = inputs["o_t"]
        for name in self.observation_names:
            log_prob = self.observation_models[name].get_log_prob({"h_t": h_t, "s_t": s_t, "o_t": o_t[name]}, sum_features=sum_features)
            observation_log_prob[name] = log_prob
        return observation_log_prob


class SymbolicEncoder(jit.ScriptModule):
    def __init__(self, observation_size: int, embedding_size: int, activation_function: str = "relu"):
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
    __constants__ = ["embedding_size"]

    def __init__(self, embedding_size: int, activation_function: str = "relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]

    @jit.script_method
    def forward(self, observation: torch.Tensor):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.reshape(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        hidden = self.fc(hidden)
        return hidden


class SoundEncoder(nn.Module):
    def __init__(self, embbed_size=250):
        super(SoundEncoder, self).__init__()
        self.embbed_size = embbed_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4), bias=False),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
            nn.Conv2d(32, 128, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
            nn.Conv2d(64, 256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.BatchNorm2d(256, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
            nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
            nn.Conv2d(64, 10, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2), bias=False),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True),
            nn.GLU(dim=1),
        )
        if embbed_size == 250:
            self.modules = [self.conv]
        else:
            self.fc = nn.Linear(250, self.embbed_size)
            self.modules = [self.conv, self.fc]

    def forward(self, spec: torch.Tensor):
        T = spec.size()[0]
        spec = spec.unsqueeze(1)
        z = self.conv(spec)
        z = z.reshape(T, -1)
        if self.embbed_size != 250:
            z = self.fc(z)
        return z


class MultimodalEncoder(nn.Module):
    __constants__ = ["embedding_size"]

    def __init__(
        self,
        observation_names,
        observation_shapes,
        embedding_size: int,
        visual_embedding_size: int,
        sound_embedding_size: int,
        symbolic_embedding_size: int,
        cnn_activation_function: str = "relu",
        dense_activation_function: str = "relu",
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.observation_names = observation_names
        self.embedding_size = embedding_size
        self.act_fns = dict()
        self.encoders = dict()
        self.modules = []
        multimodal_embedding_size = 0
        for name in self.observation_names:
            observation_shape = observation_shapes[name]

            if "image" in name:
                self.encoders[name] = VisualEncoder(visual_embedding_size, cnn_activation_function).to(device)
                multimodal_embedding_size += visual_embedding_size
                self.act_fns[name] = getattr(F, cnn_activation_function)
            elif name == "sound":
                self.encoders[name] = SoundEncoder(embbed_size=sound_embedding_size).to(device)
                multimodal_embedding_size += sound_embedding_size
                self.act_fns[name] = getattr(F, cnn_activation_function)
            else:
                self.encoders[name] = SymbolicEncoder(observation_shape[0], symbolic_embedding_size, dense_activation_function).to(device)
                multimodal_embedding_size += symbolic_embedding_size
                self.act_fns[name] = getattr(F, dense_activation_function)
            self.modules += self.encoders[name].modules
        if len(self.encoders.keys()) == 1 and multimodal_embedding_size == embedding_size:
            self.fc = lambda x: x
        else:
            self.fc = nn.Linear(multimodal_embedding_size, embedding_size)
            self.modules += [self.fc]

    def get_obs(self, observations, name):
        if name in observations.keys():
            return observations[name]
        elif (name == "observation") and ("image" in observations.keys()):
            return observations["image"]
        elif (name == "image") and ("observation" in observations.keys()):
            return observations["observation"]
        else:
            raise NotImplementedError

    # @jit.script_method
    def forward(self, observations):
        hiddens = []
        for name in self.encoders.keys():
            _obs = self.get_obs(observations, name)
            hid = self.act_fns[name](self.encoders[name](_obs))
            hiddens.append(hid)
        hidden = torch.cat(hiddens, dim=-1)
        hidden = self.fc(hidden)
        return hidden


class RSSM:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.init_models(device)
        self.init_param_list()
        self.init_optimizer()

        if cfg.main.models != "" and os.path.exists(cfg.main.models):
            self.load_model(cfg.main.models)

        self.global_prior = Normal(torch.zeros(cfg.train.batch_size, cfg.model.state_size, device=device), torch.ones(cfg.train.batch_size, cfg.model.state_size, device=device))  # Global prior N(0, I)
        # Allowed deviation in KL divergence
        self.free_nats = torch.full((1,), cfg.model.free_nats, device=device)

        self.model_modules = self.transition_model.modules + self.encoder.modules + self.observation_model.modules + self.reward_model.modules

    def init_models(self, device):
        self.transition_model = TransitionModel(self.cfg.model.belief_size, self.cfg.model.state_size, self.cfg.env.action_size, self.cfg.model.hidden_size, self.cfg.model.embedding_size, self.cfg.model.dense_activation_function).to(device=device)
        self.reward_model = RewardModel(h_size=self.cfg.model.belief_size, s_size=self.cfg.model.state_size, hidden_size=self.cfg.model.hidden_size, activation=self.cfg.model.dense_activation_function).to(device=device)

        self.observation_model = MultimodalObservationModel(
            observation_names=self.cfg.model.observation_names,
            observation_shapes=self.cfg.env.observation_shapes,
            visual_embedding_size=self.cfg.model.visual_embedding_size,
            symbolic_embedding_size=self.cfg.model.symbolic_embedding_size,
            belief_size=self.cfg.model.belief_size,
            state_size=self.cfg.model.state_size,
            cnn_activation_function=self.cfg.model.cnn_activation_function,
            dense_activation_function=self.cfg.model.dense_activation_function,
            device=device,
        ).to(device=device)

        self.encoder = MultimodalEncoder(
            observation_names=self.cfg.model.observation_names,
            observation_shapes=self.cfg.env.observation_shapes,
            embedding_size=self.cfg.model.embedding_size,
            visual_embedding_size=self.cfg.model.visual_embedding_size,
            sound_embedding_size=self.cfg.model.sound_embedding_size,
            symbolic_embedding_size=self.cfg.model.symbolic_embedding_size,
            cnn_activation_function=self.cfg.model.cnn_activation_function,
            dense_activation_function=self.cfg.model.dense_activation_function,
            device=device,
        ).to(device=device)

        if self.cfg.main.wandb:
            wandb.watch(self.transition_model)
            wandb.watch(self.observation_model)
            wandb.watch(self.reward_model)
            wandb.watch(self.encoder)

    def init_param_list(self):
        observation_model_params = list(self.observation_model.parameters())
        for model in self.observation_model.observation_models.values():
            observation_model_params += list(model.parameters())

        encoder_params = list(self.encoder.parameters())
        for model in self.encoder.encoders.values():
            encoder_params += list(model.parameters())

        self.param_list = list(self.transition_model.parameters()) + observation_model_params + list(self.reward_model.parameters()) + encoder_params

    def init_optimizer(self):
        self.model_optimizer = optim.Adam(self.param_list, lr=0 if self.cfg.model.learning_rate_schedule != 0 else self.cfg.model.model_learning_rate, eps=self.cfg.model.adam_epsilon)

    def clip_obs(self, observations, idx_start=0, idx_end=None):
        output = dict()
        for k in observations.keys():
            output[k] = observations[k][idx_start:idx_end]
        return output

    def estimate_state(self, observations_target, actions, rewards, nonterminals, batch_size=None):
        if batch_size == None:
            batch_size = self.cfg.train.batch_size

        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(batch_size, self.cfg.model.belief_size, device=self.cfg.main.device), torch.zeros(batch_size, self.cfg.model.state_size, device=self.cfg.main.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)

        obs_emb = bottle_tupele_multimodal(self.encoder, observations_target)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(init_state, actions[:-1], init_belief, obs_emb, nonterminals[:-1])

        states = dict(beliefs=beliefs, prior_states=prior_states, prior_means=prior_means, prior_std_devs=prior_std_devs, posterior_states=posterior_states, posterior_means=posterior_means, posterior_std_devs=posterior_std_devs)
        return states
        # return beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs

    def optimize_loss(self, observations_target, actions, rewards, nonterminals, states, writer, step):

        beliefs = states["beliefs"]
        prior_states = states["prior_states"]
        prior_means = states["prior_means"]
        prior_std_devs = states["prior_std_devs"]
        posterior_states = states["posterior_states"]
        posterior_means = states["posterior_means"]
        posterior_std_devs = states["posterior_std_devs"]

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        observations_loss = dict()
        observations_loss_sum = torch.tensor(0.0, device=self.cfg.main.device)

        if self.cfg.model.worldmodel_LogProbLoss:
            log_probs = self.observation_model.get_log_prob({"h_t": beliefs, "s_t": posterior_states, "o_t": observations_target}, sum_features=False)
            for name in self.cfg.model.observation_names:
                observation_shape = self.cfg.env.observation_shapes[name]
                dim = tuple(np.arange(2, 2 + len(observation_shape)))
                observation_loss = (-log_probs[name]).sum(dim=dim).mean(dim=(0, 1))

                observations_loss[name] = observation_loss
                observations_loss_sum += observation_loss
        else:
            observations_pred = self.observation_model(h_t=beliefs, s_t=posterior_states)

            for name in self.cfg.model.observation_names:
                observation_mean = observations_pred[name]["loc"]
                observation_shape = self.cfg.env.observation_shapes[name]
                dim = tuple(np.arange(2, 2 + len(observation_shape)))
                observation_loss = F.mse_loss(observation_mean, observations_target[name], reduction="none").sum(dim=dim).mean(dim=(0, 1))

                observations_loss[name] = observation_loss
                observations_loss_sum += observation_loss

        if self.cfg.model.worldmodel_LogProbLoss:
            reward_loss = -self.reward_model.get_log_prob({"h_t": beliefs, "s_t": posterior_states, "r_t": rewards[:-1]}, sum_features=False)
            reward_loss = reward_loss.mean(dim=(0, 1))
        else:
            reward_mean = self.reward_model(h_t=beliefs, s_t=posterior_states)["loc"]
            reward_loss = F.mse_loss(reward_mean, rewards[:-1], reduction="none").mean(dim=(0, 1))

        # transition loss
        div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
        # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))
        if self.cfg.model.global_kl_beta != 0:
            kl_loss += self.cfg.model.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), self.global_prior).sum(dim=2).mean(dim=(0, 1))
        # Calculate latent overshooting objective for t > 0
        if self.cfg.model.overshooting_kl_beta != 0:
            overshooting_vars = []  # Collect variables for overshooting to process in batch
            for t in range(1, self.cfg.train.chunk_size - 1):
                d = min(t + self.cfg.model.overshooting_distance, self.cfg.train.chunk_size - 1)  # Overshooting distance
                # Use t_ and d_ to deal with different time indexing for latent states
                t_, d_ = t - 1, d - 1
                # Calculate sequence padding so overshooting terms can be calculated in one batch
                seq_pad = (0, 0, 0, 0, 0, t - d + self.cfg.model.overshooting_distance)
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                overshooting_vars.append(
                    (
                        F.pad(actions[t:d], seq_pad),
                        F.pad(nonterminals[t:d], seq_pad),
                        F.pad(rewards[t:d], seq_pad[2:]),
                        beliefs[t_],
                        prior_states[t_],
                        F.pad(posterior_means[t_ + 1 : d_ + 1].detach(), seq_pad),
                        F.pad(posterior_std_devs[t_ + 1 : d_ + 1].detach(), seq_pad, value=1),
                        F.pad(torch.ones(d - t, self.cfg.train.batch_size, self.cfg.model.state_size, device=self.cfg.main.device), seq_pad),
                    )
                )  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs = self.transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
            seq_mask = torch.cat(overshooting_vars[7], dim=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss += (
                (1 / self.cfg.model.overshooting_distance)
                * self.cfg.model.overshooting_kl_beta
                * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), self.free_nats).mean(dim=(0, 1))
                * (self.cfg.train.chunk_size - 1)
            )  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
            # Calculate overshooting reward prediction loss with sequence mask
            if self.cfg.model.overshooting_reward_scale != 0:
                reward_loss += (
                    (1 / self.cfg.model.overshooting_distance) * self.cfg.model.overshooting_reward_scale * F.mse_loss(self.reward_model(beliefs, prior_states)["loc"] * seq_mask[:, :, 0], torch.cat(overshooting_vars[2], dim=1), reduction="none").mean(dim=(0, 1)) * (self.cfg.train.chunk_size - 1)
                )  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        # Apply linearly ramping learning rate schedule
        if self.cfg.model.learning_rate_schedule != 0:
            for group in self.model_optimizer.param_groups:
                group["lr"] = min(group["lr"] + self.cfg.model.model_learning_rate / self.cfg.model.learning_rate_schedule, self.cfg.model.model_learning_rate)
        model_loss = observations_loss_sum + reward_loss + kl_loss

        # Update model parameters
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.param_list, self.cfg.model.grad_clip_norm, norm_type=2)
        self.model_optimizer.step()

        # Log loss info
        loss_info = dict()
        loss_info["observations_loss_sum"] = observations_loss_sum.item()
        for name in observations_loss.keys():
            loss_info["observation_{}_loss".format(name)] = observations_loss[name].item()
        loss_info["reward_loss"] = reward_loss.item()
        loss_info["kl_loss"] = kl_loss.item()

        for name in loss_info.keys():
            writer.add_scalar(name, loss_info[name], step)

        return loss_info

    def optimize(self, observations_raw, actions, rewards, nonterminals, writer, step):
        observations = dict()
        for key in self.cfg.model.observation_names:
            observations[key] = observations_raw[key]
        observations_target = self.clip_obs(observations, idx_start=1)

        states = self.estimate_state(observations_target, actions, rewards, nonterminals)

        loss_info = self.optimize_loss(observations_target, actions, rewards, nonterminals, states, writer, step)
        return loss_info

    # def load_state_dict(self, model_path):
    #     return torch.load(model_path, map_location=torch.device(self.device))

    def load_model(self, model_path):
        model_dicts = torch.load(model_path, map_location=torch.device(self.device))
        self.load_state_dict(model_dicts)

    def load_state_dict(self, model_dicts):
        observation_model_state_dict = model_dicts["observation_model"]
        self.observation_model.load_state_dict(observation_model_state_dict["main"])
        for name in self.observation_model.observation_names:
            self.observation_model.observation_models[name].load_state_dict(observation_model_state_dict[name])

        encoder_state_dict = model_dicts["encoder"]
        for name in encoder_state_dict.keys():
            if name == "main":
                self.encoder.load_state_dict(encoder_state_dict["main"])
            else:
                self.encoder.encoders[name].load_state_dict(encoder_state_dict[name])

        self.transition_model.load_state_dict(model_dicts["transition_model"])
        self.reward_model.load_state_dict(model_dicts["reward_model"])
        self.model_optimizer.load_state_dict(model_dicts["model_optimizer"])

    def get_state_dict(self):
        observation_model_state_dict = dict(main=self.observation_model.state_dict())
        for name in self.observation_model.observation_models.keys():
            observation_model_state_dict[name] = self.observation_model.observation_models[name].state_dict()

        encoder_state_dict = dict(main=self.encoder.state_dict())
        for name in self.encoder.encoders.keys():
            encoder_state_dict[name] = self.encoder.encoders[name].state_dict()

        state_dict = {
            "transition_model": self.transition_model.state_dict(),
            "observation_model": observation_model_state_dict,
            "reward_model": self.reward_model.state_dict(),
            "encoder": encoder_state_dict,
            "model_optimizer": self.model_optimizer.state_dict(),
        }
        return state_dict

    def save_model(self, results_dir, itr):
        state_dict = self.get_state_dict()
        torch.save(state_dict, os.path.join(results_dir, "models_%d.pth" % itr))

    def imagine_fix_action(self, prev_state, prev_belief, raw_actions, transition_model: TransitionModel, planning_horizon=12, t_imag_start=0):
        """
        imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
        Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
        Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        horizon = np.min([len(prev_belief), planning_horizon])
        horizon = np.min([len(raw_actions), planning_horizon])
        flatten = lambda x: x[t_imag_start : t_imag_start + 1].reshape([-1] + list(x.size()[2:]))
        prev_belief = flatten(prev_belief)
        prev_state = flatten(prev_state)

        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = horizon + 1
        beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0] = prev_belief, prev_state

        # Loop over time sequence
        for t in tqdm(range(T - 1), desc="imagination"):
            _state = prior_states[t]
            actions = raw_actions[t_imag_start + t]

            # Compute belief (deterministic hidden state)
            hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
            beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            """
            hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
            """
            prior_states[t + 1] = transition_model.stochastic_state_model.sample({"h_t": beliefs[t + 1]}, reparam=True)["s_t"]
            loc_and_scale = transition_model.stochastic_state_model(h_t=beliefs[t + 1])
            prior_std_devs[t + 1] = loc_and_scale["scale"]
            prior_means[t + 1] = loc_and_scale["loc"]

        # Return new hidden states
        imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        return imagined_traj

    def reconstruction_multimodal(self, observations, actions, rewards, nonterminals, det=False):
        # Create initial belief and state for time t = 0
        batch_size = actions.shape[1]
        init_belief, init_state = torch.zeros(batch_size, self.cfg.model.belief_size, device=self.cfg.main.device), torch.zeros(batch_size, self.cfg.model.state_size, device=self.cfg.main.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        observations_target = self.clip_obs(observations, idx_start=1)

        obs_emb = bottle_tupele_multimodal(self.encoder, observations_target)

        # beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(
        #     init_state, actions[:-1], init_belief, obs_emb, nonterminals[:-1], det=det)
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(init_state, actions[:-1], init_belief, obs_emb, nonterminals[:-1])

        observations_pred = self.observation_model(h_t=beliefs, s_t=posterior_states)
        reward_mean = self.reward_model(h_t=beliefs, s_t=posterior_states)["loc"]

        post = {"state": posterior_states, "mean": posterior_means, "std": posterior_std_devs}
        prior = {"state": prior_states, "mean": prior_means, "std": prior_std_devs}
        states = {"belief": beliefs, "prior": prior, "post": post}

        recons = {"reward": reward_mean, "states": states}
        for name in self.cfg.model.reconstruction_names:
            recons[name] = observations_pred[name]["loc"]

        return recons

    def imagine_multimodal(self, observations, actions, rewards, nonterminals, t_imag_start, planning_horizon):
        # Create initial belief and state for time t = 0
        batch_size = actions.shape[1]
        init_belief, init_state = torch.zeros(batch_size, self.cfg.model.belief_size, device=self.cfg.main.device), torch.zeros(batch_size, self.cfg.model.state_size, device=self.cfg.main.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)

        observations_target = self.clip_obs(observations, idx_start=1)
        obs_emb = bottle_tupele_multimodal(self.encoder, observations_target)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(init_state, actions[:-1], init_belief, obs_emb, nonterminals[:-1])

        actor_states = posterior_states.detach()[: t_imag_start + 1]
        actor_beliefs = beliefs.detach()[: t_imag_start + 1]
        imagination_traj = self.imagine_fix_action(actor_states, actor_beliefs, actions, self.transition_model, planning_horizon, t_imag_start)
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj

        observations_imag = self.observation_model(h_t=imged_beliefs, s_t=imged_prior_states)
        reward_imag_mean = self.reward_model(h_t=imged_beliefs, s_t=imged_prior_states)["loc"]

        prior = {"state": imged_prior_states, "mean": imged_prior_means, "std": imged_prior_std_devs}
        states = {"belief": imged_beliefs, "prior": prior}

        imags = {"reward": reward_imag_mean, "states": states}
        for name in self.cfg.model.reconstruction_names:
            imags[name] = observations_imag[name]["loc"]
        return imags

    def observation_reconstruction_imagination(self, observations, actions, rewards, nonterminals, t_imag_start, planning_horizon):
        # Create initial belief and state for time t = 0
        batch_size = actions.shape[1]
        init_belief, init_state = torch.zeros(batch_size, self.cfg.model.belief_size, device=self.cfg.main.device), torch.zeros(batch_size, self.cfg.model.state_size, device=self.cfg.main.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        observations_target = self.clip_obs(observations, idx_start=1)
        obs_emb = bottle_tupele_multimodal(self.encoder, observations_target)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(init_state, actions[:-1], init_belief, obs_emb, nonterminals[:-1])

        # Reconstruction
        observations_pred = self.observation_model(h_t=beliefs, s_t=posterior_states)
        reward_mean = self.reward_model(h_t=beliefs, s_t=posterior_states)["loc"]

        # Clip so that it is the same as the imagination.
        idx = np.arange(t_imag_start + 1, t_imag_start + planning_horizon + 1)

        post = {"state": posterior_states[idx], "mean": posterior_means[idx], "std": posterior_std_devs[idx]}
        prior = {"state": prior_states[idx], "mean": prior_means[idx], "std": prior_std_devs[idx]}
        states = {"belief": beliefs[idx], "prior": prior, "post": post}

        recons = {"reward": reward_mean, "states": states}
        for name in self.cfg.model.reconstruction_names:
            recons[name] = observations_pred[name]["loc"]

        # Imagination
        actor_states = post["state"][:1].detach()
        actor_beliefs = states["belief"][:1].detach()

        actor_actions = actions[t_imag_start:]
        imagination_traj = self.imagine_fix_action(actor_states, actor_beliefs, actor_actions, self.transition_model, planning_horizon, 0)
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj

        observations_imag = self.observation_model(h_t=imged_beliefs, s_t=imged_prior_states)
        reward_imag_mean = self.reward_model(h_t=imged_beliefs, s_t=imged_prior_states)["loc"]

        prior = {"state": imged_prior_states, "mean": imged_prior_means, "std": imged_prior_std_devs}
        states = {"belief": imged_beliefs, "prior": prior}

        imags = {"reward": reward_imag_mean, "states": states}
        for name in self.cfg.model.reconstruction_names:
            imags[name] = observations_imag[name]["loc"]

        # Clip so that it is the same as the imagination.
        cliped_observations_as_imag = self.clip_obs(observations, idx_start=t_imag_start + 2, idx_end=t_imag_start + planning_horizon + 2)

        return cliped_observations_as_imag, recons, imags


class MRSSMSquentialInput:
    def __init__(self, cfg, device=torch.device("cpu"), predict_horizon=1) -> None:
        self.cfg = cfg
        self.device = device
        self.MRSSM = RSSM(self.cfg, self.device)
        self.predict_horizon = predict_horizon
        T = predict_horizon + 2
        (self.beliefs, self.prior_states, self.posterior_states,) = (
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
        )
        self.next_init_state = None
        self.beliefs[0] = torch.zeros(1, self.cfg.model.belief_size, device=self.device)
        self.posterior_states[0] = torch.zeros(1, self.cfg.model.state_size, device=self.device)

    def __call__(self, observations, actions):
        t = 0
        if self.next_init_state is not None:
            self.beliefs[0], self.posterior_states[0] = self.next_init_state
        _state = self.posterior_states[0]
        hidden = self.MRSSM.transition_model.act_fn(self.MRSSM.transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
        # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        self.beliefs[1] = self.MRSSM.transition_model.rnn(hidden, self.beliefs[0])
        # s_t ~ q(s_t | h_t, o_t) (Observation Model)
        obs_emb = self.MRSSM.encoder(observations)
        self.posterior_states[1] = self.MRSSM.transition_model.obs_encoder.sample({"h_t": self.beliefs[1], "o_t": obs_emb}, reparam=True)["s_t"]
        self.next_init_state = (self.beliefs[1], self.posterior_states[1])

        for t in range(1, self.predict_horizon + 1):
            _state = self.posterior_states[t] if t == 1 else self.prior_states[t]
            hidden = self.MRSSM.transition_model.act_fn(self.MRSSM.transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
            # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
            self.beliefs[t + 1] = self.MRSSM.transition_model.rnn(hidden, self.beliefs[t])
            # s_t ~ p(s_t | h_t) (Stochastic State Model)s
            self.prior_states[t + 1] = self.MRSSM.transition_model.stochastic_state_model.sample({"h_t": self.beliefs[t + 1]}, reparam=True)["s_t"]
            # loc_and_scale = self.stochastic_state_model(h_t=self.beliefs[t + 1])
            # self.prior_means[t + 1], self.prior_std_devs[t + 1] = loc_and_scale["loc"], loc_and_scale["scale"]
        pred_eepos = self.MRSSM.observation_model.get_pred_key(h_t=self.beliefs[self.predict_horizon + 1].unsqueeze_(0), s_t=self.prior_states[self.predict_horizon + 1].unsqueeze_(0), key="end_effector")["loc"].squeeze_(0)
        action = pred_eepos - observations["end_effector"]
        return action
