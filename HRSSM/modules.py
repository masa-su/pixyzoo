# Original code: https://github.com/taesupkim/vta/blob/master/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pixyz.distributions import Bernoulli, Normal, Deterministic


class Flatten(nn.Module):
    def forward(self, input_data):
        if len(input_data.size()) == 4:
            return input_data.view(input_data.size(0), -1)
        else:
            return input_data.view(input_data.size(0), input_data.size(1), -1)


class LinearLayer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 nonlinear=nn.ELU(inplace=True)):
        super(LinearLayer, self).__init__()
        # linear
        self.linear = nn.Linear(in_features=input_size,
                                out_features=output_size)

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.linear(input_data))


class ConvLayer1D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvLayer1D, self).__init__()
        # linear
        self.linear = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm1d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class ConvLayer2D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvLayer2D, self).__init__()
        # linear
        self.linear = nn.Conv2d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm2d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class ConvTransLayer2D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvTransLayer2D, self).__init__()
        # linear
        self.linear = nn.ConvTranspose2d(in_channels=input_size,
                                         out_channels=output_size,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm2d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class EncObs(Deterministic):
    def __init__(self,
                 output_size=None,
                 feat_size=128):
        super(EncObs, self).__init__(cond_var=["x"], var=["encoded_x"])
        network_list = [ConvLayer2D(input_size=3,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 16 x 16
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 8 x 8
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 4 x 4
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0),  # 1 x 1
                        Flatten()]
        if output_size is not None:
            network_list.append(LinearLayer(input_size=feat_size,
                                            output_size=output_size))
            self.output_size = output_size
        else:
            self.output_size = feat_size

        self.network = nn.Sequential(*network_list)

    def forward(self, x):
        return {"encoded_x": self.network(x)}


class PostBoundary(Deterministic):
    def __init__(self,
                 input_size=128,
                 output_size=2,
                 num_layers=3):
        super(PostBoundary, self).__init__(cond_var=["encoded_x"], var=["post_boundary_log_alpha"])
        network = list()
        for l in range(num_layers):
            network.append(ConvLayer1D(input_size=input_size,
                                       output_size=input_size))
        network.append(ConvLayer1D(input_size=input_size,
                                   output_size=output_size,
                                   normalize=False,
                                   nonlinear=nn.Identity()))
        self.network = nn.Sequential(*network)

    def forward(self, encoded_x):
        input_data_list = encoded_x
        input_data = input_data_list.permute(0, 2, 1)
        return {"post_boundary_log_alpha": self.network(input_data).permute(0, 2, 1)}


class AbsPostFwd(Deterministic):
    def __init__(self,
                 input_size=128,
                 hidden_size=128):
        super(AbsPostFwd, self).__init__(cond_var=["encoded_x_t", "abs_post_fwd_h_prev"], var=["abs_post_fwd_h"])
        # rnn cell
        self.rnn_cell = nn.GRUCell(input_size=input_size,
                                   hidden_size=hidden_size)

    def forward(self, encoded_x_t, abs_post_fwd_h_prev):
        return {"abs_post_fwd_h": self.rnn_cell(encoded_x_t, abs_post_fwd_h_prev)}


class AbsPostBwd(Deterministic):
    def __init__(self,
                 input_size=128,
                 hidden_size=128):
        super(AbsPostBwd, self).__init__(cond_var=["encoded_x_t", "abs_post_bwd_h_prev"], var=["abs_post_bwd_h"])
        # rnn cell
        self.rnn_cell = nn.GRUCell(input_size=input_size,
                                   hidden_size=hidden_size)

    def forward(self, encoded_x_t, abs_post_bwd_h_prev):
        return {"abs_post_bwd_h": self.rnn_cell(encoded_x_t, abs_post_bwd_h_prev)}


class ObsPostFwd(Deterministic):
    def __init__(self,
                 input_size=128,
                 hidden_size=128):
        super(ObsPostFwd, self).__init__(cond_var=["encoded_x_t", "obs_post_fwd_h_prev"], var=["obs_post_fwd_h"])
        # rnn cell
        self.rnn_cell = nn.GRUCell(input_size=input_size,
                                   hidden_size=hidden_size)

    def forward(self, encoded_x_t, obs_post_fwd_h_prev):
        return {"obs_post_fwd_h": self.rnn_cell(encoded_x_t, obs_post_fwd_h_prev)}


class UpdateAbsBelief(Deterministic):
    def __init__(self,
                 input_size=8,
                 hidden_size=128):
        super(UpdateAbsBelief, self).__init__(cond_var=["abs_state", "abs_belief_prev"], var=["abs_belief"])
        # rnn cell
        self.rnn_cell = nn.GRUCell(input_size=input_size,
                                   hidden_size=hidden_size)

    def forward(self, abs_state, abs_belief_prev):
        return {"abs_belief": self.rnn_cell(abs_state, abs_belief_prev)}


class PriorAbsState(Normal):
    def __init__(self,
                 input_size=128,
                 latent_size=8,
                 feat_size=None):
        super(PriorAbsState, self).__init__(cond_var=["abs_belief"], var=["prior_abs_state"])
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = LinearLayer(input_size=input_size,
                                    output_size=feat_size)

        self.mean = LinearLayer(input_size=feat_size,
                                output_size=latent_size,
                                nonlinear=nn.Identity())

        self.std = LinearLayer(input_size=feat_size,
                               output_size=latent_size,
                               nonlinear=nn.Sigmoid())

    def forward(self, abs_belief):
        feat = self.feat(abs_belief)
        return {"loc": self.mean(feat), "scale": self.std(feat)}


# p(st | ht) * p(ht | m_t-1, s_t-1, z_t, c_t, h_prev)
class AbsFeat(Deterministic):
    def __init__(self,
                 input_size=128 + 8,
                 output_size=128,
                 nonlinear=nn.Identity()):
        super(AbsFeat, self).__init__(cond_var=["abs_belief", "abs_state"], var=["abs_feat"])
        # linear
        self.linear = nn.Linear(in_features=input_size,
                                out_features=output_size)

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, abs_belief, abs_state):
        return {"abs_feat": self.nonlinear(self.linear(torch.cat([abs_belief, abs_state], 1)))}


class PostAbsState(Normal):
    def __init__(self,
                 input_size=128+128,
                 latent_size=8,
                 feat_size=None):
        super(PostAbsState, self).__init__(cond_var=["abs_post_fwd_h", "abs_post_bwd_h"], var=["post_abs_state"])
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = LinearLayer(input_size=input_size,
                                    output_size=feat_size)

        self.mean = LinearLayer(input_size=feat_size,
                                output_size=latent_size,
                                nonlinear=nn.Identity())

        self.std = LinearLayer(input_size=feat_size,
                               output_size=latent_size,
                               nonlinear=nn.Sigmoid())

    def forward(self, abs_post_fwd_h, abs_post_bwd_h):
        feat = self.feat(torch.cat([abs_post_fwd_h, abs_post_bwd_h], 1))
        return {"loc": self.mean(feat), "scale": self.std(feat)}


class AbsFeat(Deterministic):
    def __init__(self,
                 input_size=128 + 8,
                 output_size=128,
                 nonlinear=nn.Identity()):
        super(AbsFeat, self).__init__(cond_var=["abs_belief", "abs_state"], var=["abs_feat"])
        # linear
        self.linear = nn.Linear(in_features=input_size,
                                out_features=output_size)

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, abs_belief, abs_state):
        return {"abs_feat": self.nonlinear(self.linear(torch.cat([abs_belief, abs_state], 1)))}


class UpdateObsBelief(Deterministic):
    def __init__(self,
                 input_size=8 + 128,
                 hidden_size=128):
        super(UpdateObsBelief, self).__init__(cond_var=["obs_state", "abs_feat", "obs_belief_prev"], var=["obs_belief"])
        # rnn cell
        self.rnn_cell = nn.GRUCell(input_size=input_size,
                                   hidden_size=hidden_size)

    def forward(self, obs_state, abs_feat, obs_belief_prev):
        input_data = torch.cat([obs_state, abs_feat], 1)
        return {"obs_belief": self.rnn_cell(input_data, obs_belief_prev)}


class PriorObsState(Normal):
    def __init__(self,
                 input_size=128,
                 latent_size=8,
                 feat_size=None):
        super(PriorObsState, self).__init__(cond_var=["obs_belief"], var=["prior_obs_state"])
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = LinearLayer(input_size=input_size,
                                    output_size=feat_size)

        self.mean = LinearLayer(input_size=feat_size,
                                output_size=latent_size,
                                nonlinear=nn.Identity())

        self.std = LinearLayer(input_size=feat_size,
                               output_size=latent_size,
                               nonlinear=nn.Sigmoid())

    def forward(self, obs_belief):
        feat = self.feat(obs_belief)
        return {"loc": self.mean(feat), "scale": self.std(feat)}


class PostObsState(Normal):
    def __init__(self,
                 input_size=128+128,
                 latent_size=8,
                 feat_size=None):
        super(PostObsState, self).__init__(cond_var=["obs_post_fwd_h", "abs_feat"], var=["post_obs_state"])
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = LinearLayer(input_size=input_size,
                                    output_size=feat_size)

        self.mean = LinearLayer(input_size=feat_size,
                                output_size=latent_size,
                                nonlinear=nn.Identity())

        self.std = LinearLayer(input_size=feat_size,
                               output_size=latent_size,
                               nonlinear=nn.Sigmoid())

    def forward(self, obs_post_fwd_h, abs_feat):
        feat = self.feat(torch.cat([obs_post_fwd_h, abs_feat], 1))
        return {"loc": self.mean(feat), "scale": self.std(feat)}


class ObsFeat(Deterministic):
    def __init__(self,
                 input_size=128 + 8,
                 output_size=128,
                 nonlinear=nn.Identity()):
        super(ObsFeat, self).__init__(cond_var=["obs_belief", "obs_state"], var=["obs_feat"])
        # linear
        self.linear = nn.Linear(in_features=input_size,
                                out_features=output_size)

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, obs_belief, obs_state):
        return {"obs_feat": self.nonlinear(self.linear(torch.cat([obs_belief, obs_state], 1)))}


class PriorBoundary(Deterministic):
    def __init__(self,
                 input_size=128,
                 output_size=2):
        super(PriorBoundary, self).__init__(cond_var=["obs_feat"], var=["prior_boundary_log_alpha"])
        self.network = LinearLayer(input_size=input_size,
                                   output_size=output_size,
                                   nonlinear=nn.Identity())

    def forward(self, obs_feat):
        logit_data = self.network(obs_feat)
        return {"prior_boundary_log_alpha": logit_data}


class DecObs(Normal):
    def __init__(self,
                 input_size=128,
                 feat_size=128):
        super(DecObs, self).__init__(cond_var=["obs_feat"], var=["x"])
        if input_size == feat_size:
            self.linear = nn.Identity()
        else:
            self.linear = LinearLayer(input_size=input_size,
                                      output_size=feat_size,
                                      nonlinear=nn.Identity())
        self.network = nn.Sequential(ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size,
                                                      kernel_size=4,
                                                      stride=1,
                                                      padding=0),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=3,
                                                      normalize=False,
                                                      nonlinear=nn.Identity()))

    def forward(self, obs_feat):
        return {"loc": self.network(self.linear(obs_feat).unsqueeze(-1).unsqueeze(-1)), "scale": 1.0}
