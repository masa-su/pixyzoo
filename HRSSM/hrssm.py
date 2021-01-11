# Original code: https://github.com/taesupkim/vta/blob/master/hssm.py
from pixyz.models import Model
from pixyz.losses import KullbackLeibler, LogProb
import torch.optim as optim
from utils import gumbel_sampling, log_density_concrete
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from modules import EncObs, PostBoundary, AbsPostFwd, AbsPostBwd, \
                    ObsPostFwd, UpdateAbsBelief, PriorAbsState, PostAbsState, \
                    AbsFeat, UpdateObsBelief, PostObsState, \
                    PriorObsState, ObsFeat, PriorBoundary, DecObs


class HRSSM(Model):
    def __init__(self,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None,
                 hrssm_params={'seq_size': 20, 'init_size': 5, 'state_size': 8, 'belief_size': 128, 'num_layers': 3, 'max_seg_num': 5, 'max_seg_len': 10}):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seq_size = hrssm_params['seq_size']
        self.init_size = hrssm_params['init_size']
        state_size = hrssm_params['state_size']
        belief_size = hrssm_params['belief_size']
        num_layers = hrssm_params['num_layers']

        max_seg_num = hrssm_params['max_seg_num']
        max_seg_len = hrssm_params['max_seg_len']

        self.abs_belief_size = belief_size
        self.abs_state_size = state_size
        self.abs_feat_size = belief_size

        # observation level
        self.obs_belief_size = belief_size
        self.obs_state_size = state_size
        self.obs_feat_size = belief_size

        # other size
        self.num_layers = num_layers
        self.feat_size = belief_size

        # sub-sequence information
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        # for concrete distribution
        self.mask_beta = 1.0
        
        self.enc_obs = EncObs().to(self.device)
        self.post_boundary = PostBoundary().to(self.device)
        self.abs_post_fwd = AbsPostFwd().to(self.device)
        self.abs_post_bwd = AbsPostBwd().to(self.device)
        
        self.obs_post_fwd = ObsPostFwd().to(self.device)
        
        self.update_abs_belief = UpdateAbsBelief().to(self.device)
        self.prior_abs_state = PriorAbsState().to(self.device)
        self.post_abs_state = PostAbsState().to(self.device)
        
        self.abs_feat = AbsFeat().to(self.device)
        self.update_obs_belief = UpdateObsBelief().to(self.device)
        self.post_obs_state = PostObsState().to(self.device)
        
        self.prior_obs_state = PriorObsState().to(self.device)
        self.obs_feat = ObsFeat().to(self.device)
        self.prior_boundary = PriorBoundary().to(self.device)
        self.dec_obs = DecObs().to(self.device)
        
        self.log_prob_loss = - LogProb(self.dec_obs)
        self.kl_abs_loss = KullbackLeibler(self.post_abs_state, self.prior_abs_state)
        self.kl_obs_loss = KullbackLeibler(self.post_obs_state, self.prior_obs_state)
        self.kl_mask_loss = KullbackLeibler(self.post_boundary, self.prior_boundary)
        
        loss = self.log_prob_loss + self.kl_abs_loss + self.kl_obs_loss + self.kl_mask_loss
        distributions = [
            self.enc_obs,
            self.post_boundary,
            self.abs_post_fwd,
            self.abs_post_bwd,
            self.obs_post_fwd,
            self.update_abs_belief,
            self.prior_abs_state,
            self.post_abs_state,
            self.abs_feat,
            self.update_obs_belief,
            self.post_obs_state,
            self.prior_obs_state,
            self.obs_feat,
            self.prior_boundary,
            self.dec_obs
        ]

        super().__init__(loss=loss, distributions=distributions, 
                            optimizer=optimizer, optimizer_params=optimizer_params, clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)
    
    # sampler
    def boundary_sampler(self, log_alpha):
        # sample and return corresponding logit
        if self.distributions.training:
            log_sample_alpha = gumbel_sampling(log_alpha=log_alpha, temp=self.mask_beta)
        else:
            log_sample_alpha = log_alpha / self.mask_beta

        # probability
        log_sample_alpha = log_sample_alpha - torch.logsumexp(log_sample_alpha, dim=-1, keepdim=True)
        sample_prob = log_sample_alpha.exp()
        sample_data = torch.eye(2, dtype=log_alpha.dtype, device=log_alpha.device)[torch.max(sample_prob, dim=-1)[1]]

        # sample with rounding and st-estimator
        sample_data = sample_data.detach() + (sample_prob - sample_prob.detach())

        # return sample data and logit
        return sample_data, log_sample_alpha
    
    def regularize_prior_boundary(self, log_alpha_list, boundary_data_list):
        # only for training
        if not self.distributions.training:
            return log_alpha_list

        # sequence size
        num_samples = boundary_data_list.size(0)
        seq_len = boundary_data_list.size(1)

        # init seg static
        seg_num = log_alpha_list.new_zeros(num_samples, 1)
        seg_len = log_alpha_list.new_zeros(num_samples, 1)

        # get min / max logit
        one_prob = 1 - 1e-3
        max_scale = np.log(one_prob / (1 - one_prob))

        near_read_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_read_data[:, 1] = - near_read_data[:, 1]
        near_copy_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_copy_data[:, 0] = - near_copy_data[:, 0]

        # for each step
        new_log_alpha_list = []
        for t in range(seq_len):
            # (0) get length / count
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)
            seg_len = read_data * 1.0 + copy_data * (seg_len + 1.0)
            seg_num = read_data * (seg_num + 1.0) + copy_data * seg_num
            over_len = torch.ge(seg_len, self.max_seg_len).float().detach()
            over_num = torch.ge(seg_num, self.max_seg_num).float().detach()

            # (1) regularize log_alpha 
            # if read enough times (enough segments), stop
            new_log_alpha = over_num * near_copy_data + (1.0 - over_num) * log_alpha_list[:, t]

            # if length is too long (long segment), read
            new_log_alpha = over_len * near_read_data + (1.0 - over_len) * new_log_alpha

            # (2) save
            new_log_alpha_list.append(new_log_alpha)

        # return
        return torch.stack(new_log_alpha_list, dim=1)

    def decompose_sequence(self, obs_data_list):
        # q(M | X)
        # encode observation
        enc_obs_list = self.enc_obs.sample({"x": obs_data_list.view(-1, *obs_data_list.size()[2:])})["encoded_x"]
        enc_obs_list = enc_obs_list.view(self.num_samples, self.full_seq_size, -1)

        # boundary sampling
        post_boundary_log_alpha_list = self.post_boundary.sample({"encoded_x": enc_obs_list})["post_boundary_log_alpha"]
        boundary_data_list, post_boundary_sample_logit_list = self.boundary_sampler(post_boundary_log_alpha_list)
        boundary_data_list[:, :(self.init_size + 1), 0] = 1.0
        boundary_data_list[:, :(self.init_size + 1), 1] = 0.0
        boundary_data_list[:, -self.init_size:, 0] = 1.0
        boundary_data_list[:, -self.init_size:, 1] = 0.0

        return enc_obs_list, post_boundary_log_alpha_list, boundary_data_list, post_boundary_sample_logit_list

    def iterate_rnn(self, enc_obs_list, boundary_data_list):
        # calculate abs_post_fwd, abs_post_bwd, obs_post_fwd by iterating RNN
        abs_post_fwd_list = []
        abs_post_bwd_list = []
        obs_post_fwd_list = []
        abs_post_fwd = enc_obs_list.new_zeros(self.num_samples, self.abs_belief_size)
        abs_post_bwd = enc_obs_list.new_zeros(self.num_samples, self.abs_belief_size)
        obs_post_fwd = enc_obs_list.new_zeros(self.num_samples, self.obs_belief_size)
        for fwd_t, bwd_t in zip(range(self.full_seq_size), reversed(range(self.full_seq_size))):
            # forward encoding
            fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)
            abs_post_fwd = self.abs_post_fwd.sample({"encoded_x_t": enc_obs_list[:, fwd_t], "abs_post_fwd_h_prev": abs_post_fwd})["abs_post_fwd_h"]
            obs_post_fwd = self.obs_post_fwd.sample({"encoded_x_t": enc_obs_list[:, fwd_t], "obs_post_fwd_h_prev": fwd_copy_data * obs_post_fwd})["obs_post_fwd_h"]
            abs_post_fwd_list.append(abs_post_fwd)
            obs_post_fwd_list.append(obs_post_fwd)

            # backward encoding
            bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
            abs_post_bwd = self.abs_post_bwd.sample({"encoded_x_t": enc_obs_list[:, bwd_t], "abs_post_bwd_h_prev": abs_post_bwd})["abs_post_bwd_h"]
            abs_post_bwd_list.append(abs_post_bwd)
            abs_post_bwd = bwd_copy_data * abs_post_bwd
        abs_post_bwd_list = abs_post_bwd_list[::-1]

        return abs_post_fwd_list, abs_post_bwd_list, obs_post_fwd_list

    def infer_state(self, boundary_data_list, abs_post_fwd_list, abs_post_bwd_list, obs_post_fwd_list):
        # q(Z | M, X), q(S | Z, M, X)
        post_abs_state_list = []
        post_obs_state_list = []
        abs_belief_list = []
        obs_belief_list = []

        abs_belief = boundary_data_list.new_zeros(self.num_samples, self.abs_belief_size)
        abs_state = boundary_data_list.new_zeros(self.num_samples, self.abs_state_size)
        obs_belief = boundary_data_list.new_zeros(self.num_samples, self.obs_belief_size)
        obs_state = boundary_data_list.new_zeros(self.num_samples, self.obs_state_size)
        
        for t in range(self.init_size, self.init_size + self.seq_size):
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)

            if t == self.init_size:
                abs_belief = nn.Identity()(abs_post_fwd_list[t - 1])
            else:
                abs_belief = read_data * self.update_abs_belief.sample({"abs_state": abs_state, "abs_belief_prev": abs_belief})["abs_belief"] + copy_data * abs_belief
            #### q(z_t | M, X) or q(z_t | m_t-1, z_t-1, abs_fwd_t-1, abs_bwd_t) start ####
            # q(z_t | abs_fwd_t-1, abs_bwd_t), this operation happens when read_data=1 (m_t-1=1)
            post_abs_state = self.post_abs_state.sample({"abs_post_fwd_h": abs_post_fwd_list[t - 1], "abs_post_bwd_h": abs_post_bwd_list[t]}, reparam=True)["post_abs_state"]
            
            # copy_data * abs_state implements q(z_t | z_t-1), this operation happens when copy_data=1 (m_t-1=0)
            # if read_data=1 (m_t-1=1), update abs_state, copy previous abs_state otherwise (copy_data=1 (m_t-1=0))
            abs_state = read_data * post_abs_state + copy_data * abs_state
            #### q(z_t | M, X) or q(z_t | m_t-1, z_t-1, abs_fwd_t-1, abs_bwd_t) end ####

            #### q(s_t | z_t, M, X) or q(s_t | z_t, obs_fwd_t) start ####
            abs_feat = self.abs_feat.sample({"abs_belief": abs_belief, "abs_state": abs_state})["abs_feat"]

            ##### for prior state #####
            obs_belief = read_data * nn.Identity()(abs_feat) + copy_data * self.update_obs_belief.sample({"obs_state": obs_state, "abs_feat": abs_feat, "obs_belief_prev": obs_belief})["obs_belief"]

            post_obs_state = self.post_obs_state.sample({"obs_post_fwd_h": obs_post_fwd_list[t], "abs_feat": abs_feat}, reparam=True)["post_obs_state"]
            obs_state = post_obs_state
            #### q(s_t | z_t, M, X) or q(s_t | z_t, obs_fwd_t) end ####

            post_abs_state_list.append(post_abs_state)
            post_obs_state_list.append(post_obs_state)
            #### for prior state ####
            abs_belief_list.append(abs_belief)
            obs_belief_list.append(obs_belief)
        
        return post_abs_state_list, post_obs_state_list, abs_belief_list, obs_belief_list
    
    def prior_state(self, boundary_data_list, abs_belief_list=None, obs_belief_list=None):
        # p(z_t | z<t, m<t) or p(z_t | c_t) (m_t-1=1), z_t = z_t-1 (m_t-1=0)
        # p(s_t | s<t, z_t, m<t) or p(s_t | h_t)
        prior_abs_state_list = []
        prior_obs_state_list = []

        abs_belief = boundary_data_list.new_zeros(self.num_samples, self.abs_belief_size)
        abs_state = boundary_data_list.new_zeros(self.num_samples, self.abs_state_size)
        obs_belief = boundary_data_list.new_zeros(self.num_samples, self.obs_belief_size)
        obs_state = boundary_data_list.new_zeros(self.num_samples, self.obs_state_size)
        
        # for reconstruction
        if abs_belief_list:
            for t in range(self.seq_size):
                abs_belief = abs_belief_list[t]
                # p(z_t | c_t)
                prior_abs_state = self.prior_abs_state.sample({"abs_belief": abs_belief}, reparam=True)["prior_abs_state"]

                obs_belief = obs_belief_list[t]
                # p(s_t | h_t)
                prior_obs_state = self.prior_obs_state.sample({"obs_belief": obs_belief}, reparam=True)["prior_obs_state"]
                
                prior_abs_state_list.append(prior_abs_state)
                prior_obs_state_list.append(prior_obs_state)
            return prior_abs_state_list, prior_obs_state_list
        # for sampling, (generating)
        else:
            pass

    def decode(self, obs_belief_list, obs_state_list):
        # p(x_t | s_t)
        obs_feat_list = []
        obs_rec_list = []
        for t in range(self.seq_size):
            obs_belief = obs_belief_list[t]
            obs_state = obs_state_list[t]
            obs_feat = self.obs_feat.sample({"obs_belief": obs_belief, "obs_state": obs_state})["obs_feat"]
            # p(x_t | s_t)
            # obs_rec = self.dec_obs.sample({"obs_feat": obs_feat}, reparam=True)["x"]
            obs_rec = self.dec_obs.sample_mean({"obs_feat": obs_feat})
            obs_feat_list.append(obs_feat)
            obs_rec_list.append(obs_rec)
        
        return obs_feat_list, obs_rec_list

    def prior_boundary_mask(self, obs_feat_list):
        # p(m_t=1 |s_t)
        prior_boundary_log_alpha_list = []
        for t in range(self.seq_size):
            obs_feat = obs_feat_list[t]
            prior_boundary_log_alpha = self.prior_boundary.sample({"obs_feat": obs_feat})["prior_boundary_log_alpha"]

            prior_boundary_log_alpha_list.append(prior_boundary_log_alpha)
        
        return prior_boundary_log_alpha_list
    
    # forward for reconstruction
    def calculate_loss(self, obs_data_list):
        num_samples = obs_data_list.size(0)
        full_seq_size = obs_data_list.size(1)
        seq_size = self.seq_size
        init_size = self.init_size

        enc_obs_list = self.enc_obs.sample({"x": obs_data_list.view(-1, *obs_data_list.size()[2:])})["encoded_x"]
        enc_obs_list = enc_obs_list.view(num_samples, full_seq_size, -1)

        post_boundary_log_alpha_list = self.post_boundary.sample({"encoded_x": enc_obs_list})["post_boundary_log_alpha"]
        boundary_data_list, post_boundary_sample_logit_list = self.boundary_sampler(post_boundary_log_alpha_list)
        boundary_data_list[:, :(init_size + 1), 0] = 1.0
        boundary_data_list[:, :(init_size + 1), 1] = 0.0
        boundary_data_list[:, -init_size:, 0] = 1.0
        boundary_data_list[:, -init_size:, 1] = 0.0

        abs_post_fwd_list = []
        abs_post_bwd_list = []
        obs_post_fwd_list = []
        abs_post_fwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_post_bwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        obs_post_fwd = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
        for fwd_t, bwd_t in zip(range(full_seq_size), reversed(range(full_seq_size))):
            fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)
            abs_post_fwd = self.abs_post_fwd.sample({"encoded_x_t": enc_obs_list[:, fwd_t], "abs_post_fwd_h_prev": abs_post_fwd})["abs_post_fwd_h"]
            obs_post_fwd = self.obs_post_fwd.sample({"encoded_x_t": enc_obs_list[:, fwd_t], "obs_post_fwd_h_prev": fwd_copy_data * obs_post_fwd})["obs_post_fwd_h"]
            abs_post_fwd_list.append(abs_post_fwd)
            obs_post_fwd_list.append(obs_post_fwd)

            bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
            abs_post_bwd = self.abs_post_bwd.sample({"encoded_x_t": enc_obs_list[:, bwd_t], "abs_post_bwd_h_prev": abs_post_bwd})["abs_post_bwd_h"]
            abs_post_bwd_list.append(abs_post_bwd)
            abs_post_bwd = bwd_copy_data * abs_post_bwd
        abs_post_bwd_list = abs_post_bwd_list[::-1]

        prior_abs_state_list = []
        post_abs_state_list = []
        prior_obs_state_list = []
        post_obs_state_list = []
        prior_boundary_log_alpha_list = []

        abs_belief = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = obs_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_belief = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
        obs_state = obs_data_list.new_zeros(num_samples, self.obs_state_size)
        
        kl_abs_state_list = []
        kl_obs_state_list = []
        obs_feat_list = []

        for t in range(init_size, init_size + seq_size):
            # get mask data
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)

            if t == init_size:
                abs_belief = nn.Identity()(abs_post_fwd_list[t - 1])
            else:
                abs_belief = read_data * self.update_abs_belief.sample({"abs_state": abs_state, "abs_belief_prev": abs_belief})["abs_belief"] + copy_data * abs_belief
            prior_abs_state = self.prior_abs_state.sample({"abs_belief": abs_belief}, reparam=True)["prior_abs_state"]
            post_abs_state = self.post_abs_state.sample({"abs_post_fwd_h": abs_post_fwd_list[t - 1], "abs_post_bwd_h": abs_post_bwd_list[t]}, reparam=True)["post_abs_state"]
            abs_state = read_data * post_abs_state + copy_data * abs_state
            abs_feat = self.abs_feat.sample({"abs_belief": abs_belief, "abs_state": abs_state})["abs_feat"]
            
            # kl loss for abs_state
            kl_abs_state_list.append(KullbackLeibler(self.post_abs_state, self.prior_abs_state).eval({"abs_belief": abs_belief, "abs_post_fwd_h": abs_post_fwd_list[t - 1], "abs_post_bwd_h": abs_post_bwd_list[t]}))
            
            obs_belief = read_data * nn.Identity()(abs_feat) + copy_data * self.update_obs_belief.sample({"obs_state": obs_state, "abs_feat": abs_feat, "obs_belief_prev": obs_belief})["obs_belief"]
            prior_obs_state = self.prior_obs_state.sample({"obs_belief": obs_belief}, reparam=True)["prior_obs_state"]
            post_obs_state = self.post_obs_state.sample({"obs_post_fwd_h": obs_post_fwd_list[t], "abs_feat": abs_feat}, reparam=True)["post_obs_state"]
            obs_state = post_obs_state
            obs_feat = self.obs_feat.sample({"obs_belief": obs_belief, "obs_state": obs_state})["obs_feat"]
            obs_feat_list.append(obs_feat)
            
            # kl loss for obs_state
            kl_obs_state_list.append(KullbackLeibler(self.post_obs_state, self.prior_obs_state).eval({"obs_post_fwd_h": obs_post_fwd_list[t], "abs_feat": abs_feat, "obs_belief": obs_belief}))

            prior_boundary_log_alpha = self.prior_boundary.sample({"obs_feat": obs_feat})["prior_boundary_log_alpha"]

            prior_boundary_log_alpha_list.append(prior_boundary_log_alpha)
            prior_abs_state_list.append(prior_abs_state)
            post_abs_state_list.append(post_abs_state)
            prior_obs_state_list.append(prior_obs_state)
            post_obs_state_list.append(post_obs_state)
        
        # loss for observation
        obs_feat_list = torch.stack(obs_feat_list, dim=1)
        obs_cost = - LogProb(self.dec_obs).eval({"obs_feat": obs_feat_list.view(num_samples * seq_size, -1), "x": obs_data_list[:, init_size:-init_size].reshape(-1, *obs_data_list[:, init_size:-init_size].size()[2:])})

        prior_boundary_log_alpha_list = torch.stack(prior_boundary_log_alpha_list, dim=1)

        # remove padding
        boundary_data_list = boundary_data_list[:, init_size:(init_size + seq_size)]
        post_boundary_log_alpha_list = post_boundary_log_alpha_list[:, (init_size + 1):(init_size + 1 + seq_size)]
        post_boundary_sample_logit_list = post_boundary_sample_logit_list[:, (init_size + 1):(init_size + 1 + seq_size)]

        # fix prior by constraints
        prior_boundary_log_alpha_list = self.regularize_prior_boundary(prior_boundary_log_alpha_list,
                                                                       boundary_data_list)

        # compute log-density
        prior_boundary_log_density = log_density_concrete(prior_boundary_log_alpha_list,
                                                          post_boundary_sample_logit_list,
                                                          self.mask_beta)
        post_boundary_log_density = log_density_concrete(post_boundary_log_alpha_list,
                                                         post_boundary_sample_logit_list,
                                                         self.mask_beta)

        # compute boundary probability
        prior_boundary_list = F.softmax(prior_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        post_boundary_list = F.softmax(post_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        prior_boundary_list = torch.distributions.Bernoulli(probs=prior_boundary_list)
        post_boundary_list = torch.distributions.Bernoulli(probs=post_boundary_list)
        boundary_data_list = boundary_data_list[..., 0].unsqueeze(-1)

        kl_abs = []
        kl_obs = []
        for t in range(seq_size):
            # read flag
            read_data = boundary_data_list[:, t].detach()

            kl_abs_state = kl_abs_state_list[t] * read_data.squeeze()
            kl_obs_state = kl_obs_state_list[t]
            kl_abs.append(kl_abs_state)
            kl_obs.append(kl_obs_state)
        kl_abs = torch.stack(kl_abs, dim=1)
        kl_obs = torch.stack(kl_obs, dim=1)

        # kl loss for boundary
        kl_mask = (post_boundary_log_density - prior_boundary_log_density)
        total_loss = obs_cost.mean() + kl_abs.mean() + kl_obs.mean() + kl_mask.mean()

        return total_loss

    def train(self, obs_data_list):
        self.distributions.train()

        self.optimizer.zero_grad()
        loss = self.calculate_loss(obs_data_list)

        # backprop
        loss.backward()

        if self.clip_norm:
            clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
        if self.clip_value:
            clip_grad_value_(self.distributions.parameters(), self.clip_value)

        # update params
        self.optimizer.step()

        return loss.item()
    
    def test(self, obs_data_list):
        self.distributions.eval()

        with torch.no_grad():
            loss = self.calculate_loss(obs_data_list)

        return loss.item()
    
    # forward for reconstruction
    def reconstruction(self, obs_data_list):
        # obs_data_list: (B, T, C, H, W)
        self.num_samples = obs_data_list.size(0)
        self.full_seq_size = obs_data_list.size(1)

        seq_size = self.seq_size
        init_size = self.init_size

        # q(M, encoded_X| X) = q(M | encoded_X) * Encoder(encoded_X | X)
        enc_obs_list, post_boundary_log_alpha_list, boundary_data_list, post_boundary_sample_logit_list = self.decompose_sequence(obs_data_list)

        # posterior encoding
        # calculate abs_post_fwd, abs_post_bwd, obs_post_fwd by iterating RNN
        abs_post_fwd_list, abs_post_bwd_list, obs_post_fwd_list = self.iterate_rnn(enc_obs_list, boundary_data_list)

        # q(Z | M, X), q(S | Z, M, X)
        post_abs_state_list, post_obs_state_list, abs_belief_list, obs_belief_list = self.infer_state(boundary_data_list, abs_post_fwd_list, abs_post_bwd_list, obs_post_fwd_list)
        
        # P(X|Z, S)
        obs_feat_list, obs_rec_list = self.decode(obs_belief_list, post_obs_state_list)

        # P(Z, S, M)
        prior_abs_state_list, prior_obs_state_list = self.prior_state(boundary_data_list, abs_belief_list, obs_belief_list)
        prior_boundary_log_alpha_list = self.prior_boundary_mask(obs_feat_list)
        
        # stack results
        prior_boundary_log_alpha_list = torch.stack(prior_boundary_log_alpha_list, dim=1)

        # remove padding
        boundary_data_list = boundary_data_list[:, init_size:(init_size + seq_size)]
        post_boundary_log_alpha_list = post_boundary_log_alpha_list[:, (init_size + 1):(init_size + 1 + seq_size)]
        post_boundary_sample_logit_list = post_boundary_sample_logit_list[:, (init_size + 1):(init_size + 1 + seq_size)]

        # fix prior by constraints
        prior_boundary_log_alpha_list = self.regularize_prior_boundary(prior_boundary_log_alpha_list,
                                                                       boundary_data_list)

        # compute boundary probability
        prior_boundary_list = F.softmax(prior_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        post_boundary_list = F.softmax(post_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        prior_boundary_list = torch.distributions.Bernoulli(probs=prior_boundary_list)
        post_boundary_list = torch.distributions.Bernoulli(probs=post_boundary_list)
        boundary_data_list = boundary_data_list[..., 0].unsqueeze(-1)

        # return
        return {'rec_data': torch.stack(obs_rec_list, dim=1),
                'mask_data': boundary_data_list,
                'p_mask': prior_boundary_list.mean,
                'q_mask': post_boundary_list.mean,
                'p_ent': prior_boundary_list.entropy(),
                'q_ent': post_boundary_list.entropy(),
                'beta': self.mask_beta}
    
    # generation forward
    def jumpy_generation(self, init_data_list, seq_size):
        # eval mode
        self.distributions.eval()

        num_samples = init_data_list.size(0)
        init_size = init_data_list.size(1)

        abs_post_fwd = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        for t in range(init_size):
            abs_post_fwd = self.abs_post_fwd.sample({"encoded_x_t": self.enc_obs.sample({"x": init_data_list[:, t]})["encoded_x"], "abs_post_fwd_h_prev": abs_post_fwd})["abs_post_fwd_h"]
        
        # init state
        abs_belief = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = init_data_list.new_zeros(num_samples, self.abs_state_size)

        obs_rec_list = []

        for t in range(seq_size):
            if t == 0:
                abs_belief = nn.Identity()(abs_post_fwd)
            else:
                abs_belief = self.update_abs_belief.sample({"abs_state": abs_state, "abs_belief_prev": abs_belief})["abs_belief"]
                
            abs_state = self.prior_abs_state.sample({"abs_belief": abs_belief}, reparam=True)["prior_abs_state"]
            abs_feat = self.abs_feat.sample({"abs_belief": abs_belief, "abs_state": abs_state})["abs_feat"]

            obs_belief = nn.Identity()(abs_feat)
            obs_state = self.prior_obs_state.sample({"obs_belief": obs_belief}, reparam=True)["prior_obs_state"]
            obs_feat = self.obs_feat.sample({"obs_belief": obs_belief, "obs_state": obs_state})["obs_feat"]

            # obs_rec = self.dec_obs.sample({"obs_feat": obs_feat}, reparam=True)["x"]
            obs_rec = self.dec_obs.sample_mean({"obs_feat": obs_feat})
            obs_rec_list.append(obs_rec)

        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        return obs_rec_list

    # generation forward
    def full_generation(self, init_data_list, seq_size):
        # eval mode
        self.distributions.eval()

        num_samples = init_data_list.size(0)
        init_size = init_data_list.size(1)

        abs_post_fwd = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        for t in range(init_size):
            abs_post_fwd = self.abs_post_fwd.sample({"encoded_x_t": self.enc_obs.sample({"x": init_data_list[:, t]})["encoded_x"], "abs_post_fwd_h_prev": abs_post_fwd})["abs_post_fwd_h"]

        abs_belief = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = init_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_belief = init_data_list.new_zeros(num_samples, self.obs_belief_size)
        obs_state = init_data_list.new_zeros(num_samples, self.obs_state_size)

        obs_rec_list = []
        boundary_data_list = []

        read_data = init_data_list.new_ones(num_samples, 1)
        copy_data = 1 - read_data
        for t in range(seq_size):
            if t == 0:
                abs_belief = nn.Identity()(abs_post_fwd)
            else:
                abs_belief =  read_data * self.update_abs_belief.sample({"abs_state": abs_state, "abs_belief_prev": abs_belief})["abs_belief"] + copy_data * abs_belief
            abs_state = read_data * self.prior_abs_state.sample({"abs_belief": abs_belief}, reparam=True)["prior_abs_state"] + copy_data * abs_state
            abs_feat = self.abs_feat.sample({"abs_belief": abs_belief, "abs_state": abs_state})["abs_feat"]

            obs_belief = read_data * nn.Identity()(abs_feat) + copy_data * self.update_obs_belief.sample({"obs_state": obs_state, "abs_feat": abs_feat, "obs_belief_prev": obs_belief})["obs_belief"]
            obs_state = self.prior_obs_state.sample({"obs_belief": obs_belief}, reparam=True)["prior_obs_state"]
            obs_feat = self.obs_feat.sample({"obs_belief": obs_belief, "obs_state": obs_state})["obs_feat"]

            # obs_rec = self.dec_obs.sample({"obs_feat": obs_feat}, reparam=True)["x"]
            obs_rec = self.dec_obs.sample_mean({"obs_feat": obs_feat})

            obs_rec_list.append(obs_rec)
            boundary_data_list.append(read_data)

            prior_boundary = self.boundary_sampler(self.prior_boundary.sample({"obs_feat": obs_feat})["prior_boundary_log_alpha"])[0]
            read_data = prior_boundary[:, 0].unsqueeze(-1)
            copy_data = prior_boundary[:, 1].unsqueeze(-1)

        # stack results
        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        boundary_data_list = torch.stack(boundary_data_list, dim=1)
        return obs_rec_list, boundary_data_list
