import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

import numpy as np
from pixyz.distributions import Normal
from pixyz.losses import LogProb, KullbackLeibler

from utils import soft_update, create_feature_actions
from initializer import initialize_weight
from replay_buffer import ReplayBuffer
from config import SLAC_config, LOG_INTERVAL

z1_dim = SLAC_config['z1_dim']
z2_dim = SLAC_config['z2_dim']
z_dim = z1_dim + z2_dim
encoded_obs_dim = 256
num_seq = SLAC_config['num_sequences']


class Encoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=encoded_obs_dim):
        """Convolutional network used for embeddings"""
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)

    def forward(self, x):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        x = self.network(x)
        x = x.view(B, S, -1)
        return x


class Decoder(Normal):
    def __init__(self, input_dim=z_dim, std=np.sqrt(0.1)):
        """Decodes z into an observation"""
        super().__init__(
            cond_var=['z_1', 'z_2'], var=['x_decoded'])
        self.network = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, 3, 5, 2, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)
        self.std = std

    def forward(self, z_1, z_2):
        z = torch.cat([z_1, z_2], dim=-1)
        B, S, latent_dim = z.size()
        assert latent_dim == z1_dim + z2_dim
        z = z.view(B*S, latent_dim, 1, 1)
        loc = self.network(z)
        _, C, H, W = loc.size()
        loc = loc.view(B, S, C, H, W)
        scale = torch.ones_like(loc).mul_(self.std)
        return {"loc": loc, "scale": scale}


class Gaussian(Normal):
    def __init__(self, cond_var_dict: dict, var_dict: dict):
        assert len(var_dict.keys()) == 1
        super(Gaussian, self).__init__(
            cond_var=list(cond_var_dict.keys()), var=list(var_dict.keys()))
        in_dim = sum(cond_var_dict.values())
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, list(var_dict.values())[0]*2)
        ).apply(initialize_weight)
        self.inputs = cond_var_dict.keys()

    def forward(self, **kwargs):
        # sort vars in the fixed order
        x = []
        for varname in self.inputs:
            x.append(kwargs[varname])
        x = torch.cat(x, dim=-1)
        x = self.fcs(x)
        loc, scale = torch.chunk(x, 2, dim=-1)
        scale = F.softplus(scale) + 1e-5
        return {"loc": loc, "scale": scale}


class FixedGaussian(Normal):
    """
    Fixed diagonal gaussian distribution.
    """

    def __init__(self, var, output_dim, std):
        super(FixedGaussian, self).__init__(var=var, cond_var=['x'])
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        loc = torch.zeros(x.size(0), self.output_dim, device=x.device)
        scale = torch.ones(x.size(0), self.output_dim,
                           device=x.device).mul_(self.std)
        return {'loc': loc, 'scale': scale}


class QFunc(nn.Module):
    def __init__(self, z1_dim, z2_dim, num_action: int):
        super(QFunc, self).__init__()
        input_dim = z1_dim + z2_dim + num_action
        self.network1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).apply(initialize_weight)

        self.network2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).apply(initialize_weight)

    def forward(self, z, action_t_1):
        inputs = torch.cat([z, action_t_1], dim=1)
        return self.network1(inputs), self.network2(inputs)


class Pie(Normal):
    def __init__(self, in_dim, action_dim):
        super().__init__(
            cond_var=["obs_and_action"], var=['pi'])
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*action_dim)
        ).apply(initialize_weight)

    def forward(self, obs_and_action):
        loc, log_scale = torch.chunk(self.fcs(obs_and_action), 2, dim=-1)
        scale = torch.exp(log_scale.clamp(-20, 2))
        return {"loc": loc, "scale": scale}


class Actor(nn.Module):
    def __init__(self, num_action):
        super().__init__()
        # pi(a_t| x_{1:t}, a_{1:t-1})
        self.pi = Pie(in_dim=num_seq*encoded_obs_dim +
                      (num_seq - 1)*num_action, action_dim=num_action)

    def sample(self, x_encoded):
        pi = self.pi.sample({'obs_and_action': x_encoded}, reparam=True)["pi"]
        action = torch.tanh(pi)
        log_prob = self.pi.get_log_prob(
            {'obs_and_action': x_encoded, 'pi': pi}, sum_features=False)

        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def act_greedy(self, x_encoded):
        return torch.tanh(self.pi(x_encoded)['loc'])


class LatentModel(nn.Module):
    def __init__(self, num_action, obs_shape):
        super().__init__()
        self.encoder = Encoder(input_dim=obs_shape[0])
        self.decoder = Decoder()
        # p(z1(0))
        self.z1_prior_init = FixedGaussian(['z_1'], z1_dim, 1.0)

        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(
            cond_var_dict={"z_1": z1_dim}, var_dict={"z_2": z2_dim})

        # p(z1(t + 1) | z2(t), a(t))
        self.z1_prior = Gaussian(cond_var_dict={
            "z_t^2": z2_dim, "a_t": num_action}, var_dict={"z_{t + 1}^1": z1_dim})

        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(cond_var_dict={
            "z_{t + 1}^1": z1_dim, "z_t^2": z2_dim, "a_t": num_action}, var_dict={"z_{t + 1}^2": z2_dim})

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward_dist = Gaussian(cond_var_dict={
                                    "z_t^1": z1_dim, "z_t^2": z2_dim, "z_{t + 1}^1": z1_dim, "z_{t + 1}^2": z2_dim, "a_t": num_action}, var_dict={"r_t": 1})

        # q(z1(0) | x1_encoded(0))
        self.z1_posterior_init = Gaussian(
            cond_var_dict={"x_encoded": encoded_obs_dim}, var_dict={"z_1": z1_dim})

        # q(z1(t+1) | x_encoded(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(
            cond_var_dict={"x_encoded": encoded_obs_dim, "a_t": num_action, "z_t^2": z2_dim}, var_dict={"z_{t + 1}^1": z1_dim})

        self.z2_posterior_init = self.z2_prior_init
        self.z2_posterior = self.z2_prior

        # KL Divergence
        self.loss_kld_init = KullbackLeibler(
            self.z1_posterior_init, self.z1_prior_init, analytical=True)
        self.loss_kld = KullbackLeibler(
            self.z1_posterior, self.z1_prior, analytical=True)

        self.apply(initialize_weight)

    def calculate_loss(self, state, action, reward, done):
        x_encoded = self.encoder(state)
        z1, z2, loss_kld = self.sample_posterior(x_encoded, action)

        loss_img = - self.decoder.get_log_prob(
            {'z_1': z1, 'z_2': z2, 'x_decoded': state}, sum_features=False).mean(dim=0)
        loss_reward = - self.reward_dist.get_log_prob(
            {"z_t^1": z1[:, :-1], "z_t^2": z2[:, :-1], "z_{t + 1}^1": z1[:, 1:], "z_{t + 1}^2": z2[:, 1:], "a_t": action, 'r_t': reward}, sum_features=False)
        loss_reward = loss_reward.mul_(1 - done).mean(dim=0)
        return loss_kld.sum(), loss_img.sum(), loss_reward.sum()

    def sample_posterior(self, x_encoded, action):
        z1_pos_list = []
        z2_pos_list = []

        # sampling from posterior dist
        z1_pos = self.z1_posterior_init.sample(
            {'x_encoded': x_encoded[:, 0]}, reparam=True)['z_1']
        z2_pos = self.z2_posterior_init.sample(
            {'z_1': z1_pos}, reparam=True)['z_2']

        # sampling from prior dist
        z1_pri = self.z1_prior_init.sample(
            {'x': action[:, 0]}, reparam=True)['z_1']
        z2_pri = self.z2_prior_init.sample(
            {'z_1': z1_pri}, reparam=True)['z_2']

        z1_pos_list.append(z1_pos)
        z2_pos_list.append(z2_pos)

        # calc KL Divergence
        loss = self.loss_kld_init.eval(
            {'x': action[:, 0], 'x_encoded': x_encoded[:, 0]})

        for t in range(1, action.size(1) + 1):
            # q(z1(t) | x_encoded(t), z2(t-1), a(t-1))
            z1_pos = self.z1_posterior.sample(
                {'x_encoded': x_encoded[:, t],
                 'z_t^2': z2_pos,
                 'a_t': action[:, t - 1]
                 },
                reparam=True)["z_{t + 1}^1"]
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_pos = self.z2_posterior.sample(
                {"z_{t + 1}^1": z1_pos,
                 "z_t^2": z2_pos,
                 "a_t": action[:, t - 1]
                 },
                reparam=True)["z_{t + 1}^2"]
            z1_pos_list.append(z1_pos)
            z2_pos_list.append(z2_pos)

            # calc KL Divergence
            loss += self.loss_kld.eval(
                {"z_t^2": z2_pri, "a_t": action[:, t - 1], "x_encoded": x_encoded[:, t]})

            # sampling from prior dist
            z1_pri = self.z1_prior.sample(
                {"z_t^2": z2_pri, "a_t": action[:, t - 1]}, reparam=True)["z_{t + 1}^1"]
            z2_pri = self.z2_prior.sample(
                {"z_{t + 1}^1": z1_pri, "z_t^2": z2_pri, "a_t": action[:, t - 1]}, reparam=True)["z_{t + 1}^2"]

        loss = loss.mean(dim=0)
        return torch.stack(z1_pos_list, dim=1), torch.stack(z2_pos_list, dim=1), loss


class SLAC(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 seed,
                 gamma=0.99,
                 batch_size_sac=256,
                 batch_size_latent=32,
                 buffer_size=10 ** 5,
                 num_sequences=8,
                 lr_sac=3e-4,
                 lr_latent=1e-4,
                 tau=5e-3):
        super(SLAC, self).__init__()
        self.batch_size_latent = batch_size_latent
        self.batch_size_sac = batch_size_sac
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.critic = QFunc(z1_dim=z1_dim, z2_dim=z2_dim,
                            num_action=action_shape[0]).to(device)
        self.critic_target = QFunc(
            z1_dim=z1_dim, z2_dim=z2_dim, num_action=action_shape[0]).to(device)
        soft_update(self.critic_target, self.critic, 1.0)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor = Actor(num_action=action_shape[0]).to(device)
        self.latent = LatentModel(
            num_action=action_shape[0], obs_shape=obs_shape).to(device)
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.target_entropy = -float(action_shape[0])
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        # Optimizers.
        self.optim_actor = Adam(self.actor.pi.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.buffer = ReplayBuffer(
            buffer_size, num_sequences, obs_shape, action_shape, device)
        self.learning_steps_latent = 0
        self.learning_steps_sac = 0

    def update_critic(self, z, next_z, action, feature_action, reward, done, writer):
        q1, q2 = self.critic(z, action)

        with torch.no_grad():
            next_action, log_prob = self.actor.sample(feature_action)
            q1_next, q2_next = self.critic_target(next_z, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * log_prob
        target_q = reward + (1.0 - done) * self.gamma * q_next
        loss = (q1 - target_q).pow_(2).mean() + (q2 - target_q).pow_(2).mean()
        self.optim_critic.zero_grad()
        loss.backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps_sac % LOG_INTERVAL == 0:
            writer.add_scalar('loss/critic', loss, self.learning_steps_sac)

    def update_actor(self, z, feature_action, writer):
        action, log_pi = self.actor.sample(feature_action)
        q1, q2 = self.critic(z, action)
        loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi)

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        with torch.no_grad():
            entropy = -log_pi.detach().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        if self.learning_steps_sac % LOG_INTERVAL == 0:
            writer.add_scalar("loss/actor", loss_actor.item(),
                              self.learning_steps_sac)
            writer.add_scalar("loss/alpha", loss_alpha.item(),
                              self.learning_steps_sac)
            writer.add_scalar("stats/alpha", self.alpha.item(),
                              self.learning_steps_sac)
            writer.add_scalar("stats/entropy", entropy.item(),
                              self.learning_steps_sac)

    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, action_, reward_, done_ = self.buffer.sample_latent(
            self.batch_size_latent)
        loss_kld, loss_image, loss_reward = self.latent.calculate_loss(
            state_, action_, reward_, done_)

        self.optim_latent.zero_grad()
        (loss_kld + loss_image + loss_reward).backward()
        self.optim_latent.step()

        if self.learning_steps_latent % LOG_INTERVAL == 0:
            writer.add_scalar("loss/kld", loss_kld.item(),
                              self.learning_steps_latent)
            writer.add_scalar("loss/loss_image", loss_image.item(),
                              self.learning_steps_latent)
            writer.add_scalar("loss/loss_reward", loss_reward.item(),
                              self.learning_steps_latent)

    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state_, action_, reward, done = self.buffer.sample_sac(
            self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(
            state_, action_)

        self.update_critic(z, next_z, action,
                           next_feature_action, reward, done, writer)
        self.update_actor(z, feature_action, writer)
        soft_update(self.critic_target, self.critic, self.tau)

    def prepare_batch(self, state_, action_):
        with torch.no_grad():
            # f(1:t+1)
            feature_ = self.latent.encoder(state_)
            # z(1:t+1)
            z_ = torch.cat(self.latent.sample_posterior(
                feature_, action_)[:-1], dim=-1)

        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = create_feature_actions(
            feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    def preprocess(self, ob):
        state = torch.tensor(ob.state, dtype=torch.uint8,
                             device=self.device).float().div_(255.0)
        with torch.no_grad():
            feature = self.latent.encoder(state).view(1, -1)
        action = torch.tensor(ob.action, dtype=torch.float, device=self.device)
        feature_action = torch.cat([feature, action], dim=1)
        return feature_action

    def step(self, env, obs, t, is_random):
        t += 1
        if is_random:
            action = env.action_space.sample()
        else:
            action = self.explore(obs)

        assert not np.any(np.isnan(action)), 'Action should not be None'
        state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        obs.append(state, action)
        self.buffer.append(action, reward, mask, state, done)

        if done:
            t = 0
            state = env.reset()
            obs.reset_episode(state)
            self.buffer.reset_episode(state)
        return t

    def explore(self, obs):
        feature_action = self.preprocess(obs)
        with torch.no_grad():
            action = self.actor.sample(feature_action)[0]
        return action.cpu().numpy()[0]

    def exploit(self, obs):
        feature_action = self.preprocess(obs)
        with torch.no_grad():
            action = self.actor.act_greedy(feature_action)
        return action.cpu().numpy()[0]

    def save_model(self, path):
        torch.save(self.state_dict(), path)
