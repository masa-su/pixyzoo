import torch
from torch import nn
from torch.nn import functional as F

from pixyz.distributions import Normal
from pixyz.losses import LogProb, KullbackLeibler
z1_dim = 32
z2_dim = 256
embedded_obs_dim = 256
num_seq = 8


class ConvEmbedder(nn.Module):
    def __init__(self):
        """Convolutional network used for embeddings"""
        super(ConvLayers, self).__init__()
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
            nn.Conv2d(256, 256, 4),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)

    def forward(self, inputs):
        return self.network(inputs)


class Decoder(Normal):
    def __init__(self, std=0.1):
        """Decodes z into an observation"""
        super(TransitionConvLayer, self).__init__(cond_var=)
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

    def forward(self, inputs):
        loc = self.network(inputs)
        scale = torch.ones_like(inputs).mul_(self.std)
        return {"loc": loc, "scale": scale}


class Encoder(Normal):
    def __init__(self, cond_var=[""]):


class Gaussian(Normal):
    def __init__(self, cond_var_dict: dict, var_dict: dict):
        assert len(var_dict.keys()) == 1
        super(Gaussian, self).__init__(
            cond_var=cond_var_dict.keys(), var=var_dict.keys()[0])
        in_dim = sum(cond_var_dict.values())
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, var_dict.values()[0]*2)
        )
        self.inputs = cond_var_dict.keys()

    def forward(self, **kwargs):
        # sort vars in the fixed order
        x = []
        for varname in self.inputs:
            x.append(kwargs[varname])
        x = torch.cat(x, dim=-1)
        loc, scale = torch.chunk(x, 2, dim=-1)
        scale = F.softplus(std) + 1e-5
        return {"loc": loc, "scale": scale}


class QFunc(nn.Module):
    def __init__(self, z1_dim, z2_dim, num_action: int):
        super(QFunc, self).__init__()
        input_dim = z1_dim + z2_dim + num_action
        self.network1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Relu(),
            nn.Linear(256, 256),
            nn.Relu(),
            nn.Linear(256, 1)
        )
        self.network2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Relu(),
            nn.Linear(256, 256),
            nn.Relu(),
            nn.Linear(256, 1)
        )

    def forward(self, z1, z2, action_t_1):
        inputs = torch.cat([z1, z2, action_t_1])
        return self.network1(inputs), self.network2(inputs)


class Pie(Normal):
    def __init__(self, in_dim, action_dim):
        super(Actor, self).__init__(
            cond_var=["x_embedded"], name='pi')  # convでembedされた系列ですが?
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2*action_dim)
        )

    def forward(self, x_embedded):
        loc, scale = torch.chunk(self.fcs(embedded_obs), 2, dim=-1)
        self.loc = loc
        self.scale = self.scale
        return {"loc": loc, "scale": scale}


class Actor:
    def __init__(self, action_dim):
        self.pi = Pie(in_dim=num_seq*embedded_obs_dim +
                      (num_seq - 1)*action_dim)

    def sample(self, inputs):
        action = self.pi.sample(x_embedded)
        action = action["pi"]
        log_prob = self.pi.dist.log_prob()

        return torch.Tanh(action), log_prob(action)


class GenerativeModel:
    def __init__(self):
        self.initial_z1_dist = Normal(loc=torch.Tensor(0.), scale=torch.Tensor(
            1.), var=['z_1^0'], cond_var=[], name='p_0_{z1}', features_shape=[z1_dim])
        self.initial_z2_dist = Gaussian(
            cond_var_dict={"z_1": z1_dim}, var_dict={"z_2": z2_dim})
        self.z1_from_z2 = Gaussian(cond_var_dict={
            "z_t^2": z2_dim, "a_t": 1}, var_dict={"z_{t + 1}^1": z1_dim})
        self.z2_dist = Gaussian(cond_var_dict={
                                "z_{t + 1}^1": z1_dim, "z_t^2": z2_dim, "a_t": 1}, var_dict={"z_{t + 1}^2": z2_dim})
        self.reward_dist = Gaussian(cond_var_dict={
                                    "z_1^t": z1_dim, "z_2^t": z2_dim, "z_{t + 1}^1": z1_dim, "z_{t + 1}^2": z2_dim, "a_t": 1}, var_dict={"r_t": 1})
        self.decoder = Decoder()


class VariationalModel:
    def __init__(self, initial_z2_dist, z2_dist, z1_from_z2):
        self.initial_z1_dist = Gaussian(
            cond_var_dict={"x_1_embedded": embedded_obs_dim}, var_dict={"z_1": z1_dim})
        self.z1_dist = Gaussian(
            cond_var_dict={"x_1_embedded":  embedded_obs_dim, "z_1": z1_dim, "a_t": 1, "z_t^2": z2_dim}, var_dict={"z_{t + 1}^1"})  # (26)

        self.initial_z2_dist = initial_z2_dist
        self.z2_dist = z2_dist
        self.decoder = Decoder()
        self.loss =
        - LogProb(Decoder) + \
            KullbackLeibler(LogProb(self.z1_dist), LogProb(z1_from_z2))


class SLAC:
    def __init__(self, action_dim=1):
        self.critic = QFunc(z1_dim=z1_dim, z2_dim=z2_dim, num_action=1, alpha=)
        self.critic_target = QFunc(z1_dim=z1_dim, z2_dim=z2_dim, num_action=1)
        self.encoder = ConvEmbedder()
        self.actor = Actor(action_dim=action_dim)
        self.optim_critic =

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

    def update_critic(self, z, next_z, action, embedded_obs_next, reward, done):
        q1, q2 = self.critic(z, action)

        with torch.no_grad():
            action, log_prob = self.actor.sample(inputs)
            q1_next, q2_next = self.critic_target(next_z)
            q_next = torch.min(q1_next, q2_next) - self.alpha * log_prob
        target_q = reward + (1.0 - done) * self.gamma * q_next
        loss = (q1 - target_q).pow_(2).mean() + (q2 - target_q).pow_(2).mean()
        self.optim_critic.zero_grad()
        loss.backward(retain_graph=False)
        self.optim_critic.step()
