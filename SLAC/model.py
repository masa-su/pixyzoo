from pixyz.distributions import Normal
from torch import nn

z1_dim = 32
z2_dim = 256

initial_z1_dist = Normal(loc=torch.Tensor(0.), scale=torch.Tensor(
    1.), var=['z_1'], cond_var=[], name='p_0_{z1}', features_shape=[z1_dim])
initial_z2_dist = Normal(loc=torch.Tensor(0.), scale=torch.Tensor(
    1.), var=['z_2'], cond_var=[], name='p_0_{z2}', features_shape=[z2_dim])


class ConvLayers(nn.Module):
    def __init__(self):
        """Convolutional network used for """
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
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)

    def forward(self, inputs):
        return self.network(inputs)


class TransitionConvLayer(nn.Module):
    def __init__(self):
        """Convolutional network used for transition network"""
        super(TransitionConvLayer, self).__init__()
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

    def forward(self, inputs):
        return self.network(inputs)


class QFunc(nn.Module):
    def __init__(self, z1_dim, z2_dim, num_action: int):
        super(QFunc, self).__init__()
        input_dim = z1_dim + z2_dim + num_action
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Relu(),
            nn.Linear(256, 256),
            nn.Relu(),
            nn.Linear(256, 1)
        )

    def forward(self, z1, z2, action_t_1):
        inputs = torch.cat([z1, z2, action_t_1])
        return self.network(inputs)


class Actor(Normal):
    def __init__(self, convs, num_action):
        super(Actor, self).__init__(
            cond_var=["z_1", "z_2", "a_t_1"], name='\pi_{\psi}')
        self.convs = convs

    def forward(self, obs):
        inputs = torch.cat([z_1, z_2, a_t_1], dim=0)
