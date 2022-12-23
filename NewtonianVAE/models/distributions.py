import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from pixyz import distributions as dist
from pixyz.utils import epsilon


class Encoder(dist.Normal):
    """
      q(x_t | I_t) = N(x_t | I_t)
    """

    def __init__(self, input_dim: int, output_dim: int, act_func_name: str):
        super().__init__(var=["x_t"], cond_var=["I_t"], name="q")

        activation_func = getattr(nn, act_func_name)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            activation_func(),
            nn.Conv2d(32, 64, 4, stride=2),
            activation_func(),
            nn.Conv2d(64, 128, 4, stride=2),
            activation_func(),
            nn.Conv2d(128, 256, 4, stride=2),
            activation_func(),
        )

        self.loc = nn.Sequential(
            nn.Linear(1024, output_dim),
        )

        self.scale = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.Softplus()
        )

    def forward(self, I_t: torch.Tensor) -> dict:
        feature = self.encoder(I_t)
        B, C, W, H = feature.shape
        feature = feature.reshape((B, C*W*H))

        loc = self.loc(feature)
        scale = self.scale(feature) + epsilon()

        return {"loc": loc, "scale": scale}


class Decoder(dist.Normal):
    """
      p(I_t | x_t) = N(I_t | x_t)
    """

    def __init__(self, input_dim: int, output_dim: int, act_func_name: str, device: str):
        super().__init__(var=["I_t"], cond_var=["x_t"])

        activation_func = getattr(nn, act_func_name)

        self.loc = nn.Sequential(
            nn.Conv2d(input_dim+2, 64, 3, stride=1, padding=1),
            activation_func(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            activation_func(),
            nn.Conv2d(64, output_dim, 3, stride=1, padding=1),
        )

        self.image_size = 64
        a = np.linspace(-1, 1, self.image_size)
        b = np.linspace(-1, 1, self.image_size)
        x, y = np.meshgrid(a, b)
        x = x.reshape(self.image_size, self.image_size, 1)
        y = y.reshape(self.image_size, self.image_size, 1)
        self.xy = np.concatenate((x, y), axis=-1)

        self.device = device

    def forward(self, x_t: torch.Tensor) -> dict:
        batchsize = len(x_t)
        xy_tiled = torch.from_numpy(
            np.tile(self.xy, (batchsize, 1, 1, 1)).astype(np.float32)).to(self.device)

        z_tiled = torch.repeat_interleave(
            x_t, self.image_size*self.image_size, dim=0).view(batchsize, self.image_size, self.image_size, 2)

        z_and_xy = torch.cat((z_tiled, xy_tiled), dim=3)
        z_and_xy = z_and_xy.permute(0, 3, 2, 1)

        loc = self.loc(z_and_xy)

        return {"loc": loc, "scale": .01}


class Transition(dist.Normal):
    """
      p(x_t | x_{t-1}, u_{t-1}; v_t) = N(x_t | x_{t-1} + ∆t·v_t, σ^2)
    """

    def __init__(self, delta_time: float):
        super().__init__(var=["x_t"], cond_var=["x_tn1", "v_t"])

        self.delta_time = delta_time

    def forward(self, x_tn1: torch.Tensor, v_t: torch.Tensor) -> dict:

        x_t = x_tn1 + self.delta_time * v_t

        return {"loc": x_t, "scale": 0.001}


class Velocity(dist.Deterministic):
    """
      v_t = v_{t-1} + ∆t·(A·x_{t-1} + B·v_{t-1} + C·u_{t-1})
      with  [A, log(−B), log C] = diag(f(x_{t-1}, v_{t-1}, u_{t-1}))
    """

    def __init__(self, batch_size: int, delta_time: float, act_func_name: str, device: str, use_data_efficiency: bool):
        super().__init__(var=["v_t"], cond_var=[
            "x_tn1", "v_tn1", "u_tn1"], name="f")

        activation_func = getattr(nn, act_func_name)
        self.delta_time = delta_time
        self.device = device
        self.use_data_efficiency = use_data_efficiency

        if not self.use_data_efficiency:

            self.coefficient_ABC = nn.Sequential(
                nn.Linear(2*3, 2),
                activation_func(),
                nn.Linear(2, 2),
                activation_func(),
                nn.Linear(2, 2),
                activation_func(),
                nn.Linear(2, 6),
            )

        else:
            self.A = torch.zeros((batch_size, 2, 2)).to(self.device)
            self.B = torch.zeros((batch_size, 2, 2)).to(self.device)
            self.C = torch.diag_embed(torch.ones(batch_size, 2)).to(self.device)

    def forward(self, x_tn1: torch.Tensor, v_tn1: torch.Tensor, u_tn1: torch.Tensor) -> dict:

        combined_vector = torch.cat([x_tn1, v_tn1, u_tn1], dim=1)

        # For data efficiency
        if self.use_data_efficiency:
            A = self.A[:len(combined_vector)]
            B = self.B[:len(combined_vector)]
            C = self.C[:len(combined_vector)]
        else:
            _A, _B, _C = torch.chunk(self.coefficient_ABC(combined_vector), 3, dim=-1)
            A = torch.diag_embed(_A)
            B = -touch.exp(torch.diag_embed(_B))
            C = torch.exp(torch.diag_embed(_C))

        # Dynamics inspired by Newton's motion equation
        v_t = v_tn1 + self.delta_time * (torch.einsum("ijk,ik->ik", A, x_tn1) + torch.einsum(
            "ijk,ik->ik", B, v_tn1) + torch.einsum("ijk,ik->ik", C, u_tn1))

        return {"v_t": v_t}
