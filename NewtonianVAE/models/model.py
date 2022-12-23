import os

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from pixyz.losses import Parameter, LogProb, KullbackLeibler as KL, Expectation as E
from pixyz.models import Model

from models.distributions import Encoder, Decoder, Transition, Velocity

torch.backends.cudnn.benchmark = True


class NewtonianVAE(Model):
    def __init__(self,
                 encoder_param: dict = {},
                 decoder_param: dict = {},
                 transition_param: dict = {},
                 velocity_param: dict = {},
                 optimizer: str = "Adam",
                 optimizer_params: dict = {},
                 clip_grad_norm: bool = False,
                 clip_grad_value: bool = False,
                 delta_time: float = 0.5,
                 device: str = "cuda",
                 use_amp: bool = False):

        # -------------------------#
        # Define models           #
        # -------------------------#
        self.encoder = Encoder(**encoder_param).to(device)
        self.decoder = Decoder(**decoder_param).to(device)
        self.transition = Transition(**transition_param).to(device)
        self.velocity = Velocity(**velocity_param).to(device)

        # -------------------------#
        # Define hyperparams      #
        # -------------------------#
        beta = Parameter("beta")

        # -------------------------#
        # Define loss functions   #
        # -------------------------#
        recon_loss = E(self.transition, LogProb(self.decoder))
        kl_loss = KL(self.encoder, self.transition)
        self.step_loss = (beta*kl_loss - recon_loss).mean()
        print(self.step_loss)

        self.distributions = nn.ModuleList(
            [self.encoder, self.decoder, self.transition, self.velocity])

        # -------------------------#
        # Set params and optim     #
        # -------------------------#
        params = self.distributions.parameters()
        self.optimizer = getattr(optim, optimizer)(params, **optimizer_params)

        # -------------------------------------------------#
        # Set for AMP                                      #
        # Whather to use Automatic Mixture Precision [AMP] #
        # -------------------------------------------------#
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.clip_norm = clip_grad_norm
        self.clip_value = clip_grad_value
        self.delta_time = delta_time
        self.device = device

    def calculate_loss(self, input_var_dict: dict = {}):

        I = input_var_dict["I"]
        u = input_var_dict["u"]
        beta = input_var_dict["beta"]

        total_loss = 0.

        T, B, C = u.shape
        
        # x^q_{t-1} ~ p(x^q_{t-1} | I_{t-1})
        x_q_tn1 = self.encoder.sample({"I_t": I[0]}, reparam=True)["x_t"]

        for step in range(1, T-1):

            # x^q_{t} ~ p(x^q_{t} | I_{t})
            x_q_t = self.encoder.sample({"I_t": I[step]}, reparam=True)["x_t"]

            # v_t = (x^q_{t} - x^q_{t-1})/dt
            v_t = (x_q_t - x_q_tn1)/self.delta_time

            # v_{t+1} = v_{t} + dt (A*x_{t} + B*v_{t} + C*u_{t})
            v_tp1 = self.velocity(x_tn1=x_q_t, v_tn1=v_t, u_tn1=u[step])["v_t"]

            # KL[p(x^p_{t+1} | x^q_{t}, u_{t}; v_{t+1}) || q(x^q_{t+1} | I_{t+1})] - E_p(x^p_{t+1} | x^q_{t}, u_{t}; v_{t+1})[log p(I_{t+1} | x^p_{t+1})]
            step_loss, variables = self.step_loss({'x_tn1': x_q_t, 'v_t': v_tp1, 'I_t': I[step+1], 'beta': beta})

            total_loss += step_loss

            x_q_tn1 = x_q_t

        return total_loss/T

    def train(self, train_x_dict={}):

        self.distributions.train()

        with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP
            loss = self.calculate_loss(train_x_dict)

        self.optimizer.zero_grad(set_to_none=True)
        # self.optimizer.zero_grad()

        # backward
        self.scaler.scale(loss).backward()

        if self.clip_norm:
            clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
        if self.clip_value:
            clip_grad_value_(self.distributions.parameters(), self.clip_value)

        # update params
        self.scaler.step(self.optimizer)

        # update scaler
        self.scaler.update()

        return loss.item()

    def test(self, test_x_dict={}):

        self.distributions.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP
                loss = self.calculate_loss(test_x_dict)

        return loss.item()

    def estimate(self, I_t: torch.Tensor, I_tn1: torch.Tensor, u_t: torch.Tensor):
        self.distributions.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):  # AMP

                # x^q_{t-1} ~ p(x^q_{t-1) | I_{t-1))
                x_q_tn1 = self.encoder.sample_mean({"I_t": I_tn1})

                # x^q_t ~ p(x^q_t | I_t)
                x_q_t = self.encoder.sample_mean({"I_t": I_t})

                # p(I_t | x_t)
                I_t = self.decoder.sample_mean({"x_t": x_q_t})

                # v_t = (x^q_t - x^q_{t-1})/dt
                v_t = (x_q_t - x_q_tn1)/self.delta_time

                # v_{t+1} = v_{t} + dt (A*x_{t} + B*v_{t} + C*u_{t})
                v_tp1 = self.velocity(x_tn1=x_q_t, v_tn1=v_t, u_tn1=u_t)["v_t"]

                # p(x_p_{t+1} | x_q_{t}, v_{t+1})
                x_p_tp1 = self.transition.sample_mean(
                    {"x_tn1": x_q_t, "v_t": v_tp1})

                # p(I_{t+1} | x_{t+1})
                I_tp1 = self.decoder.sample_mean({"x_t": x_p_tp1})

        return I_t, I_tp1, x_q_t, x_p_tp1

    def save(self, path, filename):
        os.makedirs(path, exist_ok=True)

        torch.save({
            'distributions': self.distributions.to("cpu").state_dict(),
        }, f"{path}/{filename}")

        self.distributions.to(self.device)

    def save_ckpt(self, path, filename, epoch, loss):
        os.makedirs(path, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.distributions.to("cpu").state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, f"{path}/{filename}")

        self.distributions.to(self.device)

    def load(self, path, filename):
        self.distributions.load_state_dict(torch.load(
            f"{path}/{filename}", map_location=torch.device('cpu'))['distributions'])

    def load_ckpt(self, path, filename):
        self.distributions.load_state_dict(torch.load(
            f"{path}/{filename}", map_location=torch.device('cpu'))['distributions']['model_state_dict'])

        self.optimizer.load_state_dict(torch.load(
            f"{path}/{filename}", map_location=torch.device('cpu'))['distributions']['optimizer_state_dict'])
