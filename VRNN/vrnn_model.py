from tqdm import tqdm
import numpy as np
import random

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
#from tensorboardX import SummaryWriter
import math


from pixyz.models import Model
from pixyz.losses import KullbackLeibler, StochasticReconstructionLoss
from pixyz.losses import IterativeLoss
from pixyz.distributions import Bernoulli, Normal, Deterministic
from pixyz.utils import print_latex


batch_size = 4
epochs = 100
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

def KLGaussianGaussian(phi_mu, phi_sigma, prior_mu, prior_sigma):
    '''
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist
    '''
    kl = 0.5 * (2 * torch.log(prior_sigma) - 2 * torch.log(phi_sigma) + (phi_sigma**2 + (phi_mu - prior_mu)**2) / prior_sigma**2 - 1)
    kl = torch.sum(kl)
    return kl


def Gaussian_nll(y, mu, sigma):
    '''
    gaussian negative log-likelihood
    '''
    nll = torch.sum(torch.sqrt(y - mu) / sigma**2 + 2 * torch.log(sigma) + torch.log(2 * math.pi))
    nll = 0.5 * nll
    return nll


def bi_nll(y_hat, y):
    '''
    binary cross entropy
    '''
    nll = - (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    nll = torch.sum(nll)
    return nll


x_dim = 28
h_dim = 100
z_dim = 64
t_max = x_dim

# xのfeature_extraction, Encoderの入力となる
class Phi_x(nn.Module):
    def __init__(self):
        super(Phi_x, self).__init__()
        self.fc0 = nn.Linear(x_dim, h_dim)

    def forward(self, x):
        return F.relu(self.fc0(x))

# 潜在変数zを加工するNN, decoderの入力となる, feature_extraction
class Phi_z(nn.Module):
    def __init__(self):
        super(Phi_z, self).__init__()
        self.fc0 = nn.Linear(z_dim, h_dim)

    def forward(self, z):
        return F.relu(self.fc0(z))

f_phi_x = Phi_x().to(device)
f_phi_z = Phi_z().to(device)


# zとh_t-1に条件づけられたxt
# 入力はfeature_extractされたz, h_t-1
class Generator(Bernoulli):
    def __init__(self):
        super(Generator, self).__init__(cond_var=["z", "h_prev"], var=["x"])
        self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)
        # MNISTだから?
        # 論文上のやつの実装
        # self.fc31 = nn.Linear(h_dim, x_dim)
        # self.fc32 = nn.Linear(h_dim, x_dim)
        self.f_phi_z = f_phi_z

    def forward(self, z, h_prev):
        h = torch.cat((self.f_phi_z(z), h_prev), dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        # MNISTだから?
        # 論文上のやつの実装
        # {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}
        return {"probs": torch.sigmoid(self.fc3(h))}


# 潜在変数zのprior, 通常のVAEと違いh_prevにより平均と分散が決まる, 標準正規分布ではない
class Prior(Normal):
    def __init__(self):
        super(Prior, self).__init__(cond_var=["h_prev"], var=["z"])
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, h_prev):
        h = F.relu(self.fc1(h_prev))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


# 事後分布の推論
# feature_extractされたxtと, h_prevによる
class Inference(Normal):
    def __init__(self):
        super(Inference, self).__init__(cond_var=["x", "h_prev"], var=["z"], name="q")
        self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.f_phi_x = f_phi_x

    def forward(self, x, h_prev):
        h = torch.cat((self.f_phi_x(x), h_prev), dim=-1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


# RNNの部分, x, z, h_prevを入力として次の隠れ状態を出力する
class Recurrence(Deterministic):
    def __init__(self):
        super(Recurrence, self).__init__(cond_var=["x", "z", "h_prev"], var=["h"])
        # 1 層のGRUCell
        self.rnncell = nn.GRUCell(h_dim * 2, h_dim).to(device)
        # 隠れ状態
        self.hidden_size = self.rnncell.hidden_size
        
        # xtのfeature_extractor
        self.f_phi_x = f_phi_x
        #zのfeature_extractor
        self.f_phi_z = f_phi_z

    def forward(self, x, z, h_prev):
        extracted_x = self.f_phi_x(x)
        extracted_z = self.f_phi_z(z)
        
        rnn_input_t = torch.cat((extracted_z, extracted_x), dim=-1)
        h_next = self.rnncell(rnn_input_t, h_prev)
        return {"h": h_next}


class VRNN(nn.Module):
    def __init__(self):
        super(VRNN, self).__init__()
        self.z_prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.rnn = recurrence
        self.n_layers = 1
    
    def forward(self, x):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0
        x_ts = []
        dec_ts = []
        
        # torch.zeros(batch_size, h_dim), 隠れ状態の初期化
        h = torch.zeros(x.size(1), h_dim)
        
        # timestep t分処理を行う(x.size(0)=行数)
        for t in range(x.size(0)):
            # Encoding
            enc_t = self.encoder(x[t], h)
            enc_mean_t, enc_std_t = enc_t['loc'], enc_t['scale']
            
            # prior
            prior_t = self.z_prior(h)
            prior_mean_t, prior_std_t = prior_t['loc'], prior_t['scale']
            
            # z_sampling
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            
            # decoding
            dec_t = self.decoder(z_t, h)
            dec_mean_t = dec_t['probs']
            #dec_std_t = dec_t['scale']
            
            # recurence
            h = self.rnn(x[t], z_t, h)['h']
            
            # compute loss
            kld_loss += KLGaussianGaussian(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += bi_nll(dec_mean_t, x[t])
        return kld_loss, nll_loss
    
    
    def reparameterize(self, mean, var):
        """using std to sample"""
        eps = torch.randn(mean.size()).to(device)
        z = mean + torch.sqrt(var) * eps
        return z


if __name__ == '__main__':
    prior = Prior().to(device)
    decoder = Generator().to(device)
    encoder = Inference().to(device)
    recurrence = Recurrence().to(device)
    vrnn = VRNN().to(device)
    optimizer = optim.Adam(vrnn.parameters(), lr=0.001)
    def train(epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            for data, _ in train_loader:
                data = data.to(device).squeeze().transpose(0, 1)
                
                kld_loss, nll_loss = vrnn(data)

                optimizer.zero_grad()
                loss = kld_loss + nll_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            with open('./logs/train_loss.txt') as f:
                f.write(epoch_loss)
    train(epochs)
