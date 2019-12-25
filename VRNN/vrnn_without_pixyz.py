from tqdm import tqdm
import numpy as np
import random

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter
import math


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def KLGaussianGaussian(phi_mu, phi_sigma, prior_mu, prior_sigma, eps=1e-10):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist
    eps: small number added to variances to avoid NaNs
    """
    prior_sigma = prior_sigma + eps
    kl = 0.5 * (2 * torch.log(prior_sigma) - 2 * torch.log(phi_sigma) + (phi_sigma**2 + (phi_mu - prior_mu)**2) / prior_sigma**2 - 1)
    kl = torch.sum(kl, dim=1).mean()
    if torch.isnan(kl):
        print('kl is Nan')
    if torch.isinf(kl):
        print('kl is inf')
    return kl


def Gaussian_nll(y, mu, sigma):
    """
    gaussian negative log-likelihood
    """
    nll = torch.sum(torch.sqrt(y - mu) / sigma**2 + 2 * torch.log(sigma) + torch.log(2 * math.pi))
    nll = 0.5 * nll
    return nll


def bi_nll(y_hat, y):
    """
    binary cross entropy
    """
    eps = 1e-10
    nll = - (y * torch.log(y_hat+eps) + (1 - y) * torch.log(1 - y_hat + eps))
    nll = torch.sum(nll, dim=1).mean()
    if torch.isnan(nll):
        print('nll is nan')
        print('y_hat', y_hat)
        print('y', y)
        print('log_y_hat', torch.log(y_hat))
        print('log 1 - y_hat', torch.log(1-y_hat))
        print('y * log y_hat', y * torch.log(y_hat))
        print('1-y * log 1 - y_hat', (1 - y) * torch.log(1 - y_hat))
        print(- (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)))
        print('sum', torch.sum(- (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)), dim=1))
        print('mean', torch.sum(- (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)), dim=1).mean())
    if torch.isinf(nll):
        print('nll is inf')
    return nll

# hyper parameter
x_dim = 28
h_dim = 100
z_dim = 16
n_layers =  1
clip = 10
save_every = 10

def init_dataset(f_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    data_dir = '../data'
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda data: data[0])
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=mnist_transform),
        batch_size=f_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=mnist_transform),
        batch_size=f_batch_size, shuffle=True, **kwargs)

    fixed_t_size = 28
    return train_loader, test_loader, fixed_t_size

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


# zとh_t-1に条件づけられたxt
# 入力はfeature_extractされたz, h_t-1
class Generator(nn.Module):
    """
    Parameterizes the bernoulli(for MNIST) observation likelihood p(x_t | z_t, h_{t-1})
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)
        # For Normal
        # self.fc31 = nn.Linear(h_dim, x_dim)
        # self.fc32 = nn.Linear(h_dim, x_dim)

    def forward(self, extracted_z, h_prev):
        """
        Given the latent z at a particular time step t and hidden state,
        return the vector of probabilities taht parameterizes the bernoulli distribution
        p(x_t | z_t, h_{t-1})
        """
        h = torch.cat((extracted_z, h_prev), dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        # for Normal
        # {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

        # for MNIST(Bernoulli)
        return {"probs": torch.sigmoid(self.fc3(h))}


# 潜在変数zのprior, 通常のVAEと違いh_prevにより平均と分散が決まる, 標準正規分布ではない
class Prior(nn.Module):
    """
    Normal
    Compared to normal VAE,
    VRNN's Prior for latent z is parameterized by hidden_state h_{t-1}
    z ~ N(loc(h_{t-1}), scale(h_{t-1}))
    """
    def __init__(self):
        super(Prior, self).__init__()
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, h_prev):
        h = F.relu(self.fc1(h_prev))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


# 事後分布の推論
# feature_extractされたxtと, h_prevによる
class Inference(nn.Module):
    '''
    Normal
    Parameterizes q(z_t | h_{t-1}, x_t)
    infered z ~ N(loc(h_{t-1}, x_t), scale(h_{t-1}, x_t))
    '''
    def __init__(self):
        super(Inference, self).__init__()
        self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, extracted_x, h_prev):
        h = torch.cat((extracted_x, h_prev), dim=-1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


# RNNの部分, x, z, h_prevを入力として次の隠れ状態を出力する
class Recurrence(nn.Module):
    '''
    Deterministic
    RNN for hidden_state
    '''
    def __init__(self):
        super(Recurrence, self).__init__()
        # 1 層のGRUCell
        self.rnncell = nn.GRUCell(h_dim * 2, h_dim).to(device)
        # 隠れ状態
        self.hidden_size = self.rnncell.hidden_size
        

    def forward(self, extracted_x, extracted_z, h_prev):
        
        rnn_input_t = torch.cat((extracted_z, extracted_x), dim=-1)
        h_next = self.rnncell(rnn_input_t, h_prev)
        return {"h": h_next}


class VRNN(nn.Module):
    def __init__(self):
        super(VRNN, self).__init__()
        self.f_phi_x = Phi_x().to(device)
        self.f_phi_z = Phi_z().to(device)
        self.z_prior = Prior().to(device)
        self.encoder = Inference().to(device)
        self.decoder = Generator().to(device)
        self.rnn = Recurrence().to(device)
        self.n_layers = n_layers
    
    def forward(self, x):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0
        x_ts = []
        dec_ts = []
        
        # torch.zeros(batch_size, h_dim), 隠れ状態の初期化
        h = torch.zeros(x.size(1), h_dim).to(device)
        
        # timestep t分処理を行う(x.size(0)=行数)
        for t in range(x.size(0)):

            # Encoding
            enc_t = self.encoder(self.f_phi_x(x[t]), h)
            enc_mean_t, enc_std_t = enc_t['loc'], enc_t['scale']
            
            # prior
            prior_t = self.z_prior(h)
            prior_mean_t, prior_std_t = prior_t['loc'], prior_t['scale']
            
            # z_sampling
            z_t = self.reparameterize(enc_mean_t, enc_std_t)


            # decoding
            dec_t = self.decoder(self.f_phi_z(z_t), h)
            dec_mean_t = dec_t['probs']
            #dec_std_t = dec_t['scale']
            
            # recurence
            h = self.rnn(self.f_phi_x(x[t]), self.f_phi_z(z_t), h)['h']
            
            # compute loss
            kld_loss += KLGaussianGaussian(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += bi_nll(dec_mean_t, x[t])
        return kld_loss, nll_loss

    def sample(self):
        with torch.no_grad():
            x = []
            h = torch.zeros(batch_size, h_dim).to(device)
            for step in range(t_max):
                # prior
                prior_t = self.z_prior(h)
                prior_mean_t, prior_std_t = prior_t['loc'], prior_t['scale']
                
                # z_sampling
                z_t = self.reparameterize(prior_mean_t, prior_std_t)


                # feature extraction
                extracted_zt = self.f_phi_z(z_t)

                # decoding
                dec_t = self.decoder(extracted_zt, h)
                dec_mean_t = dec_t['probs']
                #dec_std_t = dec_t['scale']

                extracted_xt = self.f_phi_x(dec_mean_t)
                
                # recurence
                h = self.rnn(extracted_xt, extracted_zt, h)['h']
                x.append(dec_mean_t[None, :])
            x = torch.cat(x, dim=0).transpose(0, 1)
        return x

    def sample_after_nsteps(self, fixed_batch, n_steps=14):
        x = fixed_batch.transpose(0, 1)
        batch_size = fixed_batch.size()[0]
        with torch.no_grad():
            samples = []
            h = torch.zeros(batch_size, h_dim).to(device)
            for t in range(t_max):
                if t+1 <= n_steps:
                    print('timestep:', t)
                    extracted_xt = self.f_phi_x(x[t])
                    # Encoding
                    enc_t = self.encoder(extracted_xt, h)
                    enc_mean_t, enc_std_t = enc_t['loc'], enc_t['scale']
                    
                    # z_sampling
                    #z_t = self.reparameterize(enc_mean_t, enc_std_t)
                    z_t = enc_mean_t
                    # feature extraction
                    extracted_zt = self.f_phi_z(z_t)

                    # decoding
                    dec_t = self.decoder(extracted_zt, h)
                    dec_mean_t = dec_t['probs']
                    #dec_std_t = dec_t['scale']
                    
                    # recurence
                    h = self.rnn(extracted_xt, extracted_zt, h)['h']
                    samples.append(dec_mean_t[None, :])
                else:
                    print('Generate! timestep:', t)
                    # prior
                    prior_t = self.z_prior(h)
                    prior_mean_t, prior_std_t = prior_t['loc'], prior_t['scale']
                    
                    # z_sampling
                    z_t = prior_mean_t
                    # feature extraction
                    extracted_zt = self.f_phi_z(z_t)

                    # decoding
                    dec_t = self.decoder(extracted_zt, h)
                    dec_mean_t = dec_t['probs']
                    #dec_std_t = dec_t['scale']

                    extracted_xt = self.f_phi_x(dec_mean_t)
                    
                    # recurence
                    h = self.rnn(extracted_xt, extracted_zt, h)['h']
                    samples.append(dec_mean_t[None, :])
            samples = torch.cat(samples, dim=0).transpose(0, 1)
        return samples


    def reconst(self, fixed_batch):
        x = fixed_batch.transpose(0, 1)
        batch_size = fixed_batch.size()[0]
        reconst = []

        with torch.no_grad():
            h = torch.zeros(batch_size, h_dim).to(device)
            for t in range(t_max):
                # infer
                extracted_xt = self.f_phi_x(x[t])
                enc_t = self.encoder(extracted_xt, h)
                enc_mean_t, enc_std_t = enc_t['loc'], enc_t['scale']
               
                z_t = enc_mean_t


                # feature extraction
                extracted_zt = self.f_phi_z(z_t)

                # decoding
                dec_t = self.decoder(extracted_zt, h)
                dec_mean_t = dec_t['probs']
                #dec_std_t = dec_t['scale']
                h = self.rnn(extracted_xt, extracted_zt, h)['h']
                reconst.append(dec_mean_t[None, :])
            reconst_img = torch.cat(reconst, dim=0).transpose(0, 1)
        return reconst_img
   
    
    
    def reparameterize(self, mean, var):
        """using std to sample"""
        eps = torch.randn(mean.size()).to(device)
        z = mean + torch.sqrt(var) * eps
        return z


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    lr = args.lr
    seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs
    torch.manual_seed(seed)

    train_loader, test_loader, t_max = init_dataset(batch_size)

    prior = Prior().to(device)
    decoder = Generator().to(device)
    encoder = Inference().to(device)
    recurrence = Recurrence().to(device)
    vrnn = VRNN().to(device)
    optimizer = optim.Adam(vrnn.parameters(), lr=lr)
    
    writer = SummaryWriter(comment='LR_{}_SEED_{}_bsize_{}'.format(lr, seed, batch_size))
    def train():
        vrnn.train()
        epoch_loss = 0
        epoch_kld_loss = 0
        epoch_nll_loss = 0
        for data, _ in train_loader:
            b_size = data.size()[0]
            data = data.to(device).transpose(0, 1)
            #data = (data - data.min().item()) / (data.max().item() - data.min().item())
            
            kld_loss, nll_loss = vrnn(data)

            optimizer.zero_grad()
            loss = kld_loss + nll_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * b_size
            epoch_kld_loss += kld_loss.item() * b_size
            epoch_nll_loss += nll_loss.item() * b_size
        epoch_loss /= len(train_loader.dataset)
        epoch_kld_loss /= len(train_loader.dataset)
        epoch_nll_loss /= len(train_loader.dataset)
        return epoch_loss, epoch_kld_loss, epoch_nll_loss
    def test():
        vrnn.eval()
        epoch_loss = 0
        epoch_kld_loss = 0
        epoch_nll_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                b_size = data.size()[0]
                data = data.to(device).transpose(0, 1)
                #data = (data - data.min().item()) / (data.max().item() - data.min().item())
                
                kld_loss, nll_loss = vrnn(data)

                loss = kld_loss + nll_loss
                epoch_loss += loss.item() * b_size
                epoch_kld_loss += kld_loss.item() * b_size
                epoch_nll_loss += nll_loss.item() * b_size
        epoch_loss /= len(test_loader.dataset)
        epoch_kld_loss /= len(test_loader.dataset)
        epoch_nll_loss /= len(test_loader.dataset)
        return epoch_loss, epoch_kld_loss, epoch_nll_loss
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)
    for epoch in range(1, epochs):
        train_loss, train_kld_loss, train_nll_loss = train()
        test_loss, test_kld_loss, test_nll_loss = test()
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_kld_loss', train_kld_loss, epoch)
        writer.add_scalar('train_nll_loss', train_nll_loss, epoch)

        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_kld_loss', test_kld_loss, epoch)
        writer.add_scalar('test_nll_loss', test_nll_loss, epoch)
        sample = vrnn.sample()[:, None]
        writer.add_images('Image_from_latent', sample, epoch)

        sample_after_nsteps = vrnn.sample_after_nsteps(fixed_batch=_x)[:, None]
        writer.add_images('Image_after_14steps', sample_after_nsteps, epoch)

        reconst_img= vrnn.reconst(fixed_batch=_x)
        writer.add_images('reconst', reconst_img[:, None], epoch)

        writer.add_images('orignal', _x[:, None], epoch)

    if epoch % save_every == 1:
            fn = 'saves/vrnn_state_dict_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), fn)
            print('saved model to ' + fn)