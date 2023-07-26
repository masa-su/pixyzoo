
# coding: utf-8

# # Deep Markov Model

# In[1]:

from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter


# In[2]:

# In[3]:

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=128)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()
x_dim = 28
hidden_dim = 32
rnn_dim = hidden_dim * 2
transition_dim = 32
z_dim = 16
t_max = x_dim
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
seed = args.seed
torch.manual_seed(seed)


# In[4]:

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

train_loader, test_loader, t_max = init_dataset(batch_size)


# In[6]:
# Loss in https://github.com/clinicalml/dmm/blob/master/model_th/dmm.py
def KLGaussianGaussian(phi_mu, phi_sigma, prior_mu, prior_sigma, eps=1e-10):
    '''
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist
    eps: small number added to variances to avoid NaNs
    '''
    prior_sigma = prior_sigma + eps
    kl = 0.5 * (2 * torch.log(prior_sigma) - 2 * torch.log(phi_sigma) + (phi_sigma**2 + (phi_mu - prior_mu)**2) / prior_sigma**2 - 1)
    kl = torch.sum(kl, dim=1).mean()
    if torch.isnan(kl):
        print('kl is Nan')
    if torch.isinf(kl):
        print('kl is inf')
    return kl


def bi_nll(y_hat, y):
    '''
    binary cross entropy
    '''
    eps = 1e-10
    nll = - (y * torch.log(y_hat+eps) + (1 - y) * torch.log(1 - y_hat + eps))
    nll = torch.sum(nll, dim=1).mean()
    if torch.isnan(nll):
        print('nll is nan')
        #print('y_hat', y_hat)
        #print('y', y)
        #print('log_y_hat', torch.log(y_hat))
        #print('log 1 - y_hat', torch.log(1-y_hat))
        #print('y * log y_hat', y * torch.log(y_hat))
        #print('1-y * log 1 - y_hat', (1 - y) * torch.log(1 - y_hat))
        #print(- (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)))
        #print('sum', torch.sum(- (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)), dim=1))
        #print('mean', torch.sum(- (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)), dim=1).mean())
    if torch.isinf(nll):
        print('nll is inf')
    return nll
# In[7]:

class RNN(nn.Module):
    '''
    push the observed x through the rnn
    rnn output contains the hidden state at each time step
    deterministic
    '''
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(x_dim, rnn_dim, bidirectional=True)
#         self.h0 = torch.zeros(2, batch_size, self.rnn.hidden_size).to(device)
        self.h0 = nn.Parameter(torch.zeros(2, 1, self.rnn.hidden_size))
        self.hidden_size = self.rnn.hidden_size
        
    def forward(self, x):
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h0 = self.h0.expand(2, x.size(1), self.rnn.hidden_size).contiguous()
        h, _ = self.rnn(x, h0)
        return {"h": h}


# In[8]:

class Generator(nn.Module):
    '''
    Emitter
    Parameterizes the bernoulli observation likelihood p(x_t | z_t)
    Bernoulli
    '''
    def __init__(self):
        super(Generator, self).__init__()
        # initialize the two linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_hidden_to_input = nn.Linear(hidden_dim, x_dim)

        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        '''
        Given the latent z at a particular time step t, return the vector of
        probabilities taht parameterizes the bernlulli distribution p(x_t | x_t)
        '''
        h1 = self.relu(self.lin_z_to_hidden(z))
        probs = self.sigmoid(self.lin_hidden_to_input(h1))
        return {"probs": probs}


# In[9]:

class Inference(nn.Module):
    '''
    Combiner
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of te guide(i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN
    Normal
    '''
    def __init__(self):
        super(Inference, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim*2)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim*2, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim*2, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        
    def forward(self, h, z_prev):
        '''
        given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN h(x_{t:T}), return the mean and scale vectors that
        parameterize the gaussian distribution q(z_t | z_{t-1}, x_{t:T})
        '''
        # combine the rnn hideen state with a trasnformed bersion of z_{t-1}
        h_z = self.tanh(self.lin_z_to_hidden(z_prev))
        h_combined = 0.5 * (h + h_z)

        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return {"loc": loc, "scale": scale}


# In[10]:

class Prior(nn.Module):
    '''
    GatedTranstion
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    Normal
    '''
    def __init__(self):
        super(Prior, self).__init__()
        # initialize the 3 linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)

        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)

        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)

        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        # initialize the 3 non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        
    def forward(self, z_prev):
        '''
        Given the latent z_{t-1} correspoding to the time step t-1
        return the mean and scale vectors that parameterize the
        gaussian distribution p(z_t | z_{t-1})
        '''
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_prev))
        gate = self.sigmoid(self.lin_gate_hidden_to_z(_gate))

        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_prev))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)

        # assemble the actual mean used to sample z_t, which mixes
        # a linear transformation of z_{t-1} with the proposed mean
        # modulated by the gating function
        # we don't want to force the dynamics be-nonlinear
        loc = (1 - gate) * self.lin_z_to_loc(z_prev) + gate * proposed_mean
        
        # compute the scale used to sample z_t, using the proposed
        # mean from above as input. the softplus ensures that scale is positive
        # scale の元がmeanなの初めてみたよ
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        
        # return loc, scale which can be fed into Normal
        return {"loc": loc, "scale": scale}


class DMM(nn.Module):
    '''
    this pytorch module encapsulates the model as well as the 
    variational distribution (the guide) for the deep markov model
    '''
    def __init__(self):
        super(DMM, self).__init__()
        # instantiate pytorch modules used in the model and guide below
        self.emitter = Generator().to(device)
        self.trans = Prior().to(device)
        self.combiner = Inference().to(device)
        self.rnn = RNN().to(device)

        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim)).to(device)
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim)).to(device)

    # the model p(x_{1:T} | z_{1:T} p(z_{1:T}))
    def generate(self, sample_num):
        with torch.no_grad():
            x = []

            # set z_prev = z_0 to setup the recursize conditioning in p(z_t | z_{t-1})
            z_prev = self.z_0.expand(sample_num, self.z_0.size(0))

            for t in range(t_max):
                # Prior
                # first compute the parameters of the diagnal gaussian
                z = self.trans(z_prev)
                z_loc, z_scale = z['loc'], z['scale']

                # z_sampling
                z_t = self.reparameterize(z_loc, z_scale)

                # compute the probabilities that parameterize the bernoulli likelihood
                emission_probs_t = self.emitter(z_t)['probs']

                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

                x.append(emission_probs_t[None, :])
            x = torch.cat(x, dim=0).transpose(0,1)
        return x
    
    def reconst(self, img):
        reconst = []
        data = img.to(device)
        x = data.transpose(0, 1)

        with torch.no_grad():
            z_prev = self.z_q_0.expand(x.size(1), self.z_q_0.size(0))
            rnn_output = self.rnn(x)['h']
            for t in range(t_max):
                h_t = rnn_output[t]
                
                z_t = z = self.combiner(h_t, z_prev)['loc']
                
                emission_probs_t = self.emitter(z_t)['probs']
                reconst.append(emission_probs_t[None, :])
                z_prev = z_t
            reconst_img = torch.cat(reconst, dim=0).transpose(0, 1)
        return reconst_img

    def sample_after_nsteps(self, img, n_step):
        sample = []
        data = img.to(device)
        x = data.transpose(0, 1)

        with torch.no_grad():
            z_prev = self.z_q_0.expand(x.size(1), self.z_q_0.size(0))
            rnn_output = self.rnn(x)['h']
            for t in range(t_max):
                if t+1 < n_step:
                    h_t = rnn_output[t]
                    
                    z_t = z = self.combiner(h_t, z_prev)['loc']
                    
                    emission_probs_t = self.emitter(z_t)['probs']
                    sample.append(emission_probs_t[None, :])
                    z_prev = z_t
                else:
                    z_t = self.trans(z_prev)['loc']
                    emission_probs_t = self.emitter(z_t)['probs']
                    sample.append(emission_probs_t[None, :])
                    z_prev = z_t
            sample_img_after_nteps = torch.cat(sample, dim=0).transpose(0, 1)
        return sample_img_after_nteps
    
    def forward(self, x):
        kld_loss = 0
        nll_loss = 0
        # x(t_step, batch_size, features)
        T_max = x.size(0)
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in continguous gpu memory
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t | )
        z_prev = self.z_q_0.expand(x.size(1), self.z_q_0.size(0))

        # set z_prev = z_0 to setup the recursize conditioning in p(z_t | z_{t-1})
        prior_z_prev = self.z_0.expand(x.size(1), self.z_0.size(0))

        rnn_output = self.rnn(x)['h']
        for t in range(T_max):
            z = self.combiner(rnn_output[t], z_prev)
            z_loc, z_scale = z['loc'], z['scale']
            
            # sample z_t from the distribution z_dist
            z_t = self.reparameterize(z_loc, z_scale)

            # the latent sampled at this time step will be conditioned 
            # upon in the next time step so keep track of it
            z_prev = z_t

            # compute the probabilities that parameterize the bernoulli likelihood
            emission_probs_t = self.emitter(z_t)['probs']

            # Prior
            # first compute the parameters of the diagnal gaussian
            prior_z = self.trans(z_prev)
            prior_z_loc, prior_z_scale = prior_z['loc'], prior_z['scale']

            # z_sampling
            prior_z_t = self.reparameterize(prior_z_loc, prior_z_scale)

            # the latent sampled at this time step will be conditioned upon
            # in the next time step so keep track of it
            prior_z_prev = prior_z_t

            kld_loss += KLGaussianGaussian(z_loc, z_scale, prior_z_loc, prior_z_scale)
            nll_loss += bi_nll(emission_probs_t, x[t])

        return kld_loss, nll_loss


    def reparameterize(self, loc, scale):
        """using std to sample"""
        eps = torch.randn(loc.size()).to(device)
        z = loc + torch.sqrt(scale) * eps
        return z


# In[19]:
def train(model, clip_grad_value):
    model.train()
    epoch_loss = 0
    epoch_kld_loss = 0
    epoch_nll_loss = 0
    for data, _ in train_loader:
        b_size = data.size()[0]
        data = data.to(device).transpose(0, 1)
        #data = (data - data.min().item()) / (data.max().item() - data.min().item())
        
        kld_loss, nll_loss = model(data)

        optimizer.zero_grad()
        loss = kld_loss + nll_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dmm.parameters(), clip_grad_value)
        optimizer.step()
        epoch_loss += loss.item() * b_size
        epoch_kld_loss += kld_loss.item() * b_size
        epoch_nll_loss += nll_loss.item() * b_size
    epoch_loss /= len(train_loader.dataset)
    epoch_kld_loss /= len(train_loader.dataset)
    epoch_nll_loss /= len(train_loader.dataset)
    return epoch_loss, epoch_kld_loss, epoch_nll_loss
def test(model):
    model.eval()
    epoch_loss = 0
    epoch_kld_loss = 0
    epoch_nll_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            b_size = data.size()[0]
            data = data.to(device).transpose(0, 1)
            #data = (data - data.min().item()) / (data.max().item() - data.min().item())
            
            kld_loss, nll_loss = model(data)

            loss = kld_loss + nll_loss
            epoch_loss += loss.item() * b_size
            epoch_kld_loss += kld_loss.item() * b_size
            epoch_nll_loss += nll_loss.item() * b_size
    epoch_loss /= len(test_loader.dataset)
    epoch_kld_loss /= len(test_loader.dataset)
    epoch_nll_loss /= len(test_loader.dataset)
    return epoch_loss, epoch_kld_loss, epoch_nll_loss

writer = SummaryWriter(comment='LR_{}_SEED_{}_bsize_{}'.format(lr, seed, batch_size))
dmm = DMM().to(device)
optimizer = optim.RMSprop(params=dmm.parameters(), lr=lr)

# fix for checking training procedure
_x, _ = iter(test_loader).next()
_x = _x.to(device)

for epoch in range(1, epochs + 1):
    dmm.train()
    train_loss, train_kld_loss, train_nll_loss = train(model=dmm, clip_grad_value=10)
    dmm.eval()
    test_loss, test_kld_loss, test_nll_loss = test(dmm)

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_kld_loss', train_kld_loss, epoch)
    writer.add_scalar('train_nll_loss', train_nll_loss, epoch)
    
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_kld_loss', test_kld_loss, epoch)
    writer.add_scalar('test_nll_loss', test_nll_loss, epoch)

    sample = dmm.generate(batch_size)[:, None]
    writer.add_images('Image_from_latent', sample, epoch)

    reconst_img = dmm.reconst(_x)
    writer.add_images('reconst', reconst_img[:, None], epoch)

    sample_img_after_nteps = dmm.sample_after_nsteps(_x, 14)
    writer.add_images('after_nsteps', sample_img_after_nteps[:, None], epoch)

    writer.add_images('original', _x[:, None], epoch)
