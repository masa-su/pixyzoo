
# coding: utf-8

# # VRNN
# Original paper: A Recurrent Latent Variable Model for Sequential Data (https://arxiv.org/pdf/1506.02216.pdf )

# In[1]:


from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter


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
h_dim = 100
z_dim = 16
t_max = x_dim
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
seed = args.seed
torch.manual_seed(seed)

# In[2]:


# toy dataset (MNIST)
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


# In[3]:


from pixyz.models import Model
from pixyz.losses import KullbackLeibler, StochasticReconstructionLoss
from pixyz.losses import IterativeLoss
from pixyz.distributions import Bernoulli, Normal, Deterministic
from pixyz.utils import print_latex


# In[4]:

class Phi_x(nn.Module):
    def __init__(self):
        super(Phi_x, self).__init__()
        self.fc0 = nn.Linear(x_dim, h_dim)

    def forward(self, x):
        return F.relu(self.fc0(x))

class Phi_z(nn.Module):
    def __init__(self):
        super(Phi_z, self).__init__()
        self.fc0 = nn.Linear(z_dim, h_dim)

    def forward(self, z):
        return F.relu(self.fc0(z))

f_phi_x = Phi_x().to(device)
f_phi_z = Phi_z().to(device)


# In[5]:


class Generator(Bernoulli):
    def __init__(self):
        super(Generator, self).__init__(cond_var=["z", "h_prev"], var=["x"])
        self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)
        self.f_phi_z = f_phi_z

    def forward(self, z, h_prev):
        h = torch.cat((self.f_phi_z(z), h_prev), dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}

class Prior(Normal):
    def __init__(self):
        super(Prior, self).__init__(cond_var=["h_prev"], var=["z"])
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, h_prev):
        h = F.relu(self.fc1(h_prev))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

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

class Recurrence(Deterministic):
    def __init__(self):
        super(Recurrence, self).__init__(cond_var=["x", "z", "h_prev"], var=["h"])
        self.rnncell = nn.GRUCell(h_dim * 2, h_dim).to(device)
        self.f_phi_x = f_phi_x
        self.f_phi_z = f_phi_z
        self.hidden_size = self.rnncell.hidden_size

    def forward(self, x, z, h_prev):
        h_next = self.rnncell(torch.cat((self.f_phi_z(z), self.f_phi_x(x)), dim=-1), h_prev)
        return {"h": h_next}

prior = Prior().to(device)
decoder = Generator().to(device)
encoder = Inference().to(device)
recurrence = Recurrence().to(device)


# In[6]:


encoder_with_recurrence = encoder * recurrence
generate_from_prior = prior * decoder * recurrence


# In[7]:


print_latex(encoder_with_recurrence)


# In[8]:


print_latex(generate_from_prior)


# In[9]:


reconst = StochasticReconstructionLoss(encoder_with_recurrence, decoder)
kl = KullbackLeibler(encoder, prior)

step_loss = (reconst + kl).mean()
loss = IterativeLoss(step_loss, max_iter=t_max,
                     series_var=['x'],
                     update_value={"h": "h_prev"})

vrnn = Model(loss, distributions=[encoder, decoder, prior, recurrence],
             optimizer=optim.Adam, optimizer_params={'lr': lr})

print(vrnn)
print_latex(vrnn)


# In[8]:


def data_loop(epoch, loader, model, device, train_mode=False):
    mean_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        batch_size = data.size()[0]
        x = data.transpose(0, 1)
        h_prev = torch.zeros(batch_size, recurrence.hidden_size).to(device)
        if train_mode:
            mean_loss += model.train({'x': x, 'h_prev': h_prev}).item() * batch_size
        else:
            mean_loss += model.test({'x': x, 'h_prev': h_prev}).item() * batch_size

    mean_loss /= len(loader.dataset)
    if train_mode:
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, mean_loss))
    else:
        print('Test loss: {:.4f}'.format(mean_loss))
    return mean_loss


# In[9]:


def plot_image_from_latent(batch_size):
    x = []
    h_prev = torch.zeros(batch_size, recurrence.hidden_size).to(device)
    for step in range(t_max):
        samples_z = prior.sample({'h_prev': h_prev})['z']

        decoded_x = decoder(samples_z, h_prev)['probs']
        h = recurrence(decoded_x, samples_z, h_prev)['h']
        
        # hの更新にx_t使った
        h_prev = h
        x.append(decoded_x[None, :])
    x = torch.cat(x, dim=0).transpose(0, 1)
    return x


# In[10]:


writer = SummaryWriter(comment='Pix_LR_{}_SEED_{}_bsize_{}'.format(lr, seed, batch_size))

for epoch in range(1, epochs + 1):
    train_loss = data_loop(epoch, train_loader, vrnn, device, train_mode=True)
    test_loss = data_loop(epoch, test_loader, vrnn, device)

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)

    sample = plot_image_from_latent(batch_size)[:, None]
    writer.add_images('Image_from_latent', sample, epoch)

