
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
h_dim = 32
hidden_dim = 32
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


# In[5]:

from pixyz.models import Model
from pixyz.losses import KullbackLeibler, CrossEntropy, IterativeLoss
from pixyz.distributions import Bernoulli, Normal, Deterministic
from pixyz.utils import print_latex


# In[6]:


# In[7]:

class RNN(Deterministic):
    def __init__(self):
        super(RNN, self).__init__(cond_var=["x"], var=["h"])
        self.rnn = nn.GRU(x_dim, h_dim, bidirectional=True)
#         self.h0 = torch.zeros(2, batch_size, self.rnn.hidden_size).to(device)
        self.h0 = nn.Parameter(torch.zeros(2, 1, self.rnn.hidden_size))
        self.hidden_size = self.rnn.hidden_size
        
    def forward(self, x):
        h0 = self.h0.expand(2, x.size(1), self.rnn.hidden_size).contiguous()
        h, _ = self.rnn(x, h0)
        return {"h": h}


# In[8]:

class Generator(Bernoulli):
    def __init__(self):
        super(Generator, self).__init__(cond_var=["z"], var=["x"])
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, x_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        return {"probs": torch.sigmoid(self.fc2(h))}


# In[9]:

class Inference(Normal):
    def __init__(self):
        super(Inference, self).__init__(cond_var=["h", "z_prev"], var=["z"])
        self.fc1 = nn.Linear(z_dim, h_dim*2)
        self.fc21 = nn.Linear(h_dim*2, z_dim)
        self.fc22 = nn.Linear(h_dim*2, z_dim)
        
    def forward(self, h, z_prev):
        h_z = torch.tanh(self.fc1(z_prev))
        h = 0.5 * (h + h_z)
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


# In[10]:

class Prior(Normal):
    def __init__(self):
        super(Prior, self).__init__(cond_var=["z_prev"], var=["z"])
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        
    def forward(self, z_prev):
        h = F.relu(self.fc1(z_prev))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


# In[11]:

prior = Prior().to(device)
encoder = Inference().to(device)
decoder = Generator().to(device)
rnn = RNN().to(device)


# In[12]:

print(prior)
print("*"*80)
print(encoder)
print("*"*80)
print(decoder)
print("*"*80)
print(rnn)


# In[13]:

generate_from_prior = prior * decoder
print(generate_from_prior)
print_latex(generate_from_prior)


# In[14]:

step_loss = CrossEntropy(encoder, decoder) + KullbackLeibler(encoder, prior)
_loss = IterativeLoss(step_loss, max_iter=t_max, 
                      series_var=["x", "h"], update_value={"z": "z_prev"})
loss = _loss.expectation(rnn).mean()


# In[15]:

dmm = Model(loss, distributions=[rnn, encoder, decoder, prior], 
            optimizer=optim.RMSprop, optimizer_params={"lr": 5e-4}, clip_grad_value=10)


# In[16]:

print(dmm)
print_latex(dmm)


# In[17]:

def data_loop(epoch, loader, model, device, train_mode=False):
    mean_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        batch_size = data.size()[0]
        x = data.transpose(0, 1)
        z_prev = torch.zeros(batch_size, z_dim).to(device)
        if train_mode:
            mean_loss += model.train({'x': x, 'z_prev': z_prev}).item() * batch_size
        else:
            mean_loss += model.test({'x': x, 'z_prev': z_prev}).item() * batch_size
    mean_loss /= len(loader.dataset)
    if train_mode:
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, mean_loss))
    else:
        print('Test loss: {:.4f}'.format(mean_loss))
    return mean_loss


# In[18]:

def plot_image_from_latent(batch_size):
    x = []
    z_prev = torch.zeros(batch_size, z_dim).to(device)
    for step in range(t_max):
        samples = generate_from_prior.sample({'z_prev': z_prev})
        x_t = decoder.sample_mean({"z": samples["z"]})
        z_prev = samples["z"]
        x.append(x_t[None, :])
    x = torch.cat(x, dim=0).transpose(0, 1)
    return x


# In[19]:

writer = SummaryWriter(comment='Pix_LR_{}_SEED_{}_bsize_{}'.format(lr, seed, batch_size))

for epoch in range(1, epochs + 1):
    train_loss = data_loop(epoch, train_loader, dmm, device, train_mode=True)
    test_loss = data_loop(epoch, test_loader, dmm, device)

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)

    sample = plot_image_from_latent(batch_size)[:, None]
    writer.add_images('Image_from_latent', sample, epoch)


# In[ ]:




