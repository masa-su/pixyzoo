
# coding: utf-8

# # Deep Markov Model

# In[3]:


from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter


# In[4]:


batch_size = 128
epochs = 5
seed = 1
torch.manual_seed(seed)


# In[5]:


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# In[6]:


# Experiment Setting
# generate MNIST by stacking row images(consider row as time step)
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


# In[7]:


from pixyz.models import Model
from pixyz.losses import KullbackLeibler, CrossEntropy, IterativeLoss
from pixyz.distributions import Bernoulli, Normal, Deterministic
from pixyz.utils import print_latex


# In[8]:


class RNN(Deterministic):
    '''
    push the observed x through the rnn
    rnn output contains the hidden state at each time step
    '''
    def __init__(self, x_dim, rnn_dim):
        super(RNN, self).__init__(cond_var=["x"], var=["h"])
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


# In[9]:


class Generator(Bernoulli):
    '''
    Emitter
    Parameterizes the bernoulli observation likelihood p(x_t | z_t)
    '''
    def __init__(self, z_dim, hidden_dim):
        super(Generator, self).__init__(cond_var=["z"], var=["x"])
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


# In[10]:


class Inference(Normal):
    '''
    Combiner
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of te guide(i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN
    '''
    def __init__(self, z_dim, rnn_dim):
        super(Inference, self).__init__(cond_var=["h", "z_prev"], var=["z"])
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


# In[11]:


class Prior(Normal):
    '''
    GatedTranstion
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    '''
    def __init__(self, z_dim, transition_dim):
        super(Prior, self).__init__(cond_var=["z_prev"], var=["z"])
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


# In[1]:


x_dim = 28
hidden_dim = 32
rnn_dim = hidden_dim * 2
transition_dim = 32
z_dim = 16
t_max = x_dim


# In[14]:


prior = Prior(z_dim=z_dim, transition_dim=transition_dim).to(device)# p(z_t| z_{t-1})
encoder = Inference(z_dim=z_dim, rnn_dim=rnn_dim).to(device)#q(z_t | z_{t-1}, x_{t:T})
decoder = Generator(z_dim=z_dim, hidden_dim=hidden_dim).to(device)# p(x_t|z_t)
rnn = RNN(x_dim=x_dim, rnn_dim=rnn_dim).to(device)


# In[15]:


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


writer = SummaryWriter()

for epoch in range(1, epochs + 1):
    train_loss = data_loop(epoch, train_loader, dmm, device, train_mode=True)
    test_loss = data_loop(epoch, test_loader, dmm, device)

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)

    sample = plot_image_from_latent(batch_size)[:, None][1,:]
    writer.add_image('Image_from_latent', sample, epoch)

