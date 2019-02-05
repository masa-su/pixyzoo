from pixyz.distributions import Normal, Bernoulli
import torch
from torch import nn
from torch.nn import functional as F


# https://github.com/xqding/TD-VAE/blob/master/script/model.py
class DBlock(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        h = torch.tanh(self.fc1(input))
        h = h * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)
        return mu, logsigma


class Filtering(Normal):
    def __init__(self, b_size=50, h_size=50, z_size=8):
        super(Filtering, self).__init__(cond_var=["b_t1"], var=["z_t1"], name='p_b')
        self.dblock = DBlock(b_size, h_size, z_size)
        
    def forward(self, b_t1):
        mu, logsigma = self.dblock(b_t1)
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}

    
class Inference(Normal):
    def __init__(self, b_size=50, h_size=50, z_size=8):
        super(Inference, self).__init__(cond_var=["z_t2", "b_t1", "b_t2"], var=["z_t1"], name='q')
        self.dblock = DBlock(z_size+b_size*2, h_size, z_size)
    
    def forward(self, z_t2, b_t1, b_t2):
        mu, logsigma = self.dblock(torch.cat((z_t2, b_t1, b_t2), dim=1))
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}


class Transition(Normal):
    def __init__(self, h_size=50, z_size=8):
        super(Transition, self).__init__(cond_var=["z_t1"], var=["z_t2"], name='p_t')
        self.dblock = DBlock(z_size, h_size, z_size)
        
    def forward(self, z_t1):
        mu, logsigma = self.dblock(z_t1)
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}

    
class Decoder(Bernoulli):
    def __init__(self, h_size=50, x_size=1*64*64, z_size=8):
        super(Decoder, self).__init__(cond_var=["z_t2"], var=["x_t2"], name='p_d')
        self.fc1 = nn.Linear(z_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, x_size)

    def forward(self, z_t2):
        h = F.relu(self.fc1(z_t2))
        h = F.relu(self.fc2(h))
        logit = torch.sigmoid(self.fc3(h))
        return {"probs": logit}
