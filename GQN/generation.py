import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal
from pixyz.losses import NLL, KullbackLeibler
from conv_lstm import Conv2dLSTMCell

class GenerationCore(nn.Module):
    def __init__(self):
        super(GenerationCore, self).__init__()
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256, 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(7+256+3, 128, kernel_size=5, stride=1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        
    def forward(self, v, r, c_g, h_g, u, z):
        v = self.upsample_v(v.view(-1, 7, 1, 1))
        if r.size(2)!=h_g.size(2):
            r = self.upsample_r(r)
        c_g, h_g =  self.core(torch.cat((v, r, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u
        
        return c_g, h_g, u
    
class Prior(Normal):
    def __init__(self):
        super(Prior, self).__init__(cond_var=["h_g"],var=["z"])
        self.eta_pi = nn.Conv2d(128, 2*3, kernel_size=5, stride=1, padding=2)

    def forward(self, h_g):
        mu, logvar = torch.split(self.eta_pi(h_g), 3, dim=1)
        std = torch.exp(0.5*logvar)
        
        return {"loc": mu ,"scale": std}
    
class Generation(Normal):
    def __init__(self):
        super(Generation, self).__init__(cond_var=["u", "sigma"],var=["x_q"])
        self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, u, sigma):
        mu = self.eta_g(u)
        
        return {"loc": mu, "scale": sigma}
    
