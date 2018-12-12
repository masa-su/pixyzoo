import torch
from torch import nn
from torch.nn import functional as F

from pixyz.distributions import Normal
from pixyz.losses import NLL, KullbackLeibler

from conv_lstm import Conv2dLSTMCell

class InferenceCore(nn.Module):
    def __init__(self):
        super(InferenceCore, self).__init__()
        self.downsample_x = nn.Conv2d(3, 3, kernel_size=4, stride=4, padding=0, bias=False)
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256, 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.core = Conv2dLSTMCell(3+7+256+2*128, 128, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x, v, r, c_e, h_e, h_g, u):
        x = self.downsample_x(x)
        v = self.upsample_v(v.view(-1, 7, 1, 1))
        if r.size(2)!=h_e.size(2):
            r = self.upsample_r(r)
        u = self.downsample_u(u)
        c_e, h_e = self.core(torch.cat((x, v, r, h_g, u), dim=1), (c_e, h_e))
        
        return c_e, h_e
    
class Inference(Normal):
    def __init__(self):
        super(Inference, self).__init__(cond_var=["h_e"],var=["z"])
        self.eta_e = nn.Conv2d(128, 2*3, kernel_size=5, stride=1, padding=2)
        
    def forward(self, h_e):
        mu, logvar = torch.split(self.eta_e(h_e), 3, dim=1)
        std = torch.exp(0.5*logvar)
        
        return {"loc": mu, "scale": std}
    