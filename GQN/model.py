import torch
from torch import nn
from torch.nn import functional as F
from pixyz.distributions import Normal
from representation import Pyramid, Tower, Pool
from inference import InferenceCore, Inference
from generation import GenerationCore, Prior, Generation
from pixyz.losses import KullbackLeibler

class GQN(nn.Module):
    def __init__(self, representation="pool", L=12, shared_core=False):
        super(GQN, self).__init__()
        
        # Number of generative layers
        self.L = L
        
        self.shared_core = shared_core
        
        # Representation network
        self.representation = representation
        if representation=="pyramid":
            self.phi = Pyramid()
        elif representation=="tower":
            self.phi = Tower()
        elif representation=="pool":
            self.phi = Pool()
            
        # Generation network
        if shared_core:
            self.inference_core = InferenceCore()
            self.generation_core = GenerationCore()
        else:
            self.inference_core = nn.ModuleList([InferenceCore() for _ in range(L)])
            self.generation_core = nn.ModuleList([GenerationCore() for _ in range(L)])
        
        # Distribution
        self.pi = Prior()
        self.q = Inference()
        self.g = Generation()

    # EstimateELBO
    def forward(self, x, v, v_q, x_q, sigma):
        B, M, *_ = x.size()
        
        # Scene encoder
        if self.representation=="tower":
            r = x.new_zero((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        elbo = 0
        for l in range(self.L):
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior sample
            z = self.q.sample({"h_e": h_e}, reparam=True)["z"]
            
            # ELBO KL contribution update
            elbo -= KullbackLeibler(self.q, self.pi).eval({"h_e": h_e, "h_g": h_g})
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
                
        # ELBO likelihood contribution update
        elbo += self.g.log_prob().eval({"u":u, "sigma":sigma, "x_q": x_q})

        return elbo
    
    def generate(self, x, v, v_q):
        B, M, *_ = x.size()
        
        # Scene encoder
        if self.representation=="tower":
            r = x.new_zero((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k

        # Initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))
        
        for l in range(self.L):
            # Prior sample
            z = self.pi.sample({"h_g": h_g})["z"]
            
            # State update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
            
        x_q_hat = self.g.sample_mean({"u": u, "sigma": 0})

        return torch.clamp(x_q_hat, 0, 1)
    
    def kl_divergence(self, x, v, v_q, x_q):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation=="tower":
            r = x.new_zero((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        kl = 0
        for l in range(self.L):
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior sample
            z = self.q.sample({"h_e": h_e}, reparam=True)["z"]
            
            # KL divergence
            kl += KullbackLeibler(self.q, self.pi).eval({"h_e": h_e, "h_g": h_g})
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

        return kl
    
    def reconstruct(self, x, v, v_q, x_q):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation=="tower":
            r = x.new_zero((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        for l in range(self.L):
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior sample
            z = self.q.sample({"h_e": h_e}, reparam=True)["z"]
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
                
        x_q_rec = self.g.sample_mean({"u": u, "sigma": 0})

        return torch.clamp(x_q_rec, 0, 1)
    
