import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from copy import deepcopy
import os 
from tqdm import tqdm
import argparse
import imageio


if not os.path.exists("results"): os.makedirs("results")
    
class Classifier(nn.Module):
    ''' Category classifier for generation guidance. Actually, one can use any logp estimator here. '''
    def __init__(self, 
        n_categories: int = 3):
        super().__init__()
        self.n_categories = n_categories
        self.net = nn.Sequential(
            nn.Linear(2,32),nn.SiLU(),
            nn.Linear(32,32),nn.SiLU(),nn.Linear(32,n_categories))
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.net(torch.cat([x,t],-1))
    def loss(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        logits = self(x, t)
        return F.cross_entropy(logits, y)
    def logp(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        logits = self(x, t)
        logit = torch.gather(logits, dim=1, index=y.unsqueeze(1))
        logp = logit - torch.logsumexp(logits, dim=1, keepdim=True)
        return logp
    def gradients(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        x.requires_grad_()
        logp = self.logp(x, t, y)
        grad = torch.autograd.grad([logp.sum()], [x])[0]
        x.detach()
        return logp, grad
    def update(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        loss = self.loss(x, t, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

class mlp(nn.Module):
    ''' MLP as a score estimator. '''
    def __init__(self):
        super().__init__()
        self.x_layer = nn.Sequential(nn.Linear( 1, 64), nn.LayerNorm(64), nn.SiLU())
        self.n_layer = nn.Sequential(nn.Linear( 1, 64), nn.LayerNorm(64), nn.SiLU())
        self.c_layer = nn.Embedding(3, 64)
        self.mix_layer = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, x, n, c = None, c_mask = None):
        if c is not None:
            if c_mask is None: c = self.c_layer(c)
            else: c = self.c_layer(c)*c_mask
        else: c = 0.
        x, n = self.x_layer(x), self.n_layer(n)
        return self.mix_layer(x+n+c)
    
class ODE():
    def __init__(self, M=1000, device = "cpu"):
        self.device, self.M = device, M
        self.F = mlp().to(device)
        self.F.train()
        self.F_ema = deepcopy(self.F).requires_grad_(False)
        self.F_ema.eval()
        self.optim = torch.optim.Adam(self.F.parameters(), lr=5e-4)
        self.set_N(100)
        
    def ema_update(self, decay=0.99):
        for p, p_ema in zip(self.F.parameters(), self.F_ema.parameters()):
            p_ema.data = decay*p_ema.data + (1-decay)*p.data

    def set_N(self, N):
        self.N = N
        self.N, self.t_s = N, None or self.t_s
        self.sigma_s, self.scale_s = None or self.sigma_s, None or self.scale_s
        self.dot_sigma_s, self.dot_scale_s = None or self.dot_sigma_s, None or self.dot_scale_s
        if self.t_s is not None:
            self.coeff1 = (self.dot_sigma_s/self.sigma_s+self.dot_scale_s/self.scale_s)
            self.coeff2 = self.dot_sigma_s/self.sigma_s*self.scale_s
            
    def c_skip(self, sigma): raise NotImplementedError
    def c_out(self, sigma): raise NotImplementedError
    def c_in(self, sigma): raise NotImplementedError
    def c_noise(self, sigma): raise NotImplementedError
    def loss_weighting(self, sigma): raise NotImplementedError
    def sample_noise_distribution(self, N): raise NotImplementedError
    
    def D(self, x, sigma, condition = None, mask = None, use_ema = False):
        c_skip, c_out, c_in, c_noise = self.c_skip(sigma), self.c_out(sigma), self.c_in(sigma), self.c_noise(sigma)
        F = self.F_ema if use_ema else self.F
        return c_skip*x+c_out*F(c_in*x, c_noise, condition, mask)
    
    def update(self, x, condition = None):
        sigma = self.sample_noise_distribution(x.shape[0])
        eps = torch.randn_like(x) * sigma
        mask = (torch.rand_like(condition.float()) > 0.2)[:,None].float() if condition is not None else None
        loss = (self.loss_weighting(sigma) * (self.D(x + eps, sigma, condition, mask) - x)**2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.ema_update()
        return loss.item()
    
    def sample(self, n_samples, N = None, condition = None, cg = False, cfg = False, classifier = None):
        if N is not None and N != self.N: self.set_N(N)
        x = torch.randn((n_samples,1),device=self.device) * self.sigma_s[0] * self.scale_s[0]
        x_history = np.empty((self.N+1, n_samples))
        x_history[0] = x.cpu().numpy().squeeze()
        if condition is not None: condition = (torch.ones((n_samples,),device=device)*condition).long()
        if cfg: 
            mask = torch.ones((2*n_samples,1),device=device)
            mask[n_samples:] = 0
            repeat = 2
        else: mask, repeat = torch.zeros((n_samples,1),device=device), 1
        logp = None
        for i in range(self.N):
            with torch.no_grad():
                D = self.D(
                    x.repeat(repeat,1)/self.scale_s[i], 
                    torch.ones((repeat*n_samples,1),device=device)*self.sigma_s[i],
                    condition.repeat(repeat) if condition is not None else None, mask, use_ema=True)
                if cfg: D = cfg*D[:n_samples]+(1-cfg)*D[n_samples:]
            if cg and (condition is not None):
                logp, grad = classifier.gradients(
                    x/self.scale_s[i], 
                    torch.ones((n_samples,1),device=device)*self.sigma_s[i], condition)
                D = D + cg*self.scale_s[i]*(self.sigma_s[i]**2)*grad
            delta = self.coeff1[i]*x-self.coeff2[i]*D
            dt = self.t_s[i]-self.t_s[i+1] if i != self.N-1 else self.t_s[i]
            x = x - delta*dt
            x_history[i+1] = x.cpu().numpy().squeeze()
        return x, x_history, logp
    
    def save(self, path): torch.save({'model': self.F.state_dict(), 'model_ema': self.F_ema.state_dict()}, "results/"+path)
    def load(self, path):
        checkpoint = torch.load("results/"+path, map_location=self.device)
        self.F.load_state_dict(checkpoint['model'])
        self.F_ema.load_state_dict(checkpoint['model_ema'])

class VPODE(ODE):
    def __init__(self, 
        beta_min = 0.1, beta_max = 20,
        eps_s = 1e-3, eps_t = 1e-5,
        M = 1000, device="cpu"):
        self.beta_min, self.beta_max = beta_min, beta_max
        self.eps_s, self.eps_t = eps_s, eps_t
        self.beta_d = beta_max - beta_min
        super().__init__(M, device)
        
    def set_N(self, N):
        self.N = N
        self.t_s = torch.arange(N, device=device)/(N-1)*(1e-3-1)+1
        self.sigma_s = ((0.5*self.beta_d*(self.t_s**2)+self.beta_min*self.t_s).exp()-1.).sqrt()
        self.scale_s = 1/(1+self.sigma_s**2).sqrt()
        self.dot_sigma_s = (0.5*(self.sigma_s**2+1)*(self.beta_d*self.t_s+self.beta_min)/self.sigma_s)
        self.dot_scale_s = -self.sigma_s/(1+self.sigma_s**2)**1.5*self.dot_sigma_s
        super().set_N(N)
    
    def c_skip(self, sigma): return torch.ones_like(sigma)
    def c_out(self, sigma): return -sigma
    def c_in(self, sigma): return 1/(1+sigma**2).sqrt()
    def c_noise(self, sigma):
        scale = 1/(1+sigma**2).sqrt()
        t = ((self.beta_min**2-4*self.beta_d*scale.log()).sqrt()-self.beta_min)/self.beta_d
        return (self.M-1)*t
    def loss_weighting(self, sigma): return 1/(sigma**2)
    def sample_noise_distribution(self, N):
        t = torch.rand((N,1),device=self.device)*(1-self.eps_t)+self.eps_t
        return ((0.5*self.beta_d*t**2+self.beta_min*t).exp()-1).sqrt()
    
    def save(self): super().save("vpode.pt")
    def load(self): super().load("vpode.pt")
    
class VEODE(ODE):
    def __init__(self,
        sigma_min = 0.02, sigma_max = 100,
        M = 1000, device = "cpu"):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        super().__init__(M, device)
        
    def set_N(self, N):
        self.N = N
        self.sigma_s = self.sigma_max * (self.sigma_min/self.sigma_max)**(torch.arange(N, device=self.device)/(N-1))
        self.t_s = self.sigma_s**2
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = 1/(2*self.sigma_s)
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        super().set_N(N)
        
    def c_skip(self, sigma): return torch.ones_like(sigma)
    def c_out(self, sigma): return sigma
    def c_in(self, sigma): return torch.ones_like(sigma)
    def c_noise(self, sigma): return (0.5*sigma).log()
    def loss_weighting(self, sigma): return 1/(sigma**2)
    def sample_noise_distribution(self, N):
        log_sigma = torch.rand((N,1),device=self.device)*np.log(self.sigma_max/self.sigma_min)+np.log(self.sigma_min)
        return log_sigma.exp()

    def save(self): super().save("veode.pt")
    def load(self): super().load("veode.pt")

class DDIM(ODE):
    def __init__(self,
        C1 = 0.001, C2 = 0.008, j0 = 8,
        M = 1000, device = "cpu"):
        self.C1, self.C2, self.j0 = C1, C2, j0
        super().__init__(M, device)
        
    def set_N(self, N):
        self.N = N
        self.bar_alpha = torch.sin(torch.arange(self.M+1, device=self.device)/(self.M*(self.C2+1))*np.pi/2)**2
        tmp = torch.max(self.bar_alpha[:-1]/self.bar_alpha[1:], torch.tensor(self.C1, device=self.device))
        self.u = torch.empty_like(self.bar_alpha[:-1])
        for i in range(self.M):
            if i == 0: self.u[-1-i] = (1/tmp[-1-i]-1).sqrt()
            else: self.u[-1-i] = ((self.u[-i]**2+1)/tmp[-1-i]-1).sqrt()
        idx = torch.arange(N, device=self.device)
        self.t_s = self.u[torch.floor(self.j0+(self.M-1-self.j0)/(self.N-1)*idx+0.5).long()]
        self.sigma_s = self.t_s
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        super().set_N(N)
        
    def c_skip(self, sigma): return torch.ones_like(sigma)
    def c_out(self, sigma): return -sigma
    def c_in(self, sigma): return 1/(1+sigma**2).sqrt()
    def c_noise(self, sigma): return sigma
    def loss_weighting(self, sigma): return 1/(sigma**2)
    def sample_noise_distribution(self, N):
        j = torch.randint(0, self.M, (N,1), device=self.device)
        return self.u[j]

    def save(self): super().save("ddim.pt")
    def load(self): super().load("ddim.pt")
    
class EDM(ODE):
    def __init__(self,
        sigma_min = 0.002, sigma_max = 80,
        sigma_data = 0.5, rho = 7,
        p_mean = -1.2, p_std = 1.2,
        M = 1000, device = "cpu"):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.sigma_data, self.rho = sigma_data, rho
        self.p_mean, self.p_std = p_mean, p_std
        super().__init__(M, device)
        
    def set_N(self, N):
        self.N = N
        self.sigma_s = (self.sigma_max**(1/self.rho)+torch.arange(N, device=self.device)/(N-1)*\
            (self.sigma_min**(1/self.rho)-self.sigma_max**(1/self.rho)))**self.rho
        self.t_s = self.sigma_s
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        super().set_N(N)
        
    def c_skip(self, sigma): return self.sigma_data**2/(self.sigma_data**2+sigma**2)
    def c_out(self, sigma): return sigma*self.sigma_data/(self.sigma_data**2+sigma**2).sqrt()
    def c_in(self, sigma): return 1/(self.sigma_data**2+sigma**2).sqrt()
    def c_noise(self, sigma): return 0.25*(sigma).log()
    def loss_weighting(self, sigma): return (self.sigma_data**2+sigma**2)/((sigma*self.sigma_data)**2)
    def sample_noise_distribution(self, N):
        log_sigma = torch.randn((N,1),device=self.device)*self.p_std + self.p_mean
        return log_sigma.exp()

    def save(self): super().save("edm.pt")
    def load(self): super().load("edm.pt")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ode", type=str, default="all") # or "vpode", "veode", "ddim", "edm"
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--phase", type=str, default="train")

    ode_type = parser.parse_args().ode
    device = parser.parse_args().device

    if parser.parse_args().phase == "train":
        
        train_list = [ode_type] if ode_type != "all" else ["vpode", "veode", "ddim", "edm"]

        for each in train_list:
            
            if each == "vpode": ode = VPODE(device=device)
            elif each == "veode": ode = VEODE(device=device)
            elif each == "ddim": ode = DDIM(device=device)
            elif each == "edm": ode = EDM(sigma_data=np.sqrt(2/3), device=device)
            else: raise NotImplementedError
                
            classifier = Classifier(3).to(device)

            print(f'Training {each} on {device} ...')
            loss_buffer, cg_loss_buffer = [], []
            pbar = tqdm(range(100_000))
            for i in pbar:
                x = torch.randint(-1,2,(64,1),device=device).float()
                y = (x+1).long().squeeze()
                idx = torch.randint(ode.N, (64,1), device=device)
                sigma = ode.sigma_s[idx]
                scale = ode.scale_s[idx]
                
                loss = ode.update(x, y)
                loss_buffer.append(loss)
                if i < 20_000:
                    cg_loss = classifier.update(x/scale, sigma, y)
                    cg_loss_buffer.append(cg_loss)
                
                if i % 1000 == 0:
                    pbar.update(1000)
                    if i < 20_000: log = f"loss: {sum(loss_buffer)/len(loss_buffer)} cg_loss: {sum(cg_loss_buffer)/len(cg_loss_buffer)}"
                    else: log = f"loss: {sum(loss_buffer)/len(loss_buffer)}"
                    pbar.set_description(log)
                    loss_buffer, cg_loss_buffer = [], []
            ode.save()
            torch.save(classifier.state_dict(), f"results/{each}_classifier.pt")
            
    elif parser.parse_args().phase == "eval":
    
        odes = [VPODE(device=device), VEODE(device=device), DDIM(device=device), EDM(sigma_data=np.sqrt(2/3),device=device)]
        cfs = [Classifier(3).to(device) for _ in range(4)]
        names = ["vpode", "veode", "ddim", "edm"]
        for ode in odes: ode.load()
        for i, cf in enumerate(cfs): cf.load_state_dict(torch.load(f"results/{names[i]}_classifier.pt", map_location=device))
        
        print("Unconditional Sampling")
        fig, axes = plt.subplots(1,4,figsize=(20,5))
        for k in range(4):
            x, history, logp = odes[k].sample(2000, N=100, condition=None, cg=0., cfg=0.0, classifier=cfs[k])
            img = np.empty((100,history.shape[0]))
            for i in range(history.shape[0]):
                img[:,i] = np.histogram(history[i], bins=100, range=(-3,3))[0]/history.shape[-1]
            axes[k].set_yticks(np.linspace(0,99,7), np.linspace(-3,3,7))
            axes[k].imshow(1-np.exp(-15*img))
            axes[k].set_title(names[k])
            axes[k].set_xlabel("sample steps")
        plt.suptitle("Unconditional Sampling", y=0.95)
        plt.savefig("visualization/unconditional.png")
        
        print("CG Conditional Sampling")
        writer = imageio.get_writer("visualization/cg.gif", mode='I', duration=3)
        cg_schedule = (np.linspace(0.0, 1.0, 90)**5)*5
        for n in tqdm(range(90)):
            fig, axes = plt.subplots(1,4,figsize=(12,3))
            for k in range(4):
                x, history, logp = odes[k].sample(2000, N=100, condition=0, cg=cg_schedule[n], cfg=0.0, classifier=cfs[k])
                img = np.empty((100,history.shape[0]))
                for i in range(history.shape[0]):
                    img[:,i] = np.histogram(history[i], bins=100, range=(-3,3))[0]/history.shape[-1]
                axes[k].set_yticks(np.linspace(0,99,7), np.linspace(-3,3,7))
                axes[k].imshow(1-np.exp(-15*img))
                axes[k].set_title(names[k])
                axes[k].set_xlabel("sample steps")
            plt.suptitle("Classifier guidance", y=0.95)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.tostring_rgb()
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            writer.append_data(frame)
            plt.close()
        writer.close()
        
        print("CFG Conditional Sampling")
        writer = imageio.get_writer("visualization/cfg.gif", mode='I', duration=3)
        cfg_schedule = (np.linspace(0.0, 1.0, 90)**5)*5
        for n in tqdm(range(90)):
            fig, axes = plt.subplots(1,4,figsize=(12,3))
            for k in range(4):
                x, history, logp = odes[k].sample(2000, N=100, condition=0, cg=0.0, cfg=cfg_schedule[n], classifier=cfs[k])
                img = np.empty((100,history.shape[0]))
                for i in range(history.shape[0]):
                    img[:,i] = np.histogram(history[i], bins=100, range=(-3,3))[0]/history.shape[-1]
                axes[k].set_yticks(np.linspace(0,99,7), np.linspace(-3,3,7))
                axes[k].imshow(1-np.exp(-15*img))
                axes[k].set_title(names[k])
                axes[k].set_xlabel("sample steps")
            plt.suptitle("Classifier-free guidance", y=0.95)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.tostring_rgb()
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            writer.append_data(frame)
            plt.close()
        writer.close()

        print("NFE")
        nfe_schedule = np.linspace(200, 2, 100).astype(np.int32)
        fig, axes = plt.subplots(1,4,figsize=(12,3))
        for k in range(4):
            img = np.empty((100,100))
            for i in tqdm(range(100)):
                x, history, logp = odes[k].sample(2000, N=nfe_schedule[i], condition=None, cg=0.0, cfg=0.0, classifier=cfs[k])
                img[:,i] = np.histogram(x.cpu().numpy(), bins=100, range=(-3,3))[0]/history.shape[-1]
            axes[k].set_yticks(np.linspace(0,99,7), np.linspace(-3,3,7))
            axes[k].imshow(1-np.exp(-15*img))
            axes[k].set_title(names[k])
            axes[k].set_xlabel("NFE")
        plt.suptitle("NFE", y=0.95)
        plt.savefig("visualization/nfe.png")
