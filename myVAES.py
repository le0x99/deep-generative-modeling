import torch.nn as nn
import torch
import numpy as np

class Decoder(nn.Module):
    def __init__(self, M, H, D):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.dec1 = nn.Linear(in_features=self.M, out_features=self.H)
        self.dec2 = nn.Linear(in_features=self.H, out_features=self.H)
        self.dec3 = nn.Linear(in_features=self.H, out_features=self.D)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        #self.log_scale = nn.Parameter(torch.zeros(self.D))
    def forward(self, Z):
        Z = self.dec1(Z)
        Z = nn.functional.relu(Z)
        Z = self.dec2(Z)
        Z = nn.functional.relu(Z)
        mu = self.dec3(Z)
        mu = nn.functional.tanh(mu)
        std = torch.exp(self.log_scale)
        return mu, std
    
class Decoder2(nn.Module):
    def __init__(self, M, H, D):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.dec1 = nn.Linear(in_features=self.M, out_features=self.H*2)
        self.dec2 = nn.Linear(in_features=self.H*2, out_features=self.H)
        self.dec3 = nn.Linear(in_features=self.H, out_features=self.D)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        #self.log_scale = nn.Parameter(torch.zeros(self.D))
    def forward(self, Z):
        Z = self.dec1(Z)
        Z = nn.functional.relu(Z)
        Z = self.dec2(Z)
        Z = nn.functional.relu(Z)
        mu = self.dec3(Z)
        mu = nn.functional.tanh(mu)
        std = torch.exp(self.log_scale)
        return mu, std
    
class Encoder2(nn.Module):
    def __init__(self, D, H, M):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.enc1 = nn.Linear(in_features=self.D, out_features=self.H*2)
        self.enc2 = nn.Linear(in_features=self.H*2, out_features=self.H)
        self.enc3 = nn.Linear(in_features=self.H, out_features=self.M*2)
        
    def forward(self, x):
        x = self.enc1(x)
        x = nn.functional.relu(x)
        x = self.enc2(x)
        x = nn.functional.relu(x)
        x = self.enc3(x)
        x = x.view(-1, 2, self.M)
        # location
        mu = x[:, 0, :]
        # scale
        log_var = x[:, 1, :]
        std = torch.exp(log_var / 2)
        return mu, std  
    
class Encoder(nn.Module):
    def __init__(self, D, H, M):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.enc1 = nn.Linear(in_features=self.D, out_features=self.H)
        self.enc2 = nn.Linear(in_features=self.H, out_features=self.H)
        self.enc3 = nn.Linear(in_features=self.H, out_features=self.M*2)
        
    def forward(self, x):
        x = self.enc1(x)
        x = nn.functional.relu(x)
        x = self.enc2(x)
        x = nn.functional.relu(x)
        x = self.enc3(x)
        x = x.view(-1, 2, self.M)
        # location
        mu = x[:, 0, :]
        # scale
        log_var = x[:, 1, :]
        std = torch.exp(log_var / 2)
        return mu, std 

class EncoderNO2(nn.Module):
    def __init__(self, D, H, M):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.enc1 = nn.Linear(in_features=self.D, out_features=self.H)
        self.enc2 = nn.Linear(in_features=self.H, out_features=self.M*2)
        # Dist params
        self.loc_param = nn.Linear(in_features=self.M, out_features=self.M)
        self.scale_param = nn.Linear(in_features=self.M, out_features=self.M)
    def forward(self, x):
        x = self.enc1(x)
        x = nn.functional.relu(x)
        x = self.enc2(x)
        x = x.view(-1, 2, self.M)
        # location
        mu = self.loc_param(x[:, 0, :])
        # scale
        log_var = self.scale_param(x[:, 1, :])
        
        std = torch.exp(log_var / 2)
        return mu, std 
class CNNEncoder(nn.Module):
    def __init__(self, input_shape, M):
        super().__init__()
        pass
        self.D = D
        self.M = M
        self.H = H
        self.conv1 = nn.Conv2d(input_shape[1], 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * int(input_shape[-1]/2/2/2)**2 , 1)
        self.probit = nn.Sigmoid()
        self.enc1 = nn.Linear(in_features=self.D, out_features=self.H)
        self.enc2 = nn.Linear(in_features=self.H, out_features=self.M*2)
    def forward(self, x):
        x = self.enc1(x)
        x = nn.functional.relu(x)
        x = self.enc2(x)
        x = x.view(-1, 2, self.M)
        # location
        mu = x[:, 0, :]
        # scale
        log_var = x[:, 1, :]
        std = torch.exp(log_var / 2)
        return mu, std
    
class Decoder3(nn.Module):
    def __init__(self, M, H, D):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.dec1 = nn.Linear(in_features=self.M, out_features=self.H)
        self.dec2 = nn.Linear(in_features=self.H, out_features=self.H*2)
        self.dec3 = nn.Linear(in_features=self.H*2, out_features=self.H*3)
        self.dec4 = nn.Linear(in_features=self.H*3, out_features=self.D)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        #self.log_scale = nn.Parameter(torch.zeros(self.D))
    def forward(self, Z):
        Z = self.dec1(Z)
        Z = nn.functional.relu(Z)
        Z = self.dec2(Z)
        Z = nn.functional.relu(Z)
        Z = self.dec3(Z)
        Z = nn.functional.relu(Z)
        mu = self.dec4(Z)
        std = torch.exp(self.log_scale)
        return mu, std
    
class Encoder3(nn.Module):
    def __init__(self, D, H, M):
        super().__init__()
        self.D = D
        self.M = M
        self.H = H
        self.enc1 = nn.Linear(in_features=self.D, out_features=self.H*2)
        self.enc2 = nn.Linear(in_features=self.H*2, out_features=self.H)
        self.enc3 = nn.Linear(in_features=self.H, out_features=self.M*2)
        
    def forward(self, x):
        x = self.enc1(x)
        x = nn.functional.relu(x)
        x = self.enc2(x)
        x = nn.functional.relu(x)
        x = self.enc3(x)
        x = x.view(-1, 2, self.M)
        # location
        mu = x[:, 0, :]
        # scale
        log_var = x[:, 1, :]
        std = torch.exp(log_var / 2)
        return mu, std  
    
## The following code is adopted from 
## Source : https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
## All credit goes to Jakub Tomczak (JT) 
    
PI = torch.from_numpy(np.asarray(np.pi))

def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

class FlowPrior(nn.Module):
    def __init__(self, nets, nett, num_flows, D):
        super(FlowPrior, self).__init__()

        self.D = D
        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)
        s = self.s[index](xa)
        t = self.t[index](xa)

        if forward:
            #yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            #xb = f(y)
            yb = torch.exp(s) * xb + t

        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)

        return z, log_det_J

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)

        return x

    def sample(self, batch_size):
        z = torch.randn(batch_size, self.D).cuda()
        x = self.f_inv(z)
        return x.view(-1, self.D)

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        log_p = (log_standard_normal(z) + log_det_J.unsqueeze(1))
        #log_p = (torch.distributions.Normal(0,1).log_prob(z).sum() + log_det_J.unsqueeze(1))
        return log_p
