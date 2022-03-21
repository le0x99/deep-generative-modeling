import torchvision
import torch.nn.functional as F
from scipy.linalg import sqrtm
import numpy as np
import torch
import torch.nn as nn

class feature_extractor(nn.Module):

    def __init__(self,
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(feature_extractor, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.blocks = nn.ModuleList()
        inception = torchvision.models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        block1 = [
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        block2 = [
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        ]
        self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        block3 = [
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ]
        self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """
        Input tensor of shape Bx3xHxW. Values are expected to be in range (0, 1)
        """

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  

        for idx, block in enumerate(self.blocks):
            x = block(x)


        return x.reshape(x.shape[0], 2048)
    
# FID based on our own implementation using pruned IV3 (see above)

def FID(X1, X2, norm_input, gpu_inference=True):
    torch.cuda.empty_cache()
    if X1.shape[1] == 1:
        X1 = X1.repeat(1, 3, 1, 1)  
        X2 = X2.repeat(1, 3, 1, 1) 
    dev = "cuda:0" if gpu_inference else "cpu"
    fe = feature_extractor(normalize_input=norm_input).to(device=dev)
    X1 = X1.to(device=dev)
    X2 = X2.to(device=dev)
    A1 = fe(X1)
    A2 = fe(X2)

    MU1 = A1.mean(axis=0)
    MU2 = A2.mean(axis=0)

    #D1 = A1.T - A1.T.mean(axis=-1, keepdims=True)
    #COV1 = 1/(2048-1) * D1 @ D1.transpose(-1, -2)

    #D2 = A2.T - A2.T.mean(axis=-1, keepdims=True)
    #COV2 = 1/(2048-1) * D2 @ D2.transpose(-1, -2)

    COV1 = torch.from_numpy(np.cov(A1.cpu().numpy(), rowvar=False)).to(device=dev)
    COV2 = torch.from_numpy(np.cov(A2.cpu().numpy(), rowvar=False)).to(device=dev)

    SSD = ((MU1-MU2)**2.).sum()

    COVMEAN = torch.from_numpy(sqrtm(COV1.cpu().numpy() @ COV2.cpu().numpy()).real).to(device=dev)

    fid = SSD + torch.trace(COV1 + COV2 - 2 * COVMEAN)
    return fid



#FID based on an existing pytorch implementation

from ignite.metrics.gan import FID as pytorch_fid

def FID2(X1, X2, norm_input, gpu_inference=True):
    torch.cuda.empty_cache()
    dev = "cuda:0" if gpu_inference else "cpu"
    if X1.shape[1] == 1:
        X1 = X1.repeat(1, 3, 1, 1)  
        X2 = X2.repeat(1, 3, 1, 1)

    X1 = F.interpolate(X1,
                  size=(299, 299),
                  mode='bilinear',
                  align_corners=False)
    X2 = F.interpolate(X2,
                  size=(299, 299),
                  mode='bilinear',
                  align_corners=False)

    if norm_input:
        X1 = 2 * X1 - 1  
        X2 = 2 * X2 - 1  
    
    m = pytorch_fid(device=dev)
    m.update((X1,X2))
    return m.compute()