import torch
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms, datasets
import torchvision
import numpy as np
import matplotlib.pyplot as plt



def imshow(img):
    img = torchvision.utils.make_grid(img.cpu())
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation="bilinear")
    plt.show()


def svhn_data(test=False):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    out_dir = '{}/dataset'.format("./SVHN")
    if test:
        return datasets.SVHN(root=out_dir,split="test",
                                          transform=compose,
                                          download=True)
    else:
        return datasets.SVHN(root=out_dir,split="train",
                                          transform=compose,
                                          download=True)
        
    
def mnist_data(test=False):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    out_dir = '{}/dataset'.format("./MNIST")
    if test:
        return datasets.MNIST(root=out_dir, train=False, transform=compose, download=True)
    else:
        return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def load_data(name, test=False, val_frac = .2):
    data = mnist_data(test) if name == "mnist" else svhn_data(test)
    X_ = torch.stack([data.__getitem__(i)[0] for i in range(len(data))])
    if test:
        return X_
    else:
        val_i = np.random.choice(list(range(X_.shape[0])), int(X_.shape[0] * val_frac) )
        train_i = np.array([_ for _ in list(range(X_.shape[0])) if _ not in val_i ])
        val = X_[val_i]
        train = X_[train_i]
        return train, val

def n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp