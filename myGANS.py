import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNDiscriminator(nn.Module):
    def __init__(self,input_shape):
        assert len(input_shape) == 4
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[1], 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * int(input_shape[-1]/2/2/2)**2 , 1)
        self.probit = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.probit(x)
        return x
    
class Discriminator(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        assert len(input_shape) == 2
        n_features = input_shape[-1]
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    

    
class Generator(torch.nn.Module):
    def __init__(self, dim, input_shape):
        super(Generator, self).__init__()
        n_features = dim
        assert len(input_shape) == 2
        n_out = input_shape[-1]
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
    
class Discriminator2(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        assert len(input_shape) == 2
        n_features = input_shape[-1]
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 400),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(400, 300),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(300, 200),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(200, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    

    
class Generator2(torch.nn.Module):
    def __init__(self, dim, input_shape):
        super().__init__()
        n_features = dim
        assert len(input_shape) == 2
        n_out = input_shape[-1]
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 200),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(200, 300),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(300, 400),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(400, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x