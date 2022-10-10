import torch.nn as nn 
import torch.nn.functional as F
import torch
import time
import numpy as np


class ResBlock(nn.Module):
    def __init__(self) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.leakyRelu = nn.LeakyReLU(0.02)
        
    def forward(self, x) -> torch.Tensor:
        out = x
        out = self.leakyRelu(self.conv1(out))
        out = self.leakyRelu(self.conv2(out))
        out = torch.add(out, x)
        return out

class ResFCBlock(nn.Module):
    def __init__(self, num_units) -> None:
        super(ResFCBlock, self).__init__()
        self.layer1 = nn.Linear(num_units, num_units * 2)
        self.bn1 = nn.BatchNorm1d(1)
        self.leakyRelu = nn.LeakyReLU(0.2)
        
    def forward(self, x) -> torch.Tensor:
        out = self.leakyRelu(self.bn1(self.layer1(x)))
        return out
        

class HandWriteGenerator(nn.Module):
    def __init__(self, num_blocks, num_units) -> None:
        super(HandWriteGenerator, self).__init__()
        self.num_units = num_units
        self.layer1 = nn.Sequential(
            nn.Linear(10, num_units),
            nn.LeakyReLU(0.2)
            )
        
        trunk = []
        for _ in range(num_blocks):
            trunk.append(ResFCBlock(self.num_units))
            self.num_units *= 2
        self.trunk = nn.Sequential(*trunk)  
        
        self.layer2 = nn.Sequential(
            nn.Linear(self.num_units, 28*28),
            nn.Tanh()
            )
        
    def forward(self, x) -> torch.Tensor:
        out = self.layer1(x)
        out = self.trunk(out)
        out = self.layer2(out)
        out = torch.reshape(out, (-1, 1, 28, 28))
        return out

class HandWriteDiscriminator(nn.Module):
    def __init__(self, num_blocks):
        super(HandWriteDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.LeakyReLU(0.2)
            )
        
        trunk = []
        for _ in range(num_blocks):
            trunk.append(ResBlock())
        self.trunk = nn.Sequential(*trunk)  
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)
        return out

if __name__ == "__main__":
    # model = HandWriteDiscriminator(5)
    # x = torch.tensor(torch.randn((16, 1, 28, 28)))
    
    model = HandWriteGenerator(5, 64)
    x = torch.tensor(torch.randn((16, 1, 100)))
    start = time.time()
    output = model(x)
    # output1 = model1(x)
    end = time.time()
    print(end - start)
    print(output.shape)
    # print(output1.shape)

    # class_num = 10
    # input = torch.tensor(np.arange(10).reshape((10, -1)))
    # output = torch.zeros(10, 10).scatter_(1, input, 1)
    # print(input.shape)
    # print(output)
    # print(output.eval())
