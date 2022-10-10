# Copyright 2022 Hanyu Feng

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @File    :   models.py
# @Time    :   2022/10/10 12:37:31
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description : build models for generation and discrimation


import time
import torch
import numpy as np
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1),
                               padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1),
                               padding='same')
        self.leakyRelu = nn.LeakyReLU(0.02)

    def forward(self, x):
        out = x
        out = self.leakyRelu(self.conv1(out))
        out = self.leakyRelu(self.conv2(out))
        out = torch.add(out, x)
        return out


class FCBlock(nn.Module):

    def __init__(self, num_units):
        super(FCBlock, self).__init__()
        self.layer1 = nn.Linear(num_units, num_units * 2)
        self.bn1 = nn.BatchNorm1d(1)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.leakyRelu(self.bn1(self.layer1(x)))
        return out


class HandWriteGenerator(nn.Module):

    def __init__(self, num_blocks, num_units):
        super(HandWriteGenerator, self).__init__()
        self.num_units = num_units
        self.in_layer = nn.Sequential(
                            nn.Linear(10, num_units),
                            nn.LeakyReLU(0.2)
                            )

        trunk = []
        for _ in range(num_blocks):
            trunk.append(FCBlock(self.num_units))
            self.num_units *= 2
        self.trunk = nn.Sequential(*trunk)

        self.out_layer = nn.Sequential(
                            nn.Linear(self.num_units, 28*28),
                            nn.Tanh()
                            )

    def forward(self, x):
        out = self.in_layer(x)
        out = self.trunk(out)
        out = self.out_layer(out)
        out = torch.reshape(out, (-1, 1, 28, 28))
        return out


class HandWriteDiscriminator(nn.Module):

    def __init__(self, num_blocks):
        super(HandWriteDiscriminator, self).__init__()
        self.in_conv_layer = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=(3, 3),
                                  stride=(1, 1), padding='same'),
                        nn.LeakyReLU(0.2)
                        )

        trunk = []
        for _ in range(num_blocks):
            trunk.append(ResBlock())
        self.trunk = nn.Sequential(*trunk)

        self.out_conv_layer = nn.Sequential(
                        nn.Conv2d(32, 1, kernel_size=(3, 3),
                                  stride=(1, 1), padding='same'),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Sigmoid()
                        )

    def forward(self, x):
        out = self.in_conv_layer(x)
        out = self.trunk(out)
        out = self.out_conv_layer(out)
        return out


def main():
    print("[*] Check generator")
    model = HandWriteGenerator(5, 64)
    x = torch.randn((16, 1, 10))
    start = time.time()
    output = model(x)
    end = time.time()
    print("[*]", end - start)
    print("[*]", output.shape, "\n")

    print("[*] Check discriminator")
    model = HandWriteDiscriminator(5)
    x = torch.randn((16, 1, 28, 28))
    start = time.time()
    output = model(x)
    end = time.time()
    print("[*]", end - start)
    print("[*]", output.shape)


if __name__ == "__main__":
    main()
