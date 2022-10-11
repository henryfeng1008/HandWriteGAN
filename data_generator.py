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

# @File    :   dataGenerator.py
# @Time    :   2022/10/10 12:37:51
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description : Train data gerenation


import os
import sys
import yaml
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import utils

TYPE_IMG = 0
TYPE_VECTOR = 1


def get_data_as_loader(config):
    original_data_path = config['original_data_path']
    train_dataset = datasets.MNIST(root=original_data_path,
                                train=True,
                                download=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor()
                                     ]
                                    )
                                )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=5)

    test_dataset = datasets.MNIST(root=original_data_path,
                                train=False,
                                download=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor()
                                     ]
                                    )
                                )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=5)
    return train_loader, test_loader


def generate_train_pair(data_loaders, config):
    train_data_dir = config['train_data_dir']
    train_data_name = config['train_data_name']
    utils.ensure_dir(train_data_dir)
    train_data_path = os.path.join(train_data_dir, train_data_name)

    input_data = []
    target_data = []

    multiplier = config['multiplier']
    print("[*] Multiplier", multiplier)
    input_type = config['input_type']
    vector_input_dim = config["vector_input_dim"]
    img_input_dim_row = config["img_input_dim_row"]
    img_input_dim_col = config["img_input_dim_col"]
    target_dim = config["target_dim"]

    for loader_idx , data_loader in enumerate(data_loaders):
        print("[*] Dataloader ", loader_idx)
        length = len(data_loader)
        for data_idx, content in enumerate(data_loader):
            original_image, original_digit = content

            digit = original_digit.numpy()[0]
            target_image = original_image.numpy()
            target_image = np.clip(target_image * 2 - 1, -1, 1)
            target_image = target_image.reshape(-1, target_dim, target_dim)

            noise_input = None
            for _ in range(multiplier):
                if input_type == TYPE_IMG:
                    input_shape = (1, img_input_dim_row, img_input_dim_col)
                    noise_input = np.random.normal(digit * 0.1, 1, input_shape)
                    noise_input = np.clip(noise_input, -1, 1)
                elif input_type == TYPE_VECTOR:
                    noise_input = -1 * np.ones((1, vector_input_dim))
                    noise_input[0, digit] = np.random.uniform(0, 1)
                else:
                    f.close()
                    return
                input_data.append(noise_input)
                target_data.append(target_image)

            percent = int(data_idx / length * 100)
            print("\r", end="")
            print("[*] 进度: {}%: ".format(percent),
                  "*" * (percent // 2), end="")
            sys.stdout.flush()
        print()

    f = h5py.File(train_data_path, 'w')
    f['input_data'] = input_data
    f['target_data'] = target_data
    f.close()


def main():
    print("[*] Start!")
    config_path = "./config.yaml"
    config = utils.load_config(config_path)
    data_config = config['data']

    dataLoaders = []
    train_data_loader, test_data_loader = get_data_as_loader(data_config)
    dataLoaders.append(train_data_loader)
    dataLoaders.append(test_data_loader)

    generate_train_pair(dataLoaders, data_config)
    print("[*] Finished!")


if __name__ == '__main__':
    main()
