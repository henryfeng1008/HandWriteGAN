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

# @File    :   dataloader.py
# @Time    :   2022/10/10 12:38:22
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description : Load train data


import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt


class MyDataSet(torch.utils.data.Dataset):

    def __init__(self, filePath) -> None:
        h5file = h5py.File(filePath, 'r')
        self.input = h5file['input_data']
        self.target = h5file['target_data']

    def __getitem__(self, idx):
        input_data = torch.from_numpy(self.input[idx]).float()
        target_data = torch.from_numpy(self.target[idx]).float()
        return input_data, target_data

    def __len__(self):
        return self.input.shape[0]


def load_train_data_from_h5(data_path, batch_size):
    dataset = MyDataSet(data_path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    return dataloader


def main():
    data_path = './data/train_vector.h5'
    train_data = load_train_data_from_h5(data_path, 64)
    print(len(train_data))

    for _, (input_data, target_data) in enumerate(train_data):
        print(input_data.shape, target_data.shape)
        break


if __name__ == "__main__":
    main()