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

# @File    :   utils.py
# @Time    :   2022/10/10 19:51:35
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :

import os
import sys
import cv2
import yaml
import imageio
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def load_config(config_path, check=False):
    f = open(config_path)
    config = yaml.load(f, Loader=yaml.FullLoader)
    if check:
        print(config)
    return config


def ensure_dir(target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)


def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    return model.cuda()


def show_image(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()


def save_image(path, img):
    imageio.imsave(path, img)


def run_model(model, input_data_np):
    with torch.no_grad():
        input_data = torch.tensor(input_data_np).cuda()
        result = model(input_data)
        result = result.cpu().numpy().reshape(28, 28)
        result = np.uint8(np.clip((result + 1) / 2 * 255 + 0.5, 0, 255))
        result = cv2.resize(result, (100, 100))
    return np.array(result)


def merge_image(sequence):
    num_img = len(sequence)
    result = np.zeros((100, 100 * num_img)).astype(np.uint8)
    for idx, img in enumerate(sequence):
        result[:, idx * 100:(idx + 1) * 100] = img.copy()
    return result

