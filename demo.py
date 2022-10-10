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

# @File    :   demo.py
# @Time    :   2022/10/10 19:11:10
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description : string to hand write digit

import os
import numpy as np
import utils
from models import *
import utils

def str_to_list(string):
    sequence = []
    for chr in string:
        if chr.isdigit():
            digit = int(chr)
            sequence.append(digit)
        else:
            print(chr, "is not a digit")
            sequence = []
            break
    return sequence


def main():
    config_path = "./config.yaml"
    config_all = utils.load_config(config_path)
    net_config = config_all['train']

    # original_digits = input("Please input the digits:")
    original_digits = "17601638921"
    digit_sequence = str_to_list(original_digits)

    img_sequence = []
    model_ckpt_path = os.path.join('./release', 'G_00001.pth')
    if len(digit_sequence):
        generator_num_block = net_config['generator_num_block']
        generator_init_unit = net_config['generator_init_unit']
        # print(generator_num_block, generator_init_unit)
        model = HandWriteGenerator(num_blocks=generator_num_block,
                                   num_units=generator_init_unit)
        generator = utils.load_model(model, model_ckpt_path)
        for digit in digit_sequence:
            noise_input = -1 * np.ones((1, 1, 10))
            noise_input[0, 0, digit] = np.random.uniform(0, 1)
            noise_input = noise_input.astype(np.float32)
            # print(digit, noise_input)
            fake_image = utils.run_model(generator, noise_input)
            img_sequence.append(fake_image)
        pass
    else:
        print("Invalid input")
        return
    final_image = utils.merge_image(img_sequence)
    # utils.show_image(final_image)
    print(time.time())
    file_name = "./" + "%s"%(time.time()) + ".jpg"
    utils.save_image(file_name, final_image)

if __name__ == "__main__":
    main()