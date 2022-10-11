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

# @File    :   train.py
# @Time    :   2022/10/10 12:36:43
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description : train and test model for hand written digit img generation.


import os
import time
import imageio
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import utils
from models import *
from data_loader import load_train_data_from_h5


MODE_TRAIN = 0
MODE_TEST = 1


def modelTest(epoch, model, config):
    result_path = config['test_result_path']
    with torch.no_grad():
        for i in range(10):
            noise_input = -1 * np.ones((1, 1, 10))
            noise_input[0, 0, i] = np.random.uniform(0, 1)
            noise_input = torch.tensor(noise_input.astype(np.float32)).cuda()

            output = model(noise_input.cuda())
            fake_img = output.cpu().numpy()
            fake_img = fake_img.reshape(28, 28)
            fake_img = np.uint8(
                np.clip(((fake_img + 1) / 2) * 255 + 0.5, 0, 255))
            img_name = "IMG_" + str(epoch).zfill(4) +\
                "_" + str(i).zfill(2) + ".jpg"
            img_path = os.path.join(result_path, img_name)
            imageio.imsave(img_path, fake_img)


def modelTrain(train_data_loader, config):
    ckpt_path = config['ckpt_path']
    generator_num_block = config['generator_num_block']
    generator_init_unit = config['generator_init_unit']
    generator = HandWriteGenerator(num_blocks=generator_num_block,
                                   num_units=generator_init_unit).cuda()

    discriminator_num_block = config['discriminator_num_block']

    discriminator = HandWriteDiscriminator(
        num_blocks=discriminator_num_block).cuda()

    cross_entropy_loss_func = nn.BCELoss()
    l1_loss_func = nn.L1Loss()

    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4,
                             betas=[0.9, 0.999])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4,
                             betas=[0.9, 0.999])

    # pre train discriminator
    pre_train_discriminator = config["pre_train_discriminator"]
    pre_train_epoch = config["pre_train_epoch"]
    pre_train_ckpt = config["pre_train_ckpt"]
    pre_rain_discriminator_path = os.path.join(ckpt_path, pre_train_ckpt)

    scheduler_d = optim.lr_scheduler.CyclicLR(optimizer_d,
                                              base_lr=1e-5,
                                              max_lr=1e-2,
                                              mode='triangular2',
                                              cycle_momentum=False)
    if pre_train_discriminator == 1:
        if os.path.exists(pre_rain_discriminator_path):
            print("pth exist")
            ckpt = torch.load(pre_rain_discriminator_path)
            discriminator.load_state_dict(ckpt)
        else:
            print("Pre-train Discriminator not exists")
            print("[*] Train Discriminator for %d epoch" % (pre_train_epoch))
            for epoch in range(pre_train_epoch):
                for i, data in enumerate(train_data_loader):

                    input_data, target_data = data
                    n, c, h, w = target_data.shape

                    input_data = input_data.cuda()
                    target_data = target_data.cuda()

                    fake_img = generator(input_data)

                    true_label = torch.tensor(
                        np.ones((n, c, 1, 1)).astype(np.float32)).cuda()
                    false_label = torch.tensor(
                        np.zeros((n, c, 1, 1)).astype(np.float32)).cuda()

                    optimizer_d.zero_grad()
                    fake_loss = cross_entropy_loss_func(
                        discriminator(fake_img.detach()), false_label)
                    real_loss = cross_entropy_loss_func(
                        discriminator(target_data), true_label)
                    d_loss = (fake_loss + real_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()

                    if i % 10 == 0:
                        print("[%s] epoch:%d,iter:%d, d_loss:%.8f, lr:%s"%(time.asctime(time.localtime(time.time())), \
                            epoch, i, d_loss.item(), scheduler_d.get_last_lr()))
                    scheduler_d.step()
            print("Discriminator pre-train finished, save model")
            torch.save(discriminator.state_dict(), pre_rain_discriminator_path)

    print("[*] GAN Train start!")
    epochs = config['epoch']
    scheduler_g = optim.lr_scheduler.CyclicLR(optimizer_g,
                                              base_lr=5e-6,
                                              max_lr=5e-5,
                                              mode='triangular2',
                                              cycle_momentum=False)
    scheduler_d = optim.lr_scheduler.CyclicLR(optimizer_d,
                                              base_lr=5e-6,
                                              max_lr=5e-5,
                                              mode='triangular2',
                                              cycle_momentum=False)

    for epoch in range(epochs):
        for i, data in enumerate(train_data_loader):
            input_data, target_data = data
            input_data = input_data.cuda()
            target_data = target_data.cuda()

            fake_img = generator(input_data)

            # train Generator
            optimizer_g.zero_grad()

            l1_loss = l1_loss_func(fake_img, target_data)

            g_fake_logits = discriminator(fake_img)
            true_label = torch.tensor(
                np.ones(g_fake_logits.shape).astype(np.float32)).cuda()
            false_label = torch.tensor(
                np.zeros(g_fake_logits.shape).astype(np.float32)).cuda()
            g_gan_loss = l1_loss_func(g_fake_logits, true_label)
            g_loss = (g_gan_loss + l1_loss) * 0.5
            g_loss.backward()
            optimizer_g.step()

            # train Discriminator
            optimizer_d.zero_grad()

            fake_loss = l1_loss_func(
                discriminator(fake_img.detach()), false_label)
            real_loss = l1_loss_func(
                discriminator(target_data), true_label)
            d_loss = (fake_loss + real_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            if i % 100 == 0:
                print("[*] [%s] epoch:%d, iter:%d, l1_loss:%.8f, "
                      "g_gan_loss:%.8f, d_loss:%.8f, lr:%s"\
                      %(time.asctime(time.localtime(time.time())),
                      epoch, i, l1_loss.item(), g_gan_loss.item(),
                      d_loss.item(), scheduler_g.get_last_lr()))

            scheduler_g.step()
            scheduler_d.step()

        g_model_name = 'G_' + str(epoch).zfill(5) + '.pth'
        model_path = os.path.join(ckpt_path, g_model_name)
        torch.save(generator.state_dict(), model_path)

        print("[*] Run test:")
        modelTest(epoch, generator, config)
        print("[*] Test finished")

    print("[*] Train Finished")
    pass


def main():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("[*] GPU avaliable")
    else:
        print("[*] Train on CPU")
    device = torch.cuda.device("cuda" if use_gpu else "cpu")

    config_path = "./config.yaml"
    config_all = utils.load_config(config_path)
    config = config_all['train']

    mode = config['mode']

    train_data_path = config['train_data_path']
    batch_size = config['batch_size']

    ckpt_path = config['ckpt_path']
    test_result_path = config['test_result_path']
    utils.ensure_dir(ckpt_path)
    utils.ensure_dir(test_result_path)

    train_data_loader = load_train_data_from_h5(data_path=train_data_path, batch_size=batch_size)
    print("[*]", len(train_data_loader))

    if mode == MODE_TRAIN:
        print("[*] Run train")
        modelTrain(train_data_loader, config)
    elif mode == MODE_TEST:
        print("[*] Run test")
        # model = HandWriteGenerator(num_blocks=3, num_units=128).cuda()
        # ckpt = torch.load(ckptPath + '/00000.pth')
        # model.load_state_dict(ckpt)
        # modelTest(0, model, resultPath)
    else:
        print("[*] Unsopported Mode!")


if __name__ == "__main__":
    main()

