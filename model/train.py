import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from models import *
from dataloader import *
import time
import imageio


def modelTest(epoch, model, resultPath):
    with torch.no_grad():
        for i in range(10):
            # noise_input = torch.tensor(np.clip(np.random.normal(i * 0.1, 1, (1, 1, 28, 28)).astype(np.float32), 0, 1)).cuda()
            noiseInput = np.ones((1, 1, 10))
            noiseInput[0, 0, i] = np.random.uniform(-1, 0.5)
            noise_input = torch.tensor(noiseInput.astype(np.float32)).cuda()
            # noise_input = torch.tensor(np.clip(np.random.normal(i * 0.2 - 1, 1, (1, 1, 100)).astype(np.float32), -1, 1)).cuda()
            # noise_input = torch.tensor(np.clip(np.random.normal(0, 1, (1, 1, 100)).astype(np.float32), -1, 1)).cuda()
        
            output = model(noise_input.cuda())
            fake_img = output.cpu().numpy()
            fake_img = fake_img.reshape(28, 28)
            fake_img = np.uint8(np.clip(((fake_img + 1) / 2) * 255 + 0.5, 0, 255))
            imgPath = os.path.join(resultPath, "IMG_" + str(epoch).zfill(4) + "_" + str(i).zfill(2) + ".jpg")
            imageio.imsave(imgPath, fake_img)

def modelTrain(epoches, trainDataLoader, ckptPath, device):
    print(len(trainDataLoader))
    generator = HandWriteGenerator(num_blocks=3, num_units=128).cuda()
    discriminator = HandWriteDiscriminator(num_blocks=2).cuda()
    preTrainDiscriminator = os.path.join(ckptPath, "D_preTrain.pth")
    
    lossFunc = nn.BCELoss()
    l1Loss = nn.L1Loss()
    
    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=[0.9, 0.999])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=[0.9, 0.999])
    
    scheduler_d = optim.lr_scheduler.CyclicLR(optimizer_d, base_lr=1e-5, max_lr=1e-2, mode='triangular2', cycle_momentum=False)
    
    # if os.path.exists(preTrainDiscriminator):
    #     print("pth exist")
    #     ckpt = torch.load(preTrainDiscriminator)
    #     discriminator.load_state_dict(ckpt)
    # else:
    #     print("Pre-train Discriminator not exists")
    #     print("[*] Train Discriminator for %d epoch" % (preTrainEpoch))
    #     for epoch in range(preTrainEpoch):
    #         for i, data in enumerate(trainDataLoader):
                
    #             inputData, targetData = data
    #             n, c, h, w = targetData.shape

    #             inputData = inputData.cuda()
    #             targetData = targetData.cuda()
                
    #             fake_img = generator(inputData)
                
    #             true_label = torch.tensor(np.ones((n, c, 1, 1)).astype(np.float32)).cuda()
    #             false_label = torch.tensor(np.zeros((n, c, 1, 1)).astype(np.float32)).cuda()
                
    #             optimizer_d.zero_grad()
    #             fake_loss = lossFunc(discriminator(fake_img.detach()), false_label)
    #             real_loss = lossFunc(discriminator(targetData), true_label)
    #             d_loss = (fake_loss + real_loss) / 2
    #             d_loss.backward()
    #             optimizer_d.step()
                
    #             if i % 10 == 0:
    #                 print("[%s] epoch:%d,iter:%d, d_loss:%.8f, lr:%s"%(time.asctime(time.localtime(time.time())), \
    #                     epoch, i, d_loss.item(), scheduler_d.get_last_lr()))
    #             scheduler_d.step()    
    #     print("Discriminator pre-train finished, save model")
    #     torch.save(discriminator.state_dict(), preTrainDiscriminator)
    
    print("[*] GAN Train start!")
    
    scheduler_g = optim.lr_scheduler.CyclicLR(optimizer_g, base_lr=5e-6, max_lr=1e-5, mode='triangular2', cycle_momentum=False)
    scheduler_d = optim.lr_scheduler.CyclicLR(optimizer_d, base_lr=5e-6, max_lr=1e-5, mode='triangular2', cycle_momentum=False)
    
    for epoch in range(epoches):
        for i, data in enumerate(trainDataLoader):
            inputData, targetData = data
            inputData = inputData.cuda()
            targetData = targetData.cuda()
            
            fake_img = generator(inputData)

            # train Generator
            optimizer_g.zero_grad()
            
            l1_loss = l1Loss(fake_img, targetData)
            
            g_fake_logits = discriminator(fake_img)
            true_label = torch.tensor(np.ones(g_fake_logits.shape).astype(np.float32)).cuda()
            false_label = torch.tensor(np.zeros(g_fake_logits.shape).astype(np.float32)).cuda()
            g_gan_loss = lossFunc(g_fake_logits, true_label)
            g_loss = (g_gan_loss + l1_loss) / 2
            # g_loss = g_gan_loss
            g_loss.backward()
            optimizer_g.step()
        
            # train Discriminator
            optimizer_d.zero_grad()
            fake_loss = lossFunc(discriminator(fake_img.detach()), false_label)
            real_loss = lossFunc(discriminator(targetData), true_label)
            d_loss = (fake_loss + real_loss) / 2
            d_loss.backward()
            optimizer_d.step()
            
            if i % 100 == 0:
                print("[%s] epoch:%d,iter:%d, l1_loss:%.8f, g_gan_loss:%.8f, d_loss:%.8f, lr:%s"%(time.asctime(time.localtime(time.time())), \
                    epoch, i, l1_loss.item(), g_gan_loss.item(), d_loss.item(), scheduler_g.get_last_lr()))

            scheduler_g.step()
            scheduler_d.step()

        modelPath = os.path.join(ckptPath, str(epoch).zfill(5) + '.pth')
        torch.save(generator.state_dict(), modelPath)
        print("Run test:")
        modelTest(epoch, generator, resultPath)
        print("Test finished")
        
    print("Train Finished")
    pass



if __name__ == "__main__":
    mode = 'train'
    # mode = 'test'
    
    dataPath = './trainData/train_vector.h5'
    use_gpu = torch.cuda.is_available()
    device = torch.cuda.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        print("GPU avaliable")
    else:
        print("Train on CPU")
    
    
    preTrainEpoch = 1
    epoch = 10
    batchSize = 32
    trainLoader = getData(dataPath=dataPath, batchSize=batchSize)
    
    ckptPath = './ckpt'
    if not os.path.exists(ckptPath):
        os.mkdir(ckptPath)
    
    resultPath = './result'
    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
            
    if mode == 'train':
        modelTrain(epoch, trainLoader, ckptPath, device)
    elif mode == 'test':
        print("Run test")
        model = HandWriteGenerator(num_blocks=3, num_units=128).cuda()
        ckpt = torch.load(ckptPath + '/00000.pth')
        model.load_state_dict(ckpt)
        modelTest(0, model, resultPath)
    else:
        print("Unsopported Mode!")
