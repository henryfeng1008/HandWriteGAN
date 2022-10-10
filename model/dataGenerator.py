import os
import sys
import h5py
import torch
import numpy as np
from torchvision import transforms, datasets
import matplotlib.pyplot as plt



def getData():
    train_dataset = datasets.MNIST(root='../data', 
                                train=True, 
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=1, 
                                            shuffle=True,
                                            num_workers=5)

    test_dataset = datasets.MNIST(root='../data', 
                                train=False, 
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
    testLoader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=5)
    return trainLoader, testLoader

def generateTrainPair(dataloaders, trainDataPath):
    f = h5py.File(trainDataPath, 'w')
    inputData = []
    targetData = []
     
    for loaderIdx , dataloader in enumerate(dataloaders):
        print("Dataloader ", loaderIdx)
        length = len(dataloader)
        for dataIdx, content in enumerate(dataloader):
            oriImg, oriDigit = content
            n, c, h, w = oriImg.shape
            digit = oriDigit.numpy()[0]
            
            img = oriImg.numpy()
            img = np.clip(img * 2 - 1, -1, 1)
            img = img.reshape(-1, 28, 28)
            
            for i in range(multiplier):
                # noiseInput = np.clip(np.random.normal(digit * 0.1, 1, (1, h, w)), 0, 1)
                # noiseInput = np.clip(np.random.normal(digit * 0.2 - 1, 1, (1, 100)), -1, 1)
                noiseInput = np.ones((1, 10))
                noiseInput[0, digit] = np.random.uniform(-1, 0.5)
                # noiseInput = np.clip(np.random.normal(0, 1, (1, 100)), -1, 1)
                inputData.append(noiseInput)
                targetData.append(img)
            
            percent = int(dataIdx / length * 100)
            print("\r", end="")
            print("进度: {}%: ".format(percent), "*" * (percent // 2), end="")
            sys.stdout.flush()
        print()
        # break
    f['inputData'] = inputData
    f['targetData'] = targetData
    f.close()    

if __name__ == '__main__':
    targetPath = r'./trainData'
    trainDataName = 'train_vector.h5'
    multiplier = 2
    
    if not os.path.exists(targetPath):
        os.mkdir(targetPath)
    
    trainDataLoader, testDataLoader = getData()
    
    trainDataPath = os.path.join(targetPath, trainDataName)
    dataLoaders = []
    dataLoaders.append(trainDataLoader)
    dataLoaders.append(testDataLoader)
    generateTrainPair(dataLoaders, trainDataPath)