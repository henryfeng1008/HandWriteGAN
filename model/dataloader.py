import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, filePath) -> None:
        h5file = h5py.File(filePath, 'r')
        self.input = h5file['inputData']
        self.target = h5file['targetData']
    
    def __getitem__(self, idx):
        inputData = torch.from_numpy(self.input[idx]).float()
        targetData = torch.from_numpy(self.target[idx]).float()
        return inputData, targetData
    
    def __len__(self):
        return self.input.shape[0]

def getData(dataPath, batchSize):
    dataSet = MyDataSet(dataPath)
    dataloader = torch.utils.data.DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=True)
    return dataloader


if __name__ == "__main__":
    dataPath = './trainData/train.h5'
    trainData = getData(dataPath, 64)
    for idx, (inputData, targetData) in enumerate(trainData):
        print(inputData.shape, targetData.shape)
    # for i, data in enumerate(trainData):
    #     print(np.array(data[0])) 
    #     print(np.array(data[1]))
    #     plt.figure()
    #     plt.imshow(np.array(data[0][0, 0, :, :]), cmap='gra y')
    #     plt.show()
    #     break