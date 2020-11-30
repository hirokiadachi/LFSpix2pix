import os
import random
from PIL import Image

import torch

######################################
# Data Loader for training CycleGAN
######################################
class CycleGAN_Dataset(torch.utils.data.Dataset):
    def __init__(self, datapath, data_limit=2000, transforms=None):
        self.transforms = transforms
        self.A_path = os.path.join(datapath, 'trainA')
        self.B_path = os.path.join(datapath, 'trainB')
        dataA_list = sorted(os.listdir(self.A_path))[1:]
        dataB_list = sorted(os.listdir(self.B_path))[1:]
        random.shuffle(dataA_list)
        random.shuffle(dataB_list)
        self.datalength = min(data_limit, len(dataA_list), len(dataB_list))
        self.dataA = dataA_list[:self.datalength]
        self.dataB = dataB_list[:self.datalength]
        
    def __len__(self):
        return self.datalength
    
    def __getitem__(self, i):
        imgA = Image.open(os.path.join(self.A_path, self.dataA[i])).convert('RGB')
        imgB = Image.open(os.path.join(self.B_path, self.dataB[i])).convert('RGB')
        
        if self.transforms:
            imgA = self.transforms(imgA)
            imgB = self.transforms(imgB)
        
        return imgA, imgB
    
## This dataloader is for SIP dataset.
class CycleGAN_SIP_Dataset(torch.utils.data.Dataset):
    def __init__(self, datapath, situations=['sunny', 'rainy'], transforms=None):
        self.transforms = transforms
        self.A_path = os.path.join(datapath, situations[0])
        self.B_path = os.path.join(datapath, situations[1])
        dataA_list = os.listdir(self.A_path)
        dataB_list = os.listdir(self.B_path)
        self.datalength = min(len(dataA_list), len(dataB_list))
        self.dataA = dataA_list[:self.datalength]
        self.dataB = dataB_list[:self.datalength]
        
    def __len__(self):
        return self.datalength
    
    def __getitem__(self, i):
        imgA = Image.open(os.path.join(self.A_path, self.dataA[i])).convert('RGB')
        imgB = Image.open(os.path.join(self.B_path, self.dataB[i])).convert('RGB')
        
        if self.transforms:
            imgA = self.transforms(imgA)
            imgB = self.transforms(imgB)
        
        return imgA, imgB
        
        
######################################
# Data Loader for training LFSpix2pix
######################################
class LFSpix2pix_Dataset(torch.utils.data.Dataset):
    def __init__(self, datalist, transforms=None):
        self.transforms = transforms
        random.shuffle(datalist)
        self.datalist = datalist
        self.datalength = len(self.datalist)
        
    def __len__(self):
        return self.datalength
    
    def __getitem__(self, i):
        split_item = self.datalist[i].split()
        img_pathA, img_pathB = split_item[0], split_item[1]
        imgA = Image.open(img_pathA).convert('RGB')
        imgB = Image.open(img_pathB).convert('RGB')
        tgt = torch.tensor(float(split_item[2]), dtype=torch.float32)
        
        if self.transforms:
            imgA = self.transforms(imgA)
            imgB = self.transforms(imgB)
        
        return imgA, imgB, tgt

class Pix2Pix_Dataset(torch.utils.data.Dataset):
    def __init__(self, datalist, transforms=None):
        self.transforms = transforms
        random.shuffle(datalist)
        self.datalist = datalist
        self.datalength = len(self.datalist)
        
    def __len__(self):
        return self.datalength
    
    def __getitem__(self, i):
        split_item = self.datalist[i].split()
        img_pathA, img_pathB = split_item[0], split_item[1]
        imgA = Image.open(img_pathA).convert('RGB')
        imgB = Image.open(img_pathB).convert('RGB')
        if self.transforms:
            imgA = self.transforms(imgA)
            imgB = self.transforms(imgB)
        return imgA, imgB
    
#############################
# Generared images buffer
#############################
class Image_History_Buffer:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.buffer = []
    
    def get_images(self,pre_images):
        return_imgs = []
        for img in pre_images:
            img = torch.unsqueeze(img,0)
            if len(self.buffer) < self.pool_size:
                self.buffer.append(img)
                return_imgs.append(img)
            else:
                if random.randint(0,1)>0.5:
                    i = random.randint(0,self.pool_size-1)
                    tmp = self.buffer[i].clone()
                    self.buffer[i]=img
                    return_imgs.append(tmp)
                else:
                    return_imgs.append(img)
        return torch.cat(return_imgs,dim=0)
    

class loss_scheduler():
    def __init__(self, epoch_decay):
        self.epoch_decay = epoch_decay

    def f(self, epoch):
        if epoch<=self.epoch_decay:
            return 1
        else:
            scaling = 1 - (epoch-self.epoch_decay)/float(self.epoch_decay)
            return scaling