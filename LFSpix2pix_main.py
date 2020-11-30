import os
import shutil
import random
import argparse
import itertools
import multiprocessing
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import *
from Pix2Pix_models import Pix2Pix_Generator, Pix2Pix_Discriminator

def config():
    p = argparse.ArgumentParser()
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--beta1', type=float, default=0.5)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--img_w', type=int, default=512)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--decay_epoch', type=int, default=100)
    p.add_argument('--lambda_l1', type=int, default=10)
    p.add_argument('--txtpath', type=str, default='./Data/pair_data.txt')
    p.add_argument('--result', type=str, default='result_LFSpix2pix')
    p.add_argument('--log_dir', type=str, default='logs_LFSpix2pix')
    p.add_argument('--gpu', nargs='*', type=int, required=True)
    p.add_argument('--use_lfs', action='store_true')
    return p.parse_args()


class LFSpix2pix_trainer:
    def __init__(self, conf):
        self.conf = conf
        th = 0.9
        self.iters = 0
        
        os.makedirs(conf.result, exist_ok=True)
        if os.path.isdir(conf.log_dir):    shutil.rmtree(conf.log_dir)
        self.tb = SummaryWriter(log_dir=conf.log_dir)
        
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        training_transforms = transforms.Compose([
            transforms.Resize((conf.img_h, conf.img_w), Image.BICUBIC),
            transforms.RandomCrop((conf.img_h, conf.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        with open(conf.txtpath) as f:
            lines = f.readlines()
        data_th = int(th * len(lines))
        self.training_list = lines[:data_th]
        self.test_list = lines[data_th:]
        if conf.use_lfs:
            print('### A network is Pix2Pix with Latent Filter Scaling.')
            train_data = LFSpix2pix_Dataset(datalist=self.training_list, transforms=training_transforms)
        else:
            print('### A network is Pix2Pix without Latent Filter Scaling.')
            train_data = Pix2Pix_Dataset(datalist=self.training_list, transforms=training_transforms)
            
        self.training_dataset = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, 
                                           num_workers=multiprocessing.cpu_count())
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(conf.gpu) > 1:
            self.G = nn.DataParallel(Pix2Pix_Generator(in_ch=3, init_ch=64, condition_dim=2, use_lfs=conf.use_lfs).to(self.device), device_ids=conf.gpu)
            self.D = nn.DataParallel(Pix2Pix_Discriminator(in_ch=3, init_ch=64, patch_size='70x70').to(self.device), device_ids=conf.gpu)
        else:
            self.G = Pix2Pix_Generator(in_ch=3, init_ch=64, condition_dim=1, use_lfs=conf.use_lfs).to(self.device)
            self.D = Pix2Pix_Discriminator(in_ch=6, init_ch=64, patch_size='70x70').to(self.device)
        
        self.g_opt = optim.Adam(self.G.parameters(), lr=conf.lr, betas=(conf.beta1, conf.beta2))
        self.d_opt = optim.Adam(self.D.parameters(), lr=conf.lr, betas=(conf.beta1, conf.beta2))
        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_opt, lr_lambda=loss_scheduler(conf.decay_epoch).f)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_opt, lr_lambda=loss_scheduler(conf.decay_epoch).f)
        
        self.adv_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def update_generator(self, imgA, imgB, fakeB):
        self.g_opt.zero_grad()
        in_dis = torch.cat((imgA, fakeB), dim=1)
        dis_out = self.D(in_dis)
        dg_loss = self.adv_loss(dis_out, torch.tensor(1.0).expand_as(dis_out).to(self.device))
        dg_l1_loss = self.l1_loss(imgB, fakeB)
        
        g_loss = dg_loss + conf.lambda_l1 * dg_l1_loss
        g_loss.backward()
        self.g_opt.step()
        return g_loss, dg_l1_loss
        
    def update_discriminator(self, imgA, imgB, fakeB):
        self.d_opt.zero_grad()
        real_pair = torch.cat((imgA, imgB), dim=1)
        dr_out = self.D(real_pair)
        fake_pair = torch.cat((imgA, fakeB), dim=1)
        df_out = self.D(fake_pair.detach())
        dr_loss = self.adv_loss(dr_out, torch.tensor(1.0).expand_as(dr_out).to(self.device))
        df_loss = self.adv_loss(df_out, torch.tensor(0.0).expand_as(df_out).to(self.device))
        d_loss = dr_loss + df_loss
        d_loss.backward()
        self.d_opt.step()
        return d_loss
        
    def train(self, epoch):
        self.G.train()
        self.D.train()
        for idx, items in enumerate(self.training_dataset):
            if self.conf.use_lfs:
                imgA, imgB, tgt = items[0].to(self.device), items[1].to(self.device), items[2].to(self.device)[:, None]
            else:
                imgA, imgB, tgt = items[0].to(self.device), items[1].to(self.device), None
            fakeB = self.G(imgA, tgt)
                
            g_loss, l1_loss = self.update_generator(imgA, imgB, fakeB)
            d_loss = self.update_discriminator(imgA, imgB, fakeB)
            
            self.iters += 1
            if idx % 100 == 0:
                print('Training epoch: {} [{}/{} ({:.0f}%)] | D loss (A): {:.6f} | G loss: {:.6f} | L1: {:.6f} |'\
                      .format(epoch, idx * len(imgA), len(self.training_dataset.dataset),
                      100. * idx / len(self.training_dataset), d_loss.item(), g_loss.item(), l1_loss.item()))
                self.tb.add_scalars('Loss_gan', {'dis': d_loss.item(), 'gen': g_loss.item()}, self.iters)
                self.tb.add_scalar('L1_loss', l1_loss.item(), self.iters)
            
        self.g_lr_scheduler.step()
        self.d_lr_scheduler.step()
        self.model_save()
        self.image_save(epoch)
            
    def model_save(self):
        torch.save(self.G.state_dict(), os.path.join(self.conf.result, 'G'))
        torch.save(self.D.state_dict(), os.path.join(self.conf.result, 'D'))
        
    def image_save(self, epoch):
        self.G.eval()
        testlength = len(self.test_list)
        test_index = np.random.randint(testlength, size=5)
        to_tensor = transforms.ToTensor()
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        mean_tensor = torch.tensor(mean, dtype=torch.float32)[None,:,None,None].cuda()
        std_tensor = torch.tensor(std, dtype=torch.float32)[None,:,None,None].cuda()
        normalize = transforms.Normalize(mean, std)
        for idx, i in enumerate(test_index):
            test_items = self.test_list[i].split()
            test_imgA = Image.open(test_items[0]).convert('RGB')
            test_imgB = Image.open(test_items[1]).convert('RGB')
            if self.conf.use_lfs:
                test_tgt = torch.tensor(float(test_items[2]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
            else:
                test_tgt = None
            test_imgA_tensor = normalize(to_tensor(test_imgA)).unsqueeze(0).cuda()
            test_imgB_tensor = normalize(to_tensor(test_imgB)).unsqueeze(0).cuda()
            with torch.no_grad():
                #print(test_tgt.shape, test_imgA_tensor.shape)
                fake_test_B = self.G(test_imgA_tensor, test_tgt)
                
            test_imgA_tensor = (test_imgA_tensor * std_tensor) + mean_tensor
            test_imgB_tensor = (test_imgB_tensor * std_tensor) + mean_tensor
            fake_test_B = (fake_test_B * std_tensor) + mean_tensor
            if 'sunny' in test_items[1]:
                self.tb.add_image('Sunny/real_%d'%idx, test_imgA_tensor.data.cpu().squeeze(), epoch)
                self.tb.add_image('Sunny/fake_%d'%idx, fake_test_B.data.cpu().squeeze(), epoch)
            elif 'morning' in test_items[1]:
                self.tb.add_image('Morning/real_%d'%idx, test_imgB_tensor.data.cpu().squeeze(), epoch)
                self.tb.add_image('Morning/fake_%d'%idx, fake_test_B.data.cpu().squeeze(), epoch)
            elif 'night' in test_items[1]:
                self.tb.add_image('Night/real_%d'%idx, test_imgB_tensor.data.cpu().squeeze(), epoch)
                self.tb.add_image('Night/fake_%d'%idx, fake_test_B.data.cpu().squeeze(), epoch)
                
            
if __name__ == '__main__':
    conf = config()
    trainer = LFSpix2pix_trainer(conf)
    
    for epoch in range(1, conf.epochs+1):
        trainer.train(epoch)