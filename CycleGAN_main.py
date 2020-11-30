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
from CycleGAN_models import Generator, Discriminator

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
    p.add_argument('--lambda_cyc', type=int, default=10)
    p.add_argument('--lambda_id', type=float, default=0)
    p.add_argument('--datapath', type=str, default='')
    p.add_argument('--situations', nargs='*', help='You should select 2 situations in [sunny, cloudy, rainy, morning, night]. Recommended for you, when one selects a situation, one has better absolutely included in selected items since all situations other than sunny are extremely less data.')
    p.add_argument('--result', type=str, default='result')
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--gpu', nargs='*', type=int, required=True)
    return p.parse_args()


class CycleGAN_trainer:
    def __init__(self, conf):
        self.conf = conf
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        training_transforms = transforms.Compose([
            transforms.Resize((conf.img_h, conf.img_w), Image.BICUBIC),
            transforms.RandomCrop((conf.img_h, conf.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        if conf.situations:
            train_data = CycleGAN_SIP_Dataset(datapath=conf.datapath, situations=conf.situations, transforms=training_transforms)
        else:
            train_data = CycleGAN_Dataset(datapath=conf.datapath, transforms=training_transforms)
        os.makedirs(conf.result, exist_ok=True)
        if os.path.isdir(conf.log_dir):    shutil.rmtree(conf.log_dir)
        self.tb = SummaryWriter(log_dir=conf.log_dir)
        self.training_dataset = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, 
                                           num_workers=multiprocessing.cpu_count())
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(conf.gpu) > 1:
            self.G_a2b = nn.DataParallel(Generator(n_down=2, n_up=2, n_res=9, in_features=3).to(self.device), device_ids=conf.gpu)
            self.G_b2a = nn.DataParallel(Generator(n_down=2, n_up=2, n_res=9, in_features=3).to(self.device), device_ids=conf.gpu)
            self.D_a = nn.DataParallel(Discriminator(n_layers=3).to(self.device), device_ids=conf.gpu)
            self.D_b = nn.DataParallel(Discriminator(n_layers=3).to(self.device), device_ids=conf.gpu)
        else:
            self.G_a2b = Generator(n_down=2, n_up=2, n_res=9, in_features=3).to(self.device)
            self.G_b2a = Generator(n_down=2, n_up=2, n_res=9, in_features=3).to(self.device)
            self.D_a = Discriminator(n_layers=3).to(self.device)
            self.D_b = Discriminator(n_layers=3).to(self.device)
        
        self.g_opt = optim.Adam(itertools.chain(self.G_a2b.parameters(), self.G_b2a.parameters()), 
                           lr=conf.lr, betas=(conf.beta1, conf.beta2))
        self.d_A_opt = optim.Adam(self.D_a.parameters(), lr=conf.lr, betas=(conf.beta1, conf.beta2))
        self.d_B_opt = optim.Adam(self.D_b.parameters(), lr=conf.lr, betas=(conf.beta1, conf.beta2))
        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_opt, lr_lambda=loss_scheduler(conf.decay_epoch).f)
        self.d_a_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_A_opt, lr_lambda=loss_scheduler(conf.decay_epoch).f)
        self.d_b_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_B_opt, lr_lambda=loss_scheduler(conf.decay_epoch).f)
        
        self.adv_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.perceptual_loss = nn.MSELoss(reduction='sum')

        self.buffer_for_fakeA = Image_History_Buffer()
        self.buffer_for_fakeB = Image_History_Buffer()
        
    def update_generator(self, imgA, imgB, fakeA, fakeB, recA, recB, idenA=None, idenB=None, fr_mapsA=None, fr_mapsB=None):
        self.g_opt.zero_grad()
        dis_outA, ff_mapsA = self.D_a(fakeA)
        dis_outB, ff_mapsB = self.D_b(fakeB)
        dg_lossA = self.adv_loss(dis_outA, torch.tensor(1.0).expand_as(dis_outA).to(self.device))
        dg_lossB = self.adv_loss(dis_outB, torch.tensor(1.0).expand_as(dis_outB).to(self.device))
        dg_loss = dg_lossA + dg_lossB
            
        cycle_consistency_loss = self.cycle_loss(recA, imgA) + self.cycle_loss(recB, imgB)
        if self.conf.lambda_id > 0:
            identity_loss = self.identity_loss(idenA, imgA) + self.identity_loss(idenB, imgB)
            g_loss = dg_loss + conf.lambda_cyc * cycle_consistency_loss + conf.lambda_id * identity_loss
        else:
            identity_loss = None
            g_loss = dg_loss + conf.lambda_cyc * cycle_consistency_loss
            
        if fr_mapsA is not None and ff_mapsB is not None:
            lambda_per = 10
            per_lossA = 0
            for (fr_A, ff_A) in zip(fr_mapsA, ff_mapsA):
                _, c, h, w = fr_A.shape
                per_lossA += self.perceptual_loss(ff_A, fr_A.detach()) / (c*h*w)
            
            per_lossB = 0
            for (fr_B, ff_B) in zip(fr_mapsB, ff_mapsB):
                _, c, h, w = fr_B.shape
                per_lossB += self.perceptual_loss(ff_B, fr_B.detach()) / (c*h*w)
            per_loss = per_lossA + per_lossB
            g_loss = g_loss + lambda_per * per_loss
        
        g_loss.backward()
        self.g_opt.step()
        return g_loss, cycle_consistency_loss, identity_loss
        
    def update_discriminator(self, imgA, imgB, fakeA, fakeB):
        self.d_A_opt.zero_grad()
        dr_outA, fr_mapsA = self.D_a(imgA)
        fakeA_ = self.buffer_for_fakeA.get_images(fakeA)
        df_outA, _ = self.D_a(fakeA_.detach())
        dr_lossA = self.adv_loss(dr_outA, torch.tensor(1.0).expand_as(dr_outA).to(self.device))
        df_lossA = self.adv_loss(df_outA, torch.tensor(0.0).expand_as(df_outA).to(self.device))
        d_lossA = (dr_lossA + df_lossA) * 0.5
        d_lossA.backward()
        self.d_A_opt.step()
        
        self.d_B_opt.zero_grad()
        dr_outB, fr_mapsB = self.D_b(imgB)
        fakeB_ = self.buffer_for_fakeB.get_images(fakeB)
        df_outB, _ = self.D_b(fakeB_.detach())
        dr_lossB = self.adv_loss(dr_outB, torch.tensor(1.0).expand_as(dr_outB).to(self.device))
        df_lossB = self.adv_loss(df_outB, torch.tensor(0.0).expand_as(df_outB).to(self.device))
        d_lossB = (dr_lossB + df_lossB) * 0.5
        d_lossB.backward()
        self.d_B_opt.step()
        return d_lossA, d_lossB, fr_mapsA, fr_mapsB
        
    def train(self, epoch):
        self.G_a2b.train()
        self.G_b2a.train()
        self.D_a.train()
        self.D_b.train()
        for idx, (imgA, imgB) in enumerate(self.training_dataset):
            imgA, imgB = imgA.to(self.device), imgB.to(self.device)
            fakeA, fakeB = self.G_b2a(imgB), self.G_a2b(imgA)
            recA, recB = self.G_b2a(fakeB), self.G_a2b(fakeA)
            if self.conf.lambda_id > 0:
                idenA, idenB = self.G_b2a(imgA), self.G_a2b(imgB)
            else:
                idenA, idenB = None, None
            
            d_lossA, d_lossB, fr_mapA, fr_mapB = self.update_discriminator(
                imgA, imgB, fakeA, fakeB)
            
            g_loss, cyc_loss, iden_loss = self.update_generator(
                imgA, imgB, fakeA, fakeB, recA, recB, idenA, idenB, fr_mapA, fr_mapB)
            
            if idx % 100 == 0:
                print('Training epoch: {} [{}/{} ({:.0f}%)] | D loss (A): {:.6f} | D loss (B): {:.6f} | G loss: {:.6f} | Consistency: {:.6f} |'\
                      .format(epoch, idx * len(imgA), len(self.training_dataset.dataset),
                          100. * idx / len(self.training_dataset), d_lossA.item(), d_lossB.item(), g_loss.item(), cyc_loss.item()))
            
        self.g_lr_scheduler.step()
        self.d_a_lr_scheduler.step()
        self.d_b_lr_scheduler.step()
        self.model_save()
        self.image_save(epoch)
            
    def model_save(self):
        torch.save(self.G_a2b.state_dict(), os.path.join(self.conf.result, 'G_a2b'))
        torch.save(self.G_b2a.state_dict(), os.path.join(self.conf.result, 'G_b2a'))
        torch.save(self.D_a.state_dict(), os.path.join(self.conf.result, 'D_a'))
        torch.save(self.D_b.state_dict(), os.path.join(self.conf.result, 'D_b'))
        
    def image_save(self, epoch):
        if self.conf.situations:
            test_imgA_path = os.path.join(self.conf.datapath, self.conf.situations[0])
            test_imgB_path = os.path.join(self.conf.datapath, self.conf.situations[1])
        else:
            test_imgA_path = os.path.join(self.conf.datapath, 'testA')
            test_imgB_path = os.path.join(self.conf.datapath, 'testB')
        test_imgA_list = os.listdir(test_imgA_path)
        test_imgB_list = os.listdir(test_imgB_path)
        to_tensor = transforms.ToTensor()
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean, std)
        test_length = min(len(test_imgA_list), len(test_imgB_list))
        img_idx = np.random.randint(test_length, size=5)
        for idx, i in enumerate(img_idx):
            test_imgA = Image.open(os.path.join(test_imgA_path, test_imgA_list[i]))
            test_imgB = Image.open(os.path.join(test_imgB_path, test_imgB_list[i]))
            test_imgA_tensor = normalize(to_tensor(test_imgA)).unsqueeze(0).cuda()
            test_imgB_tensor = normalize(to_tensor(test_imgB)).unsqueeze(0).cuda()
            with torch.no_grad():
                fake_test_B = self.G_a2b(test_imgA_tensor)
                fake_test_A = self.G_b2a(test_imgB_tensor)
            self.tb.add_image('Domain A/real%d'%idx, test_imgA_tensor.data.cpu().squeeze(), epoch)
            self.tb.add_image('Domain A/fake%d'%idx, fake_test_A.data.cpu().squeeze(), epoch)
            self.tb.add_image('Domain B/real%d'%idx, test_imgB_tensor.data.cpu().squeeze(), epoch)
            self.tb.add_image('Domain B/fake%d'%idx, fake_test_B.data.cpu().squeeze(), epoch)
            
if __name__ == '__main__':
    conf = config()
    trainer = CycleGAN_trainer(conf)
    
    for epoch in range(1, conf.epochs+1):
        trainer.train(epoch)