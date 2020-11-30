import torch
import torch.nn as nn

class LatentFeatureScaling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LatentFeatureScaling, self).__init__()
        self.lfs = nn.Sequential(
            nn.Linear(in_ch, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_ch),
            nn.Sigmoid())
    
    def forward(self, x):
        return self.lfs(x)
    

class EncodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=True, norm='batch', act='relu'):
        """
        in_ch: number of input channel size for the convolution layer.
        out_ch: number of output channel size for the convolution layer.
        kernel_size: size of weight filter for the convolutin layer.
        stride: the interval to slide a weight filter.
        padding: padding size for feature maps.
        bias: this argument indicates whether add bias to convolved feature maps or not. (True or False)
        norm: this argument indicates used normalization techniques after the convolution. (batch: BatchNorm2d, instance: InstanceNorm2d)
        act: this argument indicates used non-linear activation function. (relu: ReLU, lrelu: LeakyReLU)
        """
        super(EncodeBlock, self).__init__()
        if norm == 'batch':    normalization = nn.BatchNorm2d(out_ch)
        elif norm == 'instance':    normalization = nn.InstanceNorm2d(out_ch)
        elif norm == 'none':    normalization = None
        
        if act == 'relu':    activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':    activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif act == 'sigmoid':    activation = nn.Sigmoid()
        if normalization:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                normalization,
                activation)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                activation)
        
    def forward(self, x):
        return self.block(x)
    
class DecodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=True, norm='batch', act='relu', dropout=False):
        """
        in_ch: number of input channel size for the convolution layer.
        out_ch: number of output channel size for the convolution layer.
        kernel_size: size of weight filter for the convolutin layer.
        stride: the interval to slide a weight filter.
        padding: padding size for feature maps.
        bias: this argument indicates whether add bias to convolved feature maps or not. (True or False)
        norm: this argument indicates used normalization techniques after the convolution. (batch: BatchNorm2d, instance: InstanceNorm2d)
        act: this argument indicates used non-linear activation function. (relu: ReLU, lrelu: LeakyReLU)
        """
        super(DecodeBlock, self).__init__()
        if norm == 'batch':    normalization = nn.BatchNorm2d(out_ch)
        elif norm == 'instance':    normalization = nn.InstanceNorm2d(out_ch)
        
        if act == 'relu':    activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':    activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif act == 'tanh':    activation = nn.Tanh()
        
        if dropout:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                normalization,
                nn.Dropout2d(p=0.5, inplace=True),
                activation)
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                normalization,
                activation)
        
    def forward(self, x):
        return self.block(x)
    
class Pix2Pix_Generator(nn.Module):
    def __init__(self, in_ch=3, init_ch=64, condition_dim=1, use_lfs=True):
        super(Pix2Pix_Generator, self).__init__()
        self.use_lfs = use_lfs
        num_lfs = 0
        network_in_channels = []
        # Difine encoder: 3->64->128->256->512->512->512->512->512
        self.e1 = EncodeBlock(in_ch, init_ch, kernel_size=4, stride=2, padding=1, bias=True, norm='none', act='lrelu')
        self.e2 = EncodeBlock(init_ch, init_ch*2, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
        self.e3 = EncodeBlock(init_ch*2, init_ch*4, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
        self.e4 = EncodeBlock(init_ch*4, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
        self.e5 = EncodeBlock(init_ch*8, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
        self.e6 = EncodeBlock(init_ch*8, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
        self.e7 = EncodeBlock(init_ch*8, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
        self.e8 = EncodeBlock(init_ch*8, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
        encoder_layers = [self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8]
        for layer in encoder_layers:
            for m in layer.modules():
                if m.__class__.__name__ == "Conv2d":
                    if m.in_channels == 3:    continue
                    num_lfs += m.in_channels
                    network_in_channels.append(m.in_channels)
        
        # Difine decoder:
        self.d1 = DecodeBlock(init_ch*8, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='relu', dropout=True)
        self.d2 = DecodeBlock(init_ch*8*2, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='relu', dropout=True)
        self.d3 = DecodeBlock(init_ch*8*2, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='relu', dropout=True)
        self.d4 = DecodeBlock(init_ch*8*2, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='relu')
        self.d5 = DecodeBlock(init_ch*8*2, init_ch*4, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='relu')
        self.d6 = DecodeBlock(init_ch*4*2, init_ch*2, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='relu')
        self.d7 = DecodeBlock(init_ch*2*2, init_ch, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='relu')
        self.d8 = DecodeBlock(init_ch*2, in_ch, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='tanh')
        decoder_layers = [self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7, self.d8]
        for layer in decoder_layers:
            for m in layer.modules():
                if m.__class__.__name__ == "ConvTranspose2d":
                    if m.in_channels == 3:    continue
                    num_lfs += m.in_channels
                    network_in_channels.append(m.in_channels)
                    
        points = 0
        self.split_points = []
        for i in range(len(network_in_channels)):
            points += network_in_channels[i]
            self.split_points.append(points)
        print(self.split_points)
        self.lfs = LatentFeatureScaling(in_ch=condition_dim, out_ch=num_lfs)
        
    def lfs_module(self, x, y, layer_idx):
        mul_cond = y[:,layer_idx-1:layer_idx,None,None].expand_as(x)
        x = x * mul_cond
        return x
        
    def pix2pix_with_lfs(self, x, y):
        proj_y = self.lfs(y)
        e1 = self.e1(x)
        e2 = self.e2(self.lfs_module(e1, proj_y, 1))
        e3 = self.e3(self.lfs_module(e2, proj_y, 2))
        e4 = self.e4(self.lfs_module(e3, proj_y, 3))
        e5 = self.e5(self.lfs_module(e4, proj_y, 4))
        e6 = self.e6(self.lfs_module(e5, proj_y, 5))
        e7 = self.e7(self.lfs_module(e6, proj_y, 6))
        e8 = self.e8(self.lfs_module(e7, proj_y, 7))
        
        d1 = self.d1(self.lfs_module(e8, proj_y, 8))
        in_d1 = torch.cat((d1, e7), dim=1)
        d2 = self.d2(self.lfs_module(in_d1, proj_y, 9))
        in_d2 = torch.cat((d2, e6), dim=1)
        d3 = self.d3(self.lfs_module(in_d2, proj_y, 10))
        in_d3 = torch.cat((d3, e5), dim=1)
        d4 = self.d4(self.lfs_module(in_d3, proj_y, 11))
        in_d4 = torch.cat((d4, e4), dim=1)
        d5 = self.d5(self.lfs_module(in_d4, proj_y, 12))
        in_d5 = torch.cat((d5, e3), dim=1)
        d6 = self.d6(self.lfs_module(in_d5, proj_y, 13))
        in_d6 = torch.cat((d6, e2), dim=1)
        d7 = self.d7(self.lfs_module(in_d6, proj_y, 14))
        in_d7 = torch.cat((d7, e1), dim=1)
        out = self.d8(self.lfs_module(in_d7, proj_y, 15))
        return out
    
    def pix2pix_without_lfs(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        d1 = self.d1(e8)
        in_d1 = torch.cat((d1, e7), dim=1)
        d2 = self.d2(in_d1)
        in_d2 = torch.cat((d2, e6), dim=1)
        d3 = self.d3(in_d2)
        in_d3 = torch.cat((d3, e5), dim=1)
        d4 = self.d4(in_d3)
        in_d4 = torch.cat((d4, e4), dim=1)
        d5 = self.d5(in_d4)
        in_d5 = torch.cat((d5, e3), dim=1)
        d6 = self.d6(in_d5)
        in_d6 = torch.cat((d6, e2), dim=1)
        d7 = self.d7(in_d6)
        in_d7 = torch.cat((d7, e1), dim=1)
        out = self.d8(in_d7)
        return out
        
    def forward(self, x, y=None):
        if self.use_lfs:    out = self.pix2pix_with_lfs(x, y)
        else:    out = self.pix2pix_without_lfs(x)
        return out
        
class Pix2Pix_Discriminator(nn.Module):
    def __init__(self, in_ch=3, init_ch=64, patch_size='70x70'):
        super(Pix2Pix_Discriminator, self).__init__()
        if patch_size == '1x1':
            self.l1 = EncodeBlock(in_ch, init_ch, kernel_size=1, stride=1, padding=0, bias=True, norm='none', act='lrelu')
            self.l2 = EncodeBlock(init_ch, init_ch*2, kernel_size=1, stride=1, padding=0, bias=True, norm='batch', act='lrelu')
            self.l3, self.l4 = None, None
            self.last = EncodeBlock(init_ch*2, 1, kernel_size=1, stride=1, padding=0, bias=True, norm='none', act='sigmoid')
        elif patch_size == '16x16':
            self.l1 = EncodeBlock(in_ch, init_ch, kernel_size=4, stride=2, padding=1, bias=True, norm='none', act='lrelu')
            self.l2 = EncodeBlock(init_ch, init_ch*2, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
            self.l3, self.l4 = None, None
            self.last = EncodeBlock(init_ch*2, 1, kernel_size=4, stride=1, padding=1, bias=True, norm='none', act='sigmoid')
        elif patch_size == '70x70':
            self.l1 = EncodeBlock(in_ch, init_ch, kernel_size=4, stride=2, padding=1, bias=True, norm='none', act='lrelu')
            self.l2 = EncodeBlock(init_ch, init_ch*2, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
            self.l3 = EncodeBlock(init_ch*2, init_ch*4, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
            self.l4 = EncodeBlock(init_ch*4, init_ch*8, kernel_size=4, stride=2, padding=1, bias=True, norm='batch', act='lrelu')
            self.last = EncodeBlock(init_ch*8, 1, kernel_size=4, stride=1, padding=1, bias=True, norm='none', act='sigmoid')
            
    def forward(self, x):
        h = self.l1(x)
        h = self.l2(h)
        if self.l3:
            h = self.l3(h)
        if self.l4:
            h = self.l4(h)
        out = self.last(h)
        return out