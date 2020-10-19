import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, features):
        """
        features: number of input and output feature dimension.
        """
        super(ResBlock, self).__init__()
        block_list = []
        self.block = self.make_block(block_list, features)
        
    def make_block(self, modules_list, features):
        modules_list.append(nn.ReflectionPad2d(1))
        modules_list.append(nn.Conv2d(features, features, kernel_size=3, stride=1, bias=True))
        modules_list.append(self.select_normalization(norm='instance', features=features))
        modules_list.append(nn.ReLU(inplace=True))
        modules_list.append(nn.ReflectionPad2d(1))
        modules_list.append(nn.Conv2d(features, features, kernel_size=3, stride=1, bias=True))
        modules_list.append(self.select_normalization(norm='instance', features=features))
        modules = nn.Sequential(*modules_list)
        return modules
        
    def select_normalization(self, norm, features):
        if norm == 'batch':
            return nn.BatchNorm2d(features)
        elif norm == 'instance':
            return nn.InstanceNorm2d(features)
        else:
            assert 0, '%s is not supported.' % norm

    def forward(self, x):
        out = x + self.block(x)
        return out

#########################
# Generator
#########################
class Generator(nn.Module):
    def __init__(self, n_down, n_up, n_res, in_features):
        super(Generator, self).__init__()
        
        out_features = 64
        first_conv = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, out_features, kernel_size=7, stride=1, bias=True),
            self.select_normalization(norm='instance', features=out_features),
            nn.ReLU(inplace=True)]
        
        down_block = []
        for _ in range(n_down):
            in_features = out_features
            out_features = in_features * 2
            down_block += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=True),
                self.select_normalization(norm='instance', features=out_features),
                nn.ReLU(inplace=True)]
            
        res_block = []
        res_features = out_features
        for _ in range(n_res):
            res_block.append(ResBlock(res_features))
            
        up_block = []
        in_features = res_features
        out_features = in_features // 2
        for _ in range(n_up):
            up_block += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                self.select_normalization(norm='instance', features=out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        
        last_conv = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, 3, kernel_size=7, stride=1, bias=True),
            nn.Tanh()]
        
        self.first_conv = nn.Sequential(*first_conv)
        self.down_block = nn.Sequential(*down_block)
        self.res_block = nn.Sequential(*res_block)
        self.up_block = nn.Sequential(*up_block)
        self.last_conv = nn.Sequential(*last_conv)
        self.init_weights(self.first_conv)
        self.init_weights(self.down_block)
        self.init_weights(self.res_block)
        self.init_weights(self.up_block)
        self.init_weights(self.last_conv)
        
    def init_weights(self, net):
        classname = net.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(net.weight.data, 0.0, 0.02)
            if hasattr(net, 'bias') and net.bias is not None:
                torch.nn.init.constant_(net.bias.data, 0.0)
    
    def select_normalization(self, norm, features):
        if norm == 'batch':
            return nn.BatchNorm2d(features)
        elif norm == 'instance':
            return nn.InstanceNorm2d(features)
        else:
            assert 0, '%s is not supported.' % norm
            
    def forward(self, x):
        h = self.first_conv(x)
        h = self.down_block(h)
        h = self.res_block(h)
        h = self.up_block(h)
        out = self.last_conv(h)   
        return out

#########################
# Discriminator
#########################
class Discriminator(nn.Module):
    def __init__(self, n_layers=3):
        super(Discriminator, self).__init__()
        out_features = 64
        modules = [nn.Conv2d(3, out_features, kernel_size=4, stride=2, padding=1, bias=True),
                   nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        for i in range(n_layers):
            in_features = out_features
            out_features = in_features * 2
            if i == n_layers - 1:    stride=1
            else:    stride=2
            modules += [nn.Conv2d(in_features, out_features, kernel_size=4, stride=stride, padding=1, bias=True),
                        self.select_normalization(norm='instance', features=out_features),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        
        modules += [nn.Conv2d(out_features, 1, kernel_size=4, stride=1, padding=1, bias=True)]
        self.layers = nn.Sequential(*modules)
        self.init_weights(self.layers)
        
    def init_weights(self, net):
        classname = net.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(net.weight.data, 0.0, 0.02)
            if hasattr(net, 'bias') and net.bias is not None:
                torch.nn.init.constant_(net.bias.data, 0.0)
    
    def select_normalization(self, norm, features):
        if norm == 'batch':
            return nn.BatchNorm2d(features)
        elif norm == 'instance':
            return nn.InstanceNorm2d(features)
        else:
            assert 0, '%s is not supported.' % norm
            
    def forward(self, x):
        f_maps = []
        out = self.layers(x)
        f_maps = None
        #h = x
        #for layer in self.layers:
        #    h = layer(h)
        #    f_maps.append(h)
        #out = h
        return out, f_maps