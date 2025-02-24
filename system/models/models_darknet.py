import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Conv2dBNLeakyReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, norm_layer=nn.BatchNorm2d):
        super(Conv2dBNLeakyReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            norm_layer(out_channels),
            # In the official implementation, negative_slope is 0.1!!!
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        assert in_channels % 2 == 0
        neck_channels = in_channels // 2
        self.conv1 = Conv2dBNLeakyReLU(in_channels, neck_channels, 1, 1, 0, False, norm_layer=nn.BatchNorm2d)
        self.conv2 = Conv2dBNLeakyReLU(neck_channels, in_channels, 3, 1, 1, False, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = inputs + x
        return x

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
        return x

class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)
    
    
class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)
    
    

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class maxpool(nn.Module):
    def __init__(self, kernel_size, stride):
        super(maxpool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.maxpool(x)
    

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = x*psi
        return out



class Darknet_MidAir_attention_md(nn.Module):   
    def __init__(self):
        super(Darknet_MidAir_attention_md, self).__init__()
        # encoder
        #norm_layer = norm_layer or nn.BatchNorm2d
        self.num_blocks = (1, 2, 4, 4, 4)

        self.conv2dbnleakyrelu = Conv2dBNLeakyReLU(3, 32, 3, 1, 1, False, norm_layer=nn.BatchNorm2d)
        in_channels = 32
        for i, num_repeat in enumerate(self.num_blocks):
            setattr(self, "layer%d" % (i + 1),
                    self._make_layers(in_channels, num_repeat))
            in_channels *= 2

        # decoder
        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(512, 256, 3, 1)
        self.disp5_layer = get_disp(256)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(256+2, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(128+2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64+2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        # attention modules
        self.attention1 = Attention_block(64, 64)
        self.attention2 = Attention_block(128, 128)
        self.attention3 = Attention_block(256, 256)
        self.attention4 = Attention_block(512, 512)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        #encoder
        x1 = self.conv2dbnleakyrelu(x)
        x2 = getattr(self, 'layer1')(x1)
        x2 = self.attention1(x2, x2)
        x3 = getattr(self, 'layer2')(x2)
        x3 = self.attention2(x3, x3)
        x4 = getattr(self, 'layer3')(x3)
        x4 = self.attention3(x4, x4)
        x5 = getattr(self, 'layer4')(x4)
        x5 = self.attention4(x5, x5)

        #decoder
        x5up = self.upconv5(x5)
        x5cat = torch.cat((x5up, x4), 1)
        x5out = self.iconv5(x5cat)
        self.disp5 = self.disp5_layer(x5out)
        self.udisp5 = nn.functional.interpolate(self.disp5, scale_factor=2, mode='bilinear', align_corners=True)

        x4up = self.upconv4(x5out)
        x4cat = torch.cat((x4up, x3, self.udisp5), 1)
        x4out = self.iconv4(x4cat)
        self.disp4 = self.disp4_layer(x4out)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        x3up = self.upconv3(x4out)
        x3cat = torch.cat((x3up, x2, self.udisp4), 1)
        x3out = self.iconv3(x3cat)
        self.disp3 = self.disp3_layer(x3out)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        x2up = self.upconv2(x3out)
        x2cat = torch.cat((x2up, x1, self.udisp3), 1)
        x2out = self.iconv2(x2cat)
        self.disp2 = self.disp2_layer(x2out)

        return self.disp2, self.disp3, self.disp4, self.disp5
    
    @staticmethod
    def _make_layers(in_channels, num_repeat):
        layers = []
        out_channels = in_channels * 2
        layers.append(Conv2dBNLeakyReLU(in_channels, out_channels, 3, 2, 1, False, norm_layer=nn.BatchNorm2d))
        for i in range(num_repeat):
            layers.append(ResidualBlock(out_channels, norm_layer=nn.BatchNorm2d))
        return nn.Sequential(*layers)



