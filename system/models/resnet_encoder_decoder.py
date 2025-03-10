import numpy as np
from pyexpat import features

import torch
from torch import nn
from torchvision import models
from torch.utils import model_zoo
import torch.nn.functional as F
class ResNetMultiImageInput(models.ResNet):
    """
    适用于多帧输入的ResNet模型，支持多个图像帧作为输入。
    该模型继承自torchversion的ResNet，实现版本与原生ResNet基本相同
    修改：第一层卷积层(conv1)以适配多帧输入
    """
    def __init__(self, block,layers,num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block,layers)
        self.inplanes = 64  # 初始化输入通道数
        # 修改第一层卷积，使其适配num_input_images帧输入
        self.conv1 = nn.Conv2d(
            num_input_images*3,# 输入通道
            64, # 输出通道
            kernel_size=7,stride=2,padding=3,
            bias=False)
        # 保持后续层不变
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


def resnet_multi_image_input(num_layers, pretrained=False,num_input_images=1):
    """
    创建一个支持多输入的ResNet模型
    Args:
        num_layers: ResNet层数
        pretrained: 是否加载预训练模型
        num_input_images: 输入的图像帧数
    Returns:
        model(ResNetMultiImageInput): 适配多帧输入的ResNet模型
    """
    assert num_layers in [18,50],"Can only run with 18 or 50 layer resnet"
    # 定义不同层数的ResNet结构
    blocks = {18:[2,2,2,2],
              50:[3,4,6,3]}[num_layers]
    block_type = {18:models.resnet.BasicBlock,
                  50:models.resnet.BasicBlock}[num_layers]
    # 创建模型
    model = ResNetMultiImageInput(block_type,blocks,num_input_images=num_input_images)

    if pretrained:
        # 加载 ImageNet 预训练模型
        loaded = model_zoo.load_url(models.resnet.model_urls[f'resnet{num_layers}'])
        # 由于 conv1 需要适配多通道输入，需要手动调整权重
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']]*num_input_images,1) / num_input_images
        # 加载调整后的权重
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """
    ResNet作为深度估计任务的编码器
    可选ResNet层数：18，34，50，101，152
    是否使用预训练模型
    运行多输入通道（num_input_images>1）
    """
    def __init__(self,num_layers=50,pretrained=True,num_input_images=1):
        super(ResnetEncoder,self).__init__()
        # 定义不同ResNet版本
        resnets = {
            18:models.resnet18,
            34:models.resnet34,
            50:models.resnet50,
            101:models.resnet101,
            152:models.resnet152
        }
        if num_layers not in resnets:
            raise ValueError(f"{num_layers} 不是有效的ResNet层数")
        # 定义ResNet编码器的通道数(ResNet50及其以上的Bottleneck结构需要调整通道数)
        self.num_ch_encoder = np.array([64,64,128,256,512])
        if num_layers > 34:
            self.num_ch_encoder[1:] *= 4  # 【64,256,512,1024,2048】

        # 处理多输入通道的情况
        if num_input_images > 1:
            self.encoder = resnet_multi_image_input(num_layers,pretrained,num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)
        # 去除ResNet的全连接层
        self.encoder.fc = None

    def forward(self,input_image):
        """
        前向传播：提取多尺度特征
        Args:
            input_image: 形状为 (B, C, H, W) 的输入图像
        Returns:
            多尺度特征列表 [C1, C2, C3, C4, C5]
        """
        self.features = []
        # 逐层提取特征
        # 0----64
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        # 1----256
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        self.features.append(x)
        # 2----512
        x = self.encoder.layer2(x)
        self.features.append(x)
        # 3----1024
        x = self.encoder.layer3(x)
        self.features.append(x)
        # 4----2048
        x = self.encoder.layer4(x)
        self.features.append(x)
        return self.features


class UpSampleBN(nn.Module):
    """
    上采样模块。使用双线性插值进行上采样
    """
    def __init__(self,in_channels,out_channels):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels,out_channels,
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x,concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)],
                             mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)



class DecoderBN(nn.Module):
    """
    解码器网络，使用批归一化(BatchNorm)进行特征处理
    该网络接收来自编码器的不同尺度特征，并通过一系列上采样层进行逐步恢复
    最终输出深度特征
    """
    def __init__(self,num_features=512,num_classes=32,bottleneck_features=2048):
        super(DecoderBN,self).__init__()
        features = int(num_features)
        # 1x1 卷积，调整通道数
        self.conv2 = nn.Conv2d(bottleneck_features,features,
                               kernel_size=1,stride=1,padding=1)
        # 上采样模块，逐步恢复空间分辨率
        self.up1 = UpSampleBN(in_channels=features//1+1024,out_channels=features//2) # in:512+1024 out:256
        self.up2 = UpSampleBN(in_channels=features//2+512,out_channels=features//4) # in:256+512 out:128
        self.up3 = UpSampleBN(in_channels=features//4+256,out_channels=features//8) # in:128+256 out:64
        self.up4 = UpSampleBN(in_channels=features//8+64,out_channels=features//16) # in:46+64 out:32=num_classes

        self.conv3 = nn.Conv2d(features//16,num_classes,
                               kernel_size=3,stride=1,padding=1)

    def forward(self, features):
        """
       前向传播过程：
       1. 接收编码器的五层特征图（x_block0 至 x_block4）。C=[64,256,512,1024,2048]
       2. 通过 1x1 卷积调整最深层特征的通道数。
       3. 逐步进行上采样，并在每一步拼接对应尺度的跳跃连接特征。
       4. 最终通过 3x3 卷积生成输出。
       """
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[0], features[1], features[2], features[3], features[4])
        # 通过 1x1 卷积调整 Bottleneck 特征的通道数
        x_d0 = self.conv2(x_block4) # 2048->512
        # 依次进行上采样，并拼接相应的编码器特征
        x_d1 = self.up1(x_d0,x_block3) # 512+1024-> 256
        x_d2 = self.up2(x_d1,x_block2) # 256+512-> 128
        x_d3 = self.up3(x_d2,x_block1) # 128+256-> 64
        x_d4 = self.up4(x_d3,x_block0) # 64+64-> 32
        # 通过最终卷积生成预测输出
        out = self.conv3(x_d4)
        return out


class ResnetEncoderDecoder(nn.Module):
    """ResNet作为编码器+解码器
    input shape is [B,3,H,W]
    first: extracts images features
    second: decodes and up sampling
    output: high resolution immediate features S and shape is [B,C,h,w]
    set h=H/2 and w=W/2
    """
    def __init__(self,num_layers=50,num_features=512,model_dim=32):
        super(ResnetEncoderDecoder,self).__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers,pretrained=True,num_input_images=1)
        self.decoder = DecoderBN(num_features=num_features,num_classes=model_dim,bottleneck_features=2048)

    def forward(self,x,**kwargs):
        x = self.encoder(x)
        x = self.decoder(x,**kwargs)
        return x

