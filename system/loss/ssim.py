import torch
from torch import nn


class SSIM(nn.Module):
    """定义结构相似性指数(structural similarity index)类，用于计算图像对之间的ssim损失"""

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7  # 设置窗口大小
        # 定义用于计算均值和方差的卷积池化操作
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        # 反射填充操作
        self.refl = nn.ReflectionPad2d(k // 2)

        # 定义SSIM公式中的常量
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        """计算输入图像x和y之间的SSIM损失"""
        # 反射填充，处理边界情况
        x = self.refl(x)
        y = self.refl(y)

        # 计算均值
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        # 计算方差和协方差
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        # 计算SSIM的分子和分母
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # 返回归一化后的SSIM值，并将其裁剪到[0,1]之间
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

