import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image


def visualize_image(image):
    """
    可视化输入的tensor图像，进行反归一化处理，恢复原始像素值
    Args:
        image (torch.Tensor): 输入图像，尺寸为(3,H,W)
    Returns:
        torch.Tensor:反归一化后的图像

    """
    # 反归一化操作，将图像从[0,1]区间
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)

    x = image * std + mean  # 反归一化
    x = torch.clamp(x, 0, 1)  # 限制到 [0,1] 范围
    return x  # 反归一化公式image = image*std+mean


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    可视化深度图，转换为彩色图并返回
    Args:
        depth (torch.Tensor): 输入的深度图，形状为(H,W)
        cmap (cv2.COLORMAP): 可选的openCV颜色映射，默认为JET
    Returns:
        转换后的深度图，尺寸为(3,H,W),范围[0,1]

    """
    x = depth.cpu().numpy()  # 将深度图放到cpu并转换为np数组
    x = np.nan_to_num(x)  # 替换nan为0，避免后续计算出现无效值
    # 获取深度图中的最大值和最小值
    mi = np.min(x)
    ma = np.max(x)
    # 对深度图进行归一化
    x = (x-mi)/(ma-mi+1e-8)  # x_normalized = (x - min) / (max-min)
    # 将深度值从[0,1]范围映射到[0,255]并转换为uint8 便于颜色映射
    x = (255*x).astype(np.uint8)
    # 使用opencv的applyColorMap将灰度深度图转换为伪彩色图
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # 转换为张量，并将图像范围从[0,255]缩放到[0,1]
    return x_
