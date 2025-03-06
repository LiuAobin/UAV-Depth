from __future__ import division
import torch
import random
import numpy as np
import cv2

"""
用于图像变换的随机函数，
能够接收一系列输入作为参数，并进行随机且一致的变换操作
"""


class Compose(object):
    def __init__(self, transforms):
        """
        初始化操作，接收一个变换列表
        """
        self.transforms = transforms

    def __call__(self, images, intrinsics=None):
        # 对每个变换函数依次进行调用
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    """
    归一化操作
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # 初始化时设置归一化的均值的标准差
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        # 对每一张图像进行归一化处理
        for tensor in images:
            shape = tensor.size()
            if shape[0] == 3:  # 判断图像是否为RGB三通道
                for t, m, s in zip(tensor, self.mean, self.std):
                    t.sub_(m).div_(s)  # 对每个通道进行均值和标准差归一化————减均值除标准差

        return images, intrinsics


class ArrayToTensor(object):
    """
    将读取的ndarray图像(H,W,C)格式，以及内参矩阵，
    转换为torch.FloatTensor类型(C,H,W)格式，内参矩阵的Tensor
    """

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            if im.ndim < 3:  # depth
                # 如果是单通道(深度图像)，则扩展维度
                im = np.expand_dims(im, axis=0)
                tensors.append(torch.from_numpy(im).float())
            else:  # 图像
                # 将(H,W,C)转换为(C,H,W)
                im = np.transpose(im, (2, 0, 1))
                tensors.append(torch.from_numpy(im).float() / 255)  # 将像素值归一化至[0,1]
        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """
    随机水平反转图像，概率为0.5
    """

    def __call__(self, images, intrinsics):
        assert intrinsics is not None  # 确保相机内参矩阵存在
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)  # 复制内参矩阵
            output_images = [np.copy(np.fliplr(im)) for im in images]  # 对每张图像进行水平反转
            w = output_images[0].shape[1]  # 获取图像的宽度
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]  # 更新内参矩阵中的cx值
        else:
            output_images = images  # 如果不反转，则直接使用原图像
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """
    随机缩放图像，缩放最大为15%，并裁剪为原图相同大小
    """

    def __call__(self, images, intrinsics):
        assert intrinsics is not None  # 内参矩阵不能为空
        output_intrinsics = intrinsics  # 复制内参矩阵
        in_h, in_w, _ = images[0].shape  # 获取输入图像的宽度和高度
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)  # 随机生成x和y方向的缩放因子
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)  # 计算缩放后的图像尺寸

        output_intrinsics[0] *= x_scaling  # 更新内参矩阵中的fx值
        output_intrinsics[1] *= y_scaling  # 更新内参矩阵中的fy值

        scaled_images = []
        for im in images:
            if im.ndim < 3:  # 深度图像
                scaled_images.append(cv2.resize(  # 使用最近邻插值进行缩放
                    im, dsize=(scaled_w, scaled_h),
                    fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST
                ))
            else:
                scaled_images.append(cv2.resize(
                    im, dsize=(scaled_w, scaled_h),
                    interpolation=cv2.INTER_LINEAR
                ))

        # 随机生成裁剪偏移量
        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        # 裁剪图像
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
                          for im in scaled_images]
        # 更新内参矩阵中的cx和cy值
        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y
        return cropped_images, output_intrinsics


class RescaleTo(object):
    """
    将图像缩放到指定的训练或者验证尺寸
    """

    def __init__(self, output_size=[256, 832]):
        # 初始化时指定输出的尺寸
        self.output_size = output_size

    def __call__(self, images, intrinsics):
        in_h, in_w, _ = images[0].shape  # 输入图像的尺寸
        out_h, out_w = self.output_size[0], self.output_size[1]  # 目标输出大小

        # 如果输入图像尺寸和目标尺寸不同，则进行缩放
        if in_h != out_h or in_w != out_w:
            scaled_images = []
            for im in images:
                if im.ndim < 3:
                    scaled_images.append(cv2.resize(
                        im, dsize=(out_w, out_h),
                        fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST
                    ))
                else:
                    scaled_images.append(cv2.resize(
                        im, dsize=(out_w, out_h),
                        interpolation=cv2.INTER_LINEAR
                    ))
        else:
            scaled_images = images  # 尺寸相同，则直接使用原图像

        if intrinsics is not None:
            # 复制并更新内参矩阵
            output_intrinsics = intrinsics
            output_intrinsics[0] *= (out_w*1.0/in_w)
            output_intrinsics[1] *= (out_h*1.0/in_h)
        else:
            output_intrinsics = None
        return scaled_images, output_intrinsics


class RandomFlip(object):
    def __call__(self, images, intrinsics):
        if random.random() < 0.5:
            flip_images = [np.copy(np.fliplr(im)) for im in images]  # 对每张图像进行水平反转
        else:
            flip_images = images
        return flip_images, intrinsics

class AugmentImagePair(object):
    def __init__(self):
        self.gamma_low = 0.8  # 0.8
        self.gamma_high = 1.2  # 1.2
        self.brightness_low = 0.5  # 0.5
        self.brightness_high = 2.0  # 2.0
        self.color_low = 0.8  # 0.8
        self.color_high = 1.2  # 1.2

    def __call__(self, images, intrinsics):
        p = np.random.uniform(0, 1, 1)
        if p > 0.5:

            gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            colors = np.random.uniform(self.color_low, self.color_high, 3)
            for img in images:
                # randomly shift gamma
                # randomly shift brightness
                img.pow_(gamma).mul_(brightness)
                for c in range(3):
                    img[c].mul_(colors[c])
                img.clamp(0, 1)
        return images, intrinsics

