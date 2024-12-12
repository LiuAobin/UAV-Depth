import numpy as np
import torch
from path import Path

from torch.utils.data import Dataset
from imageio.v2 import imread
from system.utils import load_sparse_depth, crawl_folders


class ValidationSet(Dataset):
    """
    验证集数据加载器，文件结构如下：
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
    参数：
        root (str): 数据集根目录路径。
        transform (callable, optional): 数据增强或预处理函数，默认为None。
        dataset (str): 数据集名称，支持 'nyu'、'bonn'、'tum'、'kitti'、'ddad' 等。默认为 'nyu'。
    """

    def __init__(self, root, transform=None, dataset='kitti'):
        self.root = Path(root).joinpath('training')  # 根路径
        scene_list_path = self.root.joinpath('val.txt')  # 验证集场景列表文件
        self.scenes = [self.root.joinpath(folder.strip())
                       for folder in open(scene_list_path)]  # 获取所有场景的路径
        self.transform = transform
        self.dataset = dataset
        # 加载图像和深度图
        self.imgs, self.depths = crawl_folders(self.scenes, self.dataset)

    def __getitem__(self, index):
        """
        根据索引获取对应的图像和深度图
        :param index: 索引
        :return: 包含图像和深度图的元组
        """
        # 加载图像
        img = imread(self.imgs[index]).astype(np.float32)

        # 根据数据集类型加载深度图
        # 根据数据集类型加载深度图并归一化
        if self.dataset in ['nyu']:
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float() / 5000
        elif self.dataset in ['bonn', 'tum']:
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float() / 1000
        elif self.dataset in ['ddad', 'kitti']:
            depth = torch.from_numpy(
                load_sparse_depth(self.depths[index]).astype(np.float32))
        else:
            raise ValueError('Unsupported dataset type: {}'.format(self.dataset))
            # 如果定义了数据变换函数，则对图像进行变换
        if self.transform is not None:
            img, _ = self.transform([img], None)  # 只对图像进行变换，内参保持不变
            img = img[0]  # 获取变换后的图像

        return img, depth  # 返回图像和深度图

    def __len__(self):
        """
        返回样本数量
        """
        return len(self.imgs)