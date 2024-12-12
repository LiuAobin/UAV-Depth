import random

import numpy as np
from torch.utils.data import Dataset
from path import Path
from system.utils import load_as_float


class PairSet(Dataset):
    """
    图像对序列数据加载器，文件结构如下：
        root/scene_1/0000000_0.jpg
        root/scene_1/0000001_1.jpg
        ..
        root/scene_1/cam.txt
        .
    数据对(target_image、reference_image)被组织成一个配对，
    transform函数必须接受一组图像和一个numpy数组（通常是相机内参矩阵）。
    参数:
        root (str): 数据集根目录路径。
        seed (int, optional): 随机种子，用于保证结果可复现，默认为None。
        train (bool, optional): 是否为训练集。默认为True。
        transform (callable, optional): 数据变换函数，默认为None。
    """

    def __init__(self, root, train=True, transform=None):
        self.samples = None
        self.root = Path(root)
        # 根据是否为训练集选择场景列表文件
        scene_list_path = self.root.joinpath('train.txt') if train else self.root.joinpath('val.txt')
        # 获取所有场景的文件路径
        self.scenes = [self.root.joinpath(folder.strip()) for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders()  # 遍历文件夹，收集数据对

    def crawl_folders(self):
        """
        遍历所有场景文件夹，加载配对的图像和相机内参。
        每个场景文件夹包含多对图像（tgt 和 ref），以及对应的内参文件。
        """
        pair_set = []  # 存储所有数据对（图像配对及内参）
        for scene in self.scenes:
            # imgs = sorted(scene.files('*.jpg'))  # 加载场景中的所有图像（按文件名排序）
            imgs = sorted(scene.files('*.jpg'))  # 按文件名排序加载所有图像
            intrinsics = sorted(scene.files('*.txt'))  # 按文件名排序加载所有内参文件（.txt格式）

            # 遍历图像列表，按步长为2的方式取图像配对
            for i in range(0, len(imgs) - 1, 2):
                # 加载每一对图像的内参文件，并将其转为3x3的矩阵
                intrinsic = np.genfromtxt(intrinsics[int(i / 2)]).astype(np.float32).reshape((3, 3))
                # 创建一个字典，包含当前图像对的内参、目标图像和参考图像
                sample = {'intrinsics': intrinsic, 'tgt': imgs[i], 'ref_imgs': [imgs[i + 1]]}
                pair_set.append(sample)  # 将图像对加入样本列表
        random.shuffle(pair_set)  # 随机打乱图像对
        self.samples = pair_set  # 将所有样本保存为实例变量

    def __getitem__(self, index):
        """
        根据索引获取图像对和对应的相机内参。

        参数:
            index (int): 索引，表示获取哪一对图像。

        返回:
            tuple: 包含目标图像（tgt_img）、参考图像（ref_imgs）、内参矩阵（intrinsics）和内参的逆矩阵（inv_intrinsics）的元组。
                - tgt_img (torch.Tensor): 目标图像（经过预处理或变换后的图像）。
                - ref_imgs (list of torch.Tensor): 参考图像列表（经过预处理或变换后的图像）。
                - intrinsics (np.ndarray): 相机内参矩阵。
                - inv_intrinsics (np.ndarray): 相机内参矩阵的逆矩阵。
        """
        sample = self.samples[index]  # 获取指定索引的样本
        tgt_img = load_as_float(sample['tgt'])  # 加载目标图像（并转换为浮点格式）
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]  # 加载参考图像列表

        # 如果定义了数据变换函数，则对图像进行变换
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))  # 执行数据变换
            tgt_img = imgs[0]  # 获取变换后的目标图像
            ref_imgs = imgs[1:]  # 获取变换后的参考图像
        else:
            intrinsics = np.copy(sample['intrinsics'])  # 如果没有变换函数，直接使用原始内参
        # 返回目标图像、参考图像、内参和内参的逆矩阵
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        """
        返回数据集中的样本数量。
        返回:
            int: 数据集中的样本数量，即图像对的数量。
        """
        return len(self.samples)  # 返回样本数量，即数据对的数量
