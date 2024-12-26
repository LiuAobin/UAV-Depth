import numpy as np
import torch
from path import Path
from torch.utils.data import Dataset
from imageio.v2 import imread
from system.utils import crawl_folder, load_sparse_depth


class TestSet(Dataset):
    """
    测试集数据加载器，文件结构如下：
        root/color/0000000.png
        root/depth/0000000.npz 或 0000000.png
    参数:
        root (str): 数据集根目录路径。
        transform (callable, optional): 数据增强或预处理函数，默认为None。
        dataset (str): 数据集名称，支持 'nyu', 'kitti', 'ddad'。默认为 'nyu'。
    """

    def __init__(self, root, transform=None, dataset='nyu',img_suffix='*.png',depth_suffix='*.png'):
        self.root = Path(root).joinpath('testing')
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depths = crawl_folder(self.root, self.dataset,img_suffix=img_suffix,depth_suffix=depth_suffix)

    def __getitem__(self, index):
        """
        根据指定索引获取对应的图像和深度图
        :param index: 索引
        :return: tuple:
            - img: 归一化后的图像
            - depth: 归一化处理后的深度图
        """
        # 加载图像并转换为float32类型
        img = imread(self.imgs[index]).astype(np.float32)

        # 根据数据集类型加载深度图
        if self.dataset in ['nyu']:  # NYU深度图，除以5000进行归一化
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float() / 5000
        elif self.dataset in ['kitti']:  # KITTI深度图，直接加载npy文件
            depth = torch.from_numpy(
                np.load(self.depths[index]).astype(np.float32))
        elif self.dataset in ['ddad']:
            depth = torch.from_numpy(
                load_sparse_depth(self.depths[index]).astype(np.float32))
        else:
            raise ValueError('Unsupported dataset type: {}'.format(self.dataset))

        # 如果定义了数据变换，则应用于图像
        if self.transform is not None:
            img, _ = self.transform([img], None)  # 只对图像进行变换，内参保持不变
            img = img[0]  # 获取变换后的图像
        return img, depth

    def __len__(self):
        """
        获取数据集中的样本数量
        """
        return len(self.imgs)


def main():
    import matplotlib.pyplot as plt

    # 加载稠密深度图, `filename` 是你的 npz 文件的路径
    filename = r"E:\Depth\target_dataset\SC-Depth\KITTI\2011_09_26_drive_0104_sync_02\0000000000.npy"
    dense_depth = load_sparse_depth(filename)

    # 可视化深度图
    plt.figure(figsize=(10, 10))
    plt.imshow(dense_depth, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
