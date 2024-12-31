import re

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from path import Path
from torch.utils.data import Dataset, RandomSampler, DataLoader

from system.utils import load_as_float,generate_sample_index

"""
MidAir数据集中关于深度的处理，请参考
<https://github.com/montefiore-institute/midair-dataset>
"""

def open_float16(image_path):
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img


def load_depth(image_path):
    depth = open_float16(image_path)
    depth = depth.copy()  # 创建深度的副本
    np.clip(depth,1,1250,depth)
    depth = (np.log(depth)-1) / (np.log(1025)-1)
    return depth

class MidAirSet(Dataset):
    """
    MidAir数据集加载器
    1.从配置文件中加载训练集、验证集场景
    2.根据阶段，加载对应的数据
    文件结构如下：
    root:
        scene_name:
            color_left:
                trajectory_xxxx:
                    xxxxxx.jpeg
            depth:
                trajectory_xxxx:
                        xxxxxx.jpeg
    内参矩阵
     $$\mathbf{K} = \begin{bmatrix} f_x & 0 & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \quad \text{with} \quad f_x=c_x= w/2 \ \text{and} \ f_y=c_y= h/2 $$
    """
    def __init__(self,root,mode='train',
                 sequence_length=3,
                 transform=None,
                 skip_frames=1,
                 img_suffix='*.jpeg',
                 depth_suffix='*.png'):
        """
        初始化数据集加载器，并遍历文件夹生成样本
        Args:
            root ():
            mode ():
            sequence_length ():
            transform ():
            skip_frames ():
            use_frame_index ():
            img_suffix ():
            depth_suffix ():
        """
        self.samples = None
        self.root = Path(root)
        self.mode = mode
        self.img_suffix = img_suffix
        self.depth_suffix = depth_suffix

        self.skip_frames = skip_frames
        self.sequence_length = sequence_length
        self.transform = transform
        # 场景列表文件
        scene_list_path = self.root.joinpath(f'{mode}.txt')
        # 获取所有场景路径
        self.scenes = [self.root.joinpath(folder.strip())
                       for folder in open(scene_list_path)]

        # WidAir的相机内参矩阵
        self.width = 1024
        self.height = 1024
        self.intrinsics = np.asarray([self.width/2,0,self.width/2,
                                      0,self.height/2,self.height/2,
                                      0,0,1]).astype(np.float32).reshape((3,3))
        # 遍历文件夹并生成样本
        self.crawl_folders(sequence_length)

    def crawl_folders(self,sequence_length):
        """
        遍历数据集文件夹，并生成图像序列
        Args:
            sequence_length (): 序列长度
        Returns:
        """
        sequence_set = []
        for scene in self.scenes:
            # 1. 内参矩阵
            intrinsics = self.intrinsics
            # 2. 获取图像文件
            imgs = sorted(scene.files(self.img_suffix))
            # 根据当前阶段来加载不同的数据
            if self.mode == 'train':
                if len(imgs) < sequence_length:
                    continue
                # 生成数据列表的帧索引
                sample_index_list = generate_sample_index(len(imgs),
                                                          self.skip_frames,
                                                          sequence_length)
                for sample_index in sample_index_list:
                    # 生成目标图像和参考图像
                    sample = {'intrinsics': intrinsics,
                              'tgt_img': imgs[sample_index['tgt_idx']],
                              'ref_imgs': []}
                    # 参考图像列表
                    for j in sample_index['ref_idx']:
                        sample['ref_imgs'].append(imgs[j])
                    sequence_set.append(sample)
            else:  # 非训练阶段-测试和验证阶段
                for sample_path in imgs:
                    tgt_depth = sample_path.replace('color_left', 'depth')
                    tgt_depth = re.sub(f'{re.escape(self.img_suffix.lstrip("*"))}',
                                       self.depth_suffix.lstrip("*"),
                                       tgt_depth)
                    sample = {'intrinsics':intrinsics,
                              'tgt_img':sample_path,
                              'tgt_depth':tgt_depth}
                    sequence_set.append(sample)

        self.samples = sequence_set  # 将所有样本加到列表中

    def __getitem__(self, index):
        """
        获取指定索引的样本
        Args:
            index ():
        Returns:
        """
        # 获取指定索引的样本
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt_img'])  # 目标图像
        # 根据当前阶段来加载不同的数据

        intrinsics = np.copy(sample['intrinsics'])
        if self.mode == 'train':
            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]  # 参考图像
            # 矩阵变化
            if self.transform is not None:
                img, intrinsics = self.transform([tgt_img] + ref_imgs, intrinsics)
                tgt_img = img[0]
                ref_imgs = img[1:]
            return tgt_img,ref_imgs,intrinsics
        else:
            # 矩阵变换
            tgt_depth = load_depth(sample['tgt_depth'])  # 目标图像参考深度
            if self.transform is not None:
                img, _ = self.transform([tgt_img], None)  # 只对图像进行变换，内参保持不变
                tgt_img = img[0]  # 获取变换后的图像
            return tgt_img,tgt_depth,intrinsics



    def __len__(self):
        """
        获取数据集中的样本数量
        Returns:
        """
        return len(self.samples)

def visualize_batch(images, depths):
    """
    可视化一个批次的数据 (RGB 图像和深度图)。
    :param images: 批次中的 RGB 图像，形状为 (B, C, H, W)
    :param depths: 批次中的深度图，形状为 (B, H, W)
    """
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))

    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
        depth = depths[i].cpu().numpy()  # 深度图为 (H, W)

        # 归一化到 [0, 1] 范围 (方便显示)
        image = (image - image.min()) / (image.max() - image.min())
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        # 显示 RGB 图像
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("RGB Image")
        axes[i, 0].axis("off")

        # 显示深度图
        im = axes[i, 1].imshow(depth, cmap="viridis")
        axes[i, 1].set_title("Depth Map")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataset = MidAirSet(
        root='E:/Depth/origin-dataset/MidAir',
        mode='test',
        sequence_length=3,
        transform=None,
        skip_frames=1,
        img_suffix='*.JPEG',
        depth_suffix='*.PNG'
    )
    sampler = RandomSampler(dataset,
                            replacement=True,  # 运行替换采样
                            num_samples=4 * 1002)  # 计算需要的样本数量
    data_loader = DataLoader(dataset,  # 数据集
                      batch_size=4,  # 批次大小
                      num_workers=6,  # 加载数据时使用的线程数
                      pin_memory=True,  # 使用固定内存，提升数据加载速度
                      sampler=sampler,  # 随机采样
                      drop_last=True,  # 丢弃最后一个不完整的批次
                      )
    print(dataset.__len__())
    print(data_loader.__len__())
    # 遍历 DataLoader
    for batch in data_loader:
        # 从批次中获取图像和深度图
        images, depths,ins = batch  # 假设 dataset 返回 (image, depth)

        # 检查图像和深度图形状
        print(f"Images shape: {images.shape}")  # [B, C, H, W]
        print(f"Depths shape: {depths.shape}")  # [B, H, W]

        # 可视化当前批次的数据
        visualize_batch(images, depths)

        # 仅显示一个批次，退出循环
        break