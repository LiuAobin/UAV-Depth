import re
from types import SimpleNamespace

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from path import Path
from torch.utils.data import Dataset, RandomSampler, DataLoader


from system.datasets.custom_transforms import RescaleTo, Compose, RandomFlip
from system.datasets.custom_transforms import AugmentImagePair, ArrayToTensor, RandomHorizontalFlip
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
    depth = depth.copy().astype(np.float32)  # 创建深度的副本
    # 将深度放缩到0-1之间，深度评估模式时不需要该代码
    # np.clip(depth,1,1250,depth)
    # depth = (np.log(depth)-1) / (np.log(1025)-1)
    return depth

class MidAirSet(Dataset):
    """
    MidAir数据集加载器
    1.从配置文件中加载训练集、验证集场景
    2.根据阶段，加载对应的数据
    文件结构如下：
    foggy排除
    root:
        scene_name:
            color_left:
                trajectory_xxxx:
                    xxxxxx.jpeg
            depth:
                trajectory_xxxx:
                        xxxxxx.jpeg
    内参矩阵
     $$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \quad \text{with} \quad f_x=c_x= w/2 \ \text{and} \ f_y=c_y= h/2 $$
    """
    def __init__(self, cfg, stage='train',transform=None):
        """
        初始化数据集加载器，并遍历文件夹生成样本
        Args:
            root ():
            stage ():
            sequence_length ():
            transform ():
            skip_frames ():
            img_suffix ():
            depth_suffix ():
        """
        self.samples = None
        self.root = Path(cfg.dataset_dir)
        self.stage = stage
        self.img_suffix = cfg.img_suffix
        self.depth_suffix = cfg.depth_suffix

        self.skip_frames = cfg.skip_frames
        self.sequence_length = cfg.sequence_length

        self.folder_type = cfg.folder_type
        self.cfg = cfg
        self.val_mode = cfg.val_mode
        self.transform = transform
        # 场景列表文件
        scene_list_path = self.root.joinpath(f'{stage}.txt')
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
        self.crawl_folders(self.sequence_length)

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

            # 3. 根据当前阶段来加载不同的数据
            if self.stage == 'train':
                sequence_set.extend(self._process_train(imgs, intrinsics, sequence_length))
            elif self.stage == 'val':
                sequence_set.extend(self._process_val(imgs, intrinsics, sequence_length))
            elif self.stage == 'test':
                sequence_set.extend(self._process_test(imgs))
            else:
                raise ValueError(f'Invalid stage: {self.stage}')
        # 4. 保存样本
        self.samples = sequence_set


    def __getitem__(self, index):
        """
        获取指定索引的样本
        Args:
            index ():
        Returns:
        """
        # 获取指定索引的样本
        sample = self.samples[index]
        # 根据当前阶段来加载不同的数据
        if self.stage == 'train':
            return self._get_train_sample(sample)
        elif self.stage == 'val':
            return self._get_val_sample(sample)
        elif self.stage == 'test':
            return self._get_test_sample(sample)
        else:
            raise ValueError(f'Invalid stage: {self.stage}')

    def __len__(self):
        """
        获取数据集中的样本数量
        Returns:
        """
        return len(self.samples)

    def _process_train(self,imgs,intrinsics,sequence_length):
        sequence_set = []
        if self.folder_type == 'pair':
            sequence_set.extend(
                {
                    'left_img': img,
                    'right_img': img.replace('color_left', 'color_right'),
                } for img in imgs
            )
        elif self.folder_type == 'sequence':
            if len(imgs) < sequence_length:
                return []

            # 生成数据列表的帧索引
            sample_index_list = generate_sample_index(len(imgs),
                                                      self.skip_frames,
                                                      sequence_length)
            for sample_index in sample_index_list:
                # 生成目标图像和参考图像
                sequence_set.append({'intrinsics': intrinsics,
                          'tgt_img': imgs[sample_index['tgt_idx']],
                          'ref_imgs': [ imgs[j] for j in sample_index['ref_idx'] ]
                          })
        else:
            raise ValueError(f'Invalid folder type: {self.folder_type}')
        return sequence_set

    def _process_test(self, imgs):
       return [{'tgt_img': img,
                'tgt_depth': re.sub(f'{re.escape(self.img_suffix.lstrip("*"))}',
                                    self.depth_suffix.lstrip("*"),
                                    img.replace('color_left', 'depth'))
            } for img in imgs]

    def _process_val(self, imgs, intrinsics, sequence_length):
        sequence_set = []
        if self.val_mode == 'photo':
            sequence_set.extend(self._process_train(imgs, intrinsics, sequence_length))
        elif self.val_mode == 'depth':
            sequence_set.extend(self._process_test(imgs))
        else:
            raise ValueError(f'Invalid val_mode: {self.val_mode}')
        return sequence_set

    def _get_train_sample(self, sample):
        """
                Args:
                    sample ():

                Returns:
                    left_img:
                    right_img:
                    tgt_img:
                    ref_imgs:
                    intrinsics:
                """
        if self.folder_type == 'pair':
            left_img = load_as_float(sample['left_img'])  # 左图像
            right_img = load_as_float(sample['right_img'])  # 右图像
            if self.transform is not None:
                img, _ = self.transform([left_img, right_img], None)
                left_img, right_img = img
            return left_img, right_img
        elif self.folder_type == 'sequence':
            tgt_img = load_as_float(sample['tgt_img'])  # 目标图像
            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]  # 参考图像
            intrinsics = np.copy(sample['intrinsics'])
            if self.transform is not None:
                img, intrinsics = self.transform([tgt_img] + ref_imgs, intrinsics)
                tgt_img,ref_imgs = img[0],img[1:]
            return tgt_img, ref_imgs, intrinsics

    def _get_test_sample(self, sample):
        """
        Args:
            sample ():

        Returns:
            tgt_img: 目标图像 [B,C,H,W]
            tgt_depth: 目标图像深度 [B,H,W]
        """
        tgt_img = load_as_float(sample['tgt_img'])
        tgt_depth = load_depth(sample['tgt_depth'])
        if self.transform is not None:
            img, _ = self.transform([tgt_img], None)
            tgt_img = img[0]
        return tgt_img, tgt_depth

    def _get_val_sample(self, sample):
        if self.val_mode == 'photo':
            return self._get_train_sample(sample)
        elif self.val_mode == 'depth':
            return self._get_test_sample(sample)




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
    cfg = {
        'dataset_dir' : 'E:/Depth/origin-dataset/MidAir',
    'sequence_length' : 3,
    'transform' : None,
    'skip_frames' : 1,
    'img_suffix' : '*.JPEG',
    'depth_suffix' : '*.PNG',
        'folder_type':'sequence',
        'val_mode':'depth'
    }
    train_transform = Compose([
        RescaleTo([1024,1024]),  # 重缩放图像
        RandomFlip(),
        ArrayToTensor(),
        AugmentImagePair()
    ])
    cfg = SimpleNamespace(**cfg)
    dataset = MidAirSet(
        cfg = cfg,
        stage='test',
        transform=train_transform
    )
    sampler = RandomSampler(dataset,
                            replacement=True,  # 运行替换采样
                            num_samples=1 * 1002)  # 计算需要的样本数量
    data_loader = DataLoader(dataset,  # 数据集
                      batch_size=1,  # 批次大小
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
        images,ins = batch  # 假设 dataset 返回 (image, depth)

        # 检查图像和深度图形状
        print(f"Images shape: {images.shape}")  # [B, C, H, W]
        # print(f"Depths shape: {depths.shape}")  # [B, H, W]
        #
        # # 可视化当前批次的数据
        # visualize_batch(images, depths)

        # 仅显示一个批次，退出循环
        break