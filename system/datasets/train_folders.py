import numpy as np
from path import Path
from torch.utils.data import Dataset
from system.utils import generate_sample_index, load_as_float


class TrainSet(Dataset):
    """
    一个序列数据加载器，文件结构如下：
    root/scene_1/0000000.jpg
    root/scene_1/0000001.jpg
    ...
    root/scene_1/cam.txt
    root/scene_2/0000000.jpg
    ...
    变换函数应接受图像列表和一个内参矩阵（通常是numpy数组）作为输入。
    参数:
        root (str): 数据集的根目录路径
        train (bool): 是否加载训练集，默认为True
        sequence_length (int): 每个序列的长度，默认为3
        transform (callable): 用于数据增强的变换函数，默认为None
        skip_frames (int): 跳帧数，决定参考图像和目标图像之间的间隔，默认为1
        dataset (str): 数据集的名称，默认为'kitti'
        use_frame_index (bool): 是否使用自定义的帧索引，默认为False
        with_pseudo_depth (bool): 是否使用伪深度图，默认为False
    """

    def __init__(self,
                 root,
                 train=True,
                 sequence_length=3,
                 transform=None,
                 skip_frames=1,
                 dataset='kitti',
                 use_frame_index=False,
                 with_pseudo_depth=False,
                 img_suffix='*.jpg',
                 depth_suffix='*.png'):
        self.samples = None
        self.img_suffix = img_suffix  # 图像后缀
        self.depth_suffix = depth_suffix  # 深度图后缀
        self.root = Path(root).joinpath('training')  # 数据集的根目录
        scene_list_path = self.root.joinpath('train.txt') if train else self.root.joinpath('val.txt')
        self.scenes = [self.root.joinpath(folder.strip()) for folder in open(scene_list_path)]  # 获取所有场景路径
        self.transform = transform  # 数据变换
        self.k = skip_frames
        self.dataset = dataset
        self.with_pseudo_depth = with_pseudo_depth  # 是否使用伪深度图
        self.use_frame_index = use_frame_index  # 是否使用帧索引
        self.crawl_folders(sequence_length)  # 遍历文件夹并生成样本

    def crawl_folders(self, sequence_length):
        """
        遍历数据集文件夹，并生成图像序列
        :param sequence_length: 序列长度
        :return:
        """
        sequnece_set = []
        for scene in self.scenes:
            # 1. 读取相机内参
            intrinsics = np.genfromtxt(scene.joinpath('cam.txt')).astype(np.float32).reshape((3, 3))
            # 2. 获取图像文件，按文件名排序
            imgs = sorted(scene.files(self.img_suffix))
            if self.use_frame_index:
                # 如果使用帧索引，则根据帧索引文件重新排序图像
                frame_index = [int(index) for index in open(scene.joinpath('frame_index.txt'))]
                imgs = [imgs[d] for d in frame_index]

            # 如果图像数目小于序列长度，跳过该场景
            if len(imgs) < sequence_length:
                continue

            sample_index_list = generate_sample_index(len(imgs),
                                                      self.k, sequence_length)
            for sample_index in sample_index_list:
                # 为每个样本生成目标图像和参考图像
                sample = {'intrinsics': intrinsics,
                          'tgt_img': imgs[sample_index['tgt_idx']],  # 目标图像
                          'ref_imgs': []}  # 参考图像

                # 参考图像列表
                for j in sample_index['ref_idx']:
                    sample['ref_imgs'].append(imgs[j])
                sequnece_set.append(sample)
        self.samples = sequnece_set  # 将所有样本保存到列表中

    def __getitem__(self, index):
        """
        获取指定索引的样本
        """
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt_img'])  # 目标图像
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]  # 参考图像

        # 如果定义了数据变换，则对图像进行变换
        if self.transform is not None:
            img, intrinsics = self.transform([tgt_img] + ref_imgs,
                                             np.copy(sample['intrinsics']))
            tgt_img = img[0]
            ref_imgs = img[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics

    def __len__(self):
        """
        获取数据集中的样本数量
        """
        return len(self.samples)


def main():
    train_dataset = TrainSet(root='E:/Depth/target_dataset/SC-Depth/KITTI',
                             train=False,
                             sequence_length=3,
                             transform=None,
                             skip_frames=1,
                             dataset='kitti',
                             use_frame_index=False,
                             with_pseudo_depth=False)
    print(len(train_dataset))


if __name__ == '__main__':
    main()
