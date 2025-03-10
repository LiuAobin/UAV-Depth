from pytorch_lightning import LightningDataModule
from torch.utils.data import RandomSampler, DataLoader

from .custom_transforms import (Compose, Normalize, ArrayToTensor,
                                RandomHorizontalFlip, RandomScaleCrop, RescaleTo, AugmentImagePair, RandomFlip)
from .train_folders import TrainSet
from .test_folders import TestSet
from .validation_folders import ValidationSet
from .midair_dataset import MidAirSet
from system.utils import print_log

dataset_map = {'midair':MidAirSet}
dataset_key = ['midair']

class BaseDataModule(LightningDataModule):
    """
    数据加载模块，负责加载训练集和验证集数据，并进行必要的预处理和变换
    """

    def __init__(self, cfg):
        """
        初始化模块
        :param cfg: 配置信息
                - cfg.dataset_name: 数据集名称。
                - cfg.load_pseudo_depth: 是否加载伪深度信息。
                - cfg.dataset_dir: 数据集的根目录路径。
                - cfg.sequence_length: 序列长度，用于加载数据。
                - cfg.skip_frames: 跳帧参数，决定在序列中跳过的帧数。
                - cfg.use_frame_index: 是否使用帧索引。
                - cfg.val_mode: 验证模式，'depth'或'photo'。
                - cfg.batch_size: 批次大小。
                - cfg.epoch_size: 每个epoch的样本数量。
                - cfg.folder_type: 数据类型。sequence or pair。
        """
        super().__init__()

        # 保存超参数
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.cfg = cfg
        # 获取训练数据大小
        self.training_size = [cfg.height, cfg.width]
        # 是否加载伪深度
        self.load_pseudo_depth = cfg.load_pseudo_depth

        # 数据变换————训练集
        if self.cfg.folder_type == 'sequence':
            self.train_transform = Compose([
                RandomHorizontalFlip(),  # 随机水平翻转
                RandomScaleCrop(),  # 随机缩放裁剪
                RescaleTo(self.training_size),  # 重缩放图像
                ArrayToTensor(),  # 将np数组转换为张量
                Normalize(),  # 归一化
            ])
        elif self.cfg.folder_type == 'pair':
            self.train_transform = Compose([
                RescaleTo(self.training_size),  # 重缩放图像
                RandomFlip(),
                ArrayToTensor(),
                AugmentImagePair()
            ])
        self.valid_transform = Compose([
            RescaleTo(self.training_size),  # 重缩放图像
            ArrayToTensor(),  # 将np数组转换为张量
            Normalize(),  # 归一化
        ])
        self.test_transform = Compose([
            RescaleTo(self.training_size),  # 重缩放图像
            ArrayToTensor(),  # 将np数组转换为张量
            Normalize(),  # 归一化
        ])

    def prepare_data(self):
        """
        准备数据的方法，由于数据集已经存储在磁盘上，因此此方法在此实现中不执行任何操作。
        如果需要下载数据或执行其他初始化任务，可以在此方法中添加。
        """
        pass

    def train_dataloader(self):
        """
        训练数据加载器
        :return:
        :rtype:
        """
        sampler = RandomSampler(self.train_dataset,
                                replacement=True,  # 运行替换采样
                                num_samples=self.cfg.batch_size * self.cfg.epoch_size)  # 计算需要的样本数量
        return DataLoader(self.train_dataset,  # 数据集
                          batch_size=self.cfg.batch_size,  # 批次大小
                          num_workers=self.cfg.num_threads,  # 加载数据时使用的线程数
                          pin_memory=True,  # 使用固定内存，提升数据加载速度
                          sampler=sampler,  # 随机采样
                          drop_last=True,  # 丢弃最后一个不完整的批次
                          )


    def val_dataloader(self):
        """
        验证数据加载器
        :return:
        :rtype:
        """
        return DataLoader(self.val_dataset,  # 使用验证数据集
                          shuffle=False,  # 打乱数据
                          num_workers=self.cfg.num_threads,  # 加载数据的工作线程数
                          batch_size=self.cfg.batch_size,  # 设置批次大小
                          pin_memory=True)  # 使用固定内存，提升数据加载速度

    def test_dataloader(self):
        """
        测试数据加载器
        :return:
        :rtype:
        """
        return DataLoader(self.test_dataset,  # 使用验证数据集
                          shuffle=False,  # 不打乱数据
                          num_workers=self.cfg.num_threads,  # 加载数据的工作线程数
                          batch_size=self.cfg.batch_size,  # 设置批次大小
                          pin_memory=True)  # 使用固定内存，提升数据加载速度

    def setup(self, stage=None):
        """
        设置数据集和数据加载器，根据训练或者验证模式选择适当的数据集
        :param stage:(str, optional) 数据准备阶段。可用于多阶段处理（如训练、验证或测试阶段）
        :return:
        """
        # 训练数据集
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        # 加载测试数据集
        self.test_dataset = self.get_test_dataset()

    def get_train_dataset(self):

        if self.cfg.dataset_name in dataset_key:
            return dataset_map[self.cfg.dataset_name](
                cfg=self.cfg,
                stage='train',
                transform=self.train_transform,
            )
        else:
            train_dataset = TrainSet(
                root=self.cfg.dataset_dir,
                train=True,
                sequence_length=self.cfg.sequence_length,
                transform=self.train_transform,
                skip_frames=self.cfg.skip_frames,
                dataset=self.cfg.dataset_name,
                use_frame_index=self.cfg.use_frame_index,
                with_pseudo_depth=self.load_pseudo_depth,
                img_suffix=self.cfg.img_suffix,
                depth_suffix=self.cfg.depth_suffix,
            )
            print(f'datasets---->train dataset len:{len(train_dataset)}')
            return train_dataset

    def get_val_dataset(self):
        if self.cfg.dataset_name in dataset_key:
            return dataset_map[self.cfg.dataset_name](
                cfg=self.cfg,
                stage='val',
                transform=self.valid_transform
            )
        else:  # 验证数据集——分两种，验证深度或光度损失
            if self.cfg.val_mode == 'photo':
                return TrainSet(
                    self.cfg.dataset_dir,  # 数据集根目录
                    train=False,  # 验证模式
                    transform=self.valid_transform,  # 验证集的预处理变换
                    sequence_length=self.cfg.sequence_length,  # 序列长度
                    skip_frames=self.cfg.skip_frames,  # 跳帧数
                    use_frame_index=self.cfg.use_frame_index,  # 是否使用帧索引
                    with_pseudo_depth=False  # 不加载伪深度
                )
            else:
                return ValidationSet(
                    self.cfg.dataset_dir,  # 数据集根目录
                    transform=self.valid_transform,  # 验证集的预处理变换
                    dataset=self.cfg.dataset_name,  # 数据集名称
                    img_suffix = self.cfg.img_suffix,
                    depth_suffix = self.cfg.depth_suffix,
                )

    def get_test_dataset(self):
        if self.cfg.dataset_name in dataset_key:
            return dataset_map[self.cfg.dataset_name](
                cfg=self.cfg,
                stage='test',
                transform=self.test_transform,
            )
        else:
            return TestSet(
                root=self.cfg.dataset_dir,
                transform=self.test_transform,
                dataset=self.cfg.dataset_name,
                img_suffix=self.cfg.img_suffix,
                depth_suffix=self.cfg.depth_suffix,
            )
