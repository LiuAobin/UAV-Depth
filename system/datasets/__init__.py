from .custom_transforms import (Compose, Normalize, ArrayToTensor,
                                RandomHorizontalFlip, RandomScaleCrop, RescaleTo)
from .pair_folders import PairSet
from .test_folders import TestSet
from .train_folders import TrainSet
from .validation_folders import ValidationSet
from .base_data import BaseDataModule
"""
数据集相关代码实现参考sc-depth-pl <https://github.com/JiawangBian/sc_depth_pl>
"""
dataset_map = {
    'train': TrainSet,
    'val': ValidationSet,
    'test': TestSet,
    'pair_train': PairSet,
    'pair_val': PairSet,
}
__all__ = [
    'Compose', 'Normalize', 'ArrayToTensor',
    'RandomHorizontalFlip', 'RandomScaleCrop', 'RescaleTo',
    'TrainSet', 'TestSet', 'ValidationSet',
    'dataset_map',
    'BaseDataModule'

]
