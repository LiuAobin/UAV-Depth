import numpy as np
from path import Path
from torch.utils.data import Dataset

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
    """