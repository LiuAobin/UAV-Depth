# 处理midair数数据集。
# 训练/验证/测试集比例为7/2/1
import os.path

import numpy as np

from utils.parser import create_parser,update_config
from path import Path
class MidAirLoader:
    def __init__(self,config):
        self.config = config
        self.dataset_dir = Path(config.dataset_dir)
        file = ['Kite_training','PLE_training']
        self.scenes = []
        for f in file:
            for scene in self.dataset_dir.joinpath(f).dirs():
                self.scenes.append(scene.joinpath('color_left'))
        print(self.scenes)
        print(f'total scenes collected: {len(self.scenes)}')
        # 获取所有的场景数据
        self.scenes_list = []
        for scene in self.scenes:
            self.scenes_list +=self.collect_scenes(scene)
        self.scenes_list = [path.replace(config.dataset_dir, "").replace("\\", "/") for path in self.scenes_list]
        print(self.scenes_list)


    def collect_scenes(self,scene):
        return scene.dirs()

def split_dataset(paths,train_ratio=0.8,val_ratio=0.1,test_ratio=0.1):
    # 确保比例相加为1
    assert train_ratio + val_ratio + test_ratio == 1.0, "训练、验证和测试集比例之和必须为1.0"
    np.random.shuffle(paths)
    # 计算各数据集数量
    total = len(paths)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # 确保总数一致
    # 划分数据集
    train_paths = paths[:train_size]
    val_paths = paths[train_size:train_size + val_size]
    test_paths = paths[train_size + val_size:]
    return train_paths,val_paths,test_paths

# 保存路径到文件
def save_paths_to_file(paths, file_path):
    with open(file_path, 'w') as f:
        for path in paths:
            f.write(f'{path[1:]}\n')

def main():
    args = create_parser()
    args = update_config(args)

    # 获取每个场景的文件路径
    data_loader = MidAirLoader(args)
    print('generating train/val/test lists')
    np.random.seed(args.seed)

    train_paths, val_paths, test_paths = split_dataset(data_loader.scenes_list)
    save_paths_to_file(train_paths, os.path.join(args.dataset_dir, "train.txt"))
    save_paths_to_file(val_paths, os.path.join(args.dataset_dir, "val.txt"))
    save_paths_to_file(test_paths, os.path.join(args.dataset_dir, "test.txt"))
    print("数据集划分完成，文件已保存到:", args.dataset_dir)


if __name__ == '__main__':
    main()