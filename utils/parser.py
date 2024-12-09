import argparse
from .config_utils import Config

"""
实验所需要的各种参数都在此文件进行配置
"""


def create_parser():
    parser = argparse.ArgumentParser(description="深度估计实验相关参数")
    # -------------------------------------------预处理数据集所需要的相关参数信息---------------------------------------------
    parser.add_argument('--config_file', '-c', type=str,
                        default='./data/configs/kitti_raw.py',
                        help='模型配置文件所在路径')
    parser.add_argument('--origin_data_dir', '-o', type=str,
                        default='E:/Depth_Estimation/KITTI/KITTI_Raw_Data',
                        help='数据集原始文件目录')
    parser.add_argument("--dump_dir", type=str,
                        default='E:/Depth_Estimation/KITTI/SC-Depth',
                        help='处理后的数据集的存放路径')
    parser.add_argument("--static_frames", type=str,
                        default='./configs/eigen_kitti/static_frames.txt',
                        help='指定要丢弃的静态帧列表')
    parser.add_argument("--test_scenes", type=str,
                        default='./configs/eigen_kitti/test_scenes.txt',
                        help='测试场景所需要的帧')
    parser.add_argument('--data_format', '-f', type=str,
                        default='kitti_raw',
                        choices=["kitti_raw", "cityscapes", "kitti_odom"],
                        help='数据集格式')
    parser.add_argument("--with_depth", action='store_true',
                        help='如果存在深度信息，则保存深度真值，供验证使用')
    parser.add_argument("--no_train_gt", action='store_false',
                        help='是否删除训练数据集中的深度真值以节省空间')
    parser.add_argument("--with_pose", action='store_true',
                        help='如果存在位姿信息，则保存姿态地面真值，以供验证使用')
    parser.add_argument("--height", type=int, default=256,
                        help='图像高度')
    parser.add_argument("--width", type=int, default=832,
                        help='图像宽度')
    parser.add_argument("--depth_size_ratio", type=int, default=1,
                        help='深度大小的缩放比例')
    parser.add_argument("--num_threads", type=int, default=6,
                        help='处理/加载数据时，使用的线程数')
    parser.add_argument("--min_speed", type=int, default=2,
                        help='最小速度,当没有静态帧文件时，使用该参数，去除速度小于该帧的文件')
    parser.add_argument('--seed', type=int,
                        default=8848,
                        help='随机数种子，确保结果可复现')
    return parser.parse_args()


def load_config(filename: str = None):
    """load and print config"""
    print('loading config from ' + filename + ' ...')
    try:
        configfile = Config(filename=filename)
        config = configfile._cfg_dict
    except (FileNotFoundError, IOError):
        config = dict()
        print('warning: fail to load the config!')
    return config


def update_config(args):
    config = load_config(args.config_file)
    assert isinstance(args, argparse.Namespace) and isinstance(config, dict)
    for k in config.keys():
        if hasattr(args, k):
            if getattr(args, k) != config[k] and getattr(args, k) is not None:
                print(f'overwrite config key -- {k}: {getattr(args, k)} -> {config[k]}')
                setattr(args, k, config[k])
        else:
            setattr(args, k, config[k])
    return args


def main():
    __package__ = "utils"
    args = create_parser()
    args = update_config(args)
    print(args)


if __name__ == "__main__":
    main()
