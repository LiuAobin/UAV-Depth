import argparse
from .config_utils import Config

"""
实验所需要的各种参数都在此文件进行配置
"""


def create_parser():
    parser = argparse.ArgumentParser(description="深度估计实验相关参数")
    # -------------------------------------------常用设置---------------------------------------------
    # 基础配置  --config_file是关键
    parser.add_argument('--config_file', '-c', type=str,
                        default='./data/configs/midair.py',
                        help='额外的配置文件所在路径')
    parser.add_argument('--work_dir', type=str,
                        default='./work_dirs',
                        help='工作目录')
    parser.add_argument('--exp_name', type=str,
                        default='kitti_SC_Depth',
                        help='实验名称，训练或测试过程中所有输出均在{work_dir}/{exp_name}下')
    parser.add_argument('--seed', type=int,
                        default=8848,
                        help='随机数种子，确保结果可复现')
    parser.add_argument("--num_threads", type=int,
                        default=6,
                        help='处理/加载数据时，使用的线程数')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='使用cpu or cuda 进行张量计算')
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    # 实验设置
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help="恢复训练的检查点文件")
    parser.add_argument('--resume',
                        action='store_true',
                        help='是否恢复训练')
    parser.add_argument('--log_step', type=int,
                        default=1,
                        help='设置每隔多久显示一次日志')

    # 数据集信息
    parser.add_argument('--dataset_dir', type=str,
                        default=None,
                        help="数据集根路径")
    parser.add_argument('--dataset_name', type=str,
                        default='kitti',
                        help='数据集名称')
    parser.add_argument("--channels", type=int,
                        default=3,
                        help='通道数')
    parser.add_argument("--height", type=int,
                        default=256,
                        help='图像高度')
    parser.add_argument("--width", type=int,
                        default=832,
                        help='图像宽度')
    parser.add_argument('--sequence_length', type=int,
                        default=3,
                        help='训练时使用的序列长度')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='图像序列的帧间间隔')
    parser.add_argument('--use_frame_index', action='store_true',
                        help='过滤掉视频中的静态帧')
    # 训练信息
    parser.add_argument('--epochs', type=int,
                        default=500,
                        help='训练总轮数')
    parser.add_argument('--epoch_size', type=int,
                        default=2000,
                        help='每轮的训练样本数量，如果没有设置，则根据数据集大小自动调整')
    parser.add_argument('--batch_size', type=int,
                        default=4,
                        help='每个小批次的大小')
    parser.add_argument('--val_mode', type=str, default='photo',
                        choices=['photo', 'depth'], help='验证模式')
    # 模型方法
    parser.add_argument('--method', type=str,
                        default='sc-depth',
                        help='训练所使用的模型')
    parser.add_argument('--test', action='store_true',
                        default=False,
                        help='测试模型')
    parser.add_argument('--no_display_method_info', action='store_true',
                        default=False,
                        help='是否显示方法信息,默认显示')
    parser.add_argument('--fps', default=True, type=bool,
                        help='是否显示推理速度')
    parser.add_argument('--metric_for_bestckpt', default='val_loss', type=str,
                        help='检查那个损失指标作为保持最佳检查点的信息')

    # -------------------------------------------模型训练所需要的相关参数信息---------------------------------------------

    # 训练设置 + 优化器设置
    parser.add_argument('--opt', type=str,
                        default='adamw',
                        help='优化器')
    parser.add_argument('--lr_scheduler', type=str,
                        default='onecycle',
                        help='学习率调度器')
    parser.add_argument('--opt_eps', default=None,
                        type=float,
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--momentum', type=float,
                        default=0.9,
                        help='SGD动量参数，Adam的alpha参数')
    parser.add_argument('--opt_betas', default=None,
                        help='Adam优化器的beta参数')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')
    parser.add_argument('--clip_gard', type=float,
                        default=0,
                        help='梯度裁剪')
    parser.add_argument('--lr', type=float,
                        default=1e-3,
                        help='初始化学习率')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5,
                        help='梯度衰减')
    parser.add_argument('--min_lr', type=float,
                        default=1e-6,
                        help='最小的学习率')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--decay_epoch', type=float, default=20, metavar='N',
                        help='学习率衰减周期')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='学习率衰减速率')



    # ---其他参数
    parser.add_argument('--padding_mode', type=str,
                        default='zeros', choices=['zeros', 'border'],
                        help='图像扭曲时的填充模式，对于光度差异化来说，当坐标超出目标图像范围时使用该模式.'
                             'zeros表示在超出目标图像范围时梯度为0，border表示仅在x或y坐标超出时梯度为0')
    parser.add_argument('--with_gt', action='store_true',
                        help='验证时是否使用地面真值,需要保存的npy文件，即在预处理数据时设置with_depth=True')

    parser.add_argument('--sched', default='onecycle', type=str,
                        help='LR scheduler (default: "onecycle")')
    # 损失函数相关
    parser.add_argument('--no_ssim', action='store_true',
                        help='use ssim in photometric loss')
    parser.add_argument('--no_auto_mask', action='store_true',
                        help='masking invalid static points')
    parser.add_argument('--no_dynamic_mask',
                        action='store_true', help='masking dynamic regions')
    parser.add_argument('--no_min_optimize', action='store_true',
                        help='optimize the minimum loss')
    parser.add_argument('--photo_weight', type=float,
                        default=1.0, help='photometric loss weight')
    parser.add_argument('--geometry_weight', type=float,
                        default=0.1, help='geometry loss weight')
    parser.add_argument('--smooth_weight', type=float,
                        default=0.1, help='smoothness loss weight')

    # -------------------------------------------预处理数据集所需要的相关参数信息--暂时用不上---------------------------------------------
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
    parser.add_argument("--no_train_gt", action='store_true',
                        help='是否删除训练数据集中的深度真值以节省空间')
    parser.add_argument("--with_pose", action='store_true',
                        help='如果存在位姿信息，则保存姿态地面真值，以供验证使用')

    parser.add_argument("--depth_size_ratio", type=int,
                        default=1,
                        help='深度大小的缩放比例')

    parser.add_argument("--min_speed", type=int,
                        default=2,
                        help='最小速度,当没有静态帧文件时，使用该参数，去除速度小于该帧的文件')
    # 特定于模型的一些参数
    parser.add_argument('--load_pseudo_depth',
                        action='store_true',
                        help='是否使用伪深度')
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
