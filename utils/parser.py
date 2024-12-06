import argparse
import os
"""
实验所需要的各种参数都在此文件进行配置
"""


def create_parser():
    parser = argparse.ArgumentParser(description="深度估计实验相关参数")
    parser.add_argument('--config_file', '-c', type=str,
                        default='E:\Depth_Estimation\My_Code\SC-Depth\configs\kitti_raw.txt',
                        help='模型配置文件所在路径')
    return parser


def update_config(configs):
    if not os.path.exists(configs.config_file):
        print("配置文件不存在")
        return
    with open(configs.config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            if '=' not in line:
                print(f"无法解析的行：{line}. 期望的格式是 'key=value'")
                continue
            key, value = line.strip().split('=')
            setattr(configs, key.strip(), value.strip())
    return configs


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    print(args.config_file)
    args = update_config(args)
    print(args)
