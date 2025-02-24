# -*- coding: utf-8 -*-

import warnings
import sys
# print(sys.path)
from system.api import BaseExperiment
from utils import create_parser, update_config


if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

warnings.filterwarnings("ignore")



if __name__ == "__main__":
    print("step:1---->初始化配置参数信息")
    args = create_parser()  # 获取参数信息
    config = update_config(args)  # 根据配置文件{args.config_file}的内容更新参数信息
    print('>' * 35 + ' training ' + '<' * 35)
    exp = BaseExperiment(args)
    exp.train()  # 训练过程
    print('>' * 35 + ' testing  ' + '<' * 35)
    loss = exp.test()  # 测试过程
