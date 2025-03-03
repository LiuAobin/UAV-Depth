import sys
import time


import torch
from datetime import datetime as dt
from fvcore.nn import FlopCountAnalysis, flop_count_table
from path import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import pytorch_lightning.callbacks as lc
from system.datasets import BaseDataModule
from system.methods import method_maps
from system.utils import (SetupCallback, BestCheckpointCallback, measure_throughput,
                          MyTQDMProgressBar, check_dir)

# 实验相关
"""
训练初始化——方法，优化器，优化器衰减器
回调器初始化
获取数据加载器
训练
测试
打印模型信息
"""


class BaseExperiment(object):
    def __init__(self, args, dataloaders=None, strategy='auto'):
        """
        初始化实验
        :param args: 实验参数
        :param dataloaders: 数据加载器
        :param strategy: 分布式训练策略
        """
        print('step:2---->初始化实验基本信息')
        self.config = args
        self.method = None
        self.config.method = self.config.method.lower()  # 将方法名转为小写
        # 设置保存路径和检查点路径
        base_dir = args.work_dir if args.work_dir is not None else 'work_dirs'
        base_dir = Path(base_dir)
        # save_dir = {base_dir}/{exp_name}
        save_dir = base_dir.joinpath(args.exp_name if not args.exp_name.startswith(base_dir) else
                                     args.exp_name.spilt(args.work_dir + '/')[-1])  # 保存目录
        check_dir(save_dir)  # 检查目录是否存在，不存在则创建
        # ckpt_dir = {base_dir}/{exp_name}/{checkpoints}
        ckpt_dir = save_dir.joinpath('checkpoints')  # 检查点目录
        seed_everything(self.config.seed)  # 设置随机种子，确保结果可复现
        # TODO
        self.data = self._get_data()
        # 根据方法名加载对应的训练方法
        print('step:2.2---->设置模型信息')
        self.method = method_maps[self.config.method](self.config)
        # 加载回调函数和保存目录
        print('step:2.3---->设置回调函数和保存目录')
        callbacks, self.save_dir = self._load_callbacks(args, save_dir, ckpt_dir)
        print('step:2.5---->初始化训练器')
        self.trainer = self._init_trainer(args, callbacks, strategy)

    def train(self):
        """训练模型"""
        print(f'开始训练{self.config.method}方法')
        self.trainer.fit(self.method, self.data,
                         ckpt_path=self.config.ckpt_path if self.config.ckpt_path and self.config.resume else None)

    def test(self):
        """测试模型"""
        print(f'开始测试{self.config.method}方法')
        if self.config.test:
            # 如果是测试模式，加载最佳模型检查点
            ckpt = torch.load(self.save_dir.joinpath('checkpoints', 'best.ckpt'))
            self.method.load_state_dict(ckpt['state_dict'])
        self.trainer.test(self.method, self.data)

    def _get_data(self):
        """
        准备数据集和数据加载器
        """
        print('step:2.1---->准备数据集和数据加载器')
        return BaseDataModule(self.config)

    def _init_trainer(self, args, callbacks, strategy):
        """
        初始化Pytorch Lightning 的Trainer
        Args:
            args (): 实验参数
            callbacks (): 回调函数列表
            strategy (): 分布式训练测量
        Returns:
        """
        print('初始化Trainer')
        # logger = TensorBoardLogger(
        #     save_dir=self.save_dir,
        #     name='logger'
        # )
        logger = WandbLogger(
            save_dir=self.save_dir,
            name=f'logger_{dt.now().strftime("%h_%d_%H_%M")}',
            project=self.config.exp_name, # 设置项目名称
            log_model=False, # 不记录模型
        )
        return Trainer(
            accelerator='gpu',
            strategy=strategy,  # 分布式策略，如 'ddp','deepspeed_stage_2','ddp_find_unused_parameters_false'
            devices=args.gpus,  # 使用指定的GPU
            max_epochs=args.epochs,  # 最大训练轮数
            limit_train_batches=args.epoch_size,  # 检查训练集的比例，float:百分比，int:批次数
            limit_val_batches=args.limit_val_batches,
            num_sanity_val_steps=0,  # 在训练流程开始之前，运行n个批次进行检查
            callbacks=callbacks,
            logger=logger,
            benchmark=True,  # torch.backends.cudnn.benchmark
        )

    def _load_callbacks(self, args, save_dir, ckpt_dir):
        """
        加载训练过程中的回调函数
        Args:
            args (): 实验参数
            save_dir (): 保存结果的目录
            ckpt_dir (): 检查点目录
        Returns:
        """
        print('加载训练过程中的回调函数')
        method_info = None
        # 显示方法信息
        print('step:2.3.1---->获取方法信息')
        if not args.no_display_method_info:
            method_info = self.display_method_info(args)

        # 设置训练准备的回调
        # ckpt_dir = {base_dir}/{exp_name}/{checkpoints}
        setup_callback = SetupCallback(
            prefix='train' if not args.test else 'test',  # 设置前缀
            setup_time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),  # 当前时间作为设置时间
            save_dir=save_dir,
            ckpt_dir=ckpt_dir,
            args=args,
            method_info=method_info,
            argv_content=sys.argv + ['gpus: {torch.cuda.device_count()}'],  # 获取命令行参数和GPU数量
        )
        # 设置最佳模型检查点回调
        ckpt_callback = BestCheckpointCallback(
            monitor=args.metric_for_bestckpt,  # 监控的指标
            filename='best-{epoch:02d}-{val_loss:.4f}',  # 模型文件名格式
            mode='min',  # 最小化val_loss
            save_last=True,  # 保存最后一个检查点
            dirpath=ckpt_dir,  # 检查点保存路径
            verbose=True,  # 显示日志
            save_top_k=3,
            every_n_epochs=args.log_step  # 每N个Epoch保存一次
        )

        # 进度条
        progress_bar_callback = MyTQDMProgressBar()
        # 训练结束时回调
        # epochend_callback = EpochEndCallback()
        # callbacks = [setup_callback, ckpt_callback, epochend_callback,progress_bar_callback]
        callbacks = [setup_callback, ckpt_callback, progress_bar_callback,
                     lc.LearningRateMonitor(logging_interval='step')]  # 需要学习率监控，添加学习率回调

        return callbacks, save_dir

    def display_method_info(self, args):
        """
        显示支持的训练方法的基本信息
        Args:
            args (): 实验参数
        Returns:
        """
        print(f'训练方法: {args.method}')
        device = torch.device(args.device)
        if args.device == 'cuda':
            assign_gpu = 'cuda:' + (str(args.gpus[0]) if len(args.gpus) == 1 else '0')
            device = torch.device(assign_gpu)

        # 根据不同方法构造输入数据
        if args.method in ['sc-depth']:
            input_dummy = torch.ones(1, args.channels, args.height, args.width).to(device)
        elif args.method in ['darknet']:
            input_dummy = torch.ones(1, args.channels, args.height, args.width).to(device)
        else:
            raise ValueError(f'Invalid method name {args.method}')
        # 获取方法的描述信息、计算FLOPs、获取吞吐量————只计算深度网络
        dash_line = '-' * 80 + '\n'
        info = self.method.__repr__()  # 模型信息
        flops = FlopCountAnalysis(self.method.depth_net.to(device), input_dummy)  # 计算FLOPs
        flops = flop_count_table(flops)  # 获取FLOPs表格
        if args.fps:
            fps = measure_throughput(self.method.depth_net.to(device), input_dummy)  # 计算吞吐量
            fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps)
        else:
            fps = ''
        return info, flops, fps, dash_line  # 返回方法信息、FLOPs、吞吐量以及分隔线
