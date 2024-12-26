import json
import shutil
import logging
import os.path as osp
import sys

from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar, Checkpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

from .main_utils import check_dir, print_log, collect_env, output_namespace


class SetupCallback(Callback):
    """
    用于在训练开始时执行一系列设置操作的回调函数
        :param prefix: 日志和文件的前缀（例如 'train' 或 'test'）
        :param setup_time: 当前设置的时间，用于命名文件
        :param save_dir: 结果保存目录
        :param ckpt_dir: 检查点保存目录
        :param args: 参数对象，包含实验配置信息
        :param method_info: 方法信息（包括模型信息、FLOPs和吞吐量）
        :param argv_content: 命令行参数内容
    """

    def __init__(self, prefix, setup_time, save_dir, ckpt_dir, args, method_info, argv_content=None):
        super().__init__()
        self.prefix = prefix
        self.setup_time = setup_time
        self.save_dir = save_dir
        self.ckpt_dir = ckpt_dir
        self.args = args
        self.config = args.__dict__  # 将参数转换为字典
        self.argv_content = argv_content
        self.method_info = method_info

    def on_fit_start(self, trainer, pl_module):
        """
        在训练开始时执行的操作
        Args:
            trainer (): Lightning Trainer 对象
            pl_module (): PyTorch Lightning 模型

        Returns:

        """
        # 收集环境信息
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'

        # 确保只在主线程进行操作
        if trainer.global_rank == 0:
            # 检查和创建保存目录
            self.save_dir = check_dir(self.save_dir)  # save_dir = {base_dir}/{exp_name}
            self.ckpt_dir = check_dir(self.ckpt_dir)  # ckpt_dir = {base_dir}/{exp_name}/{checkpoints}
            # 设置日志记录
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO,  # filename= {base_dir}/{exp_name}/{train/test}_{%Y%m%d_%H%M%S}.log
                                filename=osp.join(self.save_dir, f'{self.prefix}_{self.setup_time}.log'),
                                filemode='a', format='%(asctime)s - %(message)s',encoding='utf-8')  # 消息格式：时间-信息
            # 打印环境信息到日志中
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
            # 保存模型参数到JSON文件
            sv_param = osp.join(self.save_dir, 'model_param.json')
            # 打印参数信息到日志
            with open(sv_param, 'w') as file_obj:
                json.dump(self.config, file_obj)
            # 打印参数信息到日志
            print_log(output_namespace(self.args))
            # 如果有方法信息，则打印模型信息，FLOPs和吞吐量
            if self.method_info is not None:
                info, flops, fps, dash_line = self.method_info
                print_log('Model info:\n' + info + '\n' + flops + '\n' + fps + dash_line)


class EpochEndCallback(Callback):
    """
    用于记录每个epoch结束时的训练和验证信息的回调函数
    """
    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        """
        在训练结束时记录损失
        """
        self.avg_train_loss = trainer.callback_metrics.get('train/total_loss')

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        在每个验证epoch结束时打印学习率和损失信息
        """
        lr = trainer.optimizers[0].param_groups[0]['lr']  # 获取当前学习率
        avg_val_loss = trainer.callback_metrics.get('val_loss')  # 获取验证损失
        # 如果训练损失已记录，则打印损失信息和学习率
        if hasattr(self, 'avg_train_loss'):
            print_log(
                f"\nEpoch {trainer.current_epoch}: Lr: {lr:.7f} | "
                f"Train Loss: {self.avg_train_loss:.7f} | "
                f"Vali Loss: {avg_val_loss:.7f}")


class BestCheckpointCallback(ModelCheckpoint):
    """
    用于保存最佳模型检查点的回调函数
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        在每个验证epoch结束时保存最佳模型到指定路径
        """
        # 调用父类模型，处理模型检查点保存逻辑
        super().on_validation_epoch_end(trainer, pl_module)
        # 获取检查点回调的引用，确保存在有效的最佳模型路径
        checkpoint_callback = trainer.checkpoint_callback
        # 仅在主进程保存最佳模型
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path  # 获取最佳模型的文件路径

            # 将最佳模型另存为 'best.ckpt'
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))

    def on_test_end(self, trainer, pl_module):
        """
        在测试结束时将最佳模型另存为best.ckpt
        """
        # 调用父类方法处理测试结束逻辑
        super().on_test_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback

        # 仅在主进程保存最佳模型
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            # 将最佳模型另存为best.ckpt
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))


class MyTQDMProgressBar(TQDMProgressBar):
    """
    解决验证时进度条输出异常的问题
    """
    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()

    def init_validation_tqdm(self):
        bar = Tqdm(
            desc=self.validation_description,
            position=0,  # 这里固定写0
            disable=self.is_disabled,
            leave=True,  # leave写True
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
