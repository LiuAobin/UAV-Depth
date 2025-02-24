import logging

import numpy as np
import torch
import wandb
import wandb.wandb_run
from system.core import compute_metrics as metrics
from system.loss import (photo_and_geometry_loss, smooth_loss)
from .base_method import BaseMethod
from system.utils import (print_log, visualize_image, visualize_depth)
from system.models import DepthNet, PoseNet


class SCDepthV1(BaseMethod):
    def __init__(self, config):
        super(SCDepthV1, self).__init__(config)
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.test_step_outputs = []

    def _build_model(self):
        self.depth_net = DepthNet(self.hparams.config.resnet_layers)  # 深度估计网络
        self.pose_net = PoseNet()  # 位姿估计网络
        return [self.depth_net, self.pose_net]

    def forward(self, batch):
        """
        前向传播逻辑————只获取深度，这样的话，测试步骤实现就比较方便
        """
        depth = self.depth_net(batch)  # 计算深度
        return depth

    def training_step(self, batch, batch_idx):
        """
        参数:
        - batch: 一个批次的数据，包括目标图像、参考图像、相机内参。
        - batch_idx: 当前批次的索引。

        返回:
        - loss: 当前批次的总损失。
        """
        # 1. 提取目标图像、参考图像和相机内参
        tgt_img, ref_imgs, intrinsics = batch

        poses, poses_inv, ref_depths, tgt_depth = self.predict(ref_imgs, tgt_img)

        # 计算损失
        w1 = self.hparams.config.photo_weight  # 光度一致性损失权重
        w2 = self.hparams.config.geometry_weight  # 几何一致性损失权重
        w3 = self.hparams.config.smooth_weight  # 平滑损失权重
        loss_1, loss_2 = photo_and_geometry_loss(
            tgt_img, ref_imgs,
            tgt_depth, ref_depths,
            intrinsics, poses, poses_inv,
            self.hparams.config
        )  # 计算光度和几何一致性损失
        loss_3 = smooth_loss(tgt_depth, tgt_img)  # 计算深度图的平滑损失
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3  # 总损失
        # 记录日志
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)

        # 记录损失到列表中
        self.train_step_outputs.append(loss.item())  # 将损失添加到训练损失列表中
        return loss

    def predict(self, ref_imgs, tgt_img):
        # 2. 前向传播
        tgt_depth = self.depth_net(tgt_img)  # 预测目标图像的深度
        ref_depths = [self.depth_net(im) for im in ref_imgs]  # 预测参考图像的深度
        poses = [self.pose_net(tgt_img, im) for im in ref_imgs]  # 估计目标与参考图像的相对位姿
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]  # 估计反向位姿
        return poses, poses_inv, ref_depths, tgt_depth

    def validation_step(self, batch, batch_idx):
        """
        验证阶段的单步逻辑。
        """
        if self.hparams.config.val_mode == 'depth':  # 深度评估模式
            tgt_img, gt_depth = batch  # 提取目标图像和真实深度
            tgt_depth = self.depth_net(tgt_img)  # 预测深度
            errs = metrics(gt_depth, tgt_depth, self.hparams.config.dataset_name)  # 计算误差

        elif self.hparams.config.val_mode == 'photo':  # 光度损失验证模式
            tgt_img, ref_imgs, intrinsics = batch  # 提取目标图像、参考图像和相机内参
            poses, poses_inv, ref_depths, tgt_depth = self.predict(ref_imgs, tgt_img)
            loss_1, loss_2 = photo_and_geometry_loss(
                tgt_img,ref_imgs,
                tgt_depth,ref_depths,
                intrinsics,poses,poses_inv,
                self.hparams.config)
            errs = {'photo_loss': loss_1.item()}

        else:
            print('wrong validation stage')  # 错误模式提示

        self.validation_step_outputs.append(errs)
        if self.global_step < 10:  # 若步骤数小于10，直接返回误差
            return errs
        # 可视化图像和深度
        if batch_idx < 5:
            vis_img = visualize_image(tgt_img[0])  # 转换为可视化图像
            vis_depth = visualize_depth(tgt_depth[0, 0])  # 转换为可视化深度
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # 堆叠图像和深度图
            self.logger.log_image(key='val/img_depth_{}'.format(batch_idx), images=[vis_img, vis_depth],step=self.current_epoch)
            # wandb_logger.log_image(key="samples", images=[vis_img, vis_depth])
            # self.logger.experiment.add_images('val/img_depth_{}'.format(batch_idx), stack, self.current_epoch)
        return errs


    def on_train_epoch_start(self):
        # self.progress_bar.disable = True  # 禁用进度条
        # if hasattr(self, 'vali_log') and self.vali_log is not None:
        #     print_log(self.vali_log)
        # self.progress_bar.disable = False  # 恢复进度条显示
        self.train_step_outputs = []

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        """
        验证阶段结束时的逻辑
        Args:
            outputs (): 验证阶段的输出列表
        Returns:
        """
        mean_train_loss = np.mean(self.train_step_outputs)  # 计算训练损失的平均值
        lr = self.optimizers().param_groups[0]['lr']
        log_message = f'\n Epoch {self.current_epoch}: Lr: {lr:.7f} | train_loss: {mean_train_loss} | '
        # 验证阶段处理
        if self.hparams.config.val_mode == 'depth':
            log_data = {
                'val_loss': np.array([x['abs_rel'] for x in self.validation_step_outputs]).mean(),
                'val/abs_diff': np.array([x['abs_diff'] for x in self.validation_step_outputs]).mean(),
                'val/abs_rel': np.array([x['abs_rel'] for x in self.validation_step_outputs]).mean(),
                'val/sq_rel': np.array([x['sq_rel'] for x in self.validation_step_outputs]).mean(),
                'val/rmse': np.array([x['rmse'] for x in self.validation_step_outputs]).mean(),
                'val/rmse_log': np.array([x['rmse_log'] for x in self.validation_step_outputs]).mean(),
                'val/a1': np.array([x['a1'] for x in self.validation_step_outputs]).mean(),
                'val/a2': np.array([x['a2'] for x in self.validation_step_outputs]).mean(),
                'val/a3': np.array([x['a3'] for x in self.validation_step_outputs]).mean(),
                'val/log10': np.array([x['log10'] for x in self.validation_step_outputs]).mean()
            }
        elif self.hparams.config.val_mode == 'photo':
            log_data = {'val_loss': np.array([x['photo_loss'] for x in self.validation_step_outputs]).mean()}
        # 记录日志和打印
        for key, value in log_data.items():
            self.logger.experiment.log(key, value, on_epoch=True,logger=True)
        logging.info(log_message + ''.join([f'{key}: {value} | ' for key, value in log_data.items()]))


    def test_step(self, batch, batch_idx):
        """
        测试阶段的单步逻辑。
        """
        tgt_img, gt_depth = batch  # 提取目标图像和真实深度
        tgt_depth = self(tgt_img)
        errs = metrics(gt_depth, tgt_depth, self.hparams.config.dataset_name)  # 计算误差
        errs = {'abs_diff': errs[0],
                'abs_rel': errs[1], 'sq_rel': errs[2],
                'rmse': errs[3], 'rmse_log': errs[4],
                'a1': errs[5], 'a2': errs[6], 'a3': errs[7],
                'log10': errs[8]}
        self.test_step_outputs.append(errs)
        return errs

    def on_test_epoch_end(self):
        log_data = {
            'test_loss': np.array([x['abs_rel'] for x in self.test_step_outputs]).mean(),
            'test/abs_diff': np.array([x['abs_diff'] for x in self.test_step_outputs]).mean(),
            'test/abs_rel': np.array([x['abs_rel'] for x in self.test_step_outputs]).mean(),
            'test/sq_rel': np.array([x['sq_rel'] for x in self.test_step_outputs]).mean(),
            'test/rmse': np.array([x['rmse'] for x in self.test_step_outputs]).mean(),
            'test/rmse_log': np.array([x['rmse_log'] for x in self.test_step_outputs]).mean(),
            'test/a1': np.array([x['a1'] for x in self.test_step_outputs]).mean(),
            'test/a2': np.array([x['a2'] for x in self.test_step_outputs]).mean(),
            'test/a3': np.array([x['a3'] for x in self.test_step_outputs]).mean(),
            'test/log10': np.array([x['log10'] for x in self.test_step_outputs]).mean()
        }
        # 记录日志和打印
        for key, value in log_data.items():
            self.log(key, value, on_epoch=True,)
        print_log(''.join([f'{key}: {value:.4f} | ' for key, value in log_data.items()]))


