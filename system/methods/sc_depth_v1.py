import numpy as np
import torch
from system.loss import loss_functions as lossF
from .base_method import BaseMethod
from system.utils import (print_log, visualize_image, visualize_depth)
from system.models import DepthNet, PoseNet


class SCDepthV1(BaseMethod):
    def __init__(self, config):
        super(SCDepthV1, self).__init__(config)

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

        # 2. 前向传播
        tgt_depth = self.depth_net(tgt_img)  # 预测目标图像的深度
        ref_depths = [self.depth_net(im) for im in ref_imgs]  # 预测参考图像的深度

        poses = [self.pose_net(tgt_img, im) for im in ref_imgs]  # 估计目标与参考图像的相对位姿
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]  # 估计反向位姿

        # 计算损失
        w1 = self.hparams.config.photo_weight  # 光度一致性损失权重
        w2 = self.hparams.config.geometry_weight  # 几何一致性损失权重
        w3 = self.hparams.config.smooth_weight  # 平滑损失权重

        loss_1, loss_2 = lossF.photo_and_geometry_loss(
            tgt_img, ref_imgs,
            tgt_depth, ref_depths,
            intrinsics, poses, poses_inv,
            self.hparams.config
        )  # 计算光度和几何一致性损失
        loss_3 = lossF.compute_smooth_loss(tgt_depth, tgt_img)  # 计算深度图的平滑损失
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3  # 总损失
        # 记录日志
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)
        # print_log(f'\ntrain/total_loss: {loss} | train/photo_loss {loss_1} |'
        #           f' train/geometry_loss {loss_2} | train/smooth_loss {loss_3}\n')

        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证阶段的单步逻辑。
        """
        if self.hparams.config.val_mode == 'depth':  # 深度评估模式
            tgt_img, gt_depth = batch  # 提取目标图像和真实深度
            tgt_depth = self.depth_net(tgt_img)  # 预测深度
            errs = lossF.compute_errors(gt_depth, tgt_depth, self.hparams.config.dataset_name)  # 计算误差
            # [abs_diff, abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3]
            self.log('val_loss', errs[0])
            errs = {'abs_diff': errs[0],
                    'abs_rel': errs[1],
                    'sq_rel': errs[2],
                    'rmse': errs[4],
                    'rmse_log': errs[5],
                    'a1': errs[6],
                    'a2': errs[7],
                    'a3': errs[8]}

        elif self.hparams.config.val_mode == 'photo':  # 光度损失验证模式
            tgt_img, ref_imgs, intrinsics = batch  # 提取目标图像、参考图像和相机内参
            tgt_depth = self.depth_net(tgt_img)  # 预测目标深度
            ref_depths = [self.depth_net(im) for im in ref_imgs]  # 预测参考图像深度
            poses = [self.pose_net(tgt_img, im) for im in ref_imgs]  # 估计位姿
            poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]  # 估计反向位姿
            loss_1, _ = lossF.photo_and_geometry_loss(
                tgt_img, ref_imgs, tgt_depth, ref_depths, intrinsics, poses, poses_inv, self.hparams.hparams
            )
            self.log('val_loss', loss_1.item())
            errs = {'photo_loss': loss_1.item()}
        else:
            print('wrong validation mode')  # 错误模式提示
        self.validation_step_outputs.append(errs)
        # print_log(errs)
        if self.global_step < 10:  # 若步骤数小于10，直接返回误差
            return errs
        # 可视化图像和深度
        if batch_idx < 3:
            vis_img = visualize_image(tgt_img[0])  # 转换为可视化图像
            vis_depth = visualize_depth(tgt_depth[0, 0])  # 转换为可视化深度
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # 堆叠图像和深度图
            self.logger.experiment.add_images('val/img_depth_{}'.format(batch_idx), stack, self.current_epoch)

        return errs

    def on_validation_epoch_end(self):
        """
        验证阶段结束时的逻辑
        Args:
            outputs (): 验证阶段的输出列表
        Returns:
        """
        if self.hparams.config.val_mode == 'depth':  # 深度评估模式
            mean_rel = np.array([x['abs_rel'] for x in self.validation_step_outputs]).mean()
            mean_diff = np.array([x['abs_diff'] for x in self.validation_step_outputs]).mean()
            mean_sq = np.array([x['sq_rel'] for x in self.validation_step_outputs]).mean()
            mean_rmse = np.array([x['rmse'] for x in self.validation_step_outputs]).mean()
            mean_rmse_log = np.array([x['rmse_log'] for x in self.validation_step_outputs]).mean()
            mean_a1 = np.array([x['a1'] for x in self.validation_step_outputs]).mean()
            mean_a2 = np.array([x['a2'] for x in self.validation_step_outputs]).mean()
            mean_a3 = np.array([x['a3'] for x in self.validation_step_outputs]).mean()
            # # 记录日志
            self.log('val_loss', mean_rel, prog_bar=True)
            self.log('val/abs_diff', mean_diff)
            self.log('val/abs_rel', mean_rel)
            self.log('val/sq_rel', mean_sq)
            self.log('val/rmse', mean_rmse)
            self.log('val/rmse_log', mean_rmse_log)
            self.log('val/a1', mean_a1, on_epoch=True)
            self.log('val/a2', mean_a2, on_epoch=True)
            self.log('val/a3', mean_a3, on_epoch=True)
            print_log(
                f'\nval_loss: {mean_rel} | '
                f'val/abs_diff {mean_diff} | '
                f'val/abs_rel {mean_rel} | '
                f'val/sq_rel {mean_sq} | '
                f'val/rmse {mean_rmse} | '
                f'val/rmse_log {mean_rmse_log} | '
                f'val/a1 {mean_a1} |'
                f' val/a2 {mean_a2} | '
                f'val/a3 {mean_a3}\n')
        elif self.hparams.config.val_mode == 'photo':  # 光度评估模式
            mean_pl = np.array([x['photo_loss'] for x in self.validation_step_outputs]).mean()
            # self.log('val_loss', mean_pl, prog_bar=True)
            print_log(f'\nval_loss: {mean_pl}\n')

    def test_step(self, batch, batch_idx):
        """
        测试阶段的单步逻辑。
        """
        if self.hparams.config.val_mode == 'depth':  # 深度评估模式
            tgt_img, gt_depth = batch  # 提取目标图像和真实深度
            tgt_depth = self(tgt_img)
        elif self.hparams.config.val_mode == 'photo':
            tgt_img, ref_imgs, intrinsics = batch  # 提取目标图像、参考图像和相机内参
            tgt_depth = self(tgt_img)
        else:
            print('wrong validation mode')  # 错误模式提示
        return tgt_depth
