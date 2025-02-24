import numpy as np
import torch
from timm.utils import AverageMeter

from .base_method import BaseMethod
from system.models import Darknet_MidAir_attention_md
from system.loss import DepthLoss
from ..core import compute_metrics

def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def disparity_to_depth_with_intrinsics(disparity, width, height, baseline):
    """
    使用给定的相机内参计算深度图。

    参数:
        disparity (np.ndarray): 视差图，形状为 (H, W)。
        width (int): 图像宽度。
        height (int): 图像高度。
        baseline (float): 相机基线长度。

    返回:
        depth (np.ndarray): 深度图，形状为 (H, W)。
    """
    fx = width / 2  # 计算焦距 fx
    fy = height / 2  # 计算焦距 fy (可忽略，因为仅 fx 足够)
    depth = fx * baseline / (disparity + 1e-6)  # 避免视差为 0
    return depth

class DarkNet(BaseMethod):

    def __init__(self, config):
        super(DarkNet, self).__init__(config)
        self.loss_function = DepthLoss(
            n=4,
            SSIM_w=0.85,
            disp_gradient_w=0.1, lr_w=1)

    def _build_model(self):
        self.depth_net = Darknet_MidAir_attention_md()
        return self.depth_net

    def forward(self, batch):
        disps = self.depth_net(batch)
        disp = disps[0][:, 0, :, :].unsqueeze(1)
        disparities = disp[0].squeeze()
        disparities_pp = post_process_disparity(disps[0].squeeze(0).cpu().numpy())
        pred = disparity_to_depth_with_intrinsics(disparities_pp, 1024, 1024, 1)
        return pred


    def on_train_epoch_start(self) -> None:
        self.train_loss = AverageMeter()

    def training_step(self, batch, batch_idx):
        left, right = batch
        disps = self.depth_net(left)
        loss = self.loss_function(disps, [left, right])
        self.log('train/train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_loss.update(loss.item(), left.size(0))
    def on_train_epoch_end(self) -> None:
        self.log('train/train_loss_epoch', self.train_loss.avg, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_start(self) -> None:
        if self.hparams.config.val_mode == 'photo':
            self.val_loss = AverageMeter()
        elif self.hparams.config.val_mode == 'depth':
            # {'a1': 0, 'a2': 0, 'a3': 0, 'abs_diff': 0, 'abs_rel': 0,
            #    'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'log10': 0}
            self.metrics = {
                'abs_diff': AverageMeter(),
                'abs_rel': AverageMeter(),
                'sq_rel': AverageMeter(),
                'rmse': AverageMeter(),
                'rmse_log': AverageMeter(),
                'a1': AverageMeter(),
                'a2': AverageMeter(),
                'a3': AverageMeter(),
                'log10': AverageMeter()
            }
        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        if self.hparams.config.val_mode == 'photo':
            left, right = batch
            disps = self.depth_net(left)
            disp = disps[0][:, 0, :, :]
            pred = disparity_to_depth_with_intrinsics(disp, 1024, 1024, 1)
            loss = self.loss_function(disps, [left, right])
            self.val_loss.update(loss.item(), left.size(0))
        elif self.hparams.config.val_mode == 'depth':
            left, depth = batch
            disps = self.depth_net(left)
            disp = disps[0][:, 0, :, :]
            # disp = disp.unsqueeze(1)
            # disparities = disp[0].squeeze()
            # disparities_pp = post_process_disparity(disps[0].squeeze(0).cpu().numpy())
            pred = disparity_to_depth_with_intrinsics(disp, 1024,1024,1)
            # pred = pred.unsqueeze(1)
            metrics = compute_metrics(depth, pred,
                                      dataset=self.hparams.config.dataset_name)
            for key, value in metrics.items():
                self.metrics[key].update(value, 1)
        if batch_idx < 10:
            self.logger.experiment.add_images('val/img_{}'.format(batch_idx), left, self.current_epoch)
            self.logger.experiment.add_images('val/disp_{}'.format(batch_idx), disp.unsqueeze(1), self.current_epoch)
            self.logger.experiment.add_images('val/depth_{}'.format(batch_idx), pred.unsqueeze(1), self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.config.val_mode == 'photo':
            self.log('val/val_loss', self.val_loss.avg)
            self.log('val_loss', self.val_loss.avg, prog_bar=True)
        elif self.hparams.config.val_mode == 'depth':
            for key, value in self.metrics.items():
                self.log('val/' + key, value.avg)
            self.log('val_loss', self.metrics['abs_rel'].avg, prog_bar= True)


    def on_test_epoch_start(self) -> None:
        self.test_metrics = {
            'abs_diff': AverageMeter(),
            'abs_rel': AverageMeter(),
            'sq_rel': AverageMeter(),
            'a1': AverageMeter(),
            'a2': AverageMeter(),
            'a3': AverageMeter(),
            'log10': AverageMeter()
        }
    def test_step(self, batch, batch_idx):
        left, depth = batch
        disps = self.depth_net(left)
        metrics = compute_metrics(depth, disps,
                                  dataset=self.hparams.config.dataset_name)
        for key, value in metrics.items():
            self.test_metrics[key].update(value, 1)
