from typing import Any

import torch
import wandb
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import MeanMetric
import torch.nn.functional as F
from .base_method import BaseMethod
# from system.models import ResnetEncoderDecoder,Depth_Decoder_QueryTr,PoseCNN
# from system.layers import BackprojectDepth,Project3D
from system.utils import (print_log, transformation_from_parameters, convert_K_to_4x4,visualize_image,visualize_depth,
                          BackProjectDepth,Project3D)
from system.core import compute_metrics
from system.models import ResnetEncoderDecoder,PoseCNN,DepthDecoderQueryTr
from system.loss import get_smooth_loss,compute_reprojection_loss
from path import Path

class SQLDepth(BaseMethod):

    def __init__(self, config):
        print('methods>SQLDepth---->init sql_depth')
        # 初始化函数，接收一个config参数
        super().__init__(config)# 调用父类的初始化函数
        self.config = config # 将config参数赋值给self.config
        # 根据self.config.folder_type的值，设置self.num_pose_frames的值
        self.num_pose_frames = 2 if self.config.folder_type == 'pair' else self.config.sequence_length

        self.models = {}
        # 调用_build_model函数
        self._build_model()


    def _build_model(self):
        print_log('SQLDepth---->build model')
        # 选择编码器
        self.models['encoder'] = ResnetEncoderDecoder(
            num_layers = self.config.num_layers,
            num_features = self.config.num_features,
            model_dim = self.config.model_dim,
        )
        # 加载编码器预训练模型
        # encoder_path = Path('./pretrained_model/encoder.pth')
        # # print(encoder_path.absolute())
        # loaded_dict_enc = torch.load(encoder_path,map_location=self.config.device)
        # filtered_dict_enc = {k:v for k,v in loaded_dict_enc.items() if k in self.models['encoder'].state_dict()}
        # self.models['encoder'].load_state_dict(filtered_dict_enc)
        # 加载深度解码器
        self.models['depth'] = DepthDecoderQueryTr(
            in_channels = self.config.model_dim,
            patch_size=self.config.patch_size,
            dim_out=self.config.dim_out,
            embedding_dim = self.config.model_dim,
            query_nums = self.config.query_nums,
            num_heads = 4,
            min_val = self.config.min_depth,
            max_val = self.config.max_depth
        )
        # 加载深度估计模块预训练模型
        # depth_decoder_path = Path('./pretrained_model/depth.pth')
        # loaded_dict_enc = torch.load(depth_decoder_path, map_location=self.config.device)
        # filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models['depth'].state_dict()}
        # self.models['depth'].load_state_dict(filtered_dict_enc)
        # 加载姿态估计模型
        self.models['pose'] = PoseCNN(2)
        self.encoder = self.models['encoder']
        self.depth = self.models['depth']
        self.pose = self.models['pose']

    def _build_projection(self):
        # 创建一个字典，用于存储不同尺度下的反向投影深度
        # self.backproject_depth = {}
        # 创建一个字典，用于存储不同尺度下的三维投影
        # self.project_3d = {}
        # 遍历配置文件中的尺度

        # 计算当前尺度下的高度和宽度
        h = self.config.height
        w = self.config.width
        # 创建当前尺度下的反向投影深度
        self.backproject_depth = BackProjectDepth(
            height=h,width=w
        ).to(self.device)
        # 创建当前尺度下的三维投影
        self.project_3d = Project3D(
            height=h, width=w
        ).to(self.device)

    def _build_metrics(self):
        # 创建一个字典，用于存储不同尺度下的评估指标
        if self.config.val_mode == 'depth':
            self.val_metrics = {
                'de/abs_diff':MeanMetric().to(self.device),
                'de/abs_rel':MeanMetric().to(self.device),
                'de/sq_rel':MeanMetric().to(self.device),
                'de/rmse':MeanMetric().to(self.device),
                'de/log10':MeanMetric().to(self.device),
                'de/rmse_log':MeanMetric().to(self.device),
                'da/a1':MeanMetric().to(self.device),
                'da/a2':MeanMetric().to(self.device),
                'da/a3':MeanMetric().to(self.device),
            }
        elif self.config.val_mode == 'photo':
            self.val_metrics = {
                'loss':MeanMetric().to(self.device),
                'reprojection':MeanMetric().to(self.device),
                'smooth':MeanMetric().to(self.device),
            }
        else:
            raise NotImplementedError

    def on_train_start(self):
        """
        监控模型
        """
        # 将模型移动到设备上
        self.models['encoder'].to(self.device)
        self.models['depth'].to(self.device)
        self.models['pose'].to(self.device)
        # 初始化投影和损失评估指标
        self._build_projection()
        self._build_metrics()
        for name, model in self.models.items():
            wandb.watch(model, log='all', log_freq=100)

    def training_step(self, batch, batch_idx):
        # 1. 提取目标图像、参考图像和相机内参
        tgt_img, ref_imgs, intrinsics = batch
        # 2. 计算当前尺度下的深度图
        disp = self.predict_depth(tgt_img)
        # 3. 计算相机位姿
        pose_outputs = self.predict_poses(tgt_img, ref_imgs)
        outputs = {'disp':disp}
        outputs.update(pose_outputs)
        # 生成合成视角图像
        self.generate_images_pred(intrinsics,ref_imgs,outputs)
        # 计算损失
        loss,smooth_loss,reprojection_loss = self.compute_loss(tgt_img,ref_imgs,outputs)

        self.log_dict({
            'train/loss':loss,
            'train/smooth':smooth_loss,
            'train/reprojection':reprojection_loss
        },logger=True,prog_bar=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config.val_mode == 'depth':  # 深度评估模式
            tgt_img, gt_depth = batch  # 提取目标图像和真实深度
            disp = self.predict_depth(tgt_img)
            disp = F.interpolate(disp,
                                 [self.config.height, self.config.width],
                                 mode='bilinear',
                                 align_corners=False)
            depth = disp
            # 计算评估指标
            metrics = compute_metrics(gt_depth, depth,dataset=self.config.dataset_name)
            for key,value in metrics.items():
               self.val_metrics[key].update(value)
            # 可视化
            if batch_idx % 50 == 0:
                self.log_visualization(tgt_img, gt_depth,depth, batch_idx)

        elif self.config.val_mode == 'photo':  # 光度损失验证模式
            tgt_img, ref_imgs, intrinsics = batch  # 提取目标图像、参考图像和相机内参
            # 1. 提取目标图像、参考图像和相机内参
            tgt_img, ref_imgs, intrinsics = batch
            # 2. 计算当前尺度下的深度图
            disp = self.predict_depth(tgt_img)
            # 3. 计算相机位姿
            pose_outputs = self.predict_poses(tgt_img, ref_imgs)
            outputs = {'disp': disp}
            outputs.update(pose_outputs)
            # 生成合成视角图像
            self.generate_images_pred(intrinsics, ref_imgs, outputs)
            # 计算损失
            loss, smooth_loss, reprojection_loss = self.compute_loss(tgt_img, ref_imgs,outputs)
            self.val_metrics['loss'].update(loss)
            self.val_metrics['smooth'].update(smooth_loss)
            self.val_metrics['reprojection'].update(reprojection_loss)
        else:
            raise ValueError(f'Invalid validation mode: {self.config.val_mode}')

    def on_validation_epoch_end(self):
        # 计算验证集的平均指标并记录到日志
        avg_metrics = {key: metric.compute() for key, metric in self.val_metrics.items()}
        wandb.log(avg_metrics, step=self.current_epoch)

        # 记录模型验证的损失
        if self.config.val_mode == 'depth':
            self.log('val_loss', avg_metrics['de/abs_diff'], on_epoch=True, on_step=False,logger=False)
        elif self.config.val_mode == 'photo':
            self.log('val_loss', avg_metrics['loss'], on_epoch=True, on_step=False,logger=False)
        else:
            raise ValueError(f'Invalid validation mode: {self.config.val_mode}')
        # 重置验证指标
        for metric in self.val_metrics.values():
            metric.reset()

    def on_test_start(self):
        for metrics in self.val_metrics.values():
            metrics.reset()

    def test_step(self, batch, batch_idx):
        tgt_img, gt_depth = batch  # 提取目标图像和真实深度
        disp = self.predict_depth(tgt_img)
        disp = F.interpolate(disp,
                             [self.config.height, self.config.width],
                             mode='bilinear',
                             align_corners=False)
        depth = disp
        metrics = compute_metrics(gt_depth, depth)
        for key, value in metrics.items():
            self.val_metrics[key].update(value)

    def on_test_epoch_end(self):
        avg_metrics = {key: metric.compute() for key, metric in self.val_metrics.items()}
        self.log_dict(avg_metrics, logger=True, on_epoch=True, on_step=False)

    def predict_depth(self, input):
        features = self.models['encoder'](input)
        outputs = self.models['depth'](features)
        return outputs

    def forward(self, batch):
        return self.predict_depth(batch)

    def predict_poses(self, tgt_img, ref_imgs):
        outputs = {}
        # 维持时间顺序
        for idx, ref_img in enumerate(ref_imgs):
            # input (bs,c*2,H,W)
            if idx < self.config.sequence_length // 2:
                input = torch.cat([ref_img, tgt_img], dim=1)
            else:
                input = torch.cat([tgt_img, ref_img], dim=1)
            # 计算6ToF的位姿
            # axisangle(旋转，欧拉角表示) (bs,1,1,3)
            # translation(平移) 表示相机在(x,y,z)方向的位移 (bs,1,1,3)
            axisangle, translation = self.models['pose'](input)
            outputs[('axisangle', idx)] = axisangle
            outputs[('translation', idx)] = translation
            # 降维一次，移除无关的维度
            outputs[('cam_T_cam', idx)] = transformation_from_parameters(
                axisangle[:, 0],
                translation[:, 0],
                invert=(idx < self.config.sequence_length // 2))
        return outputs

    def generate_images_pred(self,intrinsics,ref_imgs,outputs):
        disp = outputs['disp']
        # 生成预测的图像
        # 在这里->disp是原始输入的一半，因此，采用双线性插值，将其恢复为原始尺度
        disp = F.interpolate(disp,
                             [self.config.height,self.config.width],
                             mode='bilinear',
                             align_corners=False)
        depth = disp
        outputs['depth'] = depth
        # 为每一帧计算重投影图像
        for idx in range(len(ref_imgs)):  # 遍历当前图像相关的其他帧
            # 获取相机变换矩阵
            cam_T_cam = outputs[('cam_T_cam', idx)]
            axisangle = outputs[('axisangle', idx)]
            translation = outputs[('translation', idx)]
            inv_depth = 1 / depth
            mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
            T = transformation_from_parameters(axisangle[:,0],
                                               translation[:,0]*mean_inv_depth[:,0],
                                               invert=(idx < self.config.sequence_length // 2))

            # 计算相机的内参矩阵(4x4)
            K = convert_K_to_4x4(intrinsics)
            outputs['K'] = K
            # 计算矩阵K的伪逆矩阵
            inv_K = torch.linalg.inv(K)
            outputs['inv_K'] = inv_K
            # 从深度图回投点云->使用深度图和反相机内参将像素坐标转换为相机坐标系下的3维点云
            cam_points = self.backproject_depth(depth, inv_K)
            # 将点云投影回图像平面->使用相机内参K和变换矩阵T将3维点云重新投影回2维平面，得到像素坐标
            # pix_coords: [bs, h, w, 2]
            pix_coords = self.project_3d(cam_points,
                                        K,T)
            outputs[('sample',idx)] = pix_coords
            # 保存重投影的颜色图像
            outputs[('color',idx)] = F.grid_sample(
                ref_imgs[idx],
                outputs[('sample',idx)],
                padding_mode='border',
                align_corners=True)

            outputs[('color_identify',idx)] = ref_imgs[idx]

    def compute_loss(self,tgt_img, ref_imgs, outputs):
        """
        计算重投影损失(reprojection loss)和平滑损失(smoothness loss)
        Args:
            outputs ():
        Returns:
        """
        loss = 0
        reprojection_losses = []
        # 获取深度，输入图像，原始视角的图像(在这里等于输入图像)
        depth = outputs['depth']
        color = tgt_img
        target = tgt_img
        # 计算重投影损失
        for idx in range(len(ref_imgs)):
            pred = outputs[('color',idx)]
            # 计算预测图像和目标图像直接的重投影损失
            reprojection_losses.append(compute_reprojection_loss(pred, target,self.config.no_ssim))

        # 连接所有视角的重投影损失，以便后续求最小值(min loss)
        reprojection_losses = torch.cat(reprojection_losses, dim=1)

        # 计算identity 重投影损失 用于比较真实图像和深度预测图像之间的误差
        if not self.config.no_auto_mask:
            identify_reprojection_losses = []
            for idx in range(len(ref_imgs)):
                pred = outputs[('color_identify',idx)]
                identify_reprojection_losses.append(compute_reprojection_loss(pred, target,self.config.no_ssim))

            identity_reprojection_losses = torch.cat(identify_reprojection_losses, 1)

        if not self.config.no_auto_mask:
            identity_reprojection_losses += torch.randn(identity_reprojection_losses.shape).to(identity_reprojection_losses.device) * 0.00001
            combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)
        else:
            combined = reprojection_losses

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)
        if not self.config.no_auto_mask:
            outputs['identity_selection'] =  (idxs > identity_reprojection_losses.shape[1] - 1).float()

        loss += to_optimise.mean()
        # 计算平滑损失
        mean_depth = depth.mean(2,True).mean(3,True)
        norm_depth = depth / (mean_depth + 1e-7)
        smooth_loss = get_smooth_loss(norm_depth, color)
        loss += smooth_loss.mean() * self.config.smooth_weight
        return loss, smooth_loss.mean(), to_optimise.mean()

    def log_visualization(self,image,gt_depth,pred_depth,batch_idx):
        image = image[0].cpu()
        gt_depth = visualize_depth(gt_depth[0])
        pred_depth = visualize_depth(pred_depth[0,0])
        # 损失以epoch形式记录
        wandb.log({
            f"input_image_{batch_idx}":wandb.Image(image,caption='RGB'),
            f"gt_depth_{batch_idx}":wandb.Image(gt_depth,caption='GT'),
            f"pred_depth_{batch_idx}":wandb.Image(pred_depth,caption='Ours')
        },step=self.current_epoch)






























































