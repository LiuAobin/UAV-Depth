import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# from mask_ranking_loss import Mask_Ranking_Loss
# from normal_ranking_loss import EdgeguidedNormalRankingLoss
from .mask_ranking_loss import Mask_Ranking_Loss
from .normal_ranking_loss import EdgeguidedNormalRankingLoss
from system.utils import inverse_warp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SSIM(nn.Module):
    """定义结构相似性指数(structural similarity index)类，用于计算图像对之间的ssim损失"""

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7  # 设置窗口大小
        # 定义用于计算均值和方差的卷积池化操作
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        # 反射填充操作
        self.refl = nn.ReflectionPad2d(k // 2)

        # 定义SSIM公式中的常量
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        """计算输入图像x和y之间的SSIM损失"""
        # 反射填充，处理边界情况
        x = self.refl(x)
        y = self.refl(y)

        # 计算均值
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        # 计算方差和协方差
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        # 计算SSIM的分子和分母
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # 返回归一化后的SSIM值，并将其裁剪到[0,1]之间
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# SSIM损失实例
compute_ssim_loss = SSIM().to(device)

# 定义Normal Ranking Loss和Mask Ranking Loss
normal_ranking_loss = EdgeguidedNormalRankingLoss().to(device)
mask_ranking_loss = Mask_Ranking_Loss().to(device)


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, hparams):
    """
    计算目标图像和参考图像之间的光度和几何损失
    Args:
        tgt_img (tensor): 目标图像，大小为 (B, 3, H, W)
        ref_img (tensor): 参考图像，大小为 (B, 3, H, W)
        tgt_depth (tensor): 目标图像深度图，大小为 (B, 1, H, W)
        ref_depth (tensor): 参考图像深度图，大小为 (B, 1, H, W)
        pose (tensor): 参考图像相对于目标图像的相机位姿变换矩阵，大小为 (B, 4, 4)
        intrinsic (tensor): 相机内参矩阵，大小为 (B, 3, 3)
        hparams (object): 超参数，包含模型设置和训练选项
    Returns:
        diff_img (tensor): 图像差异，大小为 (B, 1, H, W)
        diff_color (tensor): 颜色差异，大小为 (B, 1, H, W)
        diff_depth (tensor): 深度差异，大小为 (B, 1, H, W)
        valid_mask (tensor): 有效区域掩码，大小为 (B, 1, H, W)
    """
    # 反向重映射：将参考图像和深度图投影到目标图像的视角
    ref_img_warped,projected_depth,computed_depth = inverse_warp(
        ref_img,
        tgt_depth,ref_depth,
        pose,intrinsic,padding_mode='zeros'
    )
    # 计算深度差异（归一化绝对误差）
    # 计算深度差异公式：|d_computed - d_projected| / (d_computed + d_projected)
    diff_depth = ((computed_depth - projected_depth).abs() /
                  (computed_depth + projected_depth))

    # 对零值进行掩码，计算有效区域
    valid_mask_ref = (
            ref_img_warped.abs().mean(dim=1, keepdim=True) > 1e-3
    ).float()

    valid_mask_tgt = (
        tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3
    ).float()

    # 有效区域同时满足目标图像和参考图像有效
    valid_mask = valid_mask_tgt * valid_mask_ref

    # 计算颜色差异（目标图像与重映射后的参考图像的差异）
    # 颜色差异公式：|tgt_img - ref_img_warped|
    diff_color = (tgt_img - ref_img_warped).abs().mean(dim=1, keepdim=True)
    # 如果启用了自动掩码，则基于颜色差异和重投影误差生成掩码
    if not hparams.no_auto_mask:
        # 计算目标图像与参考图像的颜色差异
        identify_wrap_err = (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)
        # 如果颜色差异小于重投影误差，则认为是有效区域
        auto_mask = (diff_color < identify_wrap_err).float()
        valid_mask = valid_mask * auto_mask  # 更新有效区域掩码

    # 计算图像差异
    # 图像差异公式：|tgt_img - ref_img_warped|，并限制在 [0, 1] 范围内
    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)
    # 如果启用了 SSIM（结构相似性）损失，则结合 SSIM 损失
    if not hparams.no_ssim:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)  # 计算 SSIM 损失图
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)  # 加权组合图像差异和 SSIM 损失
    # 计算图像差异的均值
    diff_img = torch.mean(diff_img, dim=1, keepdim=True)

    # 为动态区域减少光度损失的权重
    if not hparams.no_dynamic_mask:
        weight_mask = (1-diff_depth).detach()  # 计算动态区域的权重，深度差异越大，权重越小
        diff_img = diff_img * weight_mask  # 应用动态区域的权重

    return diff_img, diff_color, diff_depth, valid_mask


def mean_on_mask(diff, valid_mask):
    # 计算带有掩码的平均值
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 100:
        mean_val = (diff * mask).sum() / mask.sum()
    else:
        mean_val = torch.tensor(0).float().to(device)
    return mean_val


def photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                            intrinsics, poses, poses_inv, hparams):
    """
    计算目标图像与参考图像之间的光度损失和几何损失
    Args:
        tgt_img (tensor): 目标图像，大小为 (B, 3, H, W)，B 为批量大小，H 和 W 为图像高度和宽度
        ref_imgs (list): 参考图像列表，包含多个参考图像，每个图像大小为 (B, 3, H, W)
        tgt_depth (tensor): 目标图像深度图，大小为 (B, 1, H, W)
        ref_depths (list): 参考图像的深度图列表，每个深度图大小为 (B, 1, H, W)
        intrinsics (tensor): 相机内参矩阵，大小为 (B, 3, 3)
        poses (list): 目标图像到参考图像的相机位姿变换矩阵列表
        poses_inv (list): 参考图像到目标图像的相机位姿逆变换矩阵列表
        hparams (object): 超参数，包含模型设置和训练选项
    Returns:
         photo_loss (tensor): 基于图像的照片损失
        geometry_loss (tensor): 基于深度的几何损失
        dynamic_mask (tensor): 可选，动态遮罩
    """
    diff_img_list = []  # 存储不同视图之间图像差异的列表
    diff_color_list = []  # 存储不同视图之间颜色差异的列表
    diff_depth_list = []  # 存储不同视图之间深度差异的列表
    valid_mask_list = []  # 存储有效区域的掩码列表
    # 遍历所有参考图像计算差异
    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        # 计算目标图像与参考图像的光度和几何损失
        diff_img_tmp1, diff_color_tmp1, diff_depth_temp1, valid_mask_tmp1 = compute_pairwise_loss(
            tgt_img, ref_img,
            tgt_depth, ref_depth,
            pose, intrinsics, hparams
        )
        # 计算参考图像和目标图像之间的损失（交换顺序）
        diff_img_tmp2, diff_color_tmp2, diff_depth_temp2, valid_mask_tmp2 = compute_pairwise_loss(
            ref_img, tgt_img,
            ref_depth, tgt_depth,
            pose_inv, intrinsics, hparams
        )
        # 将损失加入列表
        diff_img_list += [diff_img_tmp1, diff_img_tmp2]
        diff_color_list += [diff_color_tmp1, diff_color_tmp2]
        diff_depth_list += [diff_depth_temp1, diff_depth_temp2]
        valid_mask_list += [valid_mask_tmp1, valid_mask_tmp2]

    # 将所有损失合并为一个批次
    diff_img = torch.cat(diff_img_list, dim=1)
    diff_color = torch.cat(diff_color_list, dim=1)
    diff_depth = torch.cat(diff_depth_list, dim=1)
    valid_mask = torch.cat(valid_mask_list, dim=1)

    # 如果未禁用最小化优化，则选择每个像素的最佳匹配
    if not hparams.no_min_optimize:
        # 选择颜色差异最小的参考图像
        indices = torch.argmin(diff_color, dim=1, keepdim=True)  # argmin：选择最小值对应的索引
        # 根据选择的索引，从多个参考图像中提取最佳匹配的图像、深度和掩码
        diff_img = torch.gather(diff_img, 1, indices)  # 根据索引获取最佳图像差异
        diff_depth = torch.gather(diff_depth, 1, indices)  # 根据索引获取最佳深度差异
        valid_mask = torch.gather(valid_mask, 1, indices)  # 根据索引获取有效掩码

    # 计算光度损失和几何损失
    photo_loss = mean_on_mask(diff_img, valid_mask)  # 使用有效掩码计算图像差异的均值
    geometry_loss = mean_on_mask(diff_depth, valid_mask)  # 使用有效掩码计算深度差异的均值
    # 计算动态掩码
    return photo_loss, geometry_loss


def smooth_loss(tgt_depth, tgt_img):
    """计算平滑损失，通过颜色图像来进行边缘感知的平滑"""

    def get_smooth_loss(disp, img):
        """计算视差图的平滑损失，使用颜色信息作为边缘感知的引导"""
        # 归一化视差图
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        # 计算视差图和图像的梯度
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]),
            1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]),
            1, keepdim=True
        )
        # 通过图像梯度对视差梯度进行加权
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()
    # 计算并返回平滑损失
    loss = get_smooth_loss(tgt_depth, tgt_img)
    return loss


def main():
    # 设定测试参数
    B, C, H, W = 2, 3, 128, 128  # Batch size, Channels, Height, Width
    num_refs = 2  # 参考图像数量

    # 生成随机测试数据
    tgt_img = torch.rand(B, C, H, W)  # 目标图像
    ref_imgs = [torch.rand(B, C, H, W) for _ in range(num_refs)]  # 参考图像
    tgt_depth = torch.rand(B, 1, H, W)  # 目标深度
    ref_depths = [torch.rand(B, 1, H, W) for _ in range(num_refs)]  # 参考深度

    intrinsics = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)  # 相机内参 (B, 3, 3)
    poses = [torch.eye(4) for _ in range(num_refs)]  # 目标到参考图像的变换矩阵
    poses_inv = [torch.inverse(p) for p in poses]  # 参考图像到目标图像的逆变换

    # 伪造超参数对象
    class HParams:
        no_min_optimize = False  # 控制最小优化策略

    hparams = HParams()

    # 计算光度和几何损失
    photo_loss, geometry_loss = photo_and_geometry_loss(
        tgt_img, ref_imgs, tgt_depth, ref_depths, intrinsics, poses, poses_inv, hparams
    )

    # 打印损失结果
    print(f"Photo Loss: {photo_loss.item():.4f}")
    print(f"Geometry Loss: {geometry_loss.item():.4f}")

    # 可选：绘制损失趋势（如果需要运行多次训练迭代）
    losses = [photo_loss.item(), geometry_loss.item()]
    plt.bar(["Photo Loss", "Geometry Loss"], losses, color=["blue", "green"])
    plt.ylabel("Loss Value")
    plt.title("Loss Comparison")
    plt.show()

if __name__ == "__main__":
    main()