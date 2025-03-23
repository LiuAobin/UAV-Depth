import torch
from .ssim import SSIM


def get_smooth_loss(depth, image):
    """
    计算深度图的平滑损失，保持深度的局部平滑性
    使用输入的彩色图像 img 进行边缘感知，使得深度图在图像边缘变化较大的地方可以允许变化较大，
    而在光滑区域保持平滑。
    Args:
        depth:(batch, 1, height, width)
        image:(batch, channels, height, width)
    Returns:
    """
    # 计算深度图梯度(水平和垂直方向)
    mean_depth = depth.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    norm_depth = depth/(mean_depth+1e-7)
    depth = norm_depth
    grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:]) # 水平方向
    grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    # 计算图像梯度的均值(通道均值)
    grad_img_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    grad_img_x = torch.mean(grad_img_x,1,keepdim=True)
    grad_img_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    grad_img_y = torch.mean(grad_img_y,1,keepdim=True)

    # 通过对图像梯度的指数衰减来加权图像梯度，抑制边缘区域的平滑约束
    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)
    # 返回水平方向和垂直方向的平均损失
    return grad_depth_x.mean()+grad_depth_y.mean()




def compute_reprojection_loss(pred, target,no_ssim=False):
    """
    计算预测图像和目标图像之间的重投影损失
    Args:
        pred: 预测图像(重投影的结果) of shape [B,C,H,W]
        target: 目标图像(源图像) of shape [B,C,H,W]
        no_ssim:
    Returns:
        torch.Tensor: 计算得到的重投影损失
    """
    # L1 损失
    ssim = SSIM().to(pred.device)
    abs_diff = torch.abs(target-pred)
    l1_loss = abs_diff.mean(1,True) # 计算通道维度的均值，保持维度
    if no_ssim:
        reprojection_loss = l1_loss # 仅使用l1损失
    else:
        ssim_loss = ssim(pred, target).mean(1,True)  # 计算 SSIM 损失并求均值
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    return reprojection_loss
