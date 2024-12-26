import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
采样测量: RS (随机采样), EGS (边缘引导采样)
"""


###########
# 随机采样(Random Sampling)
# 输入:
# inputs[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs
# 返回:
# inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B
###########
def randomSamplingNormal(inputs, targets, masks, sample_num):
    # 随机采样法，找到点对(A-B)来进行法线比较和损失计算
    num_effect_pixels = torch.sum(masks).item()  # 有效像素数
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).to(device)  # 随机打乱有效像素
    valid_inputs = inputs[:, masks]  # 只保留有效的输入点
    valid_targets = targets[:, masks]  # 只保留有效的目标点
    # 从有效像素总旋转点对A和B
    inputs_A = valid_inputs[:, shuffle_effect_pixels[0: sample_num * 2: 2]]  # A点
    inputs_B = valid_inputs[:, shuffle_effect_pixels[1, sample_num * 2:2]]  # B点
    targets_A = valid_targets[:, shuffle_effect_pixels[0:sample_num * 2:2]]  # 目标A点
    targets_B = valid_targets[:, shuffle_effect_pixels[1:sample_num * 2:2]]  # 目标B点

    # 如果A点和B点数量不匹配，则取最小数量
    if inputs_A.shape[1] != inputs_B.shape[1]:
        num_min = min(targets_A.shape[1], targets_B.shape[1])
        inputs_A = inputs_A[:, :num_min]
        inputs_B = inputs_B[:, :num_min]
        targets_A = targets_A[:, :num_min]
        targets_B = targets_B[:, :num_min]
    return inputs_A, inputs_B, targets_A, targets_B


###########
# 边缘引导采样
# input:
# inputs[i,:], targets[i, :], masks[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w
# return:
# inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
###########
def ind2sub(idx, cols):
    """根据线性索引计算行列索引"""
    r = torch.div(idx, cols, rounding_mode='floor')  # 计算行
    c = idx - r * cols  # 计算列
    return r, c


def sub2ind(r, c, cols):
    """根据行列索引计算线性索引"""
    idx = r * cols + c  # 行列索引转线性索引
    return idx


def edgeGuidedSampling(inputs, targets, edges_img, thetas_img, masks, h, w):
    # 找到边缘
    edges_max = edges_img.max()
    edges_min = edges_img.min()
    edge_mask = edges_img.ge(edges_max * 0.1)  # 选择边缘区域
    edges_loc = edge_mask.nonzero(as_tuple=False)  # 获取边缘的坐标

    thetas_edge = torch.masked_select(thetas_img, edge_mask)  # 获取边缘的角度信息
    minlen = thetas_edge.size()[0]  # 最小长度

    # 选择锚点（即边缘点）
    sample_num = minlen  # 采样数量
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long).to(device)  # 随机选择锚点
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)  # 获取锚点对应的角度
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)  # 计算锚点的行列坐标

    # 计算4个点的坐标，距离在[2,30]范围内
    distance_matrix = torch.randint(3, 20, (4, sample_num)).to(device)
    pos_or_neg = torch.ones(4, sample_num).to(device)
    pos_or_neg[:2, :] = -pos_or_neg[:2, :]  # 前两个点与后两个点方向相反
    distance_matrix = distance_matrix.float() * pos_or_neg  # 计算距离
    col = (
            col_anchors.unsqueeze(0).expand(4, sample_num).long() +
            torch.round(
                distance_matrix.double() *
                torch.abs(torch.cos(theta_anchors)).unsqueeze(0)
            ).long()
    )
    row = (
            row_anchors.unsqueeze(0).expand(4, sample_num).long() +
            torch.round(
                distance_matrix.double() *
                torch.abs(torch.sin(theta_anchors)).unsqueeze(0)
            ).long()
    )
    # 限制坐标在图像范围内，注意索引要减去1
    col[col < 0] = 0
    col[col > w - 1] = w - 1
    row[row < 0] = 0
    row[row > h - 1] = h - 1

    # 计算4个点的线性索引
    a = sub2ind(row[0, :], col[0, :], w)
    b = sub2ind(row[1, :], col[1, :], w)
    c = sub2ind(row[2, :], col[2, :], w)
    d = sub2ind(row[3, :], col[3, :], w)
    A = torch.cat((a, b, c), 0)  # A点
    B = torch.cat((b, c, d), 0)  # B点

    # 获取采样点的输入和目标
    inputs_A = inputs[:, A]
    inputs_B = inputs[:, B]
    targets_A = targets[:, A]
    targets_B = targets[:, B]
    masks_A = torch.gather(masks, 0, A.long())  # 获取A点的mask
    masks_B = torch.gather(masks, 0, B.long())  # 获取B点的mask

    return (
        inputs_A, inputs_B,
        targets_A, targets_B,
        masks_A, masks_B,
        sample_num, row, col,
    )


######################################################
# EdgeguidedNormalRankingLoss
#####################################################
class EdgeguidedNormalRankingLoss(nn.Module):
    def __init__(
            self,
            point_pairs=10000,
            cos_theta1=0.25,
            cos_theta2=0.98,
            cos_theta3=0.5,
            cos_theta4=0.86,
            mask_value=-1e-8,
    ):
        super(EdgeguidedNormalRankingLoss, self).__init__()
        self.point_pairs = point_pairs  # 点对的数量
        self.mask_value = mask_value
        self.cos_theta1 = cos_theta1  # 75 度
        self.cos_theta2 = cos_theta2  # 10 度
        self.cos_theta3 = cos_theta3  # 60 度
        self.cos_theta4 = cos_theta4  # 30 度
        # 3×3卷积核，用于边缘扩展
        self.kernel = torch.tensor(
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
            requires_grad=False,
        )[None, None, :, :].to(device)

    def getEdge(self, images):
        """提取图像的边缘信息"""
        n, c, h, w = images.size()
        a = (
            torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .to(device)
            .view((1, 1, 3, 3))
            .repeat(1, 1, 1, 1)
        )
        b = (
            torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .to(device)
            .view((1, 1, 3, 3))
            .repeat(1, 1, 1, 1)
        )
        if c == 3:
            gradient_x = F.conv2d(images[:, 0, :, :].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:, 0, :, :].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))  # 计算梯度的模长
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)  # 填充边界
        thetas = torch.atan2(gradient_y, gradient_x)  # 计算梯度的角度
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    def getNormalEdge(self, normals):
        """提取法线图像的边缘信息"""
        n, c, h, w = normals.size()
        a = (
            torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .to(device)
            .view((1, 1, 3, 3))
            .repeat(3, 1, 1, 1)
        )
        b = (
            torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .to(device)
            .view((1, 1, 3, 3))
            .repeat(3, 1, 1, 1)
        )
        gradient_x = torch.abs(F.conv2d(normals, a, groups=c))
        gradient_y = torch.abs(F.conv2d(normals, b, groups=c))
        gradient_x = gradient_x.mean(dim=1, keepdim=True)  # 对每个通道求平均
        gradient_y = gradient_y.mean(dim=1, keepdim=True)  # 计算梯度的模长
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))  # 计算梯度的模长
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)  # 计算梯度的角度
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    def forward(self, gt_depths, images, inputs_normal, targets_normal):
        """
        inputs and targets: 表面法线图像
        images:RGB图像
        """
        masks = gt_depths > self.mask_value  # 有效深度区域

        inputs = inputs_normal
        targets = targets_normal
        # 从RGB图像中提取边缘
        edges_img, thetas_img = self.getEdge(images)
        # 从法线图像中提取边缘
        edges_normal, thetas_normal = self.getNormalEdge(targets)
        mask_img_border = torch.ones_like(edges_normal)  # 法线图像的边界区域
        mask_img_border[:, :, 5:-5, 5:-5] = 0  # 边缘区域不包括边框5像素
        edges_normal[mask_img_border.bool()] = 0 # 清除边框区域的法线边缘
        # 从深度图像中提取边缘
        edges_depth, _ = self.getEdge(gt_depths)
        edges_depth_mask = edges_depth.ge(edges_depth.max() * 0.1)  # 边缘深度
        edges_mask_dilate = torch.clamp(
            torch.nn.functional.conv2d(
                edges_depth_mask.float(),
                self.kernel, padding=(1, 1)
            ),
            0,
            1,
        ).bool()  # 扩展边缘掩码
        edges_normal[edges_mask_dilate] = 0  # 清除扩展后的边缘
        edges_img[edges_mask_dilate] = 0
        # =============================
        n, c, h, w = targets.size()
        inputs = inputs.contiguous().view(n, c, -1).double()  # 扁平化输入
        targets = targets.contiguous().view(n, c, -1).double()  # 扁平化目标
        masks = masks.contiguous().view(n, -1)  # 扁平化掩码
        edges_img = edges_img.contiguous().view(n, -1).double()  # 扁平化边缘图像
        thetas_img = thetas_img.contiguous().view(n, -1).double()  # 扁平化角度图像
        edges_normal = edges_normal.view(n, -1).double()  # 扁平化法线边缘
        thetas_normal = thetas_normal.view(n, -1).double()  # 扁平化法线角度

        # # initialization
        # loss = torch.DoubleTensor([0.0]).to(device)
        losses = []
        for i in range(n):
            # 执行边缘引导采样
            (
                inputs_A,
                inputs_B,
                targets_A,
                targets_B,
                masks_A,
                masks_B,
                sample_num,
                row_img,
                col_img,
            ) = edgeGuidedSampling(
                inputs[i, :],
                targets[i, :],
                edges_img[i],
                thetas_img[i],
                masks[i, :],
                h,
                w,
            )
            (
                normal_inputs_A,
                normal_inputs_B,
                normal_targets_A,
                normal_targets_B,
                normal_masks_A,
                normal_masks_B,
                normal_sample_num,
                row_normal,
                col_normal,
            ) = edgeGuidedSampling(
                inputs[i, :],
                targets[i, :],
                edges_normal[i],
                thetas_normal[i],
                masks[i, :],
                h,
                w,
            )

            # 结合EGS + EGNS（边缘引导采样 + 法线引导采样）
            inputs_A = torch.cat((inputs_A, normal_inputs_A), 1)
            inputs_B = torch.cat((inputs_B, normal_inputs_B), 1)
            targets_A = torch.cat((targets_A, normal_targets_A), 1)
            targets_B = torch.cat((targets_B, normal_targets_B), 1)
            masks_A = torch.cat((masks_A, normal_masks_A), 0)
            masks_B = torch.cat((masks_B, normal_masks_B), 0)

            # 只计算有有效GT的点对
            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A & masks_B

            # GT的顺序关系（夹角余弦值）
            target_cos = torch.abs(torch.sum(targets_A * targets_B, dim=0))  # 计算目标的夹角余弦值
            input_cos = torch.abs(torch.sum(inputs_A * inputs_B, dim=0))  # 计算输入的夹角余弦值
            # 计算损失
            losses += [torch.abs(target_cos - input_cos)]

        # 返回所有样本的平均损失
        loss = torch.cat(losses, dim=0).mean()
        return loss
