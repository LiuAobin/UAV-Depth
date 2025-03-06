import torch
from torch import nn
import numpy as np
class BackProjectDepth(nn.Module):
    """
    将深度图转换为点云
    """
    def __init__(self,batch_size, height, width):
        super(BackProjectDepth, self).__init__()
        # 保存输入
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # 创建网格坐标：通过 meshgrid 创建一个 [width, height] 的网格
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        # 形状：[2, height, width]
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.register_buffer("id_coords", torch.from_numpy(id_coords))
        # 不依赖 batch_size
        ones = torch.ones(1,1,self.height * self.width)
        self.register_buffer("ones", ones)

        pix_coords = torch.stack(
            [id_coords[0].view(-1), id_coords[1].view(-1)], 0)
        # Shape: [1, 2, H*W]
        self.register_buffer("pix_coords", pix_coords.unsqueeze(0))

    def forward(self, depth, inv_K):
        batch_size = depth.shape[0]
        # 复制坐标，使其batch维度匹配输入
        pix_coords = self.pix_coords.repeat(batch_size, -1, -1)
        ones = self.ones.repeat(batch_size, 1, 1)

        # 拼接3D像素坐标
        pix_coords = torch.cat([pix_coords, ones], 1)
        # 计算相机坐标
        cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords)  # [B, 3, H*W]
        cam_points = depth.view(batch_size, 1, -1) * cam_points  # 乘上深度
        cam_points = torch.cat([cam_points, ones], 1)  # 添加齐次坐标

        return cam_points


class Project3D(nn.Module):
    """
    将3D点投影到2D像素坐标
    该层将三维点投影到相机的二维图像平面中，使用给定的内参矩阵 K 和外参矩阵 T 进行变换。
    """

    def __init__(self,batch_size, height, width, eps=1e-7):
        """
        初始化投影层
        参数：
        - height: 图像高度
        - width: 图像宽度
        - eps: 一个小值，用于避免除零错误（默认 1e-7）
        """
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps  # 防止数值不稳定

    def forward(self, points, K, T):
        """
        前向传播：将3D点投影到2D图像平面

        参数：
        - points: 输入3D点云，形状为 [batch_size, 4, height * width]（齐次坐标）
        - K: 相机内参矩阵，形状为 [batch_size, 3, 3]
        - T: 相机位姿矩阵（4×4），形状为 [batch_size, 4, 4]

        返回：
        - pix_coords: 归一化后的2D像素坐标，形状为 [batch_size, height, width, 2]，值范围为 [-1, 1]
        """
        # 1. 投影矩阵
        # 计算投影矩阵 P = K * T，得到形状 [batch_size, 3, 4]
        P = torch.matmul(K, T)[:, :3, :]
        # 2. 计算相机坐标系下的投影点
        # 使用投影矩阵 P 变换 3D 点云，形状变为 [batch_size, 3, height * width]
        cam_points = torch.matmul(P, points)

        # 3. 计算像素坐标（归一化相机坐标）
        # 取前两个维度并除以深度 cam_points[:, 2, :]（确保除数不为零）
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)

        # 重新调整形状为 [batch_size, 2, height, width]
        pix_coords = pix_coords.view(points.shape[0], 2, self.height, self.width)

        # 调整维度顺序为 [batch_size, height, width, 2]，以符合图像数据格式
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        # 归一化像素坐标，使其范围在 [-1, 1] 之间
        pix_coords[..., 0] /= (self.width - 1)  # 归一化 x 轴
        pix_coords[..., 1] /= (self.height - 1)  # 归一化 y 轴
        pix_coords = (pix_coords - 0.5) * 2  # 转换到 [-1, 1] 范围

        return pix_coords