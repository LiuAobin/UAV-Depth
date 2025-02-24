from __future__ import division
import torch
import torch.nn.functional as F
from kornia.geometry.depth import depth_to_3d


def euler2mat(angle):
    """
    将欧拉角转换为旋转矩阵
    <https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174>
    Args:
        angle (): 旋转角度，按三个轴的顺序给出，单位是弧度--size = [B,3]
    Returns:
        对应于欧拉角的旋转矩阵，--size = [B,3,3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
    # 计算每个轴的旋转角度的余弦和正弦值
    # 1. Z轴
    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zeros = z.detach() * 0
    ones = z.detach() + 1
    # 1.1 Z轴旋转矩阵
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)
    # 2. Y轴
    cosy = torch.cos(y)
    siny = torch.sin(y)
    # 2.1 Y轴旋转矩阵
    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)
    # 3. X轴
    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)
    # 综合X，Y，Z轴旋转矩阵得到最终的旋转矩阵
    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """
    将四元数系数转换为旋转矩阵
    Args:
        quat (): 旋转前的前三个四元数系数。第四个系数将被计算并进行归一化处理 --size = [B,3]
    Returns:
        对应于四元数的旋转矩阵 --size = [B,3,3]
    """
    # 归一化四元数，确保它的范数为1
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    # 四元数系数的平方
    w2, x2, y2, z2 = x.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    # 计算旋转矩阵
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_made='euler'):
    """
    将6DoF参数转换为变换矩阵
    Args:
        vec (): 6DoF参数，顺序为tx, ty, tz, rx, ry, rz -- [B,6]
        rotation_made (): 变换模式
    Returns:
        变换矩阵 -- [B,3,4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # 提取平移部分 -- [B,3,1]
    rot = vec[:, 3:]  # 提取旋转部分
    if rotation_made == 'euler':
        rot_mat = euler2mat(rot)  # 使用欧拉角计算旋转矩阵  -- [B,3,3]
    elif rotation_made == 'quat':
        rot_mat = quat2mat(rot)  # 使用四元数计算旋转矩阵 -- [B,3,3]
    # 拼接旋转矩阵和平移向量形成变换矩阵 -- [B,3,4]
    transform_mat = torch.cat([rot_mat, translation], dim=2)
    return transform_mat


def inverse_warp(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    将源图像逆向扭曲到目标图像平面
    Args:
        img (): 源图像(带采样图像) -- [B, 3, H, W]
        depth (): 目标图像深度图 -- [B, 1, H, W]
        ref_depth (): 源图像的深度图(用于采样深度) -- [B,1,H,W]
        pose (): 从目标到源的6DoF位姿参数 -- [B,6]
        intrinsics (): 相机内参矩阵 -- [B,3,3]
        padding_mode (): 填充模式，默认0填充
    Returns:
        projected_img: 逆向扭曲到目标图像平面的源图像
        projected_depth: 从源图像采样的深度
        computed_depth: 使用目标图像深度计算得到的源图像深度
    """
    B, _, H, W = img.size()
    T = pose_vec2mat(pose)  # 获取变换矩阵  -- [B,3,4]
    P = torch.matmul(intrinsics, T)[:, :3, :]  # 计算投影矩阵

    world_points = depth_to_3d(depth, intrinsics)  # 从目标深度图获取视世界坐标 -- [B,3,H,W]
    world_points = torch.cat([world_points, torch.ones(B, 1, H, W).type_as(img)], 1)
    cam_points = torch.matmul(P, world_points.view(B, 4, -1))  # 将世界坐标转换到相机坐标

    # 将相机坐标映射到像素坐标
    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2  # 将像素坐标标准化到[-1,1]范围

    computed_depth = cam_points[:, 2, :].unsqueeze(1).view(B, 1, H, W)  # 计算深度图

    # 使用网格采样对源图像和深度图进行采样
    projected_img = F.grid_sample(img, pix_coords,
                                  padding_mode=padding_mode, align_corners=False)
    projected_depth = F.grid_sample(ref_depth, pix_coords,
                                    padding_mode=padding_mode, align_corners=False)
    return projected_img, projected_depth, computed_depth


def inverse_ration_warp(img, rot, intrinsics, padding_mode='zeros'):
    """
    使用旋转矩阵将源图像逆向扭曲
    Args:
        img (): 源图像(带采样图像) -- [B, 3, H, W]
        rot (): 旋转角度 -- [B,3] (单位：弧度)
        intrinsics (): 相机内参矩阵 -- [B,3,3]
        padding_mode ():
    Returns:
        projected_img: 逆向扭曲到目标图像平面的源图像
    """
    B, _, H, W = img.size()
    R = euler2mat(rot)  # 使用欧拉角获取旋转矩阵 [B,3,3]
    P = torch.matmul(intrinsics, R)  # 计算投影矩阵
    # 使用单位深度获取世界坐标 -- [B,3,H,W]
    world_points = depth_to_3d(torch.ones(B, 1, H, W).type_as(img), intrinsics)
    cam_points = torch.matmul(P, world_points.view(B, 3, -1))  # 将世界坐标转换为相机坐标
    # 将相机坐标映射到像素坐标
    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2  # 将像素坐标标准化到[-1.1]范围
    # 使用网格采样对源图像进行采样
    projected_img = F.grid_sample(img, pix_coords,
                                  padding_mode=padding_mode, align_corners=True)
    return projected_img
