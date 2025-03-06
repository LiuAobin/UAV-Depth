import torch





def transformation_from_parameters(axisangle,translation, invert):
    """
    将网络预测的(axisangle, translation) 转换为一个 4×4 变换矩阵
    Convert the parameters to transformation matrix
    Args:
        axisangle (Tensor): (B,1,3)
        translation (Tensor): (B,1,3)
        invert ():
    Returns:
        变换矩阵
        R,t
        0,1
    """
    # 轴角-> 旋转矩阵(B,3,3)
    R = rot_from_axisangle(axisangle)
    # 复制平移向量，避免修改原始值
    t = translation.clone()

    if invert:  # 旋转矩阵转置，平移矩阵反向
        R = R.transpose(1,2)
        t *= -1

    # 生成平移矩阵
    T = get_translation_matrix(t)

    # 计算最终变换矩阵
    if invert: # 用于相机预测，从(t=1)到(t=0)
        M = torch.matmul(T,R) # 先旋转再平移
    else: # 用于相机预测，从(t=0)到(t=1)
        M = torch.matmul(R,T) # 先平移再旋转
    return M


def rot_from_axisangle(vec):
    """
    将轴角（axis-angle）表示的旋转转换为 4×4 旋转矩阵。
    (adapted from https://github.com/Wallacoloo/printipi)
    Args:
        vec (Tensor): (B,1,3)
    Returns:(B,4,4)
    """
    # 计算旋转角度和旋转轴
    angle = torch.norm(vec, 2, 2, True)  # 计算旋转角度(B,1,1)
    axis = vec / (angle + 1e-7)  # 归一化旋转轴(B,1,3)
    #  计算旋转矩阵的 Rodrigues 公式参数
    ca = torch.cos(angle)  # 计算 cos(θ) (B,1,1)
    sa = torch.sin(angle)  # 计算 sin(θ) (B,1,1)
    C = 1 - ca  # 计算 1 - cos(θ) (B,1,1)

    # 提取旋转轴的x,y,z分量
    x = axis[...,0].unsqueeze(1)
    y = axis[...,1].unsqueeze(1)
    z = axis[...,2].unsqueeze(1)

    # 构建旋转矩阵的各个元素
    xs = x * sa # x * sin(θ)
    ys = y * sa # y * sin(θ)
    zs = z * sa # z * sin(θ)
    xC = x * C # x * (1 - cos(θ))
    yC = y * C # y * (1 - cos(θ))
    zC = z * C # z * (1 - cos(θ))
    xyC = x * yC # x * y * (1 - cos(θ))
    yzC = y * zC # y * z * (1 - cos(θ))
    zxC = z * xC # z * x * (1 - cos(θ))
    # 构建旋转矩阵
    rot = torch.zeros((vec.shape[0],4,4)).to(vec.device)
    # 填充旋转矩阵
    rot[:,0,0] = torch.squeeze(x * xC + ca) # R[0,0]
    rot[:,0,1] = torch.squeeze(xyC - zs) # R[0,1]
    rot[:,0,2] = torch.squeeze(zxC + ys) # R[0,2]

    rot[:,1,0] = torch.squeeze(xyC + zs) # R[1,0]
    rot[:,1,1] = torch.squeeze(y * yC + ca) # R[1,1]
    rot[:,1,2] = torch.squeeze(yzC - xs) # R[1,2]
    rot[:,2,0] = torch.squeeze(zxC - ys) # R[2,0]
    rot[:,2,1] = torch.squeeze(yzC + xs) # R[2,1]
    rot[:,2,2] = torch.squeeze(z * zC + ca) # R[2,2]
    rot[:,3,3] = 1
    return rot

def get_translation_matrix(translation):
    """
    将 3D 平移向量 转换为 4×4 齐次变换矩阵
    Args:
        translation (Tensor):(B,3)

    Returns:

    """
    T = torch.zeros((translation.shape[0],4,4)).to(translation.device)
    # 提取并调整平移向量的形状
    t = translation.contiguous().view(-1,3,1)
    # 填充齐次平移矩阵的对角线
    T[:,0,0] = 1
    T[:,1,1] = 1
    T[:,2,2] = 1
    T[:,3,3] = 1
    T[:,0:3,3,None] = t  # 填充平移部分

    return T


def convert_K_to_4x4(K_3x3):
    """
    将 3x3 内参矩阵 转换为 4x4 齐次变换矩阵
    Args:
        K_3x3 ():
    Returns:
    """
    K_4x4 = torch.zeros((K_3x3.shape[0],4,4)).to(K_3x3.device)
    K_4x4[:,:3,:3] = K_3x3
    # 设置最后一列和最后一行
    K_4x4[:,3,3] = 1.0
    return K_4x4












































