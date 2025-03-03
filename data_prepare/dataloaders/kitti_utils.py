import numpy as np


def rotx(t):
    """
    绕x轴旋转
    """
    c = np.cos(t)  # 旋转角度t的余弦和正弦值
    s = np.sin(t)
    # 返回旋转矩阵
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """
    绕y轴旋转
    """
    c = np.cos(t)  # 旋转角度t的余弦和正弦值
    s = np.sin(t)
    # 返回旋转矩阵
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """
    绕x轴旋转
    """
    c = np.cos(t)  # 旋转角度t的余弦和正弦值
    s = np.sin(t)
    # 返回旋转矩阵
    return np.array([[c, -s, 0],  # 返回旋转矩阵，绕 z 轴旋转
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    """
    从旋转矩阵和平移向量生成变换矩阵
    :param R: 旋转矩阵
    :param t: 平移向量
    :return:
    """
    R = R.reshape(3, 3)  # 旋转矩阵转换为3x3矩阵
    t = t.reshape(3, 1)  # 位移向量转换为列向量
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))  # 组合成4×4矩阵


def pose_from_oxts_packet(metadata, scale):
    """
    计算从全局坐标系到机器人当前位姿的变换矩阵
    Helper method to compute a SE(3) pose matrix from an OXTS packet.
    Taken from https://github.com/utiasSTARS/pykitti
    :return:
    :rtype:
    """
    # 解包元数据-(纬度，经度，海拔，滚转，俯仰，偏航角)
    lat, lon, alt, roll, pitch, yaw = metadata
    er = 6378137.  # 地球半径(单位：m)

    # 使用墨卡托投影来计算平移向量
    ty = lat * np.pi * er / 180.  # 纬度转换为米
    tx = scale * lon * np.pi * er / 180.  # 经度转换为m
    tz = alt  # 海拔/高度
    t = np.array([tx, ty, tz]).reshape(-1, 1)  # 形成平移向量

    # 使用欧拉角(roll,pitch,yaw)计算旋转矩阵
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))  # 组合旋转矩阵
    return transform_from_rot_trans(R, t)


def read_calib_file(path):
    """
    读取校准文件并解析
    来自 https://github.com/hunse/kitti
    :param path:
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)  # 按‘:’分割键值对
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    pass  # 如果转换失败，则跳过
    return data