import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np
import logging
import os

import torch
import torchvision


def concat_image_seq(seq):
    """
    将一系列图像进行水平堆叠-即将图像拼接到一张图中
    要求：：所有图像具有相同的高度
    :param seq: 带堆叠的图像序列列表
    :type seq:
    :return: 堆叠后的图像
    :rtype:
    """
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
        return res


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    return path


def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    """
    格式化命名空间中的配置信息
    Args:
        namespace (): 存储配置的命名空间对象

    Returns: 格式化后的字符串信息

    """
    configs = namespace.__dict__  # 将命名空间转换为字典
    message = ''
    # 遍历字典中的键值对，并格式化为字符串
    for key, value in configs.items():
        message += f'\n{key}: {str(value)}'
    return message


def collect_env():
    """
    收集运行环境的相关信息
    Returns:
        env_info (dict):包含当前环境信息的字典
    """
    env_info = {
        'sys.platform': sys.platform,  # 平台系统信息
        'Python': sys.version.replace('\n', '')  # Python版本信息
    }

    # 检查是否支持CUDA
    cuda_available = torch.cuda.is_available()
    env_info['cuda_available'] = cuda_available
    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        # CUDA按照目录
        env_info['CUDA_HOME'] = CUDA_HOME
        # 检查cuda_home是否有效
        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                # 获取NVCC版本信息
                nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
                nvcc = subprocess.check_output('"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc
        # 获取可用的GPU信息
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name,devids in devices.items():
            env_info['GPU'+','.join(devids)] = name
            env_info['GPU Total Memory'] = torch.cuda.get_device_properties(int(devids[0])).total_memory

    # 获取GCC的版本信息
    try:
        # 检测 GCC 版本
        gcc_version = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc_version = gcc_version.decode('utf-8').strip()
        env_info['GCC Version'] = gcc_version

        # 检测 GCC 编译器路径
        gcc_path = subprocess.check_output('which gcc', shell=True)
        gcc_path = gcc_path.decode('utf-8').strip()
        env_info['GCC Path'] = gcc_path
    except subprocess.SubprocessError:
        env_info['GCC Version'] = 'Not Available'
        env_info['GCC Path'] = 'Not Available'

    # Pytorch相关信息
    env_info['Pytorch'] = torch.__version__
    env_info['Pytorch compiling details'] = torch.__config__.show()
    # TorchVision版本
    env_info['TorchVersion'] = torchvision.__version__
    # Deep Learning 框架特定信息（如 NCCL, cuDNN 等）
    try:
        env_info['cuDNN Version'] = torch.backends.cudnn.version()
        env_info['cuDNN Enabled'] = torch.backends.cudnn.is_available()
    except AttributeError:
        env_info['cuDNN Version'] = 'Not Available'
    # OpenCV版本
    env_info['OpenCV'] = cv2.__version__

    return env_info


def measure_throughput(model,input_dummy):
    """
    测量模型的吞吐量（Throughput）
    Args:
        model (): 要评估的深度学习模型
        input_dummy (): 模拟输入参数，
    Returns:
        Throughout: 模型每秒处理的样本数
    """
    def get_batch_size(H,W):
        """
        根据输入的高度H和宽度W计算批次大小和重复次数(repetitions)
        Args:
            H (): 输入张量的高度
            W (): 输入张量的宽度
        Returns:
            bs: 批量大小
            repetitions: 测量重复次数
        """
        max_side = max(H,W)  # 获取输入的最大边长
        if max_side >= 128:
            bs = 10  # 当最大边长较大时，批量大小较小
            repetitions = 1000  # 重复次数较多
        else:
            bs = 100  # 当最大边长较小时，批量大小较大
            repetitions = 100  # 重复次数较大
        return bs,repetitions

    # 判断输入是否是元组类型
    if isinstance(input_dummy,tuple):
        input_dummy = list(input_dummy)  # 将元组转变为列表，以便修改
        _,C,H,W = input_dummy[0].shape  # 获取输入张量的形状信息
        bs,repetitions = get_batch_size(H,W)  # 根据形状计算批量大小和重复次数
        # 创建具有相同设备和形状的随机张量作为新的模拟输入
        _input = torch.rand(bs,C,H,W).to(input_dummy[0].device)
        input_dummy[0] = _input # 替换列表中的第一个张量
        input_dummy = tuple(input_dummy)  # 将列表转换为元组
    else:
        # 如果输入是单个张量，直接获取其形状信息
        _,C,H,W = input_dummy.shape
        bs,repetitions = get_batch_size(H,W)  # 计算批量大小和重复次数
        # 创建具有相同设备和形状的随机张量作为新的模拟输入
        input_dummy = torch.rand(bs,C,H,W).to(input_dummy.device)

    total_time = 0  # 初始化总耗时为0
    with torch.no_grad():  # 禁用梯度计算以加速推理
        for _ in range(repetitions):  # 重复进行推理
            # 创建CUDA事件用于计时
            starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            starter.record()  # 开始记录事件
            if isinstance(input_dummy,tuple):
                _ = model(*input_dummy)  # 使用元组解包的方式进行推理
            else:
                _ = model(input_dummy)  # 单张量输入推理
            ender.record()
            torch.cuda.synchronize()  # 确保所有CUDA操作完成
            curr_time = starter.elapsed_time(ender)/1000  # 计算单次推理耗时 （单位：S）
            total_time += curr_time  # 累计总时间

    # 计算吞吐量：总样本数/总时间
    Throughput = (bs*repetitions)/total_time
    return Throughput

