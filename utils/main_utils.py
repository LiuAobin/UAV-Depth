import numpy as np
import logging
import os


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