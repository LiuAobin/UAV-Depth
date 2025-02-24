import numpy as np
from imageio.v2 import imread
from scipy import sparse


def load_as_float(path):
    """
    读取图像并转换为float32类型
    :param path:文件路径
    :return: 图像的np数组
    """
    return imread(path).astype(np.float32)


def generate_sample_index(num_frames, skip_frames=1, sequence_length=3):
    """
    生成每个场景中样本的索引，返回包含一个目标图像和参考图像索引的列表
    :param num_frames: 总帧数
    :param skip_frames:跳帧数，决定参考图像与目标图像之间的间隔
    :param sequence_length:每个样本的序列长度
    :return:
        list:包含目标图像索引和参考图像索引的字典列表
    """
    sample_index_list = []
    k = skip_frames
    demi_length = (sequence_length - 1) // 2  # 序列长度的一半
    shifts = list(range(-demi_length * k, demi_length * k + 1, k))  # 根据skip_frames生成偏移量
    shifts.pop(demi_length)  # 去掉目标图像对应的偏移量

    if num_frames > sequence_length:
        for i in range(demi_length * k, num_frames - demi_length * k):
            # 遍历每一帧，生成目标图像和参考图像的索引
            sample_index = {'tgt_idx': i, 'ref_idx': []}
            for j in shifts:
                sample_index['ref_idx'].append(i + j)
            sample_index_list.append(sample_index)

    return sample_index_list


def load_sparse_depth(filename):
    """
    加载稀疏深度图数据，并将其转换为密集模式(dense)
    :param filename:(str) 深度图的路径，npz文件
    :return:(ndarray)转换后的密集深度图
    """
    sparse_depth = sparse.load_npz(filename)
    depth = sparse_depth.todense()
    return np.array(depth)


def crawl_folder(folder, dataset='ddad',img_suffix='*.jpg',depth_suffix='*.png'):
    """
    遍历给定的文件夹，加载图像和对应的深度图
    :param folder: 数据集所在文件夹路径
    :param dataset: 数据集名称
    :return:(tuple)包含图像路径和深度图路径的元组。图像路径列表和深度图路径列表
    """
    # 加载图像文件
    imgs = sorted(folder.joinpath('color').files(img_suffix))
    # 根据不同数据集加载不同深度图文件
    if dataset in ['nyu']:  # NYU数据集使用PNG格式的深度图
        depths = sorted(folder.joinpath('depth').files(depth_suffix))
    elif dataset in ['kitti']:  # KITTI使用NPY格式的深度图
        depths = sorted(folder.joinpath('depth').files('*.npy'))
    elif dataset in ['ddad']:  # DDAD使用NPZ格式数据集
        depths = sorted(folder.joinpath('depth').files(depth_suffix))
    else:
        raise ValueError('Unsupported dataset type: {}'.format(dataset))
    # 返回图像和深度图路径列表
    return imgs, depths


def crawl_folders(folders_lost, dataset='nyu',img_suffix='*.jpg',depth_suffix='*.png'):
    """
    遍历文件夹列表，加载图像和对应的深度图
    :param folders_lost: 文件夹路径列表，每个文件夹包含多个图像和深度图文件
    :param dataset: 数据集名称
    :return: 包含图像和深度图路径的元组
    Args:
        img_suffix (): 图像文件后缀
        depth_suffix (): 深度文件后缀
    """
    imgs = []
    depths = []
    for folder in folders_lost:
        # 加载当前文件夹下的所有图像文件，jpg格式
        current_imgs = sorted(folder.files(img_suffix))
        # 根据数据集类型选择不同的深度图文件格式
        if dataset in ['nyu', 'bonn', 'tum', 'cityscapes']:  # PNG格式深度图
            current_depth = sorted(folder.joinpath('depth').files(depth_suffix))
        elif dataset in ['ddad', 'kitti']:  # NPZ格式深度图
            current_depth = sorted(folder.joinpath('depth').files(depth_suffix))
        imgs.extend(current_imgs)  # 添加当前文件夹的图像路径
        depths.extend(current_depth)  # 添加当前文件夹的深度图路径
    return imgs, depths  # 返回所有图像和深度图的路径列表
