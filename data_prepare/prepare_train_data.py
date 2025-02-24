import os
import shutil
from glob import glob

import imageio.v2
import numpy as np  # 用于数值计算和数组处理
from pebble import ProcessPool  # 用于并行处理
from tqdm import tqdm  # 用于实现进度条

from system.utils import check_dir
from utils import create_parser, update_config
from data.dataloaders import dataloader_map
"""
用于处理深度数据集，将每个场景的图像和相关信息保存到指定位置，并生成训练和验证的文件列表
验证集与测试集依照<https://github.com/JiawangBian/SC-SfMLearner-Release>进行处理
"""


def dump_example(args, scene, data_loader=None):
    """
    用于保存每个数据场景
    :param args: 
    :param scene: 
    :param data_loader: 
    """
    # 获取该场景的所有数据
    scene_list = data_loader.collect_scenes(scene)

    for scene_data in scene_list:  # 遍历每个场景数据
        dump_dir = os.path.join(str(args.dump_dir), scene_data['rel_path'])
        check_dir(dump_dir)  # 检查文件夹是否存在

        # 相机内参
        intrinsics = scene_data['intrinsics']
        # 相机内参文件路径
        dump_cam_file = os.path.join(dump_dir, 'cam.txt')
        np.savetxt(dump_cam_file, intrinsics)  # 保存相机内参

        # 状态信息
        poses_file = os.path.join(dump_dir, 'pose.txt')
        poses = []  # 用于存储姿态信息

        for sample in data_loader.get_scene_imgs(scene_data):  # 获取场景数据中的所有图像
            # 保存图像
            img, frame_nb = sample['img'], sample['id']  # 获取图像和帧编号
            dump_img_file = os.path.join(dump_dir, f'{frame_nb}.jpg')
            imageio.imsave(dump_img_file, img)

            #  保存姿态信息
            if 'pose' in sample.keys():
                poses.append(sample['pose'].tolist())
            # 保存深度
            if 'depth' in sample.keys():
                dump_depth_file = os.path.join(dump_dir, f'{frame_nb}.npy')
                # imageio.imsave(dump_depth_file, sample['depth'].astype(np.uint8))
                np.save(dump_depth_file, sample['depth'])

        # 将姿态信息写入到文件系统
        if len(poses) > 0:
            np.savetxt(poses_file, np.array(poses).reshape(-1, 12), fmt='%.6e')

        if len(glob(os.path.join(dump_dir, '*.jpg'))) < 3:
            shutil.rmtree(dump_dir)


def main():
    args = create_parser()
    args = update_config(args)
    print(args.__dict__)
    check_dir(args.dump_dir)  # 创建存储预处理数据结果的根目录

    data_loader = dataloader_map[args.data_format](args)
    n_scenes = len(data_loader.scenes)  # 获取场景数量
    print(f'Found {n_scenes} potential scenes')
    print('Retrieving frames')
    # 开始预处理数据
    if args.num_threads == 1:
        for scene in tqdm(data_loader.scenes):
            dump_example(args, scene, data_loader=data_loader)
    else:  # 多线程
        with ProcessPool(max_workers=args.num_threads) as pool:  # 创建一个线程池
            tasks = pool.map(dump_example, [args] * n_scenes, data_loader.scenes, [data_loader] * n_scenes)
            try:
                for _ in tqdm(tasks.result(), total=n_scenes):  # 显示任务进度
                    pass
            except KeyboardInterrupt as e:  # 捕获中断异常
                tasks.cancel()  # 取消任务
                raise e

    print('Generating train val lists')
    np.random.seed(args.seed)  # 设置随机数种子
    subfolders = os.listdir(args.dump_dir)
    subfolders_prefix = set(subfolder[:-2] for subfolder in subfolders)
    with open(os.path.join(args.dump_dir, 'train.txt'), 'w') as tf:
        with open(os.path.join(args.dump_dir, 'val.txt'), 'w') as vf:
            for pr in tqdm(subfolders_prefix):
                corresponding_dirs = glob(os.path.join(args.dump_dir, pr + '*'))
                if np.random.random() < 0.1:  # 以10%的概率放入验证集
                    for s in corresponding_dirs:
                        vf.write(f'{os.path.basename(s)}\n')
                else:
                    for s in corresponding_dirs:
                        tf.write(f'{os.path.basename(s)}\n')  # 写入训练集
                        # 如果需要深度信息且不保留训练集中的深度GT
                        if args.with_depth and args.no_train_gt:
                            for gt_file in glob(os.path.join(s, '*.npy')):
                                os.remove(gt_file)


if __name__ == '__main__':
    main()
