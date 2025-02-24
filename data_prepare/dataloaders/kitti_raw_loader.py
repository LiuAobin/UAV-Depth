import os
from collections import Counter
from glob import glob

import cv2
import imageio.v2
import numpy as np

from .base_loader import BaseLoader
from utils import *


class KittiRawLoader(BaseLoader):
    """
    Kitti数据集加载类
    """

    def __init__(self, config):
        super().__init__(config)

        # 静态帧处理
        self.from_speed = config.static_frames is None
        if config.static_frames is not None:  # 如果提供了静态帧文件
            self.collect_static_frames(config.static_frames)  # 收集静态帧
        # 测试场景数据处理
        # 测试场景数据
        test_scene_file = config.test_scenes
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t.strip() for t in test_scenes]  # :-1去除末尾的换行符
        # 设置相机id和日期列表
        self.cam_ids = ['02', '03']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        self.collect_train_folders()

    def collect_train_folders(self):
        """
        收集训练文件夹
        :return: 完整的文件目录列表
        """
        self.scenes = []
        for date in self.date_list:
            drive_set = [d for d in os.scandir(os.path.join(self.dataset_dir, date)) if d.is_dir()]
            for dr in drive_set:
                if dr.name[:-5] not in self.test_scenes:
                    self.scenes.append(dr.path)

    def collect_static_frames(self, static_frames):
        """
        收集静态帧
        :param static_frames:
        :type static_frames:
        :return:
        :rtype:
        """
        with open(static_frames, 'r') as f:  # 打开文件
            frames = f.readlines()  # 读取所有静态帧
        self.static_frames = {}  # 初始化静态帧列表
        for fr in frames:
            if fr == '\n':
                continue  # 跳过空行
            date, device, frame_id = fr.split(' ')  # 提取日期，驱动器和帧ID
            curr_fid = f'{np.int16(frame_id[:-1]):d}'  # :-1 是为了去除回车
            if device not in self.static_frames.keys():
                self.static_frames[device] = []
            self.static_frames[device].append(curr_fid)

    def collect_scenes(self, drive):
        """
        收集一个驾驶记录的所有帧数据
        """
        train_scenes = []  # 初始化一个空列表来保存场景数据
        for c in self.cam_ids:  # 变量摄像头
            # 获取oxt系统数据 获取IMU数据文件列表
            oxts = sorted(glob(os.path.join(drive, 'oxts', 'data', '*.txt')))

            # 初始化摄像机场景数据
            scene_data = {
                'cid': c,
                'dir': drive,
                'speed': [],
                'frame_id': [],
                'pose': [],
                'rel_path': os.path.basename(drive) + '_' + c
            }
            scale = None  # 比例尺
            origin = None  # 初始化原点
            # 读取标定文件
            imu2velo = read_calib_file(os.path.join(os.path.dirname(drive), 'calib_imu_to_velo.txt'))
            velo2cam = read_calib_file(os.path.join(os.path.dirname(drive), 'calib_velo_to_cam.txt'))
            cam2cam = read_calib_file(os.path.join(os.path.dirname(drive), 'calib_cam_to_cam.txt'))
            # 转换标定矩阵
            velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])  # 雷达到摄像头
            imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])  # IMU到雷达
            cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))  # 摄像头到校正图像
            imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat  # 计算IMU到摄像头的总转换矩阵 @:矩阵乘法

            # 处理每一帧imu数据
            for n, f in enumerate(oxts):
                metadata = np.genfromtxt(f)  # 使用np读取imu数据文件
                speed = metadata[8:11]  # 提取速度数据 即[vf,vl,vu]
                scene_data['speed'].append(speed)
                scene_data['frame_id'].append(f'{n:010d}')  # 生成帧ID，并添加到元组
                lat = metadata[0]  # 纬度
                if scale is None:  # 如果没有定义比例尺，则根据维度计算比例尺
                    scale = np.cos(lat * np.pi / 180.)

                # 计算姿态矩阵
                pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
                if origin is None:  # 如果没有定义原点，则以第一帧的姿态矩阵作为原点
                    origin = pose_matrix
                # 计算相对于原点的位姿
                odo_pose = imu2cam @ np.linalg.inv(origin) @ pose_matrix @ np.linalg.inv(imu2cam)
                scene_data['pose'].append(odo_pose[:3])  # 添加相对位姿

            # 加载图片样本
            sample = self.load_image(scene_data, 0)  # 加载第一帧的图像
            if sample is None:
                return []
            # 获取内参矩阵
            scene_data['P_rect'] = self.get_P_rect(scene_data, sample[1], sample[2])
            scene_data['intrinsics'] = scene_data['P_rect'][:, :3]
            train_scenes.append(scene_data)

        return train_scenes

    def load_image(self, scene_data, tgt_idx):
        """
        加载图像并进行缩放
        :param scene_data:场景信息
        :param tgt_idx:索引，目标帧在帧id列表中的位置
        :return:缩放后的图像
        """
        # 加载图像并进行缩放
        img_file = os.path.join(scene_data['dir'], f'image_{scene_data["cid"]}', 'data',
                                f'{scene_data["frame_id"][tgt_idx]}.png')
        if not os.path.isfile(img_file):
            return None  # 检测文件是否存在
        img = imageio.v2.imread(img_file)
        # 计算缩放因子
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
        img = cv2.resize(img, (self.img_width, self.img_height))
        return img, zoom_x, zoom_y

    def get_P_rect(self, scene_data, zoom_x, zoom_y):
        # 获取相机的投影矩阵
        calib_file = os.path.join(os.path.dirname(scene_data['dir']), 'calib_cam_to_cam.txt')
        file_data = read_calib_file(calib_file)
        P_rect = np.reshape(file_data['P_rect_' + scene_data['cid']], (3, 4))
        # 根据缩放因子调整矩阵
        P_rect[0] *= zoom_x
        P_rect[1] *= zoom_y
        return P_rect

    def get_scene_imgs(self, scene_data):
        """
        获取一个场景中的图像和深度信息
        :param scene_data: 场景数据
        :return:
        :rtype:
        """

        def construct_sample(scene_data, i, frame_id):
            sample = {'img': self.load_image(scene_data, i)[0],
                      'id': frame_id}
            if self.get_depth:
                sample['depth'] = self.generate_depth_map(scene_data, i)
            if self.get_pose:
                sample['pose'] = scene_data['pose'][i]
            return sample

        if self.from_speed:  # 从速度数据生成，累积速度并根据最小速度生成
            cum_speed = np.zeros(3)
            for i, speed in enumerate(scene_data['speed']):
                speed_mag = np.linalg.norm(speed)
                if speed_mag > self.min_speed:
                    frame_id = scene_data['frame_id'][i]
                    yield construct_sample(scene_data, i, frame_id)
                    cum_speed *= 0
        else:
            drive = str(os.path.basename(scene_data['dir']))
            for (i, frame_id) in enumerate(scene_data['frame_id']):
                if (drive not in self.static_frames.keys()) or (frame_id not in self.static_frames[drive]):
                    yield construct_sample(scene_data, i, frame_id)

    def generate_depth_map(self, scene_data, tgt_idx):
        # 生成深度图
        def sub2bind(maxtrixSize, rowSub, colSub):
            """
            用于将矩阵坐标转换为索引
            :param maxtrixSize: 矩阵大小
            :param rowSub: 行坐标
            :param colSub: 列坐标
            :return:
            """
            m, n = maxtrixSize
            return rowSub * (n - 1) + colSub - 1

        # 创建一个单位矩阵，用于投影
        R_cam2rect = np.eye(4)

        # 1. 读取标定文件
        calib_dir = os.path.dirname(scene_data['dir'])
        cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        # 将点云数据转化为左侧的相机坐标系，也用做相机中点云坐标系的表示
        velo2cam = np.hstack(  # 将旋转矩阵R和平移矩阵T平坦化
            (velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis])
        )
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # 2. 计算点云到图像平面的投影矩阵
        P_rect = np.copy(scene_data['P_rect'])
        P_rect[0] /= self.depth_size_ratio  # 根据深度尺寸进行比例处理
        P_rect[1] /= self.depth_size_ratio
        # 校准旋转矩阵，使得图像平面共面
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        # 计算点云到图像的投影矩阵
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        # 3. 加载点云图像
        velo_file_name = os.path.join(scene_data['dir'], 'velodyne_points','data', f'{scene_data["frame_id"][tgt_idx]}.bin')
        velo = np.fromfile(velo_file_name, dtype=np.float32).reshape(-1, 4)  # 前三个值为(x,y,z)，最后一个值是反射率信息
        velo[:, 3] = 1  # 将最后一列置为1，这用于在之后的转换中构建齐次坐标
        velo = velo[velo[:, 0] >= 0, :]  # 仅保留x轴正半轴的数据，因为只有这部分在摄像头的视野中

        # 4. 点云投影
        # 施加投影矩阵，转换到图像平面
        velo_pts_im = np.dot(P_velo2im, velo.T).T
        # 最后一列元素（此处为齐次坐标的一部分）用于归一化前两列，使得其在相应的图像平面内
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, -1:]
        # 检查投影结果是否在图像边界
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < self.img_width / self.depth_size_ratio)
        val_inds = val_inds & (velo_pts_im[:, 1] < self.img_height / self.depth_size_ratio)
        velo_pts_im = velo_pts_im[val_inds, :]  # 确保投影在图像内

        # 创建一个全0矩阵depth，其大小与原图比例缩小self.depth_size_ratio倍，数据类型为float32
        depth = np.zeros((self.img_height // self.depth_size_ratio,
                         self.img_width // self.depth_size_ratio)).astype(np.float32)
        # 利用投影到图像的坐标，填充深度图中的像素深度
        depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]
        # 重复点问题，找到最近的深度值
        inds = sub2bind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])  # 将行列坐标转换为索引
        dupe_inds = [item for item, count in Counter(inds).items() if count > 0]
        for dd in dupe_inds:  # 遍历重复索引列表
            pts = np.where(inds == dd)[0]  # 找出所有和当前索引重复的点
            x_loc = int(velo_pts_im[pts[0], 0])  # 获取重复点坐标
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()  # 设置重复点深度值最小的为该点的真实深度值
        depth[depth < 0] = 0  # 将负值设置为0
        return depth


if __name__ == '__main__':
    from utils import create_parser

    config = create_parser()
    kitti_raw_loader = KittiRawLoader(config)
    for scene in kitti_raw_loader.scenes:
        scene_list = kitti_raw_loader.collect_scenes(scene)
        for scene_data in scene_list:
            samples = kitti_raw_loader.get_scene_imgs(scene_data)
            for sample in samples:
                print(sample['id'])
