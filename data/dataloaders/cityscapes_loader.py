import json
import os

import imageio
import numpy as np
from path import Path
from skimage import transform

from .base_loader import BaseLoader


class CityscapesLoader(BaseLoader):
    def __init__(self,
                 config):
        """
        初始化Cityscape数据集
        :param config: 配置参数
        """
        super().__init__(config)
        self.split = config.split  # 数据集划分
        # Crop out the bottom 25% of the image to remove the car logo
        self.crop_bottom = config.crop_bottom  # 是否裁剪底部车标——裁剪图片25%
        # 获取该划分下的所有城市场景序列
        self.scenes = Path(os.path.join(config.dataset_dir, 'leftImg8bit_sequence', self.split)).dirs()
        print('Total scenes collected: {}'.format(len(self.scenes)))

    def collect_scenes(self, city):
        """
        收集并整理城市中的所有场景数据
        :param city: 城市对象
        :return: 连接的场景数据列表
        """
        # 获取所有的图像文件,按名称排序
        img_files = sorted(city.files('*.png'))
        scenes = {}  # 包含所有帧
        connex_scenes = {}  # 分段连续帧
        connex_scene_data_list = []
        # 将文件按场景id分组
        for f in img_files:
            scene_id, frame_id = f.basename().split('_')[1:3]
            if scene_id not in scenes.keys():
                scenes[scene_id] = []
            scenes[scene_id].append(frame_id)

        # 将每个场景划分为连接的子序列
        for scene_id in scenes.keys():
            previous = None
            connex_scenes[scene_id] = []
            for id in scenes[scene_id]:
                # 如果当前帧和上一帧不是连续的，则开始新的子序列
                if previous is None or int(id) - int(previous) > 1:
                    current_list = []
                    connex_scenes[scene_id].append(current_list)
                current_list.append(id)
                previous = id

        # 为每个连接的子场景创建数据字典，并每隔两帧取一个子场景
        for scene_id in connex_scenes.keys():
            intrinsics = self.load_intrinsics(city, scene_id)
            for subscene in connex_scenes[scene_id]:
                # 获取每个子场景的帧速率
                frame_speeds = [self.load_speed(city, scene_id, frame_id) for frame_id in subscene]
                connex_scene_data_list.append({'city': city,
                                               'scene_id': scene_id,
                                               'rel_path': f'{city.basename}_{scene_id}_{subscene[0]}_0',
                                               'intrinsics': intrinsics,
                                               'frame_ids': subscene[0::2],  # 每隔两帧取一个子序列
                                               'speeds': frame_speeds[0::2]})
                connex_scene_data_list.append({'city': city,
                                               'scene_id': scene_id,
                                               'rel_path': f'{city.basename}_{scene_id}_{subscene[0]}_1',
                                               'intrinsics': intrinsics,
                                               'frame_ids': subscene[1::2],  # 每隔两帧取一个子序列
                                               'speeds': frame_speeds[1::2]})
        return connex_scene_data_list

    def load_intrinsics(self, city, scene_id):
        """
        加载相机内参
        :param city: 城市对象
        :param scene_id: 场景ID
        :return: 相机内参矩阵
        """
        city_name = city.basename()
        # 相机内参文件路径
        camera_folder = Path(self.dataset_dir).joinpath('camera',self.split,city_name)
        camera_file = camera_folder.files(f'{city_name}_{scene_id}_*_camera.json')[0]
        frame_id = camera_file.split('_')[2]  # 获取帧ID
        # 获取对应帧的图像路径
        frame_path = city.joinpath(f'{city_name}_{scene_id}_{frame_id}_leftImg8bit.png')

        # 加载相机内参文件
        with open(camera_file, 'r') as f:
            camera = json.load(f)

        # 获取内参矩阵的元素
        fx = camera['intrinsic']['fx']
        fy = camera['intrinsic']['fy']
        u0 = camera['intrinsic']['u0']
        v0 = camera['intrinsic']['v0']

        # 构造相机内参矩阵
        intrinsics = np.array([[fx, 0, u0],
                               [0, fy, v0],
                               [0, 0, 1]])

        # 读取图像以便调整内参矩阵
        img = imageio.v2.imread(frame_path)
        h, w, _ = img.shape
        zoom_y = self.img_height / h
        zoom_x = self.img_width / w

        # 调整内参矩阵
        intrinsics[0] *= zoom_x
        intrinsics[1] *= zoom_y
        return intrinsics

    def load_speed(self, city, scene_id, frame_id):
        """
        加载指定帧的车辆速度
        :param city: 城市对象
        :param scene_id: 场景ID
        :param frame_id: 帧ID
        :return: 车辆速度
        """
        city_name = city.basename()
        vehicle_folder = Path(self.dataset_dir).joinpath('vehicle_sequence',self.split,city_name)
        vehicle_file = vehicle_folder.joinpath(f'{city_name}_{scene_id}_{frame_id}_vehicle.json')
        with open(vehicle_file, 'r') as f:
            vehicle = json.load(f)
        return vehicle['speed']

    def get_scene_imgs(self, scene_data):
        """
        获取指定场景的所有图像
        :param scene_data: 场景数据
        :yield: 处理后的图像及帧ID
        """
        cum_speed = np.zeros(3)  # 累积速度向量
        print(scene_data['city'].basename(), scene_data['scene_id'], scene_data['frame_ids'][0])
        for i, frame_id in enumerate(scene_data['frame_ids']):
            cum_speed += scene_data['speeds'][i]  # 更新累积速度
            speed_mag = np.linalg.norm(cum_speed)  # 计算速度的大小
            if speed_mag > self.min_speed:  # 如果速度大于最小速度，则返回图像
                yield {"img": self.load_image(scene_data['city'], scene_data['scene_id'], frame_id),
                       "id": frame_id}
                cum_speed *= 0  # 重置累积速度

    def load_image(self, city, scene_id, frame_id):
        """
        加载指定的图像
        :param city: 城市对象
        :param scene_id: 场景ID
        :param frame_id: 帧ID
        :return: 处理后的图像
        """
        img_file = city.joinpath(f'{city.basename()}_{scene_id}_{frame_id}_leftImg8bit.png')
        if not img_file.isfile():
            return None  # 如果文件不存在，返回None
        img = imageio.v2.imread(img_file)
        # 调整图像大小，并裁剪掉底部的部分（如果设置了crop_bottom）
        img = transform.resize(img, (self.img_height, self.img_width))[:int(self.img_height * 0.75)]
        return img
