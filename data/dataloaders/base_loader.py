
class BaseLoader(object):
    def __init__(self,config):
        """
        dataset_dir,
        static_frames_file=None,
        img_height=128,
        img_width=416,
        min_speed=2,
        get_depth=False,
        get_pose=False,
        depth_size_ratio=1
        :param config:
        :type config:
        """
        self.config = config
        self.dataset_dir = config.origin_data_dir  # 设置原始数据路径
        self.img_width = config.width  # 设置图像宽度
        self.img_height = config.height  # 设置图像高度
        self.min_speed = config.min_speed  # 设置最小速度
        self.get_depth = config.with_depth  # 是否需要真实深度
        self.get_pose = config.with_pose  # 是否需要真实位姿
        self.depth_size_ratio = config.depth_size_ratio  # 深度大小的放缩比例

    def collect_scenes(self,drive):
        pass

    def get_scene_imgs(self,scene_data):
        """
        获取指定场景的所有图像
        :param scene_data: 场景数据
        :yield: 处理后的图像及帧ID，以及其他相关信息
        """
        pass