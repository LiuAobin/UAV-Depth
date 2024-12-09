
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

    def generate_depth_map(self,scene_data,tgt_idx):
        # 生成深度图
        pass