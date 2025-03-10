from torch import nn


class PoseCNN(nn.Module):
    """
    用于位姿估计（Pose Estimation）
    输入：多个连续帧图像
    输出：相邻帧之间的旋转(axis angle)和平移(translation)
    """
    def __init__(self,num_input_frames):
        super(PoseCNN, self).__init__()
        self.num_input_frames = num_input_frames
        # 定义卷积层
        self.convs = {}
        self.convs[0] = nn.Conv2d(3*num_input_frames,16,
                                  7,2,3)
        self.convs[1] = nn.Conv2d(16,32,
                                  5,2,2)
        self.convs[2] = nn.Conv2d(32,64,
                                  3,2,1)
        self.convs[3] = nn.Conv2d(64,128,
                                  3,2,1)
        self.convs[4] = nn.Conv2d(128,256,
                                  3,2,1)
        self.convs[5] = nn.Conv2d(256,256,
                                  3,2,1)
        self.convs[6] = nn.Conv2d(256,256,
                                  3,2,1)
        # 最终用于位姿回归的卷积层，输出6*(num_input_frames-1)个通道
        self.pose_conv = nn.Conv2d(256,6*(num_input_frames-1),1)

        self.num_convs = len(self.convs) # 卷积层数量
        self.relu = nn.ReLU(inplace=True) # 激活函数
        # 将卷积层存入 ModuleList，保证参数可被训练
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, x):
        """
        前向传播计算
        Args:
            x: 输入的图像张量，形状为(B,3*num_input_frames,H,W)
        Returns:
            轴角和平移。
            shape:[B,num_input_frames-1,1,6]
        """
        # 特征提取
        for i in range(self.num_convs):
            x = self.net[i](x)
            x = self.relu(x)
        # 回归最终位姿
        out = self.pose_conv(x)
        # 进行全局平均池化
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1,self.num_input_frames-1,1,6)
        axisangle = out[...,:3]
        translation = out[...,3:]
        return axisangle, translation
