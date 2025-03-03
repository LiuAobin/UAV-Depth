#------------------------ 基础配置
# config_file = './configs/sc_depth/kitti.py'  # 额外的配置文件所在路径
work_dir='./work_dirs'  # 工作目录
exp_name = f'depth_midair_sc'  # 实验名称
# seed = 8848  # 随机数种子
# num_threads = 6  # 加载数据时使用的线程数
# device = 'cuda'  # 使用gpu进行运算
# gpus=[0]
resume=True  # 是否恢复训练
ckpt_path="/data/coding/SC-Depth/work_dirs/depth_midair_sc/checkpoints/last-v9.ckpt"  # 模型恢复时检查点路径
# log_step=1  # 设置每隔多久保存一次最佳检查点
resnet_layers = 18  # resnet层数
#------------------------ 数据集信息
dataset_dir='/data/coding/MidAir'  # 数据集根路径
# dataset_dir='E:/Depth/origin-dataset/MidAir'  # 数据集根路径
dataset_name='midair'  # 数据集名称[kitti,midair]
channels=3  # 输入图像通道数
height=1024  # 输入图像高度
width=1024  # 输入图像宽度
img_suffix='*.JPEG'
depth_suffix='*.PNG'
limit_val_batches=0.5  # 验证时限制的批次大小float类型，1.0表示使用全部验证数据
sequence_length=3  # 训练时使用的序列长度
skip_frames=1  # 序列的帧间间隔
# use_frame_index=False  # 是否使用帧索引
#------------------------ 训练信息
epochs=500  # 训练回合数
epoch_size=2500  # 每回合大小
batch_size=4  # 每个批次大小
val_mode='depth'  # 验证模式，'depth' or 'photo'
folder_type='sequence'  # 数据集类型，'sequence' or 'pair'
method='sc-depth'  # 模型
# test=False  # 是否只进行测试
# no_display_method_info=False  # 是否显示方法信息
# fps=True  # 显示帧率
# metric_for_bestckpt='val_loss'  # 检查那个损失指标作为保持最佳检查点的信息
#------------------------ 优化器信息
opt='adamw'  # 优化器
lr= 1e-3  # 学习率
warmup_lr= 4e-5  # 预热学习率
min_lr= 1e-6  # 最小的学习率
lr_scheduler='onecycle'  # 学习率调度器
final_div_factor=1e4  # min_lr = initial_lr/final_div_factor for onecycle scheduler
# decay_epoch=20  # 学习率衰减周期
# decay_rate=0.1  # 学习率衰减速率
# warmup_epoch=10  # 预热周期
# lr_k_decay=1.0  # learning rate k-decay for cosine/poly (default: 1.0)
# opt_eps=None  # Optimizer epsilon (default: None, use opt default)
# momentum=0.9  # SGD动量参数，Adam的alpha参数
# opt_betas=None  # Adam优化器的beta参数
# filter_bias_and_bn=False  # Whether to set the weight decay of bias and bn to 0
# weight_decay=1e-5  # 梯度衰减值
# clip_gard=0  # 梯度裁剪范数
#------------------------ 其他信息
# 'zeros表示在超出目标图像范围时梯度为0，border表示仅在x或y坐标超出时梯度为0
# padding_mode='zeros'  #图像扭曲时的填充模式，对于光度差异化来说，当坐标超出目标图像范围时使用该模式.
# with_gt=True  # 验证时是否使用地面真值
# 损失函数相关
# no_ssim=False  # use ssim in photometric loss
# no_auto_mask=False  # masking invalid static points
# no_dynamic_mask=False  # masking dynamic regions
# no_min_optimize=False  # optimize the minimum loss
photo_weight=1.0  # photometric loss weight
geometry_weight=0.1  # geometry loss weight
smooth_weight=0.1  # smoothness loss weight
# sc-depth-v3的参数
# load_pseudo_depth=False  # 是否使用伪深度

#------------------------ 数据预处理相关信息
# origin_data_dir=''  # 数据集原始文件目录
# dump_dir = ''  # 处理后的数据集的存放路径
# static_frames=''  # 指定要丢弃的静态帧列表
# test_scenes=''  # 测试场景所需要的帧
# data_format=''  # 数据集格式
# with_depth=''  # 如果存在深度信息，则保存深度真值，供验证使用
# no_train_gt=False  # 是否删除训练数据集中的深度真值以节省空间
# with_pose=False  # 如果存在位姿信息，则保存姿态地面真值，以供验证使用
# depth_size_ratio=1  # 深度大小的缩放比例
# min_speed=2  # 最小速度,当没有静态帧文件时，使用该参数，去除速度小于该帧的文件
