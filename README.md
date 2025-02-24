# 深度估计相关实验

1. 已完成：配置文件加载和读取
2. KITTI和cityscapes数据集的预处理

##  [数据集](./data/DATA.md)

## Model
TAU:Squeeze-and-excitation networks
VAN:Visual Attention Network
![img.png](imgs/img.png)
![img.png](imgs/img2.png)

![img.png](imgs/img3.png)
# 训练
## MidAir数据集
```shell
python -m tools.train --config_file ./configs/sc_depth/midair.py
```
# 可视化
Tensorboard
```shell
tensorboard --logdir=./work_dirs --host=0.0.0.0
```
Wandb

api_key:4186dee52f004546f3d3caaa4113d1a907afa21d
```shell
wandb.login(key=[your_api_key])
```

