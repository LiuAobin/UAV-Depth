import json
from torch import optim
# 导入各种优化器
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
# 导入各种学习率调度器
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler
# 导入优化器参数
from .optim_constant import optim_parameters
from ..utils import print_log

# 定义可用的学习率调度器列表
timm_schedulers = [
    CosineLRScheduler, MultiStepLRScheduler, StepLRScheduler, TanhLRScheduler
]


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    """
    获取模型参数组
    """

    parameter_group_names = {}  # 用于存储参数组的名称
    parameter_group_vars = {}  # 用于存储每个参数组的变量
    # 遍历模型所有参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 如果参数不需要梯度（被冻结）——跳过
        # 判断是否是偏置项或者需要跳过的参数
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"  # 偏置和一维参数不使用权重衰减
            this_weight_decay = 0.
        else:  # 其他参数使用权重衰减
            group_name = "decay"
            this_weight_decay = weight_decay
        # 如果提供了get_num_layer函数，根据层号调整组名
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        # 如果该组还没有初始化，则创建该组
        if group_name not in parameter_group_names:
            if get_layer_scale is not None:  # 如果提供了get_layer_scale函数，获取该层的缩放因子
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.
            # 初始化该组的参数
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
        # 将当前参数加入对应的参数组
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # 打印参数组信息
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())  # 返回所有参数组


def get_optim_scheduler(args, epoch, models, steps_per_epoch):
    """
    获取优化器和学习率调度器的函数
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    opt_lower = args.opt.lower()  # 获取优化器名字并转换为小写
    weight_decay = args.weight_decay  # 获取梯度衰减系数
    parameters = []
    for model in models:
        # 过滤掉偏置和归一化项
        if args.filter_bias_and_bn:
            if hasattr(model, 'no_weight_decay'):
                skip = model.no_weight_decay()  # 获取需要跳过的参数
            else:
                skip = {}
            # 获取带有权重衰减的参数
            parameters += get_parameter_groups(model, weight_decay, skip)
            weight_decay = 0.  # 不使用权重衰减
        else:  # 对所有参数进行权重衰减
            parameters += model.parameters()

    # 获取优化器参数
    opt_args = optim_parameters.get(opt_lower, dict())

    opt_args.update(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    # 根据优化器名称选择对应的优化器
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    else:  # 不存在优化器，则报错
        assert False and "Invalid optimizer"

    # 如果优化器名称包含lookahead，包裹优化器
    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    # 获取学习率调度器的类型
    sched_lower = args.sched.lower()
    total_steps = epoch * steps_per_epoch  # 计算总步骤数量
    by_epoch = True  # 默认按epoch更新学习率
    if sched_lower == 'onecycle':
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps // len(args.gpus),
            final_div_factor=getattr(args, 'final_div_factor', 1e4))
        by_epoch = False
    elif sched_lower == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epoch,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch,
            t_in_epochs=True,  # 按epoch更新学习率
            k_decay=getattr(args, 'lr_k_decay', 1.0))
    elif sched_lower == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=epoch,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch,
            t_in_epochs=True)  # 按epoch更新学习率
    elif sched_lower == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epoch,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch)
    elif sched_lower == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=args.decay_epoch,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch)
    else:
        assert False and "Invalid scheduler"  # 如果调度器不合法，则报错

    # 返回优化器、学习率调度器和是否按epoch更新学习率
    print_log(f'lr_scheduler: {lr_scheduler}')
    print_log(f'optimizer: {optimizer}')
    return optimizer, lr_scheduler, by_epoch
