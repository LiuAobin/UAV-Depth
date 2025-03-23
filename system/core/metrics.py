# 评估函数
import torch
import torch.nn.functional as F
import numpy as np


def get_crop_mask_and_depth_range(dataset, gt):
    """
    根据数据集类型返回裁剪掩码和最大深度值和最小深度值。
    """
    crop_mask = (gt[0] != gt[0])  # 初始化无效掩码
    min_depth=0.1
    if dataset == 'kitti':
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[x1:x2, y1:y2] = 1
        max_depth = 80
    elif dataset == 'ddad':
        crop_mask[:, :] = 1
        max_depth = 200
    elif dataset == 'nyu':
        crop = np.array([45, 471, 41, 601]).astype(np.int32)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        max_depth = 10
    elif dataset in ['bonn', 'tum']:
        crop_mask[:, :] = 1
        max_depth = 10
    elif dataset == 'midair':
        crop_mask[:, :] = 1
        min_depth,max_depth = 1,200
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    return crop_mask, max_depth, min_depth


def compute_depth_errors(valid_gt, valid_pred, min_depth, max_depth):
    """
    计算各种深度误差指标。
    """
    # 对预测深度值进行尺度对齐，确保预测值和真实值在同一尺度范围
    # torch.median() 计算张量中所有元素的中位数。
    valid_pred_median = torch.median(valid_pred)
    valid_gt_median = torch.median(valid_gt)

    # 防止除以零或中位数为零
    if valid_pred_median == 0 or torch.isnan(valid_pred_median):
        return None  # 返回None表示无效

    valid_pred = valid_pred * valid_gt_median / valid_pred_median
    valid_pred = valid_pred.clamp(min=min_depth, max=max_depth)  # 将预测深度值限制在合法的深度范围内，去除异常值

    # 计算误差指标
    thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
    metrics = {
        'da/a1': (thresh < 1.25       ).float().mean(),
        'da/a2': (thresh < (1.25 ** 2)).float().mean(),
        'da/a3': (thresh < (1.25 ** 3)).float().mean(),
        'de/abs_diff': torch.mean(torch.abs(valid_gt - valid_pred)),
        'de/abs_rel': torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt.clamp(min=1e-8)),
        'de/sq_rel': torch.mean(torch.pow(valid_gt - valid_pred, 2) / valid_gt.clamp(min=1e-8)),
        'de/rmse': torch.sqrt(torch.mean(torch.pow(valid_gt - valid_pred, 2))),
    }
    # 对数误差（避免对数中的零或负值）
    if torch.any(valid_gt <= 0) or torch.any(valid_pred <= 0):
        return None  # 返回None表示无效

    # rmse_log = torch.sqrt(torch.mean(torch.pow(torch.log(valid_gt) - torch.log(valid_pred), 2)))
    # log10 = torch.mean(torch.abs(torch.log10(valid_gt) - torch.log10(valid_pred)))
    metrics['de/rmse_log'] = torch.sqrt(torch.mean(torch.pow(torch.log(valid_gt) - torch.log(valid_pred), 2)))
    metrics['de/log10'] = torch.mean(torch.abs(torch.log10(valid_gt) - torch.log10(valid_pred)))
    # return a1, a2, a3, abs_diff, abs_rel, sq_rel, rmse, rmse_log, log10
    return metrics

@torch.no_grad()
def compute_metrics(gt, pred, dataset):
    """
    计算预测深度与真实深度之间的误差
    Args:
        gt (): 真实深度 [batch_size, H, W]
        pred (): 预测深度 [batch_size, H, W]
        dataset (): 数据集类型
    Returns: 所有误差指标的平均值
    mean[abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3，log10]
    """
    # abs_diff = abs_rel = sq_rel = rmse = rmse_log = a1 = a2 = a3 = log10 = 0
    # 使用字典来存储所有指标的累加值

    metrics = {'da/a1': 0, 'da/a2': 0, 'da/a3': 0, 'de/abs_diff': 0, 'de/abs_rel': 0,
               'de/sq_rel': 0, 'de/rmse': 0, 'de/rmse_log': 0, 'de/log10': 0}
    batch_size, h, w = gt.shape
    valid_samples = 0  # 用于统计有效样本的数量

    # 如果预测深度图和真实深度图大小不一致，进行插值
    if pred.nelement() != gt.nelement():
        pred = F.interpolate(pred, [h, w], mode='bilinear', align_corners=False)
    pred = pred.view(batch_size, h, w)

    # 根据不同的数据集类型定义有效的掩码
    crop_mask,max_depth,min_depth = get_crop_mask_and_depth_range(dataset,gt)
    # 遍历计算评价指标
    for current_gt, current_pred in zip(gt, pred):
        # 根据深度范围筛选有效像素
        valid = (current_gt > min_depth) & (current_gt < max_depth)
        valid = valid & crop_mask  # 结合裁剪掩码，筛选有效像素
        # 从真实深度图和预测深度图中提取有效像素对应的深度值。
        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        # 如果有效像素数为0，跳过该样本
        if valid_gt.numel() == 0 or valid_pred.numel() == 0:
            continue
        # 计算误差指标
        sample_metrics = compute_depth_errors(valid_gt, valid_pred, min_depth, max_depth)
        if sample_metrics is None:
            continue  # 如果计算出无效结果，则跳过该样本
        # 累加每个误差指标
        for key in metrics:
            metrics[key] += sample_metrics.get(key, 0)
        valid_samples += 1  # 增加有效样本数
        # 如果没有有效样本，则返回0
    if valid_samples == 0:
        return {key: 0.0 for key in metrics}
    # 返回所有误差指标的平均值mean[abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3，log10]
    # return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, log10]]
    # 使用有效样本数对所有误差指标进行归一化
    return {key: value / valid_samples for key, value in metrics.items()}


if __name__ == '__main__':
    batch_size, h, w = 2, 1024, 1024
    dataset = 'midair'
    total_metrics = {
        'a1': 0.0, 'a2': 0.0, 'a3': 0.0,
        'abs_diff': 0.0, 'abs_rel': 0.0, 'sq_rel': 0.0,
        'rmse': 0.0, 'rmse_log': 0.0, 'log10': 0.0
    }
    for i in range(10):
        # 随机生成真实深度图 GT，值在 [0.1, 1000] 之间
        gt = torch.rand((batch_size, h, w), dtype=torch.float32) * 999.9 + 0.1

        # 随机生成预测深度图 Pred，值在 [0.1, 1000] 之间
        pred = torch.rand((batch_size, h, w), dtype=torch.float32) * 999.9 + 0.1
        metrics = compute_metrics(gt, pred, dataset)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            total_metrics[metric] += metrics.get(metric, 0.0)
        # 计算平均误差
    avg_metrics = {key: value / 10 for key, value in total_metrics.items()}

    # 输出平均误差
    print(f"经过 {10} 次测试后的平均误差：")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")