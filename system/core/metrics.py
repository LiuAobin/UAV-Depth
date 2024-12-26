import torch
import torch.nn.functional as F
import numpy as np
@torch.no_grad()
def compute_metrics(gt, pred, dataset):
    """
    计算预测深度与真实深度之间的误差
    Args:
        gt (): 真实深度
        pred (): 预测深度
        dataset (): 数据集类型
    Returns: 所有误差指标的平均值
    mean[abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3，log10]
    """
    abs_diff = abs_rel = sq_rel = rmse = rmse_log = a1 = a2 = a3 = log10 = 0
    batch_size, h, w = gt.shape

    # 如果预测深度图和真实深度图大小不一致，进行插值
    if pred.nelement() != gt.nelement():
        pred = F.interpolate(pred, [h, w], mode='bilinear', align_corners=False)

    pred = pred.view(batch_size, h, w)

    # 根据不同的数据集类型定义有效的掩码
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[x1:x2, y1:y2] = 1
        max_depth = 80
    elif dataset == 'ddad':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 200
    elif dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        crop = np.array([45, 471, 41, 601]).astype(np.int32)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        max_depth = 10
    elif dataset == 'bonn':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 10
    elif dataset == 'tum':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 10
    else:
        raise ValueError('Unknown dataset')

    min_depth = 0.1
    # 遍历计算评价指标
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > min_depth) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        # 进行尺度对齐
        valid_pred = valid_pred * torch.median(valid_gt) / torch.median(valid_pred)

        valid_pred = valid_pred.clamp(min=min_depth, max=max_depth)
        # 计算各类误差
        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = valid_gt - valid_pred
        abs_diff += torch.mean(
            torch.abs(diff_i)
        )
        abs_rel += torch.mean(
            torch.abs(diff_i) / valid_gt
        )
        sq_rel += torch.mean(
            torch.pow(diff_i, 2) / valid_gt
        )
        rmse += torch.sqrt(
            torch.mean(
                torch.pow(diff_i, 2)
            ))
        rmse_log += torch.sqrt(
            torch.mean(
                torch.pow(torch.log(valid_gt) - torch.log(valid_pred), 2)
            ))
        log10 += torch.mean(
            torch.abs(
                torch.log10(valid_gt) - torch.log10(valid_pred)
            )
        )
    # 返回所有误差指标的平均值mean[abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3，log10]
    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, log10]]
