import torch
from torch import nn

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class Mask_Ranking_Loss(nn.Module):
    def __init__(self, sample_ratio=0.1, filter_depth=1e-8):
        """
        初始化 Mask Ranking Loss 类。
        Args:
            sample_ratio (): 用于控制随机采样的比例，绝对了生成掩码的数量
            filter_depth (): 用于过滤深度图中的无效值，避免除零错误
        """
        super(Mask_Ranking_Loss, self).__init__()
        self.sample_ratio = sample_ratio
        self.filter_depth = filter_depth

    def generate_global_target(self, depth, pred, theta=0.15):
        """
        生成全局目标，用于计算全局对比损失
        Args:
            depth (): 真实深度图
            pred (): 预测深度图
            theta (): 对比损失的阈值，用于确定正负样本
        Returns:
            用于计算损失的预测深度，目标和掩码
        """
        B, C, H, W = depth.shape
        # 随机生成掩码
        mask_A = torch.rand(C, H, W).to(device)
        mask_A[mask_A >= (1 - self.sample_ratio)] = 1
        mask_A[mask_A < (1 - self.sample_ratio)] = 0
        idx = torch.randperm(mask_A.nelement())
        mask_B = mask_A.view(-1)[idx].view(mask_A.size())
        # 扩展掩码并与输入的深度图匹配
        mask_A = mask_A.repeat(B, 1, 1).view(depth.shape) == 1
        mask_B = mask_B.repeat(B, 1, 1).view(depth.shape) == 1
        # 提取掩码对应的深度值
        za_gt = depth[mask_A]
        zb_gt = depth[mask_B]

        # 过滤掉深度小于设定阈值的区域
        mask_ignoreb = zb_gt > self.filter_depth
        mask_ignorea = za_gt > self.filter_depth
        mask_ignore = mask_ignorea | mask_ignoreb
        za_gt = za_gt[mask_ignore]
        zb_gt = zb_gt[mask_ignore]

        # 计算相对深度的对比标志
        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt

        # 根据相对深度的对比，生成正负样本
        mask1 = flag1 > 1 + theta
        mask2 = flag2 > 1 + theta
        target = torch.zeros(za_gt.size()).to(device)
        target[mask1] = 1
        target[mask2] = -1

        return pred[mask_A][mask_ignore], pred[mask_B][mask_ignore], target

    def generate_percentMask_target(self, depth, pred, invalid_mask, theta=0.15):
        """
        生成基于百分比掩码的目标，用于计算对比损失
        Args:
            depth (): 真实深度图
            pred (): 预测深度图
            invalid_mask (): 无效像素的掩码
            theta (): 对比损失的阈值
        Returns:
            用于计算损失的预测深度，目标和掩码
        """
        B, C, H, W = depth.shape
        valid_mask = ~invalid_mask  # 取反，得到有效区域的掩码
        # 初始化容器
        gt_inval, gt_val, pred_inval, pred_val = None, None, None, None
        for bs in range(B):
            gt_invalid = depth[bs, :, :, :]
            pred_invalid = pred[bs, :, :, :]
            # 获取无效区域的深度和预测
            mask_invalid = invalid_mask[bs, :, :, :]
            gt_invalid = gt_invalid[mask_invalid]
            pred_invalid = pred_invalid[mask_invalid]

            gt_valid = depth[bs, :, :, :]
            pre_valid = pred[bs, :, :, :]
            # 获取有效区域的蛇毒和预测
            mask_valid = valid_mask[bs, :, :, :]
            gt_valid = gt_valid[mask_valid]
            pre_valid = pre_valid[mask_valid]

            # 生成随机索引，用于选择相同数量的有效样本
            # generate the sample index. index range -> (0, len(gt_valid)). The amount -> gt_invalid.size()
            idx = torch.randint(0, len(gt_valid), gt_invalid.size())
            gt_valid = gt_valid[idx]
            pre_valid = pre_valid[idx]

            if bs == 0:
                gt_inval, gt_val, pred_inval, pred_val = gt_invalid, gt_valid, pred_invalid, pre_valid
                continue
            # 合并每一批次的样本
            gt_inval = torch.cat((gt_inval, gt_invalid), dim=0)
            gt_val = torch.cat((gt_val, gt_valid), dim=0)
            pred_inval = torch.cat((pred_inval, pred_invalid), dim=0)
            pred_val = torch.cat((pred_val, pre_valid), dim=0)
        # 计算相对深度的对比标志
        za_gt = gt_inval
        zb_gt = gt_val

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 >= 1 + theta
        mask2 = flag2 > 1 + theta
        target = torch.zeros(za_gt.size()).to(device)
        target[mask1] = 1
        target[mask2] = -1

        return pred_inval, pred_val, target

    def cal_ranking_loss(self, z_A, z_B, target):
        """
        计算给定像素对的排列误差
        Args:
            z_A (): A像素的预测深度
            z_B (): B像素的预测深度
            target (): A和B之间的相对深度(-1,0,1)
        Returns: 排序损失和有效点的数量

        """
        pred_depth = z_A - z_B
        log_loss = torch.sum(  # 计算排序顺手，使用对数损失函数
            torch.log(1 + torch.exp(-target[target != 0] * pred_depth[target != 0])))
        pointNum = len(target[target != 0])
        return log_loss, pointNum

    def get_unreliable(self, tgt_valid_weight):
        """
        生成无效区域掩码，根据目标的有效权重，标记为无效的区域
        Args:
            tgt_valid_weight (): 目标有效权重
        Returns:无效区域掩码
        """
        B, C, H, W = tgt_valid_weight.shape
        unreliable_percent = 0.2  # 设定无效区域的百分比
        invalidMask = torch.ones_like(tgt_valid_weight)
        for bs in range(B):
            weight = tgt_valid_weight[bs]
            maskIv = invalidMask[bs]
            weight = weight.view(-1)
            maskIv = maskIv.view(-1)

            weight_sorted, indices = torch.sort(weight)
            # 根据权重排序，选择最小的百分比作为无效区域
            # each item in indices represent an index(valid)
            indices[:int(unreliable_percent*H*W)] = indices[H*W-1]
            # use indices for the selection. mask=0 -> valid
            maskIv[indices] = 0

        return invalidMask > 0

    def get_textureWeight(self, tgt_img):
        """
        计算纹理权重，用于估计图像的纹理信息
        Args:
            tgt_img (): 目标图像
        Returns: 计算的纹理权重
        """
        grad_img_x = torch.mean(
            torch.abs(tgt_img[:, :, :, :-1] - tgt_img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(tgt_img[:, :, :-1, :] - tgt_img[:, :, 1:, :]), 1, keepdim=True)
        # 计算纹理权重
        textureWeight = torch.zeros_like(tgt_img[:, :1, :, :])
        textureWeight[:, :, :, :-1] = textureWeight[:, :, :, :-1] + grad_img_x
        textureWeight[:, :, :-1, :] = textureWeight[:, :, :-1, :] + grad_img_y
        textureWeight = textureWeight / 2.0
        return textureWeight

    def forward(self, pred_depth, gt_depth, tgt_valid_weight):
        """
        计算整个 Mask Ranking Loss
        Args:
            pred_depth (): 预测深度图
            gt_depth (): 真实深度图
            tgt_valid_weight (): 目标有效权重
        Returns: 计算得到的总损失
        """
        # 动态掩码
        unreliableMask = self.get_unreliable(tgt_valid_weight)

        # 计算基于百分比掩码的损失
        za_1, zb_1, target_1 = self.generate_percentMask_target(
            gt_depth, pred_depth, unreliableMask)
        loss_percentMask, pointNum_1 = self.cal_ranking_loss(
            za_1, zb_1, target_1)

        # 计算全局对比损失
        za_2, zb_2, target_2 = self.generate_global_target(
            gt_depth, pred_depth)
        loss_global, pointNum_2 = self.cal_ranking_loss(za_2, zb_2, target_2)
        # 计算总损失
        total_loss = (loss_global + loss_percentMask)/(pointNum_2 + pointNum_1)
        return total_loss
