# 文件路径: /data/wangbo/CissMOS/utils/new_loss.py
# 最终修正版

import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    一个更鲁棒的交叉熵损失函数，能处理不同的输入形状并稳健地处理设备问题。
    """
    def __init__(self, weight=None, ignore_index=-100):
        # __init__ 中不再调用 super().__init__，因为我们将在 forward 中完全重写逻辑
        # 以便更好地控制权重设备
        super(RobustCrossEntropyLoss, self).__init__(weight=None, ignore_index=ignore_index)
        # 将权重注册为 buffer
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.FloatTensor(weight)
        self.register_buffer('weight', weight)

    def forward(self, input, target, reduction=None):
        # 调整目标张量的形状
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target.squeeze(1)
        target = target.long()

        # --- 根本性修正 ---
        # 获取内部权重，并确保它和输入张量在同一个设备上
        current_weight = self.weight
        if current_weight is not None:
            current_weight = current_weight.to(input.device)
        # --- 修正结束 ---
        final_reduction = reduction if reduction is not None else self.reduction
        # 直接调用底层的 functional cross_entropy，并传入正确设备上的权重
        return F.cross_entropy(input, target,
                               weight=current_weight,
                               ignore_index=self.ignore_index,
                               reduction=final_reduction)

    def update_weight(self, weight):
        """
        用于动态更新类别权重的方法。
        """
        if weight is not None:
            # 只更新 buffer 的值，设备问题将在 forward 中处理
            self.weight = weight


class DiceLoss(nn.Module):
    """
    计算Dice损失。期望输入为logits，目标为索引图（index map）。
    """
    def __init__(self, n_classes, weight=None, soft_max=True, target_is_soft=False):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.soft_max = soft_max
        self.target_is_soft = target_is_soft
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.FloatTensor(weight)
        self.register_buffer('weight', weight)

    def forward(self, inputs, target):
        if not self.target_is_soft:
            # 默认行为：将 target 从 index map 转为 one-hot
            target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=self.n_classes).permute(0, 4, 1, 2, 3).float()
        else:
            # 新增行为：当 target 已经是软概率图时，直接使用
            target_one_hot = target
        
        if self.soft_max:
            inputs = F.softmax(inputs, dim=1)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(inputs * target_one_hot, dims)
        cardinality = torch.sum(inputs + target_one_hot, dims)
        dice_score = 2. * intersection / (cardinality + 1e-6)

        dice_loss_per_class = 1. - dice_score
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            dice_loss_per_class = weight.view(1, -1) * dice_loss_per_class if len(weight.shape) == 1 else weight * dice_loss_per_class
        
        return dice_loss_per_class.mean() 

    def update_weight(self, weight):
        """
        用于动态更新类别权重的方法。
        """
        if weight is not None:
            self.weight = weight


class DC_and_CE_loss(nn.Module):
    """
    Dice损失和交叉熵损失的组合。
    """
    def __init__(self, n_classes, w_dc=None, w_ce=None, ignore_idx=-100):
        super().__init__()
        self.ce = RobustCrossEntropyLoss(weight=w_ce, ignore_index=ignore_idx)
        self.dc = DiceLoss(n_classes=n_classes, weight=w_dc)

    def forward(self, pred, target):
        dc_loss = self.dc(pred, target)
        ce_loss = self.ce(pred, target)
        return dc_loss + ce_loss

    def update_weight(self, weight_ce, weight_dc):
        """
        将权重更新的指令传递给内部的CE和Dice两个组件。
        """
        self.ce.update_weight(weight_ce)
        self.dc.update_weight(weight_dc)