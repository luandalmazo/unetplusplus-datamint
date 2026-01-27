from matplotlib import use
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, softmax
import numpy as np

"""
    Taken from: https://github.com/aswahd/TMJ-Disk-Dislocation-Classification/blob/main/UNetPPTMJ/dataloading/TMJDataset.py
"""
class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        """
        input and targets of shape:
                (N, C, H, W)
        """
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 0.001
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weights=None, use_softmax=False, **kwargs):
        super().__init__()
        self.weights = weights
        self.kwargs = kwargs
        self.softmax = use_softmax

    def forward(self, input, target):
        """
        input tesor of shape = (N, C, H, W)
        target tensor of shape = (N, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        nclass = input.shape[1]
        if self.weights is None:
            weights = torch.tensor(np.ones(nclass))
        else:
            weights = torch.tensor(self.weights)
        weights = weights / weights.sum()
        
        target = target.to(input.device)
        target = oneHot(target, n_classes=nclass)

        assert input.shape == target.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        if self.softmax:
            logits = softmax(input, dim=1)
        else:
            logits = input

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(nclass):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i]) * weights[i]
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss


def oneHot(targets: torch.Tensor, n_classes: int):
    """
    targets: (N, H, W) - torch.long
    returns: (N, C, H, W) - one-hot encoded
    """
    device = targets.device

    targets = targets.long()
    targets = targets.unsqueeze(1)  # (N,1,H,W)

    one_hot = torch.zeros(
        targets.size(0),
        n_classes,
        targets.size(2),
        targets.size(3),
        device=device,
        dtype=torch.float32,
    )

    one_hot.scatter_(1, targets, 1.0)
    return one_hot
