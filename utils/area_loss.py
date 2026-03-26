import torch
import torch.nn as nn

# ADD
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [N, C, H, W] (经过Sigmoid/Softmax)
        # target: [N, C, H, W] (one-hot或二值)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

# ADD
class DynamicLossScheduler:
    def __init__(self, 
                 base_iou_weight=0.2,      # 初始IoU权重
                 max_iou_weight=1.0,       # 最大IoU权重
                 warmup_epochs=1,          # 预热阶段epoch数
                 step_epochs=2):           # 每隔多少epoch提升权重
        self.base_iou_weight = base_iou_weight
        self.max_iou_weight = max_iou_weight
        self.warmup_epochs = warmup_epochs
        self.step_epochs = step_epochs
        self.current_epoch = 0
        self.effective_weight = 0  # 初始有效权重

    def update(self, epoch):
        """在每个epoch开始时调用此方法更新权重"""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            self.effective_weight = 0.0
        else:
            # 计算已过的完整step数
            steps_completed = (epoch - self.warmup_epochs) // self.step_epochs
            # 线性增长到最大值
            self.effective_weight = min(
                self.base_iou_weight + steps_completed * self.step_epochs * 0.3,
                self.max_iou_weight
            )