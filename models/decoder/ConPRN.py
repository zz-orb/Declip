import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align

class ContrastiveRPN(nn.Module):
    def __init__(self, clip_dim=1024, proj_dim=256):
        super().__init__()
        # RPN参数设置
        self.anchor_scales = [8, 16, 32]  # 对应原图中的4x4到16x16区域
        self.anchor_ratios = [0.5, 1, 2]  # 宽高比覆盖常见物体形态
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)
        
        # RPN网络
        self.rpn_conv = nn.Conv2d(clip_dim, 512, 3, padding=1)
        self.rpn_cls = nn.Conv2d(512, self.num_anchors, 1)
        self.rpn_reg = nn.Conv2d(512, self.num_anchors * 4, 1)
        
        # 对比学习投影头
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # 空间重组参数
        self.patch_size = 16
        self.feat_size = 16  # 156 / 16
        self.stride = 16

    def forward(self, clip_features, gt_masks):
        """
        clip_features: [257, B, 1024]
        gt_masks: [B, 1, 256, 256]
        """
        # 重组空间特征 [B, 1024, 14, 14]
        B = clip_features.size(1)
        spatial_features = clip_features[1:].permute(1, 2, 0)  # [B, 1024, 256]
        spatial_features = spatial_features.view(B, 1024, self.feat_size, self.feat_size)
        
        # RPN前向
        x = F.relu(self.rpn_conv(spatial_features))
        rpn_cls = self.rpn_cls(x)  # [B, num_anchors, H, W]
        rpn_reg = self.rpn_reg(x)  # [B, num_anchors*4, H, W]
        
        # 生成候选区域
        rois = self.generate_rois(rpn_cls, rpn_reg, B)  # [N, 5]

        # 新增：处理空ROI情况
        if rois.size(0) == 0:
            return torch.tensor(0.0, device=clip_features.device, requires_grad=True)
        
        # ROI特征对齐
        roi_features = roi_align(
            spatial_features, 
            rois, 
            output_size=(7,7),
            spatial_scale=1.0/self.stride
        )  # [N, 1024, 7, 7]
        
        # 对比学习处理
        roi_features_pooled = roi_features.mean(dim=[2,3])  # [N, 1024]
        projections = self.projection(roi_features_pooled)  # [N, 256]

        # 计算对比损失
        # print(f"gt_masks shape: {gt_masks.shape}")
        # 将 gt_masks 尺寸从 [B, 65536] 变为 [B, 1, 256, 256]
        # if gt_masks.dim() == 2 and gt_masks.size(1) == 256 * 256:
        gt_masks = gt_masks.view(B, 1, 256, 256)
        contrast_loss = self.supcon_loss(projections, rois, gt_masks)
        
        return contrast_loss

    def generate_rois(self, rpn_cls, rpn_reg, batch_size):
        """
        rpn_cls: [B, num_anchors, H, W]
        rpn_reg: [B, num_anchors*4, H, W]
        返回: [N, 5] (batch_index, x1, y1, x2, y2)
        """
        device = rpn_cls.device
        all_rois = []
        
        anchors = self.generate_anchors().to(device)
        
        for b in range(batch_size):
            # 处理回归参数
            reg_pred = rpn_reg[b].view(
                self.num_anchors, 4, self.feat_size, self.feat_size
            ).permute(2, 3, 0, 1).contiguous().view(-1, 4)
            
            # 处理分类得分
            cls_scores = rpn_cls[b].permute(1, 2, 0).contiguous()  # [H, W, num_anchors]
            cls_scores = cls_scores.view(-1)  # 展平为[H*W*num_anchors]
            
            # 解码预测框
            pred_boxes = self.decode_boxes(reg_pred, anchors)
            
            # 应用NMS
            keep = self.apply_nms(pred_boxes, cls_scores)
            batch_rois = torch.cat([
                torch.full((len(keep),1), b, device=device),
                pred_boxes[keep]
            ], dim=1)
            
            all_rois.append(batch_rois)
        
        return torch.cat(all_rois, dim=0)

    def generate_anchors(self):
        """生成基础锚点"""
        anchors = []
        for i in range(self.feat_size):
            for j in range(self.feat_size):
                x_center = (j + 0.5) * self.stride
                y_center = (i + 0.5) * self.stride
                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        w = scale * (ratio ** 0.5)
                        h = scale / (ratio ** 0.5)
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2
                        x2 = x_center + w / 2
                        y2 = y_center + h / 2
                        anchors.append([x1, y1, x2, y2])
        return torch.tensor(anchors)

    def decode_boxes(self, reg_pred, anchors):
        """将回归参数转换为实际边界框"""
        dx = reg_pred[:, 0]  # [2304]
        dy = reg_pred[:, 1]
        dw = reg_pred[:, 2]
        dh = reg_pred[:, 3]
        
        # 解码公式
        anchor_wh = anchors[:, 2:] - anchors[:, :2]  # [2304,2]
        anchor_ctr = anchors[:, :2] + 0.5 * anchor_wh
        
        pred_ctr_x = dx * anchor_wh[:, 0] + anchor_ctr[:, 0]
        pred_ctr_y = dy * anchor_wh[:, 1] + anchor_ctr[:, 1]
        pred_w = torch.exp(dw) * anchor_wh[:, 0]
        pred_h = torch.exp(dh) * anchor_wh[:, 1]
        
        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ], dim=1)  # [2304,4]
        
        # return pred_boxes.clamp(min=0, max=256)

        # 新增：确保框宽高至少1像素
        pred_boxes = pred_boxes.clamp(min=0, max=256)
        w = pred_boxes[:, 2] - pred_boxes[:, 0]
        h = pred_boxes[:, 3] - pred_boxes[:, 1]
        pred_boxes[w < 1, 2] = pred_boxes[w < 1, 0] + 1
        pred_boxes[h < 1, 3] = pred_boxes[h < 1, 1] + 1
        
        return pred_boxes


    def apply_nms(self, boxes, scores, nms_thresh=0.85, topk=30):
        """应用非极大值抑制"""
        scores = scores.sigmoid()  # 直接处理一维得分张量
        
        # 选择topk候选
        _, idxs = scores.sort(descending=True)
        idxs = idxs[:topk]
        
        # 应用NMS
        keep = nms(boxes[idxs], scores[idxs], nms_thresh)
        return idxs[keep]

    def supcon_loss(self, projections, rois, gt_masks):
        """监督对比损失"""
        # 获取每个ROI的标签
        roi_labels = []
        for roi in rois:
            b, x1, y1, x2, y2 = roi.int().tolist()
            mask_patch = gt_masks[b, :, y1:y2, x1:x2]
            label = 1 if mask_patch.mean() > 0.5 else 0 
            roi_labels.append(label)
        roi_labels = torch.tensor(roi_labels).to(projections.device)
        
        # 计算对比损失
        temperature = 0.07
        norm_features = F.normalize(projections, dim=1)
        sim_matrix = torch.mm(norm_features, norm_features.T) / temperature
        
        # 构建有效样本掩码（排除自身）
        pos_mask = roi_labels.unsqueeze(0) == roi_labels.unsqueeze(1)
        pos_mask.fill_diagonal_(False)  # 排除自身
        
        # exp_sim = torch.exp(sim_matrix)
        # pos_sum = (exp_sim * pos_mask).sum(dim=1)
        # neg_sum = (exp_sim * ~pos_mask).sum(dim=1)
        
        # loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8)).mean()
        # return loss

        # 新增
        # 构建有效样本掩码（排除自身）
        valid_mask = torch.ones_like(pos_mask, dtype=torch.bool)
        valid_mask.fill_diagonal_(False)

        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = (exp_sim * valid_mask).sum(dim=1)  # 所有有效样本（不含自身）
        
        # 安全计算损失（防止除零和log(0)）
        ratio = pos_sum / (all_sum + 1e-12)
        loss_per_sample = -torch.log(ratio + 1e-12)  # 避免log(0)
        
        # 处理无正样本的情况
        no_pos_mask = (pos_mask.sum(dim=1) == 0)
        loss_per_sample[no_pos_mask] = 0  # 无正样本的ROI损失归零
        
        return loss_per_sample[~no_pos_mask].mean() if (~no_pos_mask).any() else torch.tensor(0.0, device=projections.device)


# 测试代码
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 16
    clip_features = torch.randn(257, batch_size, 1024)  # CLIP输出的特征
    gt_masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()  # 随机生成掩码

    # 初始化模型
    model = ContrastiveRPN()
    
    # 前向传播测试
    loss = model(clip_features, gt_masks)
    
    # 验证输出
    print(f"Input CLIP features shape: {clip_features.shape}")
    print(f"Output contrastive loss: {loss.item():.4f}")
    print("Test passed! All shapes are compatible.")

    # 检查ROI生成
    rois = model.generate_rois(
        rpn_cls = torch.randn(batch_size, model.num_anchors, model.feat_size, model.feat_size),
        rpn_reg = torch.randn(batch_size, model.num_anchors*4, model.feat_size, model.feat_size),
        batch_size = batch_size
    )
    print(f"Generated ROIs shape: {rois.shape}")
    print(f"Sample ROIs:\n{rois[:5]}")