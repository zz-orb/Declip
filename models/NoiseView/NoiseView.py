import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DCTHighPass(nn.Module):
    def __init__(self, kernel_size=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AdaptiveAvgPool2d(kernel_size)
        
    def forward(self, x):
        # 输入尺寸: [B,3,H,W]
        b, c, h, w = x.shape
        
        # 转换为灰度图
        gray = 0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]
        gray = gray.unsqueeze(1)  # [B,1,H,W]
        
        # 分块处理（保持批次维度）
        chunks = []
        for i in range(0, h, self.kernel_size):
            for j in range(0, w, self.kernel_size):
                chunk = gray[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                
                # 处理不规则块（填充）
                pad_h = (self.kernel_size - chunk.shape[2] % self.kernel_size) % self.kernel_size
                pad_w = (self.kernel_size - chunk.shape[3] % self.kernel_size) % self.kernel_size
                chunk = F.pad(chunk, (0, pad_w, 0, pad_h))
                
                chunks.append(chunk)
        
        # 批量处理DCT
        dct_chunks = []
        for chunk in chunks:
            # 转换为复数输入
            chunk_complex = torch.fft.fft2(chunk.squeeze(1))  # [B,k,k] → 移除通道维度
            
            # 计算幅度谱和相位谱
            magnitude = torch.abs(chunk_complex)  # [B,k,k]
            # phase = torch.angle(chunk_complex)
            
            # 高通滤波
            rows, cols = magnitude.shape[-2:]  # 正确获取空间维度
            crow, ccol = rows//2, cols//2
            mask = torch.ones_like(magnitude)
            mask[crow-1:crow+2, ccol-1:ccol+2] = 0  # 3x3中心屏蔽
            
            filtered = magnitude * mask  # 仅保留幅度信息
            filtered_chunk = filtered.unsqueeze(1)  # 恢复通道维度 [B,1,k,k]
            
            dct_chunks.append(filtered_chunk)
        
        # 重构特征图
        output = torch.cat(dct_chunks, dim=2)  # [B,1,H,W]
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        
        return output

class DWTHighPass(nn.Module):
    def __init__(self, levels=2, combine_mode="l1", eps=1e-6):
        super().__init__()
        self.levels = levels
        self.combine_mode = combine_mode
        self.eps = eps

        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        self.register_buffer("ll_filter", ll.view(1, 1, 2, 2))
        self.register_buffer("lh_filter", lh.view(1, 1, 2, 2))
        self.register_buffer("hl_filter", hl.view(1, 1, 2, 2))
        self.register_buffer("hh_filter", hh.view(1, 1, 2, 2))

    def _to_gray(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return gray.unsqueeze(1)

    def _pad_even(self, x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def _detail_response(self, lh, hl, hh):
        if self.combine_mode == "l2":
            return torch.sqrt(lh.pow(2) + hl.pow(2) + hh.pow(2) + self.eps)
        return (lh.abs() + hl.abs() + hh.abs()) / 3.0

    def forward(self, x):
        _, _, h, w = x.shape
        current = self._to_gray(x)
        detail_maps = []

        for _ in range(self.levels):
            current = self._pad_even(current)
            ll = F.conv2d(current, self.ll_filter, stride=2)
            lh = F.conv2d(current, self.lh_filter, stride=2)
            hl = F.conv2d(current, self.hl_filter, stride=2)
            hh = F.conv2d(current, self.hh_filter, stride=2)

            detail = self._detail_response(lh, hl, hh)
            detail = F.interpolate(detail, size=(h, w), mode="bilinear", align_corners=False)
            detail_maps.append(detail)
            current = ll

        if not detail_maps:
            return torch.zeros(x.size(0), 1, h, w, device=x.device, dtype=x.dtype)

        return torch.stack(detail_maps, dim=0).mean(dim=0)
# Light
class SpatialFrequencyFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # 通道注意力融合层
        self.channel_attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

    def forward(self, rgb_features, dct_features):
        # 维度对齐调整（将序列维度置于最后）
        rgb = rgb_features.permute(1, 0, 2)  # [B, 257, 1024]
        dct = dct_features.permute(1, 0, 2)  # [B, 257, 1024]
        
        # 通道注意力计算
        channel_weights = self.channel_attention(
            torch.mean(rgb, dim=1) + torch.mean(dct, dim=1)  # 全局特征
        ).unsqueeze(1)  # [B, 1, 1024]
        
        # 自适应特征融合
        fused = (rgb * channel_weights) + (dct * (1 - channel_weights))
        
        # 恢复原始维度 [257, B, 1024]
        return fused.permute(1, 0, 2)


class CBAMFusion(nn.Module):
    def __init__(self, in_dim=1024, reduction_ratio=8):
        super().__init__()
        # 通道注意力（原始CBAM改进）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 针对序列维度池化
            nn.Conv1d(in_dim, in_dim//reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(in_dim//reduction_ratio, in_dim, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力（适配序列特征）
        self.spatial_att = nn.Sequential(
            nn.Conv1d(2, 1, 7, padding=3),  # 序列维度卷积
            nn.LayerNorm([257*2]),  # 序列长度归一化
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, dct_feat):
        # rgb_feat/dct_feat: [257, B, 1024]
        concat_feat = torch.cat([rgb_feat, dct_feat], dim=0)  # [514, B, 1024]
        
        # 通道注意力
        channel_weights = self.channel_att(
            concat_feat.permute(1,2,0)  # [B,1024,514]
        ).permute(2,0,1)  # [514,B,1024]
        channel_refined = concat_feat * channel_weights
        
        # 空间注意力
        avg_pool = torch.mean(channel_refined, dim=2, keepdim=True)  # [514,B,1]
        max_pool = torch.max(channel_refined, dim=2, keepdim=True)[0]  # [514,B,1]
        spatial_weights = self.spatial_att(
            torch.cat([avg_pool, max_pool], dim=2).permute(1,2,0)  # [B,2,514]
        ).permute(2,0,1)  # [514,B,1]
        
        fused_feat = channel_refined * spatial_weights
        
        # 分割还原
        rgb_out = fused_feat[:257]  # [257,B,1024]
        dct_out = fused_feat[257:]
        return (rgb_out + dct_out)/2

class GroupCBAMEnhancer(nn.Module):
    def __init__(self, in_dim=1024, reduction_ratio=8, groups=4):
        """
        Args:
            in_dim: 输入特征维度
            reduction_ratio: 通道压缩比例
            groups: 分组数量 (建议4或8的倍数)
        """
        super().__init__()
        self.in_dim = in_dim
        self.groups = groups
        assert in_dim % groups == 0, "in_dim must be divisible by groups"
        self.group_dim = in_dim // groups
        
        # 分组通道注意力模块
        self.group_channel_att = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(self.group_dim, self.group_dim//reduction_ratio, 1),
                nn.ReLU(),
                nn.Conv1d(self.group_dim//reduction_ratio, self.group_dim, 1)
            ) for _ in range(groups)
        ])
        
        # 分组空间注意力模块
        self.group_spatial_att = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2, 1, 7, padding=3),
                nn.LayerNorm([257*2]),
            ) for _ in range(groups)
        ])
        
        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(1))
        
    def _weight_redistribution(self, weights):
        """
        权重重分配策略: 高于均值的置1，低于均值的保留原值
        """
        mean_val = weights.mean()
        enhanced_weights = torch.where(weights > mean_val, 
                                     torch.ones_like(weights), 
                                     weights)
        return torch.tanh(enhanced_weights)  # Tanh门控

    def forward(self, rgb_feat, dct_feat):
        """
        Args:
            rgb_feat: RGB特征 [257, B, 1024]
            dct_feat: DCT特征 [257, B, 1024]
        Returns:
            增强后的融合特征 [257, B, 1024]
        """
        # 特征拼接 [514, B, 1024]
        concat_feat = torch.cat([rgb_feat, dct_feat], dim=0)
        B = concat_feat.size(1)
        
        # 保存原始特征用于残差连接
        original_feat = concat_feat
        
        # 沿通道维度分组 [514, B, groups, group_dim]
        grouped_feat = concat_feat.view(514, B, self.groups, self.group_dim)
        
        # 各组独立处理
        group_outputs = []
        for g in range(self.groups):
            # 当前组特征 [514, B, group_dim]
            group_feat = grouped_feat[:, :, g, :]
            
            # --- 通道注意力 ---
            channel_weights = self.group_channel_att[g](
                group_feat.permute(1,2,0)  # [B, group_dim, 514]
            ).permute(2,0,1)  # [514, B, group_dim]
            
            # 权重重分配
            channel_weights = self._weight_redistribution(channel_weights)
            channel_refined = group_feat * channel_weights
            
            # --- 空间注意力 ---
            avg_pool = torch.mean(channel_refined, dim=2, keepdim=True)  # [514,B,1]
            max_pool = torch.max(channel_refined, dim=2, keepdim=True)[0]
            spatial_weights = self.group_spatial_att[g](
                torch.cat([avg_pool, max_pool], dim=2).permute(1,2,0)  # [B,2,514]
            ).permute(2,0,1)  # [514,B,1]
            
            spatial_weights = torch.sigmoid(spatial_weights)  # Sigmoid门控
            refined_feat = channel_refined * spatial_weights
            
            # 残差连接
            refined_feat = refined_feat + group_feat
            group_outputs.append(refined_feat)
        
        # 合并分组结果 [514, B, 1024]
        fused_feat = torch.cat(group_outputs, dim=2)
        
        # 全局残差连接与缩放
        enhanced_feat = original_feat + self.scale * fused_feat
        
        # 分割并平均 [257, B, 1024]
        rgb_out = enhanced_feat[:257]
        dct_out = enhanced_feat[257:]
        return (rgb_out + dct_out) / 2


class GroupSpatialFrequencyFusion(nn.Module):
    def __init__(self, num_groups=8):
        super().__init__()
        self.num_groups = num_groups  # 分组数量（默认为4组）

        # 原始通道注意力融合层
        self.channel_attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

        # 新增分组注意力模块
        self.group_attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # 输出单值权重
            nn.Sigmoid()
        )

        # 可学习的缩放参数（用于残差连接）
        self.scale = nn.Parameter(torch.tensor(0.1))  # 初始值设为0.1

    def forward(self, rgb_features, dct_features):
        # === 步骤1: 原始特征融合 ===
        rgb = rgb_features.permute(1, 0, 2)  # [B, 257, 1024]
        dct = dct_features.permute(1, 0, 2)  # [B, 257, 1024]
        
        # 通道注意力计算
        channel_weights = self.channel_attention(
            torch.mean(rgb, dim=1) + torch.mean(dct, dim=1)
        ).unsqueeze(1)  # [B, 1, 1024]
        
        # 基础融合特征
        fused_base = (rgb * channel_weights) + (dct * (1 - channel_weights))  # [B, 257, 1024]

        # === 步骤2: 分组局部注意力增强 ===
        B, L, D = fused_base.shape  # L=257 (1全局+256局部)
        global_feat = fused_base[:, :1, :]  # 全局特征 [B, 1, 1024]
        local_feat = fused_base[:, 1:, :]   # 局部特征 [B, 256, 1024]

        # 将256个块均匀分组
        group_feat = local_feat.reshape(B, self.num_groups, -1, D)  # [B, G, K, 1024]
        G, K = group_feat.shape[1], group_feat.shape[2]  # G=组数, K=每组块数

        # 计算每组独立注意力权重
        flat_feat = group_feat.reshape(B * G * K, D)  # 展平 [B*G*K, 1024]
        attn_flat = self.group_attention(flat_feat)  # [B*G*K, 1]
        attn = attn_flat.reshape(B, G, K, 1)  # 恢复形状 [B, G, K, 1]

        # === 步骤3: 权重重分配（均值门控）===
        mean_weights = torch.mean(attn, dim=2, keepdim=True)  # 组内均值 [B, G, 1, 1]
        enhanced_attn = torch.where(
            attn > mean_weights,
            torch.ones_like(attn),  # 高于均值置1
            attn                     # 低于均值保留原值
        )

        # === 步骤4: 残差连接增强 ===
        weighted_feat = group_feat * enhanced_attn  # 加权特征 [B, G, K, 1024]
        residual_feat = group_feat + self.scale * weighted_feat  # 残差连接

        # 恢复局部特征形状
        enhanced_local = residual_feat.reshape(B, G * K, D)  # [B, 256, 1024]

        # 合并全局+局部特征
        enhanced_fused = torch.cat([global_feat, enhanced_local], dim=1)  # [B, 257, 1024]

        # 恢复原始维度 [257, B, 1024]
        return enhanced_fused.permute(1, 0, 2)


# 使用示例
if __name__ == "__main__":
    # 模拟输入特征
    rgb_feat = torch.randn(257, 8, 1024)  # [257, B, 1024]
    dct_feat = torch.randn(257, 8, 1024)
    
    # 初始化增强器
    # enhancer = GroupCBAMEnhancer(in_dim=1024, groups=4)
    enhancer = GroupSpatialFrequencyFusion(num_groups=8)
    
    # 特征融合
    fused_feat = enhancer(rgb_feat, dct_feat)
    
    print(f"输出特征形状: {fused_feat.shape}")  # 应输出 [257, 8, 1024]
