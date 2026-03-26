import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP_ASPP_Adapter(nn.Module):
    def __init__(self, 
                 in_dim=1024,
                 out_dim=1024,  # 保持输出维度与输入一致
                 hidden_dim=256,
                 grid_size=16,
                 BatchNorm=nn.BatchNorm2d):
        super().__init__()
        
        # 空间特征重构参数
        self.grid_size = grid_size
        self.spatial_proj = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        
        # 优化后的ASPP参数
        self.aspp = ASPP(
            inplanes=in_dim,
            outplanes=hidden_dim,
            dilations=[1, 2, 4, 6],  # 适配16x16的膨胀率
            BatchNorm=BatchNorm
        )
        
        # 维度恢复
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, in_dim, 3, padding=1),
            BatchNorm(in_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """维度变换流程：
        输入: [257, B, 1024] → 
        输出: [257, B, 1024]
        """
        # 分离分类token与空间token
        cls_token = x[:1]        # [1, B, 1024]
        spatial_tokens = x[1:]   # [256, B, 1024]
        B = spatial_tokens.size(1)
        
        # 转换为Conv1d需要的3D格式 [B, C, L]
        spatial_features = spatial_tokens.permute(1, 2, 0)  # [B, 1024, 256]
        
        # 特征增强（关键修改点）
        spatial_features = self.spatial_proj(spatial_features)  # [B, 1024, 256]
        
        # 重建为2D特征图 [B, C, H, W]
        spatial_features = spatial_features.view(
            B, -1, self.grid_size, self.grid_size
        )  # [B, 1024, 16, 16]

        # 通过ASPP处理
        aspp_features = self.aspp(spatial_features)  # [B, 256, 16, 16]
        
        # 恢复原始通道维度
        restored_features = self.final_conv(aspp_features)  # [B, 1024, 16, 16]
        
        # 转换回token序列
        restored_tokens = restored_features \
            .permute(2, 3, 0, 1)  \
            .contiguous() \
            .view(-1, B, 1024)        # [256, B, 1024]
        
        # 合并分类token
        return torch.cat([cls_token, restored_tokens], dim=0)  # [257, B, 1024]

class ASPP(nn.Module):
    def __init__(self, 
                 inplanes=1024,
                 outplanes=256,
                 dilations=[1, 2, 4, 6],  # 专为16x16优化
                 BatchNorm=nn.BatchNorm2d):
        super().__init__()
        
        # 多尺度分支
        self.aspp_modules = nn.ModuleList([
            nn.Conv2d(inplanes, outplanes, 1, bias=False),
            self._make_aspp(inplanes, outplanes, 3, dilations[1]),
            self._make_aspp(inplanes, outplanes, 3, dilations[2]),
            self._make_aspp(inplanes, outplanes, 3, dilations[3])
        ])
        
        # 全局上下文
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, outplanes, 1, bias=False),
            BatchNorm(outplanes),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(outplanes*5, outplanes, 1),
            BatchNorm(outplanes),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self._init_weights()

    def _make_aspp(self, in_c, out_c, kernel, dilation):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        features = []
        for module in self.aspp_modules:
            features.append(module(x))
        
        # 全局特征
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        features.append(global_feat)
        
        return self.fusion(torch.cat(features, dim=1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# 使用示例
if __name__ == "__main__":
    # 模拟输入
    clip_features = torch.randn(257, 16, 1024)  # CLIP输出特征
    
    # 初始化
    aspp_adapter  = CLIP_ASPP_Adapter()
    enhanced_features = aspp_adapter(clip_features)
    print("Enhanced features shape:", enhanced_features.shape)  # 应为 [257, 16, 1024]
    
    