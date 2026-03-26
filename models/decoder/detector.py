import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
import numpy as np

class ForgeryDetector(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, dropout=0.5):
        """
        图像伪造检测模块
        
        参数:
            feature_dim (int): 输入特征维度 (CLIP特征默认1024)
            hidden_dim (int): 隐藏层维度
            dropout (float): Dropout概率
        """
        super(ForgeryDetector, self).__init__()
        
        # 特征聚合层（综合使用所有特征）
        self.feature_aggregator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 注意力机制用于加权聚合特征
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 伪造检测分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        # 损失函数
        self.loss_fn = nn.BCELoss()
        
        # 存储计算结果
        self.latest_loss = None
        self.latest_metrics = {
            'f1': None,
            'auc': None,
            'mcc': None
        }
    
    def forward(self, features):
        """
        前向传播
        综合使用所有特征，而不仅仅是CLS token
        
        参数:
            features (torch.Tensor): CLIP特征, 形状 (batch_size, seq_len, feature_dim)
        
        返回:
            torch.Tensor: 伪造概率, 形状 (batch_size,)
        """
        # seq_len, batch_size, feat_dim = features.shape
        # 转换为 (batch_size, seq_len, feat_dim)
        features = features.permute(1, 0, 2)
        batch_size, seq_len, feat_dim = features.shape

        # 展平序列维度
        flat_features = features.reshape(batch_size * seq_len, feat_dim)
        
        # 特征变换
        transformed = self.feature_aggregator(flat_features)
        transformed = transformed.view(batch_size, seq_len, -1)
        
        # 注意力权重
        attn_weights = self.attention(transformed)
        
        # 加权特征聚合
        weighted_features = torch.sum(transformed * attn_weights, dim=1)
        
        # 伪造概率预测
        pred_probs = self.classifier(weighted_features).squeeze(1)
        return pred_probs
    
    def get_image_labels(self, mask):
        """
        从像素级标签生成图像级标签
        
        参数:
            mask (torch.Tensor): 像素级标签, 形状 (batch_size, 1, H, W)
        
        返回:
            torch.Tensor: 图像级标签 (0=真实, 1=伪造), 形状 (batch_size,)
        """
        # 检查每个图像中是否有伪造像素
        if isinstance(mask, list):
            batch_size = len(mask)
            device = mask[0].device
        else:
            batch_size = mask.shape[0]
            device = mask.device
        image_labels = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # 计算每个图像的伪造像素比例
        # 如果有任何伪造像素(>0)，则标记为伪造图像
        for i in range(batch_size):
            if torch.any(mask[i] > 0):
                image_labels[i] = 1.0
        
        return image_labels
    
    def compute_loss_and_metrics(self, pred_probs, image_labels):
        """
        计算损失和评估指标，并将结果存储在self变量中
        
        参数:
            pred_probs (torch.Tensor): 预测的伪造概率, 形状 (batch_size,)
            image_labels (torch.Tensor): 真实的图像级标签, 形状 (batch_size,)
        
        返回:
            torch.Tensor: 计算得到的损失
        """
        # 计算二元交叉熵损失
        # print(f"pred_probs device: {pred_probs.device}, image_labels device: {image_labels.device}")
        # print(f"Predicted probabilities shape: {pred_probs.shape}, Image labels shape: {image_labels.shape}")
        loss = self.loss_fn(pred_probs, image_labels)
        
        # 准备指标计算
        pred_probs_np = pred_probs.detach().cpu().numpy()
        image_labels_np = image_labels.detach().cpu().numpy()
        
        # 计算二值预测 (阈值0.5)
        pred_labels = (pred_probs_np > 0.5).astype(int)
        
        # 计算指标
        metrics = {}
        
        # F1分数
        try:
            metrics['f1'] = f1_score(image_labels_np, pred_labels, zero_division=0)
        except:
            metrics['f1'] = 0.0
        
        # AUC (如果只有单一类别则跳过)
        try:
            if len(np.unique(image_labels_np)) > 1:
                metrics['auc'] = roc_auc_score(image_labels_np, pred_probs_np)
            else:
                metrics['auc'] = 0.0
        except:
            metrics['auc'] = 0.0
        
        # Matthews相关系数
        try:
            metrics['mcc'] = matthews_corrcoef(image_labels_np, pred_labels)
        except:
            metrics['mcc'] = 0.0
        
        # 存储计算结果
        self.latest_loss = loss.item()
        self.latest_metrics = metrics
        
        return loss, metrics.copy()
    
    def compute_metrics_with_imgpath(self, pred_probs, images_path):
        # print(f"pred_probs:{pred_probs}")
        if isinstance(pred_probs, list):
            pred_probs=torch.stack(pred_probs)
        # print(f"pred_probs stack:{pred_probs}")

        # 根据images_path生成图像级标签
        image_labels = torch.zeros(len(images_path), dtype=torch.float32, device=pred_probs.device)
        for i, img_path in enumerate(images_path):
            image_labels[i] = 0.0 if "real" in img_path else 1.0

        # 准备指标计算
        pred_probs_np = pred_probs.detach().cpu().numpy()
        image_labels_np = image_labels.detach().cpu().numpy()

        # 计算二值预测 (阈值0.5)
        pred_labels = (pred_probs_np > 0.5).astype(int)

        # 计算指标
        metrics = {}

        # F1分数
        try:
            metrics['f1'] = f1_score(image_labels_np, pred_labels, zero_division=0)
        except:
            metrics['f1'] = 0.0
        
        # AUC (如果只有单一类别则跳过)
        try:
            if len(np.unique(image_labels_np)) > 1:
                metrics['auc'] = roc_auc_score(image_labels_np, pred_probs_np)
            else:
                metrics['auc'] = 0.0
        except:
            metrics['auc'] = 0.0
        
        # Matthews相关系数
        try:
            metrics['mcc'] = matthews_corrcoef(image_labels_np, pred_labels)
        except:
            metrics['mcc'] = 0.0

        return metrics.copy()

    def get_latest_results(self):
        """
        获取最近一次计算的损失和指标
        
        返回:
            dict: 包含最新损失和指标的字典
        """
        return self.latest_loss, self.latest_metrics.copy()


# 测试代码
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 模拟输入数据
    batch_size = 16
    seq_len = 257
    feature_dim = 1024
    features = torch.randn(seq_len, batch_size, feature_dim)  # CLIP特征
    mask = torch.zeros(batch_size, 1, 256, 256)  # 像素级标签
    
    # 创建伪造图像 (随机选择一半作为伪造图像)
    for i in range(batch_size):
        if i % 2 == 0:  # 偶数索引为伪造图像
            # 在随机位置添加伪造区域
            x1, y1 = torch.randint(0, 200, (2,))
            x2, y2 = x1 + 56, y1 + 56
            mask[i, 0, x1:x2, y1:y2] = 1.0
    
    # 初始化模块
    detector = ForgeryDetector()
    
    # 前向传播 - 使用所有特征
    pred_probs = detector(features)
    print("预测概率:", pred_probs)
    
    # 获取图像级标签
    image_labels = detector.get_image_labels(mask)
    print("图像级标签:", image_labels)
    
    # 计算损失和指标
    loss = detector.compute_loss_and_metrics(pred_probs, image_labels)
    
    # 获取存储的结果
    loss, metrics = detector.get_latest_results()
    
    # 打印结果
    print("\n损失和指标结果:")
    print(f"Loss: {loss:.4f}")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # 测试多次调用
    print("\n测试多次调用结果存储:")
    for i in range(3):
        # 生成新数据
        new_features = torch.randn(seq_len, batch_size, feature_dim)
        new_pred_probs = detector(new_features)
        new_image_labels = detector.get_image_labels(mask)
        
        # 计算新损失和指标
        detector.compute_loss_and_metrics(new_pred_probs, new_image_labels)
        loss, metrics = detector.get_latest_results()
        
        print(f"Iteration {i+1} - Loss: {loss:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, MCC: {metrics['mcc']:.4f}")