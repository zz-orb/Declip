import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models import get_model
from options.test_options import TestOptions


MEAN_CLIP = [0.48145466, 0.4578275, 0.40821073]
STD_CLIP = [0.26862954, 0.26130258, 0.27577711]

class DeclipPreprocessor:
    """DeClip图像预处理器"""
    
    def __init__(self):
        self.transform = self._build_transform()
        
    def _build_transform(self):
        """构建图像预处理流水线"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN_CLIP,
                std=STD_CLIP
            )
        ])
    
    def process(self, image_input):
        """
        图像预处理
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图像不存在: {image_input}")
            pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise TypeError("不支持的图像输入格式")
            
        return self.transform(pil_image).unsqueeze(0)


class DeclipResultProcessor:
    """DeClip结果处理器"""
    
    @staticmethod
    def process_detection_output(raw_output):
        """处理检测分支输出"""
        detection_prob = raw_output
        return {
            'score': detection_prob.item(),
            'is_forged': detection_prob.item() > 0.5
        }
    
    @staticmethod
    def process_localization_output(raw_output, target_size):
        """处理定位分支输出"""
        # Reshape到二维
        h_w = int(raw_output.size(1) ** 0.5)
        reshaped_output = raw_output.view(raw_output.size(0), h_w, h_w)
        
        # 上采样到目标尺寸
        upsampled_mask = F.interpolate(
            reshaped_output.unsqueeze(1),
            size=target_size,
            mode='bilinear',
            align_corners=True
        ).squeeze(1)
        
        # 应用sigmoid
        probability_map = torch.sigmoid(upsampled_mask)
        
        return {
            'heatmap': probability_map.cpu().numpy()[0],
            'shape': probability_map.shape[-2:]
        }


class DeclipDetector:
    """
    DeClip伪造检测与定位器
    
    支持单张图像的快速伪造检测与精确定位
    """
    def __init__(self, ckpt='checkpoints/GeoRSCLIP_DOTA_PS_deepfake_all/model_epoch_best.pth'):        
        # 模型参数
        self.opt = TestOptions().parse(print_options=False)
        self.opt.ckpt = ckpt
        self.opt.arch='CLIP:GeoRSCLIP'
        self.opt.use_aspp = True
        self.opt.use_simdet = True
        self.fully_supervised = True
        print('self.opt:', self.opt)

        # 核心组件
        self.model = None
        self.preprocessor = DeclipPreprocessor()
        self.processor = DeclipResultProcessor()

        # 初始化
        self.initialize_model()
        self.device = 'cuda:0'
            
    def initialize_model(self):
        """完整模型初始化"""
        try:
            state_dict = torch.load(self.opt.ckpt, map_location='cpu')
            try:
                self.opt.feature_layer = state_dict['feature_layer']
                self.opt.decoder_type = state_dict['decoder_type']
            except:
                print('No feature_layer or decoder_type in the checkpoint state_dict, using the info from feature_layer and decoder_type args')
            self.model = get_model(self.opt)
            self.model.load_state_dict(state_dict['model'], strict=False)
            self.model.eval()
            self.model.cuda()

            print("✅ PSCC-Net 系统初始化完成")
            
        except Exception as e:
            raise RuntimeError(f"❌ 模型初始化失败: {str(e)}")
            
    def preprocess_image(self, image_input):
        """
        图像预处理
        image_input: 图像路径/PIL图像
        """
        # 获取原始尺寸
        if isinstance(image_input, str):
            with Image.open(image_input) as img:
                orig_width, orig_height = img.size
        else:
            orig_width, orig_height = image_input.size
            
        processed_tensor = self.preprocessor.process(image_input)

        temp = processed_tensor.float()
        mean_value = torch.mean(temp)
        mapped_value = mean_value * 0.01 + 0.05
        mapped_value = torch.clamp(mapped_value, 0.0, 0.1)
        
        return {
            'tensor': processed_tensor,
            'orig_size': (orig_height, orig_width),
            'mapped_value': mapped_value
        }
        
    def run_inference(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            raw_output = self.model(image_tensor)
            
            # 分离检测和定位结果
            # detection_raw = torch.mean(raw_output, dim=1)
            detection_raw = self.model.det_pred_probs
            localization_raw = raw_output
            
            return {
                'detection_raw': detection_raw,
                'localization_raw': localization_raw
            }
            
    def postprocess_results(self, raw_results, orig_size):
        """后处理推理结果"""
        # 处理检测结果
        detection_result = self.processor.process_detection_output(
                raw_results['detection_raw'])
                
        # 处理定位结果  
        localization_result = self.processor.process_localization_output(
                raw_results['localization_raw'],
                orig_size
            )
            
        return {
            'detection': detection_result,
            'localization': localization_result
        }
        
    def detect_forgery(self, image_input):
        """
        对外接口：伪造检测与定位
        """
        try:
            preprocessing_result = self.preprocess_image(image_input)
            mapped_value = preprocessing_result['mapped_value']
            # print('mapped_value:', mapped_value)

            # 推理
            raw_inference_result = self.run_inference(preprocessing_result['tensor'])
            
            # 后处理
            final_result = self.postprocess_results(
                    raw_inference_result,
                    preprocessing_result['orig_size']
                )
            
            return {
                'success': True,
                'detection': final_result['detection'],
                'localization': final_result['localization']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'detection': None,
                'localization': None
            }


def convert_numpy_to_python(obj):
    """递归地将NumPy数组和其他NumPy数据类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


# 使用示例
if __name__ == "__main__":
    # 实例化检测器
    detector = DeclipDetector()
    # 执行伪造分析
    test_image_path = "/nas/datasets/DOTA-PS/copymove/batch1-output/9-fake.png"
    result = detector.detect_forgery(test_image_path)

    if result['success']:
        print(f"🔍 检测结果: {'伪造' if result['detection']['is_forged'] else '真实'} "
              f"(置信度: {result['detection']['probability']:.4f})")
        print(f"📍 定位热力图尺寸: {result['localization']['shape']}")

        # 可视化
        output_dir = './output_masks'
        heatmap_data = result['localization']['heatmap']
        # 1. 二值掩码（黑白）- 使用0.5阈值
        binary_threshold = 0.5
        binary_mask = (heatmap_data > binary_threshold).astype(np.uint8) * 255
        
        # 2. 灰度掩码（黑白灰）- 直接将概率值映射到0-255
        grayscale_mask = (heatmap_data * 255).astype(np.uint8)

        # 保存二值掩码
        binary_filename = f"{output_dir}/{os.path.basename(test_image_path)}_binary_mask.png"
        Image.fromarray(binary_mask, mode='L').save(binary_filename)
        
        # 保存灰度掩码
        grayscale_filename = f"{output_dir}/{os.path.basename(test_image_path)}_grayscale_mask.png"
        Image.fromarray(grayscale_mask, mode='L').save(grayscale_filename)

        print(f"✅ 掩码图像已保存至 {output_dir}:")
        print(f"   - 二值掩码: {binary_filename}")
        print(f"   - 灰度掩码: {grayscale_filename}")