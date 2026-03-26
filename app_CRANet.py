from flask import Flask, request, jsonify
import os
import uuid
import sys
from infer import DeclipDetector, convert_numpy_to_python
import numpy as np
from PIL import Image

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = './system/uploads'
OUTPUT_FOLDER = './system/output_masks'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB限制

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 全局检测器实例
detector = None

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_directories():
    """初始化必要的目录"""
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        os.makedirs(folder, exist_ok=True)


def init_detector():
    """初始化检测器"""
    global detector
    try:
        detector = DeclipDetector()
        print("✅ 伪造检测器初始化成功")
    except Exception as e:
        print(f"❌ 检测器初始化失败: {e}")
        raise


@app.route('/health_ours', methods=['GET'])
def health_check():
    """健康检查端点"""
    status = True if detector is not None else False
    message = "服务运行正常" if detector is not None else "检测器未就绪"
    
    return jsonify({
        'status': status,
        'service': 'ours-api',
        'message': message
    })

@app.route('/detect_ours', methods=['POST'])
def detect_forgery():
    """
    伪造检测API接口
    接收图片并进行伪造检测与定位
    """
    # 检查请求中是否包含文件
    if 'image' not in request.files:
        print("未找到图像文件，请使用image字段上传")
        return jsonify({
            'success': False,
            'error': '未找到图像文件，请使用"image"字段上传'
        }), 400

    file = request.files['image']
    
    # 检查文件名
    if file.filename == '':
        print("无效的文件名")
        return jsonify({
            'success': False,
            'error': '无效的文件名'
        }), 400

    # 验证文件类型
    if not allowed_file(file.filename):
        print('不支持的文件格式。允许的格式')
        return jsonify({
            'success': False,
            'error': f'不支持的文件格式。允许的格式: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        
        # 保存上传的文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # print(f"文件已保存到: {filepath}")
        # 执行伪造检测
        result = detector.detect_forgery(filepath)
        
        # # 如果成功，保存掩码图像
        # if result['success'] and result.get('localization'):
        #     save_mask_images(result, file.filename)
            
        # # 清理上传的文件
        # os.remove(filepath)

        converted_result = convert_numpy_to_python(result)
        
        return jsonify(converted_result)
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return jsonify({
            'success': False,
            'error': f'处理过程中发生错误: {str(e)}'
        }), 500


def save_mask_images(result, original_filename):
    """保存掩码图像"""
    try:
        heatmap_data = result['localization']['heatmap']
        
        # 生成基础文件名
        base_name = os.path.splitext(original_filename)[0]
        
        # 1. 二值掩码
        binary_threshold = 0.5
        binary_mask = (heatmap_data > binary_threshold).astype(np.uint8) * 255
        
        # 2. 灰度掩码
        grayscale_mask = (heatmap_data * 255).astype(np.uint8)
        
        # 保存二值掩码
        binary_filename = f"{base_name}_binary_mask.png"
        binary_filepath = os.path.join(OUTPUT_FOLDER, binary_filename)
        Image.fromarray(binary_mask, mode='L').save(binary_filepath)
        
        # 保存灰度掩码
        grayscale_filename = f"{base_name}_grayscale_mask.png"
        grayscale_filepath = os.path.join(OUTPUT_FOLDER, grayscale_filename)
        Image.fromarray(grayscale_mask, mode='L').save(grayscale_filepath)
        
        # 更新结果中包含保存的文件路径
        result['localization']['mask_files'] = {
            'binary': binary_filename,
            'grayscale': grayscale_filename
        }
        
    except Exception as e:
        print(f"⚠️ 保存掩码图像失败: {e}")


@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({
        'success': False,
        'error': '文件大小超过限制 (最大16MB)'
    }), 413


if __name__ == '__main__':
    # 初始化
    init_directories()
    init_detector()
    
    # 启动服务
    print("🚀 CRA-Net 伪造检测服务启动中...")
    print(f"📁 上传目录: {UPLOAD_FOLDER}")
    print(f"📁 输出目录: {OUTPUT_FOLDER}")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True 
    )