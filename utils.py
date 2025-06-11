import cv2
import numpy as np


def read_image(file_path):
    """
    读取图像文件，支持常见格式

    参数:
        file_path: 图像文件路径

    返回:
        numpy数组: 灰度图像
    """
    # 普通图像文件
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像文件: {file_path}")
    return img


def prepare_image_for_display(img):
    """
    准备图像用于在PyQt界面显示

    参数:
        img: 输入图像 (灰度或BGR)

    返回:
        numpy数组: RGB格式图像
    """
    if len(img.shape) == 2:  # 灰度图
        # 转换为伪彩色以增强可视化
        colored = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # BGR彩色图
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:  # BGRA彩色图
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        # 创建默认图像
        default_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(default_img, "Invalid Image", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return default_img