import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from typing import Dict, Tuple, Optional, List


def analyze_mammo_image(img: np.ndarray, k: int = 3, lesion_is_bright: bool = True, 
                       morph_kernel_size: Tuple[int, int] = (5, 5), 
                       min_lesion_size: int = 100,
                       visualize: bool = False) -> Dict:
    """
    对输入的乳腺钼靶图像进行分析，识别病灶区域并提供详细特征
    
    参数:
        img: 灰度图像 (numpy数组)
        k: 聚类数量
        lesion_is_bright: 病灶是否表现为较亮区域
        morph_kernel_size: 形态学操作核大小
        min_lesion_size: 最小病灶面积过滤阈值(像素)
        visualize: 是否显示中间结果
    
    返回:
        dict: 包含分析结果的字典
            - original_img: 原始灰度图像
            - segmented_img: 聚类分割后的图像
            - mask_img: 病灶区域掩码
            - highlighted_img: 高亮病灶区域的图像
            - lesion_percentage: 病灶区域占比
            - target_cluster: 病灶聚类索引
            - lesion_count: 检测到的病灶数量
            - lesion_areas: 各病灶面积列表
            - lesion_coordinates: 各病灶中心坐标
            - lesion_features: 各病灶特征(面积、周长、圆形度等)
    """
    # 验证输入
    if img is None or len(img.shape) != 2:
        raise ValueError("输入图像应为灰度图像")
    
    # 图像预处理
    img_enhanced = cv2.equalizeHist(img)
    
    # 高斯滤波减少噪声
    img_smooth = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    
    # K-Means聚类分割
    pixel_values = img_smooth.reshape((-1, 1)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)
    
    # 重构聚类图像
    segmented_img = centers[labels.flatten()].reshape(img.shape)
    
    # 识别病灶聚类
    target_cluster = np.argmax(centers) if lesion_is_bright else np.argmin(centers)
    mask_img = (labels.reshape(img.shape) == target_cluster).astype(np.uint8) * 255
    
    # 形态学操作优化掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
    
    # 移除小面积噪声区域
    labeled_mask, num_features = ndimage.label(mask_img)
    sizes = ndimage.sum(mask_img, labeled_mask, range(num_features + 1))
    mask_size = mask_img.shape[0] * mask_img.shape[1]
    
    # 过滤小区域
    min_size = min_lesion_size
    for i in range(num_features + 1):
        if sizes[i] < min_size or sizes[i] > 0.8 * mask_size:  # 同时过滤过大区域
            mask_img[labeled_mask == i] = 0
    
    # 计算病灶区域占比
    lesion_percentage = np.sum(mask_img > 0) / mask_img.size * 100
    
    # 高亮病灶区域
    highlighted_img = img.copy()
    highlighted_img[mask_img > 0] = 255
    
    # 病灶特征提取
    lesion_features = extract_lesion_features(mask_img)
    
    # 可视化中间结果
    if visualize:
        plot_results(img, img_enhanced, segmented_img, mask_img, highlighted_img)
    
    return {
        'original_img': img,
        'segmented_img': segmented_img,
        'mask_img': mask_img,
        'highlighted_img': highlighted_img,
        'lesion_percentage': lesion_percentage,
        'target_cluster': target_cluster,
        'lesion_count': len(lesion_features),
        'lesion_features': lesion_features
    }


def extract_lesion_features(mask_img: np.ndarray) -> List[Dict]:
    """提取病灶区域的特征"""
    labeled_mask, num_labels = ndimage.label(mask_img)
    regions = measure.regionprops(labeled_mask)
    
    features = []
    for region in regions:
        # 计算基本特征
        area = region.area
        perimeter = region.perimeter
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        major_axis_length = region.major_axis_length
        minor_axis_length = region.minor_axis_length
        eccentricity = region.eccentricity
        solidity = region.solidity
        
        # 计算边界框和中心
        min_row, min_col, max_row, max_col = region.bbox
        centroid_row, centroid_col = region.centroid
        
        features.append({
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'major_axis_length': major_axis_length,
            'minor_axis_length': minor_axis_length,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'bounding_box': (min_row, min_col, max_row, max_col),
            'centroid': (centroid_row, centroid_col)
        })
    
    # 按面积排序
    features.sort(key=lambda x: x['area'], reverse=True)
    return features


def plot_results(original, enhanced, segmented, mask, highlighted):
    """可视化分析结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title('增强图像')
    axes[1].axis('off')
    
    axes[2].imshow(segmented, cmap='gray')
    axes[2].set_title('聚类分割')
    axes[2].axis('off')
    
    axes[3].imshow(mask, cmap='gray')
    axes[3].set_title('病灶掩码')
    axes[3].axis('off')
    
    axes[4].imshow(highlighted, cmap='gray')
    axes[4].set_title('病灶高亮')
    axes[4].axis('off')
    
    # 隐藏最后一个子图
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()