import cv2
import numpy as np
from sklearn.cluster import KMeans


def analyze_mammo_image(img, k=3):
    """
    对输入的乳腺钼靶图像进行分析，识别病灶区域

    参数:
        img: 灰度图像 (numpy数组)
        k: 聚类数量

    返回:
        dict: 包含分析结果的字典
            - original_img: 原始灰度图像
            - segmented_img: 聚类分割后的图像
            - mask_img: 病灶区域掩码
            - highlighted_img: 高亮病灶区域的图像
            - lesion_percentage: 病灶区域占比
            - target_cluster: 病灶聚类索引
    """
    # 验证输入
    if img is None or len(img.shape) != 2:
        raise ValueError("输入图像应为灰度图像")

    # 1. 预处理：直方图均衡化增强对比度
    img_enhanced = cv2.equalizeHist(img)

    # 2. 图像展平
    pixel_values = img_enhanced.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # 3. K-Means聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)

    # 4. 重构聚类图像
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(img.shape)

    # 5. 识别病灶区域（灰度最高的聚类）
    target_cluster = np.argmax(centers)
    mask = (labels.flatten() == target_cluster)
    mask_img = mask.reshape(img.shape).astype(np.uint8) * 255  # 转换为0-255范围

    # 6. 形态学操作优化掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)

    # 7. 高亮病灶区域（白色显示）
    highlighted_img = img.copy()
    highlighted_img[mask_img > 0] = 255  # 将目标区域变白

    # 8. 计算病灶区域占比
    lesion_percentage = np.sum(mask_img > 0) / mask_img.size * 100

    return {
   
        'highlighted_img': highlighted_img,
        'lesion_percentage': lesion_percentage,
        'target_cluster': target_cluster
    }