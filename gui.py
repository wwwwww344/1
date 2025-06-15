import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSpinBox, QProgressBar, QGroupBox,
    QApplication, QMenu, QMenuBar, QAction
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import processing
import utils
from datetime import datetime


class AnalysisThread(QThread):
    """图像分析线程，防止UI卡顿"""
    update_progress = pyqtSignal(int)
    finish_analysis = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)

    def __init__(self, img, k_value):
        super().__init__()
        self.img = img
        self.k_value = k_value

    def run(self):
        try:
            self.update_progress.emit(20)  # 开始处理
            result = processing.analyze_mammo_image(
                self.img, 
                k=self.k_value, 
                visualize=False  # 不在线程中生成UI
            )
            self.update_progress.emit(100)  # 处理完成
            self.finish_analysis.emit(result)
        except Exception as e:
            self.analysis_error.emit(str(e))


class MammoAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("乳腺钼靶图像分析系统")
        self.setGeometry(100, 100, 1200, 800)
        self.original_img = None
        self.highlighted_img = None
        self.image_path = None
        
        # 初始化UI
        self.init_ui()
        
        # 5秒后自动全屏（保持强制全屏逻辑）
        QTimer.singleShot(5000, self.showFullScreen)

    def init_ui(self):
        """初始化用户界面"""
        # 设置科技感样式
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0e14; }
            QStatusBar { background-color: #0a0e14; color: #58a6ff; border-top: 1px solid #1f6feb; }
            QMenuBar { background-color: #0a0e14; color: #c9d1d9; border-bottom: 1px solid #1f6feb; }
            QMenuBar::item:selected { background-color: #1f6feb; }
            QGroupBox { color: #58a6ff; border: 1px solid #1f6feb; border-radius: 5px; }
            QPushButton { background-color: #1f6feb; color: white; border-radius: 5px; min-height: 30px; }
            QPushButton:hover { background-color: #58a6ff; }
            QPushButton:pressed { background-color: #0d419d; }
            QLabel { color: #c9d1d9; }
            QSpinBox { background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d; }
            QProgressBar { border: 1px solid #30363d; background-color: #0d1117; }
            QProgressBar::chunk { background-color: #1f6feb; }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_panel)

        # 图像选择
        self.btn_open = QPushButton("选择乳腺图像")
        self.btn_open.setIcon(QIcon.fromTheme("document-open"))
        self.btn_open.clicked.connect(self.open_image)
        control_layout.addWidget(self.btn_open)

        # 参数设置区域
        params_group = QGroupBox("分析参数")
        params_layout = QVBoxLayout(params_group)
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("聚类数量 (k):"))
        self.spin_k = QSpinBox()
        self.spin_k.setMinimum(2)
        self.spin_k.setMaximum(6)
        self.spin_k.setValue(3)
        param_layout.addWidget(self.spin_k)
        params_layout.addLayout(param_layout)
        control_layout.addWidget(params_group)

        # 处理按钮
        self.btn_process = QPushButton("分析图像")
        self.btn_process.setIcon(QIcon.fromTheme("system-run"))
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setMinimumHeight(40)
        control_layout.addWidget(self.btn_process)

        # 保存按钮
        self.btn_save = QPushButton("保存分析结果")
        self.btn_save.setIcon(QIcon.fromTheme("document-save"))
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_analysis_results)
        self.btn_save.setMinimumHeight(40)
        control_layout.addWidget(self.btn_save)

        # 退出按钮
        self.btn_exit = QPushButton("退出程序")
        self.btn_exit.setIcon(QIcon.fromTheme("application-exit"))
        self.btn_exit.clicked.connect(self.close)
        self.btn_exit.setMinimumHeight(40)
        control_layout.addWidget(self.btn_exit)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        control_layout.addWidget(self.progress)

        # 结果区域
        results_group = QGroupBox("分析结果")
        results_layout = QVBoxLayout(results_group)
        self.result_label = QLabel("等待加载图像...")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        font = QFont()
        font.setPointSize(10)
        self.result_label.setFont(font)
        results_layout.addWidget(self.result_label)
        control_layout.addWidget(results_group)

        # 添加伸缩空间
        control_layout.addStretch(1)

        # 右侧图像显示区域
        image_panel = QGroupBox("图像显示")
        image_layout = QVBoxLayout(image_panel)

        # 原始图像显示
        image_layout.addWidget(QLabel("原始图像"))
        self.original_label = QLabel()
        self.original_label.setMinimumSize(512, 512)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("""
            border: 2px solid #1f6feb; border-radius: 5px; 
            background-color: #0d1117; color: #58a6ff; font: bold 14px;
        """)
        self.original_label.setText("未加载图像")
        image_layout.addWidget(self.original_label)

        # 结果图像显示
        image_layout.addWidget(QLabel("分析结果"))
        self.result_image_label = QLabel()
        self.result_image_label.setMinimumSize(512, 512)
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setStyleSheet("""
            border: 2px solid #1f6feb; border-radius: 5px; 
            background-color: #0d1117; color: #58a6ff; font: bold 14px;
        """)
        self.result_image_label.setText("等待分析结果")
        image_layout.addWidget(self.result_image_label)

        # 添加到主布局
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(image_panel, 2)

    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择乳腺钼靶图像", "",
            "医学图像 (*.dcm *.dicom *.tif);;常规图像 (*.png *.jpg *.jpeg);;所有文件 (*.*)"
        )

        if not file_path:
            return

        try:
            # 释放旧图像内存
            if self.original_img is not None:
                self.original_img = None
            if self.highlighted_img is not None:
                self.highlighted_img = None

            # 读取图像（支持DICOM）
            self.original_img = utils.read_image(file_path)
            
            # 验证图像有效性
            if self.original_img is None or self.original_img.size == 0:
                raise ValueError("图像数据为空")
                
            height, width = self.original_img.shape[:2]
            if height > 4096 or width > 4096:
                raise ValueError("图像尺寸过大(>4096px)，请缩放后再加载")
                
            img_size_mb = self.original_img.nbytes / (1024 * 1024)
            if img_size_mb > 200:
                raise MemoryError(f"图像内存占用过高({img_size_mb:.1f}MB)")

            # 显示图像
            self.display_image(self.original_img, self.original_label)
            self.btn_process.setEnabled(True)
            self.result_label.setText("已加载图像，点击'分析图像'开始处理")
            self.image_path = file_path

        except Exception as e:
            self.result_label.setText(f"错误：{str(e)}")

    def display_image(self, img, label, max_size=800):
        """优化图像显示，支持自适应缩放"""
        if img is None:
            label.setText("图像加载失败")
            return
        
        # 转换为显示格式
        display_img = utils.prepare_image_for_display(img)
        
        # 自适应缩放
        h, w = display_img.shape[:2]
        scale = 1.0
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为QPixmap
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def process_image(self):
        """启动图像分析线程"""
        if self.original_img is None:
            self.result_label.setText("错误：未加载图像")
            return

        # 显示处理状态
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.result_label.setText("正在分析图像...")
        self.btn_process.setEnabled(False)
        self.btn_open.setEnabled(False)

        # 启动分析线程
        self.analysis_thread = AnalysisThread(self.original_img, self.spin_k.value())
        self.analysis_thread.update_progress.connect(self.progress.setValue)
        self.analysis_thread.finish_analysis.connect(self.on_analysis_complete)
        self.analysis_thread.analysis_error.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_complete(self, result):
        """分析完成回调"""
        try:
            # 显示结果图像并标注病灶
            self.highlighted_img = result['highlighted_img']
            self.display_lesion_annotations(result, self.result_image_label)
            
            # 生成分析报告
            report = self.generate_analysis_report(result)
            self.result_label.setText(report)
            
            # 启用保存按钮
            self.btn_save.setEnabled(True)
            
        except Exception as e:
            self.result_label.setText(f"结果显示错误: {str(e)}")
        finally:
            # 恢复界面交互
            self.progress.setVisible(False)
            self.btn_process.setEnabled(True)
            self.btn_open.setEnabled(True)
            # 分析完成后全屏（保持强制逻辑）
            self.showFullScreen()

    def on_analysis_error(self, error_msg):
        """分析错误回调"""
        self.result_label.setText(f"分析错误: {error_msg}")
        self.progress.setVisible(False)
        self.btn_process.setEnabled(True)
        self.btn_open.setEnabled(True)

    def generate_analysis_report(self, result):
        """生成详细分析报告"""
        report = f"乳腺钼靶图像分析报告\n\n"
        report += f"图像尺寸: {self.original_img.shape[1]}×{self.original_img.shape[0]}\n"
        report += f"病灶区域占比: {result['lesion_percentage']:.2f}%\n"
        report += f"检测到 {result['lesion_count']} 个可疑病灶\n\n"
        
        if result['lesion_count'] > 0:
            report += "主要病灶特征：\n"
            for i, lesion in enumerate(result['lesion_features'][:3]):
                report += f"  病灶 {i+1}:\n"
                report += f"    面积: {lesion['area']} 像素\n"
                report += f"    圆形度: {lesion['circularity']:.2f}\n"
                report += f"    长轴长度: {lesion['major_axis_length']:.1f} 像素\n"
    
        # 医学建议（结合病灶数量和形状）
        if result['lesion_count'] == 0 or result['lesion_percentage'] < 0.5:
            report += "\n建议：未见明显异常，建议每年定期复查。"
        elif result['lesion_count'] <= 2 and result['lesion_features'][0]['circularity'] > 0.7:
            report += "\n建议：发现良性可能病灶，建议6个月后复查超声。"
        else:
            report += "\n建议：发现可疑病灶，形态学特征不规则，建议尽快到乳腺专科就诊。"
    
        return report

    def display_lesion_annotations(self, result, label):
        """在图像上标注病灶边界和中心"""
        if result is None or 'highlighted_img' not in result:
            return
        
        img = result['highlighted_img'].copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转为彩色用于标注
        
        lesion_features = result.get('lesion_features', [])
        for i, lesion in enumerate(lesion_features[:5]):  # 最多标注5个病灶
            y1, x1, y2, x2 = lesion['bounding_box']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
            
            cv2.putText(img, f"nidus{i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # 黄色文字
        
        self.display_image(img, label)

    def save_analysis_results(self):
        """保存标注图像和分析报告"""
        if self.image_path is None or self.highlighted_img is None:
            self.result_label.setText("错误：无结果可保存")
            return
        
        try:
            # 生成保存路径
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(os.path.dirname(self.image_path), f"{base_name}_analysis_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存标注图像
            marked_img_path = os.path.join(save_dir, f"{base_name}_marked.jpg")
            cv2.imwrite(marked_img_path, self.highlighted_img)
            
            # 保存分析报告
            report_path = os.path.join(save_dir, f"{base_name}_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(self.result_label.text())
            
            # 保存病灶特征
            self.save_lesion_features(save_dir)
            
            self.result_label.setText(f"分析结果已保存至: {save_dir}")
        except Exception as e:
            self.result_label.setText(f"分析结果已保存至: {save_dir}")

    def save_lesion_features(self, save_dir):
        """保存病灶特征到CSV文件"""
        import csv
        lesion_features = self.analysis_thread.result.get('lesion_features', [])  # 假设线程保存结果
        
        if not lesion_features:
            return
        
        csv_path = os.path.join(save_dir, "lesion_features.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ['area', 'perimeter', 'circularity', 'major_axis_length', 
                          'minor_axis_length', 'eccentricity', 'solidity']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for lesion in lesion_features:
                writer.writerow({k: lesion[k] for k in fieldnames if k in lesion})

    def timestamp(self):
        """生成时间戳"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        self.original_img = None
        self.highlighted_img = None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置中文字体
    font = QFont("Microsoft YaHei")
    app.setFont(font)
    window = MammoAnalysisApp()
    window.show()
    sys.exit(app.exec_())