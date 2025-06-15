import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QSpinBox, QProgressBar, QGroupBox,
                             QApplication, QMessageBox)
from PyQt5.QtGui import QIcon, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import processing
from datetime import datetime

# 导入用户提供的辅助函数
import utils as utils  # 请替换为实际模块名


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
                k=self.k_value
            )
            self.update_progress.emit(100)  # 处理完成
            self.finish_analysis.emit(result)
        except Exception as e:
            self.analysis_error.emit(str(e))


class MammoAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("乳腺钼靶图像分析系统")
        self.original_img = None
        self.highlighted_img = None
        self.image_path = None
        self.analysis_result = None
        
        # 初始化UI
        self.init_ui()
        
        # 5秒后自动全屏
        QTimer.singleShot(5000, self.show_full_screen)

    def init_ui(self):
        """初始化用户界面"""
        # 设置科技感样式
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0e14; }
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

        # 左侧控制面板 - 使用百分比布局确保全屏适配
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMinimumWidth(300)  # 设置最小宽度防止布局过窄

        # 图像选择
        self.btn_open = QPushButton("选择乳腺图像")
        self.btn_open.setIcon(QIcon.fromTheme("document-open"))
        self.btn_open.clicked.connect(self.open_image)
        control_layout.addWidget(self.btn_open)

        # 参数设置
        params_group = QGroupBox("分析参数")
        params_layout = QVBoxLayout(params_group)
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("聚类数量 (k):"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 6)
        self.spin_k.setValue(3)
        param_layout.addWidget(self.spin_k)
        params_layout.addLayout(param_layout)
        control_layout.addWidget(params_group)

        # 处理按钮
        self.btn_process = QPushButton("分析图像")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setMinimumHeight(40)
        control_layout.addWidget(self.btn_process)

        # 保存按钮
        self.btn_save = QPushButton("保存分析结果")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_analysis_results)
        self.btn_save.setMinimumHeight(40)
        control_layout.addWidget(self.btn_save)

        # 退出按钮
        self.btn_exit = QPushButton("退出程序")
        self.btn_exit.clicked.connect(self.close)
        self.btn_exit.setMinimumHeight(40)
        control_layout.addWidget(self.btn_exit)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        control_layout.addWidget(self.progress)

        # 结果显示
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

        # 伸缩空间
        control_layout.addStretch(1)

        # 右侧图像显示区域
        image_panel = QGroupBox("图像显示")
        image_layout = QVBoxLayout(image_panel)

        # 原始图像
        image_layout.addWidget(QLabel("原始图像"))
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("""
            border: 2px solid #1f6feb; border-radius: 5px; 
            background-color: #0d1117; color: #58a6ff; font: bold 14px;
        """)
        self.original_label.setText("未加载图像")
        image_layout.addWidget(self.original_label)

        # 分析结果图像
        image_layout.addWidget(QLabel("分析结果"))
        self.result_image_label = QLabel()
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setStyleSheet("""
            border: 2px solid #1f6feb; border-radius: 5px; 
            background-color: #0d1117; color: #58a6ff; font: bold 14px;
        """)
        self.result_image_label.setText("等待分析结果")
        image_layout.addWidget(self.result_image_label)

        # 添加到主布局 - 左侧占30%，右侧占70%
        main_layout.addWidget(control_panel, 3)
        main_layout.addWidget(image_panel, 7)

    def show_full_screen(self):
        """显示全屏界面"""
        self.showFullScreen()
        # 调整图像显示区域的最小尺寸以适应全屏
        self.original_label.setMinimumSize(800, 800)
        self.result_image_label.setMinimumSize(800, 800)

    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择乳腺钼靶图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )

        if not file_path:
            return

        try:
            # 释放旧资源
            self.original_img = None
            self.highlighted_img = None
            self.analysis_result = None
            
            # 读取图像（使用用户提供的辅助函数）
            self.original_img = utils.read_image(file_path)
            
            # 验证图像
            if self.original_img is None or self.original_img.size == 0:
                raise ValueError("图像数据为空")
                
            height, width = self.original_img.shape[:2]
            if height > 4096 or width > 4096:
                raise ValueError("图像尺寸过大(>4096px)")
                
            img_size_mb = self.original_img.nbytes / (1024 * 1024)
            if img_size_mb > 200:
                raise MemoryError(f"图像内存占用过高({img_size_mb:.1f}MB)")

            # 显示图像（使用用户提供的辅助函数）
            self.display_image(self.original_img, self.original_label)
            self.btn_process.setEnabled(True)
            self.result_label.setText("已加载图像，点击'分析图像'开始处理")
            self.image_path = file_path

        except Exception as e:
            self.result_label.setText(f"错误：{str(e)}")
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")

    def display_image(self, img, label, max_size=None):
        """使用用户提供的辅助函数准备并显示图像，支持全屏自适应"""
        if img is None:
            label.setText("图像加载失败")
            return
        
        # 使用用户提供的函数准备图像
        display_img = utils.prepare_image_for_display(img)
        
        # 全屏模式下动态计算最大尺寸
        if max_size is None:
            max_size = min(label.width(), label.height()) - 50
        
        # 自适应缩放
        h, w = display_img.shape[:2]
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
            # 保存分析结果
            self.analysis_result = result
            
            # 显示结果图像
            self.highlighted_img = result['highlighted_img']
            self.display_lesion_annotations(result, self.result_image_label)
            
            # 生成分析报告
            report = self.generate_analysis_report(result)
            self.result_label.setText(report)
            
            # 启用保存按钮
            self.btn_save.setEnabled(True)
            
        except Exception as e:
            self.result_label.setText(f"结果显示错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示分析结果失败: {str(e)}")
        finally:
            # 恢复界面交互
            self.progress.setVisible(False)
            self.btn_process.setEnabled(True)
            self.btn_open.setEnabled(True)
            # 分析完成后确保全屏显示
            self.show_full_screen()

    def on_analysis_error(self, error_msg):
        """分析错误回调"""
        self.result_label.setText(f"分析错误: {error_msg}")
        self.progress.setVisible(False)
        self.btn_process.setEnabled(True)
        self.btn_open.setEnabled(True)
        QMessageBox.critical(self, "错误", f"图像分析失败: {error_msg}")

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
    
        # 医学建议
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
            # 使用用户提供的函数转换为RGB
            img = utils.prepare_image_for_display(img)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        lesion_features = result.get('lesion_features', [])
        for i, lesion in enumerate(lesion_features[:5]):  # 最多标注5个病灶
            y1, x1, y2, x2 = lesion['bounding_box']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
            
            cv2.putText(img, f"nidus{i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # 黄色文字
        
        self.display_image(img, label)

    def save_analysis_results(self):
        """保存标注图像和分析报告"""
        if self.image_path is None or self.analysis_result is None:
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
            # 转换为BGR格式保存
            if len(self.highlighted_img.shape) == 2:
                cv2.imwrite(marked_img_path, self.highlighted_img)
            else:
                cv2.imwrite(marked_img_path, cv2.cvtColor(self.highlighted_img, cv2.COLOR_RGB2BGR))
            
            # 保存分析报告
            report_path = os.path.join(save_dir, f"{base_name}_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(self.result_label.text())
            
            # 保存病灶特征
            processing.save_lesion_features(
                self.analysis_result.get('lesion_features', []), 
                save_dir
            )
            
            self.result_label.setText(f"分析结果已保存至: {save_dir}")
            QMessageBox.information(self, "保存成功", f"分析结果已保存至:\n{save_dir}")
        except Exception as e:
            self.result_label.setText(f"保存失败: {str(e)}")
            QMessageBox.critical(self, "保存失败", f"保存分析结果时出错: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        self.original_img = None
        self.highlighted_img = None
        self.analysis_result = None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置中文字体
    font = QFont("Microsoft YaHei")
    app.setFont(font)
    window = MammoAnalysisApp()
    window.show()
    sys.exit(app.exec_())