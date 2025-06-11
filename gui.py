from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSpinBox, QProgressBar, QGroupBox,
    QApplication
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QTimer
import processing
import cv2
import numpy as np
import os
import utils
import sys


class MammoAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("乳腺钼靶图像分析系统")
        self.setGeometry(100, 100, 1200, 800)
        self.original_img = None  # 初始化图像属性
        
        # 设置窗口图标
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.ico")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                print(f"提示: 自定义图标未找到 ({icon_path})，将使用默认图标")
                # 使用Qt默认图标
                self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        except Exception as e:
            print(f"警告: 加载图标时出错 - {str(e)}，使用默认图标")
            self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

        # 初始化UI
        self.init_ui()

        # 初始显示为普通窗口
        
        # 延迟1秒后切换到全屏模式
        QTimer.singleShot(5000, self.showFullScreen)

        # 初始化变量
        self.image_path = None
        self.original_img = None
        self.highlighted_img = None

    def init_ui(self):
        """初始化用户界面"""
        # 创建菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        
        open_action = file_menu.addAction("打开图像")
        open_action.triggered.connect(self.open_image)
        
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

        # 设置科技感样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0e14;
            }
            QStatusBar {
                background-color: #0a0e14;
                color: #58a6ff;
                border-top: 1px solid #1f6feb;
                font-family: "Microsoft YaHei";
            }
            QMenuBar {
                background-color: #0a0e14;
                color: #c9d1d9;
                border-bottom: 1px solid #1f6feb;
                font-family: "Microsoft YaHei";
            }
            QMenuBar::item {
                padding: 5px 10px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #1f6feb;
            }
            QMenuBar::item:pressed {
                background-color: #0d419d;
            }
            QMenu {
                background-color: #0a0e14;
                color: #c9d1d9;
                border: 1px solid #1f6feb;
            }
            QMenu::item:selected {
                background-color: #1f6feb;
            }
            QGroupBox {
                color: #58a6ff;
                font: bold 12px;
                border: 1px solid #1f6feb;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #1f6feb;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-family: "Microsoft YaHei";
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #58a6ff;
            }
            QPushButton:pressed {
                background-color: #0d419d;
            }
            QLabel {
                color: #c9d1d9;
                font-family: "Microsoft YaHei";
            }
            QSpinBox {
                background-color: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 3px;
                padding: 3px;
            }
            QProgressBar {
                border: 1px solid #30363d;
                border-radius: 3px;
                text-align: center;
                background-color: #0d1117;
            }
            QProgressBar::chunk {
                background-color: #1f6feb;
                width: 10px;
            }
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

        # 聚类参数
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

        # 退出程序按钮
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
        # 设置字体
        font = QFont()
        font.setPointSize(10)
        self.result_label.setFont(font)
        results_layout.addWidget(self.result_label)
        control_layout.addWidget(results_group)

        # 添加伸缩空间
        control_layout.addStretch(1)

        # 添加到主布局
        main_layout.addWidget(control_panel, 1)

        # 右侧图像显示区域
        image_panel = QGroupBox("图像显示")
        image_layout = QVBoxLayout(image_panel)

        # 原始图像显示
        image_layout.addWidget(QLabel("原始图像"))
        self.original_label = QLabel()
        self.original_label.setMinimumSize(512, 512)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("""
            border: 2px solid #1f6feb;
            border-radius: 5px;
            background-color: #0d1117;
            color: #58a6ff;
            font: bold 14px;
        """)
        self.original_label.setText("未加载图像")
        image_layout.addWidget(self.original_label)

        # 结果图像显示
        image_layout.addWidget(QLabel("分析结果"))
        self.result_image_label = QLabel()
        self.result_image_label.setMinimumSize(512, 512)
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setStyleSheet("""
            border: 2px solid #1f6feb;
            border-radius: 5px;
            background-color: #0d1117;
            color: #58a6ff;
            font: bold 14px;
        """)
        self.result_image_label.setText("等待分析结果")
        image_layout.addWidget(self.result_image_label)

        main_layout.addWidget(image_panel, 2)

    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择乳腺钼靶图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.tif *.dcm);;所有文件 (*.*)"
        )

        if file_path:
            self.image_path = file_path

            try:
                # 检查文件大小
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                if file_size > 50:
                    self.result_label.setText("错误：图像文件过大(>50MB)，请选择较小文件")
                    return

                # 释放之前的内存
                if hasattr(self, 'original_img') and self.original_img is not None:
                    del self.original_img
                if hasattr(self, 'highlighted_img') and self.highlighted_img is not None:
                    del self.highlighted_img

                # 读取图像
                self.original_img = utils.read_image(file_path)

                if self.original_img is not None:
                    # 检查图像内存占用
                    img_size = self.original_img.nbytes / (1024 * 1024)  # MB
                    if img_size > 100:
                        self.result_label.setText("错误：图像内存占用过大(>100MB)，请选择较小图像")
                        self.original_img = None
                        return

                    # 显示原始图像
                    self.display_image(self.original_img, self.original_label)
                    self.btn_process.setEnabled(True)
                    self.result_label.setText("已加载图像，请点击'分析图像'按钮进行处理")

                    # 重置结果图像
                    self.result_image_label.clear()
                    self.result_image_label.setText("等待分析结果")
                    self.highlighted_img = None
                else:
                    self.result_label.setText("错误：无法加载图像文件")
            except MemoryError:
                self.result_label.setText("错误：内存不足，无法加载图像")
            except Exception as e:
                self.result_label.setText(f"加载错误: {str(e)}")

    def display_image(self, img, label):
        """在QLabel中显示OpenCV图像"""
        if img is None:
            label.setText("图像加载失败")
            return

        # 转换图像为适合显示的格式
        display_img = utils.prepare_image_for_display(img)

        # 转换为QPixmap
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 保持宽高比缩放
        scaled_pixmap = pixmap.scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def exit_fullscreen(self):
        """退出全屏模式"""
        self.showNormal()
        self.btn_exit_fullscreen.setEnabled(False)
        self.btn_exit_fullscreen.setEnabled(False)

    def process_image(self):
        """处理图像并显示结果"""
        if self.original_img is None:
            self.result_label.setText("错误：未加载图像")
            return

        # 准备处理状态
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(10)
        self.result_label.setText("正在分析图像...")
        self.btn_process.setEnabled(False)
        self.btn_open.setEnabled(False)

        # 强制更新UI
        QApplication.processEvents()

        try:
            # 获取聚类数量
            k = self.spin_k.value()

            # 更新进度
            self.progress.setValue(30)
            QApplication.processEvents()

            # 调用处理函数
            result = processing.analyze_mammo_image(self.original_img, k)

            # 更新进度
            self.progress.setValue(80)
            QApplication.processEvents()

            # 显示结果
            self.highlighted_img = result['highlighted_img']
            self.display_image(self.highlighted_img, self.result_image_label)

            # 显示分析结果
            self.result_label.setText(
                f"分析完成!\n\n"
                f"图像尺寸: {self.original_img.shape[1]}×{self.original_img.shape[0]}\n"
                f"病灶区域占比: {result['lesion_percentage']:.2f}%\n"
                f"最大灰度聚类: {result['target_cluster'] + 1}/{k}\n\n"
                f"建议: {self.get_recommendation(result['lesion_percentage'])}"
            )

            # 进入全屏模式
            self.showFullScreen()

            # 完成进度
            self.progress.setValue(100)

        except Exception as e:
            self.result_label.setText(f"处理错误: {str(e)}")
        finally:
            # 恢复按钮状态
            self.btn_process.setEnabled(True)
            self.btn_open.setEnabled(True)
            # 延迟隐藏进度条
            QTimer.singleShot(1500, lambda: self.progress.setVisible(False))

    def get_recommendation(self, percentage):
        """根据病灶占比生成建议"""
        if percentage < 0.5:
            return "未见明显异常，建议定期复查。"
        elif percentage < 2:
            return "发现可疑小病灶，建议3-6个月内复查。"
        elif percentage < 5:
            return "发现明显病灶，建议进一步做超声或MRI检查。"
        else:
            return "发现较大范围病灶，建议立即进行活检并专科就诊。"

    def resizeEvent(self, event):
        """窗口大小变化时重新调整图像"""
        super().resizeEvent(event)
        if hasattr(self, 'original_img') and self.original_img is not None:
            self.display_image(self.original_img, self.original_label)
        if hasattr(self, 'highlighted_img') and self.highlighted_img is not None:
            self.display_image(self.highlighted_img, self.result_image_label)