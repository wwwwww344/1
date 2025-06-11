import sys
import os
from PyQt5.QtWidgets import QApplication, QLabel
from PIL import Image, ImageTk, ImageSequence
import tkinter as tk
import time
from PyQt5.QtCore import Qt, QTimer
from gui import MammoAnalysisApp

if __name__ == "__main__":
    # 创建应用实例
    app = QApplication(sys.argv)

    # 设置应用名称
    app.setApplicationName("乳腺钼靶分析系统")
    app.setApplicationDisplayName("乳腺钼靶图像分析系统")

    # 创建主窗口(但不立即显示)
    window = MammoAnalysisApp()
    
    # 创建启动动画
    image_path = "E:/new/new/asdf.jpg"  # 使用实际启动图片
    
    # 创建临时启动窗口
    startup_window = QLabel()
    startup_window.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    # 设置启动窗口尺寸(800x600)并居中
    startup_window.resize(800, 600)
    screen_geometry = app.desktop().screenGeometry()
    startup_window.move(
        (screen_geometry.width() - 800) // 2,
        (screen_geometry.height() - 600) // 2
    )
    # 设置科技感样式
    startup_window.setStyleSheet("""
        background-color: #0a0e14;
        border: 2px solid #1f6feb;
        border-radius: 10px;
    """)
    
    # 加载并显示图片
    print(f"正在加载图片: {image_path}")
    try:
        # 创建Tkinter窗口
        root = tk.Tk()
        root.overrideredirect(True)  # 移除窗口装饰（标题栏、边框等）
        root.attributes('-topmost', True)  # 确保窗口在最前
        
        # 使用Pillow加载动画
        with Image.open(image_path) as img:
            print(f"动画加载成功 - 格式: {img.format}, 尺寸: {img.width}x{img.height}")
            # 设置窗口大小为图片尺寸并居中显示
            root.geometry(f"{img.width}x{img.height}+{int((root.winfo_screenwidth()-img.width)/2)}+{int((root.winfo_screenheight()-img.height)/2)}")
            
            # 准备动画帧
            frames = []
            for frame in ImageSequence.Iterator(img):
                frames.append(ImageTk.PhotoImage(frame))
            
            # 创建标签显示动画
            label = tk.Label(root)
            label.pack()
            
            # 检查图片是否存在且格式正确
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
            if not image_path.lower().endswith(valid_extensions):
                raise ValueError(f"不支持的图片格式，请使用: {', '.join(valid_extensions)}")

            # 添加科技感欢迎文字
            text_label = tk.Label(
                root,
                text="生医软控欢迎您的使用",
                font=("Microsoft YaHei", 24, "bold"),
                fg="#1f6feb",
                bg="#0a0e14",
                bd=0,
                highlightthickness=0,
                relief="flat"
            )
            # 添加发光效果
            text_label.config(
                highlightbackground="#1f6feb",
                highlightcolor="#1f6feb",
                highlightthickness=2
            )
            # 将文字放置在窗口底部中央
            text_label.place(relx=0.5, rely=0.95, anchor="center")
            
            # 动画播放控制变量
            start_time = time.time()
            play_duration = 4  # 播放4秒
            
            # 动画播放函数
            def play_animation(frame_num=0):
                frame = frames[frame_num]
                label.config(image=frame)
                label.image = frame
                
                # 检查是否达到播放时长
                if time.time() - start_time < play_duration:
                    root.after(100, play_animation, (frame_num + 1) % len(frames))
                else:
                    root.destroy()  # 关闭窗口
            
            # 开始播放动画
            play_animation()
            root.mainloop()
            
            print(f"图片加载成功 - 格式: {img.format}, 尺寸: {img.width}x{img.height}")
            
            # 显示主窗口
            window.show()
        
    except Exception as e:
        print(f"图片加载出错: {str(e)}")
        print(f"无法加载启动图片: {image_path}")
        print("请检查: 1.图片是否存在 2.图片格式是否正确(jpg/png/bmp)")
        window.show()

    # 启动事件循环
    sys.exit(app.exec_())