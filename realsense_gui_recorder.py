"""
RealSense GUI 录制程序
功能：
1. 程序启动后先打开相机并显示实时预览，但不会自动录制；
2. 点击“开始录制”后才创建 mp4 文件并开始写入视频帧；
3. 点击“停止录制”后关闭当前视频文件，但程序和相机预览继续运行；
4. 可以再次点击“开始录制”生成一个新的视频文件；
5. 默认保存彩色画面，深度流仍然开启并对齐到彩色流，便于后续扩展。
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class RealSenseWorker(QThread):
    """RealSense采集线程。

    该线程只负责两件事：
    1. 持续从RealSense读取帧，并把预览图像发送给GUI；
    2. 当recording=True时，把彩色帧写入VideoWriter。

    注意：相机采集不能放在GUI主线程，否则界面会卡死。
    """

    image_ready = Signal(object)              # 发送RGB图像，类型为numpy.ndarray
    status_msg = Signal(str)                  # 发送状态信息到界面日志
    camera_started = Signal(int, int, int)    # width, height, fps
    recording_started = Signal(str)           # 视频保存路径
    recording_stopped = Signal(str, int, float, float)  # path, frames, seconds, real_fps
    camera_error = Signal(str)                # 错误信息

    def __init__(self, parent=None):
        super().__init__(parent)

        # RealSense参数，与原代码保持一致
        self.width = 640
        self.height = 480
        self.fps = 30

        self.pipeline = None
        self.align = None
        self.running = False

        # 录制相关变量
        self.recording = False
        self.video_writer = None
        self.video_path = ""
        self.frame_count = 0
        self.record_start_time = 0.0

        # 保护录制状态，避免GUI线程和采集线程同时访问VideoWriter
        self.record_lock = threading.Lock()

    def run(self):
        """线程入口函数：启动相机并循环读取图像。"""
        self.running = True

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()

            # 彩色流：用于显示和保存
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            # 深度流：当前只用于对齐，默认不保存深度图
            config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )

            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.camera_started.emit(self.width, self.height, self.fps)
            self.status_msg.emit("相机启动成功，当前为预览状态，尚未开始录制。")

        except Exception as e:
            self.running = False
            self.camera_error.emit(f"相机启动失败：{e}")
            return

        while self.running:
            try:
                # wait_for_frames超时不要太长，便于关闭程序时快速退出线程
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                if not color_frame:
                    continue

                # RealSense彩色帧为BGR格式，VideoWriter也使用BGR
                color_image = np.asanyarray(color_frame.get_data())

                # 录制时写入原始彩色帧；预览时叠加文字，不影响保存视频
                with self.record_lock:
                    is_recording = self.recording and self.video_writer is not None
                    if is_recording:
                        self.video_writer.write(color_image)
                        self.frame_count += 1
                        frame_count = self.frame_count
                    else:
                        frame_count = self.frame_count

                # 构造预览画面
                display_image = color_image.copy()
                if is_recording:
                    cv2.putText(
                        display_image,
                        f"REC  Frame: {frame_count}",
                        (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    cv2.circle(display_image, (20, 60), 8, (0, 0, 255), -1)
                else:
                    cv2.putText(
                        display_image,
                        "Preview - Not Recording",
                        (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Qt显示需要RGB格式
                rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                self.image_ready.emit(rgb_image.copy())

            except RuntimeError as e:
                # 偶发超时不直接退出，连续没有帧时界面仍保持可关闭
                self.status_msg.emit(f"读取相机帧超时或失败：{e}")
                continue
            except Exception as e:
                self.camera_error.emit(f"运行时错误：{e}")
                break

        # 退出线程前释放资源
        self._release_writer_if_needed()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                self.status_msg.emit("相机已关闭。")
            except Exception as e:
                self.status_msg.emit(f"关闭相机时出现异常：{e}")

    def start_recording(self, save_dir: str):
        """开始录制视频。

        Args:
            save_dir: 视频保存文件夹。
        """
        with self.record_lock:
            if self.recording:
                self.status_msg.emit("当前已经在录制中，请先停止当前录制。")
                return

            save_path = Path(save_dir).expanduser().resolve()
            save_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = str(save_path / f"realsense_video_{timestamp}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, self.fps, (self.width, self.height)
            )

            if not self.video_writer.isOpened():
                self.video_writer = None
                self.video_path = ""
                self.status_msg.emit("无法创建MP4文件，请检查编码器或保存路径权限。")
                return

            self.frame_count = 0
            self.record_start_time = time.time()
            self.recording = True

        self.recording_started.emit(self.video_path)
        self.status_msg.emit(f"开始录制：{self.video_path}")

    def stop_recording(self):
        """停止当前录制，但不关闭相机，也不退出程序。"""
        with self.record_lock:
            if not self.recording:
                self.status_msg.emit("当前没有正在录制的视频。")
                return

            path = self.video_path
            frames = self.frame_count
            seconds = max(time.time() - self.record_start_time, 1e-6)
            real_fps = frames / seconds

            if self.video_writer is not None:
                self.video_writer.release()

            self.video_writer = None
            self.video_path = ""
            self.frame_count = 0
            self.record_start_time = 0.0
            self.recording = False

        self.recording_stopped.emit(path, frames, seconds, real_fps)
        self.status_msg.emit(
            f"录制已停止：{path}，共保存 {frames} 帧，时长 {seconds:.2f} s，实际帧率 {real_fps:.2f} fps。"
        )

    def stop_worker(self):
        """通知线程退出。"""
        self.running = False

    def _release_writer_if_needed(self):
        """线程退出时释放可能还未关闭的视频文件。"""
        with self.record_lock:
            if self.video_writer is not None:
                self.video_writer.release()
            self.video_writer = None
            self.recording = False


class MainWindow(QMainWindow):
    """RealSense录制GUI主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSense 视频录制工具")
        self.resize(980, 760)

        self.worker = RealSenseWorker()
        self.is_closing = False
        self._build_ui()
        self._connect_signals()

        # 程序启动后自动打开相机预览，但不自动录制
        self.worker.start()

    def _build_ui(self):
        """创建界面控件。"""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        self.video_label = QLabel("正在启动相机...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "QLabel { background-color: #202020; color: white; border: 1px solid #555; }"
        )
        main_layout.addWidget(self.video_label)

        # 保存路径区域
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("保存文件夹："))

        default_save_dir = str((Path.cwd() / "videos").resolve())
        self.save_dir_edit = QLineEdit(default_save_dir)
        path_layout.addWidget(self.save_dir_edit, stretch=1)

        self.choose_dir_btn = QPushButton("选择文件夹")
        path_layout.addWidget(self.choose_dir_btn)
        main_layout.addLayout(path_layout)

        # 控制按钮区域
        btn_layout = QHBoxLayout()
        self.start_record_btn = QPushButton("开始录制")
        self.stop_record_btn = QPushButton("停止录制")
        self.exit_btn = QPushButton("退出程序")

        self.start_record_btn.setEnabled(False)  # 等相机启动成功后再允许录制
        self.stop_record_btn.setEnabled(False)

        self.start_record_btn.setMinimumHeight(42)
        self.stop_record_btn.setMinimumHeight(42)
        self.exit_btn.setMinimumHeight(42)

        btn_layout.addWidget(self.start_record_btn)
        btn_layout.addWidget(self.stop_record_btn)
        btn_layout.addWidget(self.exit_btn)
        main_layout.addLayout(btn_layout)

        self.status_label = QLabel("状态：正在启动相机")
        main_layout.addWidget(self.status_label)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(130)
        main_layout.addWidget(self.log_edit)

    def _connect_signals(self):
        """绑定按钮和线程信号。"""
        self.choose_dir_btn.clicked.connect(self.choose_save_dir)
        self.start_record_btn.clicked.connect(self.start_recording)
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.exit_btn.clicked.connect(self.close)

        self.worker.image_ready.connect(self.update_image)
        self.worker.status_msg.connect(self.append_log)
        self.worker.camera_started.connect(self.on_camera_started)
        self.worker.recording_started.connect(self.on_recording_started)
        self.worker.recording_stopped.connect(self.on_recording_stopped)
        self.worker.camera_error.connect(self.on_camera_error)

    @Slot()
    def choose_save_dir(self):
        """选择视频保存文件夹。"""
        current_dir = self.save_dir_edit.text().strip() or str(Path.cwd())
        directory = QFileDialog.getExistingDirectory(self, "选择视频保存文件夹", current_dir)
        if directory:
            self.save_dir_edit.setText(directory)

    @Slot()
    def start_recording(self):
        """点击开始录制按钮后的处理。"""
        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            QMessageBox.warning(self, "保存路径为空", "请先选择视频保存文件夹。")
            return

        self.worker.start_recording(save_dir)

    @Slot()
    def stop_recording(self):
        """点击停止录制按钮后的处理。"""
        self.worker.stop_recording()

    @Slot(object)
    def update_image(self, rgb_image: np.ndarray):
        """把采集线程传来的RGB图像显示到QLabel上。"""
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width

        # copy()很重要：避免numpy数组生命周期结束后QImage引用失效
        q_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    @Slot(int, int, int)
    def on_camera_started(self, width: int, height: int, fps: int):
        """相机启动成功后更新界面状态。"""
        self.start_record_btn.setEnabled(True)
        self.status_label.setText(f"状态：相机预览中，分辨率 {width}x{height}，{fps} FPS")
        self.append_log(f"相机预览已启动：{width}x{height}@{fps}FPS。")

    @Slot(str)
    def on_recording_started(self, path: str):
        """录制开始后更新按钮状态。"""
        self.start_record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)
        self.status_label.setText(f"状态：正在录制 -> {path}")

    @Slot(str, int, float, float)
    def on_recording_stopped(self, path: str, frames: int, seconds: float, real_fps: float):
        """录制停止后恢复按钮状态。"""
        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.status_label.setText(
            f"状态：录制已停止，保存 {frames} 帧，实际帧率 {real_fps:.2f} FPS"
        )
        if not self.is_closing:
            QMessageBox.information(
                self,
                "录制完成",
                f"视频已保存：\n{path}\n\n帧数：{frames}\n时长：{seconds:.2f} s\n实际帧率：{real_fps:.2f} FPS",
            )

    @Slot(str)
    def on_camera_error(self, msg: str):
        """显示相机或运行错误。"""
        self.append_log(msg)
        self.status_label.setText(f"状态：错误 - {msg}")
        QMessageBox.critical(self, "运行错误", msg)
        self.start_record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)

    @Slot(str)
    def append_log(self, msg: str):
        """在日志框中追加一行信息。"""
        now = datetime.now().strftime("%H:%M:%S")
        self.log_edit.append(f"[{now}] {msg}")

    def closeEvent(self, event):
        """关闭窗口时，安全停止录制、关闭相机线程。"""
        self.is_closing = True
        if self.worker.isRunning():
            self.worker.stop_recording()
            self.worker.stop_worker()
            self.worker.wait(3000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
