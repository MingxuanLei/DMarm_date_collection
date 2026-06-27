from __future__ import annotations
import contextlib
import os
import queue
import sys
import threading
import time
import traceback
from typing import Optional

from intelrealsense_episode_recorder import D435iRecorder

from PySide6.QtCore import QObject, Signal, Qt, QTimer
from PySide6.QtGui import QTextCursor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
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

try:
    import teleoperation_episode_data_collection as core
except Exception as exc:  # pragma: no cover - only used at runtime on user machine
    core = None
    CORE_IMPORT_ERROR = exc
else:
    CORE_IMPORT_ERROR = None
class QtLogWriter:
    """把 print/stdout/stderr 重定向到 Qt 日志框。"""

    def __init__(self, signal: Signal):
        self.signal = signal
        self._buffer = ""
        self._lock = threading.RLock()

    def write(self, text: str) -> int:
        if text is None:
            return 0
        text = str(text)
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self.signal.emit(line)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buffer.strip():
                self.signal.emit(self._buffer.rstrip("\n"))
            self._buffer = ""


class TeleoperationWorker(QObject):
    log_signal = Signal(str)
    mode_signal = Signal(str)
    finished_signal = Signal(bool, str)
    running_signal = Signal(bool)

    def __init__(self, camera_recorder: Optional[D435iRecorder] = None):
        super().__init__()
        self.camera_recorder = camera_recorder
        self._thread: Optional[threading.Thread] = None
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._running_lock = threading.RLock()
        self._running = False

    @property
    def running(self) -> bool:
        with self._running_lock:
            return self._running

    def _set_running(self, value: bool) -> None:
        with self._running_lock:
            self._running = bool(value)
        self.running_signal.emit(bool(value))

    def start_or_switch(
        self,
        cmd: str,
        record_file: Optional[str],
        replay_file: Optional[str],
        replay_source: str,
        replay_speed: float,
        enable_tool_teleop: bool,
        enable_weak_bilateral: bool,
        enable_error_feedback: bool,
        enable_torque_feedback: bool,
        enable_tool_weak_feedback: bool,
        enable_d435i_recording: bool,
    ) -> None:
        if core is None:
            self.log_signal.emit(f"[ERR] 导入 teleoperation_episode_data_collection.py 失败: {CORE_IMPORT_ERROR}")
            return

        if cmd not in (core.CMD_TELEOP, core.CMD_RECORD, core.CMD_REPLAY, core.CMD_EXIT):
            self.log_signal.emit(f"[WARN] 未知命令: {cmd}")
            return

        if self.running:
            self._cmd_queue.put(cmd)
            self.log_signal.emit(f"[CMD] 已发送切换命令: {cmd}")
            return

        self._stop_event.clear()
        self._cmd_queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            args=(
                cmd,
                record_file,
                replay_file,
                replay_source,
                replay_speed,
                enable_tool_teleop,
                enable_weak_bilateral,
                enable_error_feedback,
                enable_torque_feedback,
                enable_tool_weak_feedback,
                enable_d435i_recording,
            ),
            name="teleoperation_gui_worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self.running:
            self._cmd_queue.put(core.CMD_EXIT if core is not None else "0")
            self.log_signal.emit("[CMD] 已发送安全退出命令 0")
        else:
            self.log_signal.emit("[INFO] 当前没有正在运行的遥操作/回放任务")

    def _apply_runtime_options(
        self,
        enable_tool_teleop: bool,
        enable_weak_bilateral: bool,
        enable_error_feedback: bool,
        enable_torque_feedback: bool,
        enable_tool_weak_feedback: bool,
        enable_d435i_recording: bool,
    ) -> None:
        # 这些是 teleoperation_episode_data_collection.py 中的全局开关。
        core.ENABLE_TOOL_TELEOP = bool(enable_tool_teleop)
        core.ENABLE_WEAK_BILATERAL = bool(enable_weak_bilateral)
        core.ENABLE_ERROR_FEEDBACK = bool(enable_error_feedback)
        core.ENABLE_TORQUE_FEEDBACK = bool(enable_torque_feedback)
        core.ENABLE_TOOL_WEAK_FEEDBACK = bool(enable_tool_weak_feedback)
        core.ENABLE_D435I_RECORDING = bool(enable_d435i_recording)

    def _run(
        self,
        initial_cmd: str,
        record_file: Optional[str],
        replay_file: Optional[str],
        replay_source: str,
        replay_speed: float,
        enable_tool_teleop: bool,
        enable_weak_bilateral: bool,
        enable_error_feedback: bool,
        enable_torque_feedback: bool,
        enable_tool_weak_feedback: bool,
        enable_d435i_recording: bool,
    ) -> None:
        self._set_running(True)
        ok = False
        msg = "任务结束"
        writer = QtLogWriter(self.log_signal)

        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                self._apply_runtime_options(
                    enable_tool_teleop=enable_tool_teleop,
                    enable_weak_bilateral=enable_weak_bilateral,
                    enable_error_feedback=enable_error_feedback,
                    enable_torque_feedback=enable_torque_feedback,
                    enable_tool_weak_feedback=enable_tool_weak_feedback,
                    enable_d435i_recording=enable_d435i_recording,
                )

                print("=" * 70)
                print("七电机弱双向遥操作示教 GUI 已启动（直接调用 teleoperation_episode_data_collection.py）")
                print(f"第7工具电机遥操作：{'开启' if core.ENABLE_TOOL_TELEOP else '关闭'}")
                print(f"弱双向反馈：{'开启' if core.ENABLE_WEAK_BILATERAL else '关闭'}")
                print(f"误差反馈：{'开启' if core.ENABLE_ERROR_FEEDBACK else '关闭'}")
                print(f"电机力矩反馈：{'开启' if core.ENABLE_TORQUE_FEEDBACK else '关闭'}")
                print(f"第7工具电机弱反馈：{'开启' if core.ENABLE_TOOL_WEAK_FEEDBACK else '关闭'}")
                print(f"D435i RGB同步录制：{'开启' if core.ENABLE_D435I_RECORDING else '关闭'}")
                print(f"数据根目录：{core.DATA_DIR}")
                print("保存结构：data/episode_xx/trajectory 与 data/episode_xx/video")
                print("=" * 70)

                current_cmd = initial_cmd
                while current_cmd != core.CMD_EXIT and not self._stop_event.is_set():
                    current_mode = core.mode_from_command(current_cmd)
                    if current_mode is None:
                        print(f"[WARN] 未知模式命令: {current_cmd}")
                        break

                    mode_name = core.MODE_CN_NAME.get(current_mode, str(current_mode))
                    self.mode_signal.emit(mode_name)
                    print()
                    print(f"========== 当前模式：{mode_name} ==========")
                    print("可通过 GUI 按钮切换模式，或点击安全退出。")
                    print()

                    if current_mode in (core.RUN_MODE_TELEOP, core.RUN_MODE_RECORD):
                        # 不在 GUI/Worker 中提前创建 episode；真正进入记录模式时由核心文件创建。
                        current_cmd = core.run_teleop_record_session(
                            self._cmd_queue,
                            current_mode,
                            record_file,
                            enable_camera_recording=enable_d435i_recording,
                            camera_recorder=(self.camera_recorder if enable_d435i_recording else None),
                        )
                    elif current_mode == core.RUN_MODE_REPLAY:
                        resolved_replay_file = (
                            core.resolve_replay_file_path(replay_file)
                            or core.LAST_RECORD_FILE
                            or core.find_latest_teach_record_file()
                        )

                        if resolved_replay_file is None:
                            print("[ERR] 未指定回放文件，也没有在 data/episode_xx/trajectory 文件夹中找到 CSV。")
                            print("[ERR] 请先进入遥操作记录模式生成轨迹文件，或在界面中指定回放文件。")
                            msg = "没有可回放的轨迹文件"
                            break

                        print(f"[REPLAY] 本次使用的回放文件: {resolved_replay_file}")
                        current_cmd = core.run_replay_session(
                            self._cmd_queue,
                            resolved_replay_file,
                            replay_source=replay_source,
                            replay_speed=replay_speed,
                        )
                    else:
                        raise RuntimeError(f"未知运行模式: {current_mode}")

                    if current_cmd is None:
                        current_cmd = core.CMD_EXIT

                ok = True
                msg = "已安全结束"
                self.mode_signal.emit("空闲")
                print("[EXIT] GUI 任务已结束")

        except Exception:
            ok = False
            msg = "运行异常"
            self.log_signal.emit(traceback.format_exc())
            self.mode_signal.emit("异常")
        finally:
            try:
                writer.flush()
            except Exception:
                pass
            self._set_running(False)
            self.finished_signal.emit(ok, msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("七电机双向力反馈示教记录回放界面（Episode版+D435i RGB）")
        self.resize(1320, 900)

        self.camera_recorder: Optional[D435iRecorder] = None

        self.worker = TeleoperationWorker()
        self.worker.log_signal.connect(self.append_log)
        self.worker.mode_signal.connect(self.on_mode_changed)
        self.worker.running_signal.connect(self.on_running_changed)
        self.worker.finished_signal.connect(self.on_worker_finished)

        self._build_ui()
        self._init_camera_preview()
        self.worker.camera_recorder = self.camera_recorder

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_labels)
        self._refresh_timer.start(500)

        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._refresh_camera_preview)
        self._camera_timer.start(33)

        if core is None:
            self.append_log(f"[ERR] 导入 teleoperation_episode_data_collection.py 失败: {CORE_IMPORT_ERROR}")
            self.append_log("[ERR] 请确认 GUI 文件与 teleoperation_episode_data_collection.py 位于同一目录。")
        else:
            self.append_log("[INFO] GUI 初始化完成。当前界面直接调用 teleoperation_episode_data_collection.py，第7工具电机弱反馈默认开启。")
            self.append_log(f"[INFO] 数据根目录: {core.DATA_DIR}")
            self.append_log("[INFO] 每次记录自动创建 data/episode_xx/trajectory 与 data/episode_xx/video。")
            self.append_log("[INFO] 记录CSV包含从端实际关节速度 slave_actual_v1~v6 和 slave_tool_actual_vel。")

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        # 顶部状态
        status_group = QGroupBox("运行状态")
        status_layout = QGridLayout(status_group)
        self.mode_label = QLabel("当前模式：空闲")
        self.running_label = QLabel("运行状态：未运行")
        self.record_dir_label = QLabel("轨迹目录：每次记录自动生成 data/episode_xx/trajectory")
        self.video_dir_label = QLabel("视频目录：每次记录自动生成 data/episode_xx/video")
        if core is not None:
            self.record_dir_label.setText("轨迹目录：data/episode_xx/trajectory（记录时自动创建）")
            self.video_dir_label.setText("视频目录：data/episode_xx/video（记录时自动创建）")
        status_layout.addWidget(self.mode_label, 0, 0)
        status_layout.addWidget(self.running_label, 0, 1)
        status_layout.addWidget(self.record_dir_label, 1, 0, 1, 2)
        status_layout.addWidget(self.video_dir_label, 2, 0, 1, 2)
        root.addWidget(status_group)

        # 控制按钮
        control_group = QGroupBox("模式控制")
        control_layout = QGridLayout(control_group)
        self.btn_teleop = QPushButton("1 遥操作模式\n双向力反馈")
        self.btn_record = QPushButton("2 遥操作记录模式\n双向力反馈 + 保存轨迹")
        self.btn_replay = QPushButton("3 示教回放模式")
        self.btn_stop = QPushButton("0 安全退出/停止当前任务")
        self.btn_teleop.clicked.connect(lambda: self.start_or_switch(core.CMD_TELEOP if core else "1"))
        self.btn_record.clicked.connect(lambda: self.start_or_switch(core.CMD_RECORD if core else "2"))
        self.btn_replay.clicked.connect(lambda: self.start_or_switch(core.CMD_REPLAY if core else "3"))
        self.btn_stop.clicked.connect(self.worker.stop)
        control_layout.addWidget(self.btn_teleop, 0, 0)
        control_layout.addWidget(self.btn_record, 0, 1)
        control_layout.addWidget(self.btn_replay, 0, 2)
        control_layout.addWidget(self.btn_stop, 0, 3)
        root.addWidget(control_group)

        # 文件设置
        file_group = QGroupBox("记录与回放文件")
        file_layout = QGridLayout(file_group)
        self.record_file_edit = QLineEdit()
        self.record_file_edit.setPlaceholderText("留空：自动保存为 data/episode_xx/trajectory/teach_record_时间.csv；如填写只取文件名")
        self.replay_file_edit = QLineEdit()
        self.replay_file_edit.setPlaceholderText("留空：自动读取本次记录文件或 data/episode_xx/trajectory 中最新记录")
        self.btn_browse_record = QPushButton("选择记录文件名")
        self.btn_browse_replay = QPushButton("选择回放CSV")
        self.btn_browse_record.clicked.connect(self.browse_record_file)
        self.btn_browse_replay.clicked.connect(self.browse_replay_file)

        self.replay_source_combo = QComboBox()
        self.replay_source_combo.addItem("从端实际轨迹 actual", "actual")
        self.replay_source_combo.addItem("从端目标轨迹 target", "target")

        self.replay_speed_spin = QDoubleSpinBox()
        self.replay_speed_spin.setRange(0.05, 10.0)
        self.replay_speed_spin.setDecimals(2)
        self.replay_speed_spin.setSingleStep(0.1)
        self.replay_speed_spin.setValue(1.0)

        file_layout.addWidget(QLabel("记录文件"), 0, 0)
        file_layout.addWidget(self.record_file_edit, 0, 1)
        file_layout.addWidget(self.btn_browse_record, 0, 2)
        file_layout.addWidget(QLabel("回放文件"), 1, 0)
        file_layout.addWidget(self.replay_file_edit, 1, 1)
        file_layout.addWidget(self.btn_browse_replay, 1, 2)
        file_layout.addWidget(QLabel("回放源"), 2, 0)
        file_layout.addWidget(self.replay_source_combo, 2, 1)
        file_layout.addWidget(QLabel("回放速度倍率"), 2, 2)
        file_layout.addWidget(self.replay_speed_spin, 2, 3)
        root.addWidget(file_group)

        # 反馈设置
        fb_group = QGroupBox("遥操作与力反馈开关")
        fb_layout = QGridLayout(fb_group)
        self.check_tool_teleop = QCheckBox("启用第7工具电机遥操作")
        self.check_weak_bilateral = QCheckBox("启用弱双向力反馈")
        self.check_error_feedback = QCheckBox("启用从端跟踪误差反馈")
        self.check_torque_feedback = QCheckBox("启用从端电机力矩反馈")
        self.check_tool_weak_feedback = QCheckBox("启用第7工具电机弱反馈")
        for cb in (
            self.check_tool_teleop,
            self.check_weak_bilateral,
            self.check_error_feedback,
            self.check_torque_feedback,
            self.check_tool_weak_feedback,
        ):
            cb.setChecked(True)
        fb_layout.addWidget(self.check_tool_teleop, 0, 0)
        fb_layout.addWidget(self.check_weak_bilateral, 0, 1)
        fb_layout.addWidget(self.check_error_feedback, 1, 0)
        fb_layout.addWidget(self.check_torque_feedback, 1, 1)
        fb_layout.addWidget(self.check_tool_weak_feedback, 2, 0)
        root.addWidget(fb_group)

        # D435i RGB 实时预览与同步采集设置
        camera_group = QGroupBox("D435i RGB实时预览与同步采集")
        camera_layout = QGridLayout(camera_group)
        self.check_d435i_recording = QCheckBox("记录模式下同步录制D435i RGB视频")
        self.check_d435i_recording.setChecked(True)
        camera_layout.addWidget(self.check_d435i_recording, 0, 0)
        camera_layout.addWidget(QLabel("保存：data/episode_xx/video/teach_record_时间_color.mp4、camera_timestamps.csv、camera_meta.json"), 0, 1)

        self.camera_rgb_label = QLabel("D435i RGB画面初始化中...")
        self.camera_rgb_label.setAlignment(Qt.AlignCenter)
        self.camera_rgb_label.setMinimumSize(640, 360)
        self.camera_rgb_label.setStyleSheet("background-color: #202020; color: #DDDDDD; border: 1px solid #666666;")
        camera_layout.addWidget(self.camera_rgb_label, 1, 0, 1, 2)
        root.addWidget(camera_group)

        # 日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.clicked.connect(self.log_box.clear)
        log_layout.addWidget(self.log_box)
        log_layout.addWidget(self.btn_clear_log, 0, Qt.AlignRight)
        root.addWidget(log_group, 1)

        self.setCentralWidget(central)

    def _init_camera_preview(self) -> None:
        """启动 D435i RGB 预览。预览和记录分离，记录模式下才写入视频。"""
        try:
            width = core.D435I_COLOR_WIDTH if core is not None else 640
            height = core.D435I_COLOR_HEIGHT if core is not None else 480
            fps = core.D435I_FPS if core is not None else 30
            self.camera_recorder = D435iRecorder(enable=True, color_width=width, color_height=height, fps=fps)
            ok = self.camera_recorder.start_camera()
            if ok:
                self.append_log(f"[D435i] RGB实时预览已启动：{width}x{height}@{fps}")
            else:
                self.append_log("[D435i][WARN] RGB实时预览启动失败，请检查相机连接和权限。")
        except Exception as exc:
            self.camera_recorder = None
            self.append_log(f"[D435i][WARN] RGB实时预览初始化异常：{exc}")

    def _refresh_camera_preview(self) -> None:
        """在 Qt 主线程中定时刷新 QLabel，避免后台线程直接操作 GUI。"""
        if self.camera_recorder is None:
            if hasattr(self, "camera_rgb_label"):
                self.camera_rgb_label.setText("D435i RGB相机未启动")
            return

        frame_bgr = self.camera_recorder.get_latest_preview()
        if frame_bgr is None:
            self.camera_rgb_label.setText("等待D435i RGB画面...")
            return

        try:
            h, w = frame_bgr.shape[:2]
            if hasattr(QImage, "Format_BGR888"):
                qimg = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format_BGR888).copy()
            else:
                frame_rgb = frame_bgr[:, :, ::-1].copy()
                qimg = QImage(frame_rgb.data, w, h, frame_rgb.strides[0], QImage.Format_RGB888).copy()

            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(
                self.camera_rgb_label.width(),
                self.camera_rgb_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.camera_rgb_label.setPixmap(pixmap)
        except Exception as exc:
            self.camera_rgb_label.setText(f"D435i RGB画面显示异常：{exc}")

    def browse_record_file(self) -> None:
        # 记录路径由核心文件自动放入新的 episode_xx/trajectory；这里选择的路径只用于取文件名。
        if core is not None:
            os.makedirs(core.DATA_DIR, exist_ok=True)
            start_dir = core.DATA_DIR
        else:
            start_dir = os.getcwd()
        path, _ = QFileDialog.getSaveFileName(self, "选择记录文件名（只使用文件名，目录自动放入episode）", start_dir, "CSV Files (*.csv)")
        if path:
            self.record_file_edit.setText(os.path.basename(path))

    def browse_replay_file(self) -> None:
        if core is not None:
            latest = core.find_latest_teach_record_file()
            if latest:
                start_dir = os.path.dirname(latest)
            else:
                os.makedirs(core.DATA_DIR, exist_ok=True)
                start_dir = core.DATA_DIR
        else:
            start_dir = os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "选择回放CSV文件", start_dir, "CSV Files (*.csv)")
        if path:
            self.replay_file_edit.setText(path)

    def _get_record_file(self) -> Optional[str]:
        text = self.record_file_edit.text().strip()
        return text or None

    def _get_replay_file(self) -> Optional[str]:
        text = self.replay_file_edit.text().strip()
        return text or None

    def start_or_switch(self, cmd: str) -> None:
        if core is None:
            QMessageBox.critical(self, "错误", f"无法导入 teleoperation_episode_data_collection.py：\n{CORE_IMPORT_ERROR}")
            return

        self.worker.start_or_switch(
            cmd=cmd,
            record_file=self._get_record_file(),
            replay_file=self._get_replay_file(),
            replay_source=str(self.replay_source_combo.currentData()),
            replay_speed=float(self.replay_speed_spin.value()),
            enable_tool_teleop=self.check_tool_teleop.isChecked(),
            enable_weak_bilateral=self.check_weak_bilateral.isChecked(),
            enable_error_feedback=self.check_error_feedback.isChecked(),
            enable_torque_feedback=self.check_torque_feedback.isChecked(),
            enable_tool_weak_feedback=self.check_tool_weak_feedback.isChecked(),
            enable_d435i_recording=self.check_d435i_recording.isChecked(),
        )

    def append_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")
        self.log_box.moveCursor(QTextCursor.End)

    def on_mode_changed(self, mode: str) -> None:
        self.mode_label.setText(f"当前模式：{mode}")

    def on_running_changed(self, running: bool) -> None:
        self.running_label.setText("运行状态：运行中" if running else "运行状态：未运行")
        # 运行时仍允许按模式按钮进行切换；只禁用部分配置项，避免中途改参数造成误解。
        for w in (
            self.record_file_edit,
            self.replay_file_edit,
            self.btn_browse_record,
            self.btn_browse_replay,
            self.replay_source_combo,
            self.replay_speed_spin,
            self.check_tool_teleop,
            self.check_weak_bilateral,
            self.check_error_feedback,
            self.check_torque_feedback,
            self.check_tool_weak_feedback,
            self.check_d435i_recording,
        ):
            w.setEnabled(not running)

    def on_worker_finished(self, ok: bool, msg: str) -> None:
        self.append_log(f"[DONE] {msg}，状态={'正常' if ok else '异常'}")

    def _refresh_labels(self) -> None:
        if core is not None:
            self.record_dir_label.setText("轨迹目录：data/episode_xx/trajectory（记录时自动创建）")
            self.video_dir_label.setText("视频目录：data/episode_xx/video（记录时自动创建）")

    def closeEvent(self, event) -> None:
        if self.worker.running:
            QMessageBox.warning(
                self,
                "提示",
                "当前遥操作/回放任务仍在运行。请先点击“0 安全退出/停止当前任务”，等待运行状态变为未运行后再关闭窗口。",
            )
            event.ignore()
            return
        try:
            if hasattr(self, "_camera_timer"):
                self._camera_timer.stop()
            if self.camera_recorder is not None:
                self.camera_recorder.close()
                self.camera_recorder = None
                self.worker.camera_recorder = None
        except Exception as exc:
            self.append_log(f"[D435i][WARN] 关闭RGB相机异常：{exc}")
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
