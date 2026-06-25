from __future__ import annotations

import csv
import json
import os
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs


class D435iRecorder:
    """
    Intel RealSense D435i RGB 实时预览与同步录制模块。

    设计方式：
    1. 相机采集在线程中运行，不阻塞遥操作控制循环；
    2. GUI 可通过 get_latest_preview() 获取最新 RGB 画面用于 QLabel 显示；
    3. 进入记录模式时调用 start_record()，只保存 RGB 视频和帧时间戳；
    4. 退出记录模式时调用 stop_record()，但相机预览线程继续运行。

    默认输出文件：
        <record_stem>_color.mp4
        <record_stem>_camera_timestamps.csv
        <record_stem>_camera_meta.json
    """

    def __init__(
        self,
        enable: bool = True,
        color_width: int = 640,
        color_height: int = 480,
        fps: int = 30,
        video_codec: str = "mp4v",
        warmup_frames: int = 15,
    ) -> None:
        self.enable = bool(enable)
        self.color_width = int(color_width)
        self.color_height = int(color_height)
        self.fps = int(fps)
        self.video_codec = str(video_codec)
        self.warmup_frames = int(max(0, warmup_frames))

        self.pipeline: Optional[Any] = None
        self.profile: Optional[Any] = None
        self.meta: dict[str, Any] = {}

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipeline_started = False
        self._recording = False

        self._record_start_time: Optional[float] = None
        self._video_dir: Optional[str] = None
        self._record_stem: Optional[str] = None
        self._frame_idx = 0

        self._color_writer: Optional[Any] = None
        self._timestamp_file: Optional[Any] = None
        self._timestamp_writer: Optional[csv.DictWriter] = None

        self.color_video_path: Optional[str] = None
        self.timestamp_csv_path: Optional[str] = None
        self.meta_json_path: Optional[str] = None
        self.last_error: Optional[str] = None

        # 最新预览帧，BGR uint8，便于 OpenCV 写视频，也便于 Qt 使用 Format_BGR888 显示。
        self._latest_color_bgr: Optional[np.ndarray] = None
        self._latest_system_time_s: Optional[float] = None
        self._latest_frame_number: Optional[int] = None
        self._latest_realsense_timestamp_ms: Optional[float] = None

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._pipeline_started

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def _set_error(self, msg: str) -> None:
        self.last_error = msg
        print(f"[D435i][WARN] {msg}")

    def start_camera(self) -> bool:
        """启动 RealSense RGB pipeline 和后台采集线程。可重复调用。"""
        if not self.enable:
            return False

        with self._lock:
            if self._pipeline_started:
                return True

        pipeline = None
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)
            profile = pipeline.start(config)

            for _ in range(self.warmup_frames):
                try:
                    pipeline.wait_for_frames(1000)
                except Exception:
                    break

            meta = self._build_meta(profile)

            with self._lock:
                self.pipeline = pipeline
                self.profile = profile
                self.meta = meta
                self._pipeline_started = True
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._capture_loop, name="d435i_rgb_capture_thread", daemon=True)
                self._thread.start()

            print(f"[D435i] RGB相机已启动并预热：color={self.color_width}x{self.color_height}@{self.fps}")
            return True

        except Exception as exc:
            self._set_error(f"启动 D435i RGB相机失败: {exc}")
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            return False

    # 兼容旧代码中 start_pipeline() 的调用。
    def start_pipeline(self) -> bool:
        return self.start_camera()

    def _intrinsics_to_dict(self, intr: Any) -> dict[str, Any]:
        try:
            coeffs = [float(x) for x in intr.coeffs]
        except Exception:
            coeffs = []
        return {
            "width": int(getattr(intr, "width", 0)),
            "height": int(getattr(intr, "height", 0)),
            "ppx": float(getattr(intr, "ppx", 0.0)),
            "ppy": float(getattr(intr, "ppy", 0.0)),
            "fx": float(getattr(intr, "fx", 0.0)),
            "fy": float(getattr(intr, "fy", 0.0)),
            "model": str(getattr(intr, "model", "")),
            "coeffs": coeffs,
        }

    def _build_meta(self, profile: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "camera": "Intel RealSense D435i",
            "stream": "color_only",
            "color_width": self.color_width,
            "color_height": self.color_height,
            "fps": self.fps,
            "video_codec": self.video_codec,
            "save_color_video": True,
        }
        try:
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            meta["color_intrinsics"] = self._intrinsics_to_dict(color_stream.get_intrinsics())
        except Exception as exc:
            meta["color_intrinsics_error"] = str(exc)
        return meta

    def start_record(self, video_dir: str, record_stem: str, record_start_time: float) -> bool:
        """
        开始保存本次记录的 RGB 视频和相机帧时间戳。

        参数：
            video_dir: data/video 目录
            record_stem: 与 teach_record_xxx.csv 对应的文件名前缀
            record_start_time: 遥操作 CSV 使用的同一个 time.time() 零点
        """
        if not self.enable:
            return False
        if not self.start_camera():
            return False

        os.makedirs(video_dir, exist_ok=True)
        record_stem = os.path.splitext(os.path.basename(str(record_stem)))[0]

        color_video_path = os.path.join(video_dir, f"{record_stem}_color.mp4")
        timestamp_csv_path = os.path.join(video_dir, f"{record_stem}_camera_timestamps.csv")
        meta_json_path = os.path.join(video_dir, f"{record_stem}_camera_meta.json")

        with self._lock:
            self._close_record_outputs_locked()
            self._video_dir = video_dir
            self._record_stem = record_stem
            self._record_start_time = float(record_start_time)
            self._frame_idx = 0
            self.color_video_path = color_video_path
            self.timestamp_csv_path = timestamp_csv_path
            self.meta_json_path = meta_json_path

            self._timestamp_file = open(timestamp_csv_path, "w", newline="", encoding="utf-8-sig")
            fieldnames = [
                "frame_idx",
                "camera_time_s",
                "system_time_s",
                "perf_counter_s",
                "color_realsense_timestamp_ms",
                "color_frame_number",
                "color_video",
            ]
            self._timestamp_writer = csv.DictWriter(self._timestamp_file, fieldnames=fieldnames)
            self._timestamp_writer.writeheader()

            meta = dict(self.meta)
            meta.update(
                {
                    "record_stem": record_stem,
                    "record_start_time_system_s": float(record_start_time),
                    "color_video_path": color_video_path,
                    "timestamp_csv_path": timestamp_csv_path,
                    "created_time_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }
            )
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            self._recording = True

        print(f"[D435i] 已开始同步录制RGB视频，保存目录: {video_dir}")
        print(f"[D435i] RGB视频: {color_video_path}")
        print(f"[D435i] 帧时间戳: {timestamp_csv_path}")
        return True

    def stop_record(self) -> None:
        """停止当前 RGB 视频保存，但不关闭 RealSense pipeline。"""
        with self._lock:
            if not self._recording and self._timestamp_file is None and self._color_writer is None:
                return
            self._recording = False
            frame_count = self._frame_idx
            color_path = self.color_video_path
            timestamp_path = self.timestamp_csv_path
            self._close_record_outputs_locked()

        print(f"[D435i] 已停止RGB录制，共保存约 {frame_count} 帧")
        if color_path:
            print(f"[D435i] RGB视频已保存: {color_path}")
        if timestamp_path:
            print(f"[D435i] 帧时间戳已保存: {timestamp_path}")

    def get_latest_preview(self) -> Optional[np.ndarray]:
        """返回最新 RGB 预览帧，格式为 BGR uint8。返回的是 copy，GUI 可安全读取。"""
        with self._lock:
            if self._latest_color_bgr is None:
                return None
            return self._latest_color_bgr.copy()

    def get_latest_preview_info(self) -> dict[str, Any]:
        """返回最新预览帧的辅助时间信息。"""
        with self._lock:
            return {
                "system_time_s": self._latest_system_time_s,
                "frame_number": self._latest_frame_number,
                "realsense_timestamp_ms": self._latest_realsense_timestamp_ms,
            }

    def close(self) -> None:
        """停止录制、退出采集线程并释放相机。"""
        self.stop_record()
        self._stop_event.set()

        with self._lock:
            th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)

        with self._lock:
            pipeline = self.pipeline
            self.pipeline = None
            self.profile = None
            self._thread = None
            self._pipeline_started = False
            self._latest_color_bgr = None

        if pipeline is not None:
            try:
                pipeline.stop()
                print("[D435i] RGB相机 pipeline 已释放")
            except Exception as exc:
                print(f"[D435i][WARN] 释放RGB相机 pipeline 异常: {exc}")

    def _close_record_outputs_locked(self) -> None:
        if self._timestamp_file is not None:
            try:
                self._timestamp_file.flush()
                self._timestamp_file.close()
            except Exception:
                pass
        self._timestamp_file = None
        self._timestamp_writer = None

        if self._color_writer is not None:
            try:
                self._color_writer.release()
            except Exception:
                pass
        self._color_writer = None

    def _ensure_video_writer_locked(self, color_image: np.ndarray) -> None:
        if self._color_writer is not None or not self.color_video_path:
            return
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        h, w = color_image.shape[:2]
        self._color_writer = cv2.VideoWriter(self.color_video_path, fourcc, float(self.fps), (int(w), int(h)), True)
        if not self._color_writer.isOpened():
            print(f"[D435i][WARN] RGB VideoWriter 打开失败: {self.color_video_path}")
            try:
                self._color_writer.release()
            except Exception:
                pass
            self._color_writer = None

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                pipeline = self.pipeline
            if pipeline is None:
                time.sleep(0.02)
                continue

            try:
                frames = pipeline.wait_for_frames(1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                system_time_s = time.time()
                perf_counter_s = time.perf_counter()
                frame_number = int(color_frame.get_frame_number())
                rs_timestamp_ms = float(color_frame.get_timestamp())

                with self._lock:
                    self._latest_color_bgr = color_image.copy()
                    self._latest_system_time_s = system_time_s
                    self._latest_frame_number = frame_number
                    self._latest_realsense_timestamp_ms = rs_timestamp_ms

                    if not self._recording or self._record_start_time is None:
                        continue

                    self._ensure_video_writer_locked(color_image)

                    if self._color_writer is not None:
                        self._color_writer.write(color_image)

                    if self._timestamp_writer is not None:
                        self._timestamp_writer.writerow(
                            {
                                "frame_idx": int(self._frame_idx),
                                "camera_time_s": f"{system_time_s - self._record_start_time:.8f}",
                                "system_time_s": f"{system_time_s:.8f}",
                                "perf_counter_s": f"{perf_counter_s:.8f}",
                                "color_realsense_timestamp_ms": f"{rs_timestamp_ms:.4f}",
                                "color_frame_number": frame_number,
                                "color_video": os.path.basename(self.color_video_path or ""),
                            }
                        )
                        if self._timestamp_file is not None and self._frame_idx % max(1, self.fps) == 0:
                            self._timestamp_file.flush()

                    self._frame_idx += 1

            except Exception as exc:
                print(f"[D435i][WARN] RGB采集线程异常: {exc}")
                time.sleep(0.05)
