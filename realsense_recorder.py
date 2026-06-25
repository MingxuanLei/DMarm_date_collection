from __future__ import annotations

import csv
import json
import os
import threading
import time
from typing import Any, Optional

try:
    import pyrealsense2 as rs  # type: ignore
except Exception as exc:  # pragma: no cover - depends on runtime machine
    rs = None  # type: ignore
    RS_IMPORT_ERROR = exc
else:
    RS_IMPORT_ERROR = None

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - depends on runtime machine
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None  # type: ignore
    NP_IMPORT_ERROR = exc
else:
    NP_IMPORT_ERROR = None


class D435iRecorder:
    """
    D435i 深度相机录制模块。

    设计目标：
    1. 相机采集和视频编码在独立线程中运行，不阻塞 500Hz 遥操作控制循环；
    2. start_record() 接收遥操作 CSV 的 record_start_time，使视频帧时间戳和 CSV time_s 使用同一时间零点；
    3. pyrealsense2 / OpenCV / D435i 不可用时，只返回 False 并打印警告，不让遥操作主程序崩溃。

    默认输出文件：
        <record_stem>_color.mp4
        <record_stem>_depth_vis.mp4
        <record_stem>_camera_timestamps.csv
        <record_stem>_camera_meta.json
    """

    def __init__(
        self,
        enable: bool = True,
        color_width: int = 640,
        color_height: int = 480,
        depth_width: int = 640,
        depth_height: int = 480,
        fps: int = 30,
        save_color_video: bool = True,
        save_depth_vis_video: bool = True,
        align_depth_to_color: bool = True,
        video_codec: str = "mp4v",
        warmup_frames: int = 15,
        depth_vis_alpha: float = 0.03,
    ) -> None:
        self.enable = bool(enable)
        self.color_width = int(color_width)
        self.color_height = int(color_height)
        self.depth_width = int(depth_width)
        self.depth_height = int(depth_height)
        self.fps = int(fps)
        self.save_color_video = bool(save_color_video)
        self.save_depth_vis_video = bool(save_depth_vis_video)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.video_codec = str(video_codec)
        self.warmup_frames = int(max(0, warmup_frames))
        self.depth_vis_alpha = float(depth_vis_alpha)

        self.pipeline: Optional[Any] = None
        self.align: Optional[Any] = None
        self.profile: Optional[Any] = None
        self.depth_scale: Optional[float] = None
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
        self._depth_writer: Optional[Any] = None
        self._timestamp_file: Optional[Any] = None
        self._timestamp_writer: Optional[csv.DictWriter] = None

        self.color_video_path: Optional[str] = None
        self.depth_vis_video_path: Optional[str] = None
        self.timestamp_csv_path: Optional[str] = None
        self.meta_json_path: Optional[str] = None
        self.last_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.enable and rs is not None and cv2 is not None and np is not None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def _set_error(self, msg: str) -> None:
        self.last_error = msg
        print(f"[D435i][WARN] {msg}")

    def start_pipeline(self) -> bool:
        """启动 RealSense pipeline 和后台采集线程。可重复调用。"""
        if not self.enable:
            return False
        if rs is None:
            self._set_error(f"无法导入 pyrealsense2: {RS_IMPORT_ERROR}")
            return False
        if cv2 is None:
            self._set_error(f"无法导入 OpenCV cv2: {CV2_IMPORT_ERROR}")
            return False
        if np is None:
            self._set_error(f"无法导入 numpy: {NP_IMPORT_ERROR}")
            return False

        with self._lock:
            if self._pipeline_started:
                return True

        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.fps)
            profile = pipeline.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
            align = rs.align(rs.stream.color) if self.align_depth_to_color else None

            # 预热若干帧，让曝光和深度稳定。
            for _ in range(self.warmup_frames):
                try:
                    pipeline.wait_for_frames(1000)
                except Exception:
                    break

            meta = self._build_meta(profile, depth_scale)

            with self._lock:
                self.pipeline = pipeline
                self.profile = profile
                self.align = align
                self.depth_scale = depth_scale
                self.meta = meta
                self._pipeline_started = True
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._capture_loop, name="d435i_capture_thread", daemon=True)
                self._thread.start()

            print(
                f"[D435i] 相机已启动并预热：color={self.color_width}x{self.color_height}@{self.fps}, "
                f"depth={self.depth_width}x{self.depth_height}@{self.fps}, depth_scale={depth_scale:.8f}"
            )
            return True

        except Exception as exc:
            self._set_error(f"启动 D435i 失败: {exc}")
            try:
                pipeline.stop()  # type: ignore[name-defined]
            except Exception:
                pass
            return False

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

    def _build_meta(self, profile: Any, depth_scale: float) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "camera": "Intel RealSense D435i",
            "color_width": self.color_width,
            "color_height": self.color_height,
            "depth_width": self.depth_width,
            "depth_height": self.depth_height,
            "fps": self.fps,
            "align_depth_to_color": self.align_depth_to_color,
            "depth_scale": float(depth_scale),
            "video_codec": self.video_codec,
            "save_color_video": self.save_color_video,
            "save_depth_vis_video": self.save_depth_vis_video,
            "depth_vis_alpha": self.depth_vis_alpha,
        }
        try:
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            meta["color_intrinsics"] = self._intrinsics_to_dict(color_stream.get_intrinsics())
        except Exception as exc:
            meta["color_intrinsics_error"] = str(exc)
        try:
            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            meta["depth_intrinsics"] = self._intrinsics_to_dict(depth_stream.get_intrinsics())
        except Exception as exc:
            meta["depth_intrinsics_error"] = str(exc)
        return meta

    def start_record(self, video_dir: str, record_stem: str, record_start_time: float) -> bool:
        """
        开始保存本次记录的视频和相机时间戳。

        参数：
            video_dir: data/video 目录
            record_stem: 与 teach_record_xxx.csv 对应的文件名前缀
            record_start_time: 遥操作 CSV 使用的同一个 time.time() 零点
        """
        if not self.enable:
            return False
        if not self.start_pipeline():
            return False

        os.makedirs(video_dir, exist_ok=True)
        record_stem = os.path.splitext(os.path.basename(str(record_stem)))[0]

        color_video_path = os.path.join(video_dir, f"{record_stem}_color.mp4")
        depth_vis_video_path = os.path.join(video_dir, f"{record_stem}_depth_vis.mp4")
        timestamp_csv_path = os.path.join(video_dir, f"{record_stem}_camera_timestamps.csv")
        meta_json_path = os.path.join(video_dir, f"{record_stem}_camera_meta.json")

        with self._lock:
            self._close_record_outputs_locked()
            self._video_dir = video_dir
            self._record_stem = record_stem
            self._record_start_time = float(record_start_time)
            self._frame_idx = 0
            self.color_video_path = color_video_path
            self.depth_vis_video_path = depth_vis_video_path
            self.timestamp_csv_path = timestamp_csv_path
            self.meta_json_path = meta_json_path

            self._timestamp_file = open(timestamp_csv_path, "w", newline="", encoding="utf-8-sig")
            fieldnames = [
                "frame_idx",
                "camera_time_s",
                "system_time_s",
                "perf_counter_s",
                "color_realsense_timestamp_ms",
                "depth_realsense_timestamp_ms",
                "color_frame_number",
                "depth_frame_number",
                "color_video",
                "depth_vis_video",
            ]
            self._timestamp_writer = csv.DictWriter(self._timestamp_file, fieldnames=fieldnames)
            self._timestamp_writer.writeheader()

            meta = dict(self.meta)
            meta.update(
                {
                    "record_stem": record_stem,
                    "record_start_time_system_s": float(record_start_time),
                    "color_video_path": color_video_path,
                    "depth_vis_video_path": depth_vis_video_path,
                    "timestamp_csv_path": timestamp_csv_path,
                    "created_time_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }
            )
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            self._recording = True

        print(f"[D435i] 已开始同步录制视频，保存目录: {video_dir}")
        print(f"[D435i] RGB视频: {color_video_path}")
        print(f"[D435i] 深度伪彩色视频: {depth_vis_video_path}")
        print(f"[D435i] 帧时间戳: {timestamp_csv_path}")
        return True

    def stop_record(self) -> None:
        """停止当前视频文件保存，但不关闭 RealSense pipeline。"""
        with self._lock:
            if not self._recording and self._timestamp_file is None and self._color_writer is None and self._depth_writer is None:
                return
            self._recording = False
            frame_count = self._frame_idx
            color_path = self.color_video_path
            depth_path = self.depth_vis_video_path
            timestamp_path = self.timestamp_csv_path
            self._close_record_outputs_locked()

        print(f"[D435i] 已停止录制，共保存约 {frame_count} 帧")
        if color_path:
            print(f"[D435i] RGB视频已保存: {color_path}")
        if depth_path:
            print(f"[D435i] 深度伪彩色视频已保存: {depth_path}")
        if timestamp_path:
            print(f"[D435i] 帧时间戳已保存: {timestamp_path}")

    def close(self) -> None:
        """停止录制、退出采集线程并释放相机。"""
        self.stop_record()
        self._stop_event.set()

        th = None
        with self._lock:
            th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)

        with self._lock:
            pipeline = self.pipeline
            self.pipeline = None
            self.profile = None
            self.align = None
            self._thread = None
            self._pipeline_started = False

        if pipeline is not None:
            try:
                pipeline.stop()
                print("[D435i] 相机 pipeline 已释放")
            except Exception as exc:
                print(f"[D435i][WARN] 释放相机 pipeline 异常: {exc}")

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

        if self._depth_writer is not None:
            try:
                self._depth_writer.release()
            except Exception:
                pass
        self._depth_writer = None

    def _ensure_video_writers_locked(self, color_image: Any, depth_vis_image: Any) -> None:
        if cv2 is None:
            return
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)

        if self.save_color_video and self._color_writer is None and self.color_video_path:
            h, w = color_image.shape[:2]
            self._color_writer = cv2.VideoWriter(self.color_video_path, fourcc, float(self.fps), (int(w), int(h)), True)
            if not self._color_writer.isOpened():
                print(f"[D435i][WARN] RGB VideoWriter 打开失败: {self.color_video_path}")
                try:
                    self._color_writer.release()
                except Exception:
                    pass
                self._color_writer = None

        if self.save_depth_vis_video and self._depth_writer is None and self.depth_vis_video_path and depth_vis_image is not None:
            h, w = depth_vis_image.shape[:2]
            self._depth_writer = cv2.VideoWriter(self.depth_vis_video_path, fourcc, float(self.fps), (int(w), int(h)), True)
            if not self._depth_writer.isOpened():
                print(f"[D435i][WARN] 深度 VideoWriter 打开失败: {self.depth_vis_video_path}")
                try:
                    self._depth_writer.release()
                except Exception:
                    pass
                self._depth_writer = None

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                pipeline = self.pipeline
                align = self.align
            if pipeline is None:
                time.sleep(0.02)
                continue

            try:
                frames = pipeline.wait_for_frames(1000)
                if align is not None:
                    frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # bgr8 输出可直接写入 OpenCV VideoWriter。
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_vis = None
                if cv2 is not None and self.save_depth_vis_video:
                    depth_8u = cv2.convertScaleAbs(depth_image, alpha=self.depth_vis_alpha)
                    depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

                system_time_s = time.time()
                perf_counter_s = time.perf_counter()

                with self._lock:
                    if not self._recording or self._record_start_time is None:
                        continue

                    self._ensure_video_writers_locked(color_image, depth_vis)

                    if self._color_writer is not None:
                        self._color_writer.write(color_image)
                    if self._depth_writer is not None and depth_vis is not None:
                        self._depth_writer.write(depth_vis)

                    if self._timestamp_writer is not None:
                        self._timestamp_writer.writerow(
                            {
                                "frame_idx": int(self._frame_idx),
                                "camera_time_s": f"{system_time_s - self._record_start_time:.8f}",
                                "system_time_s": f"{system_time_s:.8f}",
                                "perf_counter_s": f"{perf_counter_s:.8f}",
                                "color_realsense_timestamp_ms": f"{float(color_frame.get_timestamp()):.4f}",
                                "depth_realsense_timestamp_ms": f"{float(depth_frame.get_timestamp()):.4f}",
                                "color_frame_number": int(color_frame.get_frame_number()),
                                "depth_frame_number": int(depth_frame.get_frame_number()),
                                "color_video": os.path.basename(self.color_video_path or ""),
                                "depth_vis_video": os.path.basename(self.depth_vis_video_path or ""),
                            }
                        )
                        if self._timestamp_file is not None and self._frame_idx % max(1, self.fps) == 0:
                            self._timestamp_file.flush()

                    self._frame_idx += 1

            except Exception as exc:
                # 避免偶发相机超时导致整个遥操作线程崩溃。
                print(f"[D435i][WARN] 采集线程异常: {exc}")
                time.sleep(0.05)
