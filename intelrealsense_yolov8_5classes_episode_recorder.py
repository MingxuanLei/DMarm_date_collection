from __future__ import annotations

import csv
import copy
import json
import math
import os
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs


# 只检测以下 5 个类别：person, mouse, keyboard, cell phone, cup
CLASSES = [0, 64, 66, 67, 41]


class D435iYoloRecorder:
    """
    Intel RealSense D435i RGB-D + YOLOv8 实时检测、测距、预览与 Episode 录制模块。

    线程结构：
    1. Camera thread：持续采集 RGB + Depth，并将 Depth 对齐到 RGB；
    2. YOLO thread：只处理最新帧，检测物体并估计深度/XYZ/空间距离；
    3. GUI thread：通过 get_latest_preview() 读取已经画框的最新图像。

    记录模式默认输出：
        <video_dir>/<record_stem>_color.mp4
        <video_dir>/<record_stem>_camera_timestamps.csv
        <video_dir>/<record_stem>_camera_meta.json
        <video_dir>/<record_stem>_object_detections.csv

    说明：
    - MP4 保存原始 RGB 图像，不把检测框烧录进原始视频；
    - object_detections.csv 保存 YOLO 检测与距离/XYZ，时间零点与机械臂 CSV 共用；
    - YOLO 模型加载失败时，相机预览和录像仍然继续工作。
    """

    def __init__(
        self,
        enable: bool = True,
        color_width: int = 640,
        color_height: int = 480,
        depth_width: Optional[int] = None,
        depth_height: Optional[int] = None,
        fps: int = 30,
        video_codec: str = "mp4v",
        warmup_frames: int = 15,
        *,
        enable_yolo: bool = True,
        yolo_model_path: str = "yolov8n.pt",
        yolo_conf: float = 0.50,
        yolo_iou: float = 0.45,
        yolo_imgsz: int = 640,
        yolo_device: Optional[str] = None,
        yolo_max_hz: float = 15.0,
        show_xyz: bool = True,
        depth_roi_ratio: float = 0.40,
        min_depth_m: float = 0.10,
        max_depth_m: float = 5.0,
    ) -> None:
        self.enable = bool(enable)
        self.color_width = int(color_width)
        self.color_height = int(color_height)
        self.depth_width = int(depth_width if depth_width is not None else color_width)
        self.depth_height = int(depth_height if depth_height is not None else color_height)
        self.fps = int(fps)
        self.video_codec = str(video_codec)
        self.warmup_frames = int(max(0, warmup_frames))

        self.pipeline: Optional[Any] = None
        self.profile: Optional[Any] = None
        self.align: Optional[Any] = None
        self.color_intrinsics: Optional[Any] = None
        self.depth_scale: float = 0.001
        self.meta: dict[str, Any] = {}

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._yolo_thread: Optional[threading.Thread] = None
        self._pipeline_started = False
        self._recording = False

        self._record_start_time: Optional[float] = None
        self._video_dir: Optional[str] = None
        self._record_stem: Optional[str] = None
        self._frame_idx = 0

        self._color_writer: Optional[Any] = None
        self._timestamp_file: Optional[Any] = None
        self._timestamp_writer: Optional[csv.DictWriter] = None
        self._detection_file: Optional[Any] = None
        self._detection_writer: Optional[csv.DictWriter] = None

        self.color_video_path: Optional[str] = None
        self.timestamp_csv_path: Optional[str] = None
        self.meta_json_path: Optional[str] = None
        self.detection_csv_path: Optional[str] = None
        self.last_error: Optional[str] = None

        # Camera thread 产出的最新 RGB-D 数据。
        self._latest_color_bgr: Optional[np.ndarray] = None
        self._latest_depth_m: Optional[np.ndarray] = None
        self._latest_system_time_s: Optional[float] = None
        self._latest_perf_counter_s: Optional[float] = None
        self._latest_frame_number: Optional[int] = None
        self._latest_realsense_timestamp_ms: Optional[float] = None
        self._latest_frame_seq = 0

        # YOLO thread 产出的最新结果。
        self._latest_annotated_bgr: Optional[np.ndarray] = None
        self._latest_detections: list[dict[str, Any]] = []
        self._latest_detection_frame_number: Optional[int] = None
        self._latest_inference_ms: Optional[float] = None
        self._yolo_status = "未加载"

        # YOLO runtime config。
        self._enable_yolo = bool(enable_yolo)
        self._yolo_model_path = str(yolo_model_path).strip() or "yolov8n.pt"
        self._yolo_conf = float(np.clip(yolo_conf, 0.01, 1.0))
        self._yolo_iou = float(np.clip(yolo_iou, 0.01, 1.0))
        self._yolo_imgsz = int(max(64, yolo_imgsz))
        self._yolo_device = yolo_device
        self._yolo_max_hz = float(max(0.1, yolo_max_hz))
        self._show_xyz = bool(show_xyz)
        self._depth_roi_ratio = float(np.clip(depth_roi_ratio, 0.05, 0.95))
        self._min_depth_m = float(max(0.0, min_depth_m))
        self._max_depth_m = float(max(self._min_depth_m + 1e-6, max_depth_m))
        self._model_generation = 0

    # ------------------------------------------------------------------
    # Properties and runtime settings
    # ------------------------------------------------------------------
    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._pipeline_started

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def yolo_enabled(self) -> bool:
        with self._lock:
            return self._enable_yolo

    def set_yolo_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enable_yolo = bool(enabled)
            if not self._enable_yolo:
                self._latest_annotated_bgr = None
                self._latest_detections = []
                self._yolo_status = "已关闭"
            else:
                self._yolo_status = "等待模型"
        print(f"[YOLO] 实时检测：{'开启' if enabled else '关闭'}")

    def set_yolo_confidence(self, conf: float) -> None:
        with self._lock:
            self._yolo_conf = float(np.clip(conf, 0.01, 1.0))
        print(f"[YOLO] 置信度阈值已设置为 {self._yolo_conf:.2f}")

    def set_show_xyz(self, enabled: bool) -> None:
        with self._lock:
            self._show_xyz = bool(enabled)

    def set_yolo_model(self, model_path: str) -> None:
        model_path = str(model_path).strip()
        if not model_path:
            raise ValueError("YOLO 模型路径不能为空")
        with self._lock:
            self._yolo_model_path = model_path
            self._model_generation += 1
            self._yolo_status = "等待重新加载模型"
        print(f"[YOLO] 已请求加载模型: {model_path}")

    def get_latest_detections(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._latest_detections)

    def get_yolo_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enable_yolo,
                "status": self._yolo_status,
                "model_path": self._yolo_model_path,
                "confidence": self._yolo_conf,
                "detection_count": len(self._latest_detections),
                "inference_ms": self._latest_inference_ms,
                "frame_number": self._latest_detection_frame_number,
            }

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    def _set_error(self, msg: str) -> None:
        self.last_error = msg
        print(f"[D435i][WARN] {msg}")

    def start_camera(self) -> bool:
        """启动 RGB-D pipeline、对齐器、相机线程和 YOLO 推理线程。可重复调用。"""
        if not self.enable:
            return False

        with self._lock:
            if self._pipeline_started:
                return True

        pipeline = None
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.depth,
                self.depth_width,
                self.depth_height,
                rs.format.z16,
                self.fps,
            )
            config.enable_stream(
                rs.stream.color,
                self.color_width,
                self.color_height,
                rs.format.bgr8,
                self.fps,
            )
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)

            for _ in range(self.warmup_frames):
                try:
                    frames = pipeline.wait_for_frames(1000)
                    align.process(frames)
                except Exception:
                    break

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intrinsics = color_stream.get_intrinsics()

            depth_scale = 0.001
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = float(depth_sensor.get_depth_scale())
            except Exception as exc:
                print(f"[D435i][WARN] 读取 depth_scale 失败，使用默认 0.001 m/unit: {exc}")

            meta = self._build_meta(profile, depth_scale)

            with self._lock:
                self.pipeline = pipeline
                self.profile = profile
                self.align = align
                self.color_intrinsics = color_intrinsics
                self.depth_scale = depth_scale
                self.meta = meta
                self._pipeline_started = True
                self._stop_event.clear()
                self._capture_thread = threading.Thread(
                    target=self._capture_loop,
                    name="d435i_rgbd_capture_thread",
                    daemon=True,
                )
                self._yolo_thread = threading.Thread(
                    target=self._yolo_loop,
                    name="d435i_yolov8_inference_thread",
                    daemon=True,
                )
                self._capture_thread.start()
                self._yolo_thread.start()

            print(
                f"[D435i] RGB-D 相机已启动并预热："
                f"color={self.color_width}x{self.color_height}@{self.fps}, "
                f"depth={self.depth_width}x{self.depth_height}@{self.fps}"
            )
            return True

        except Exception as exc:
            self._set_error(f"启动 D435i RGB-D 相机失败: {exc}")
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            return False

    # 兼容旧代码。
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

    def _build_meta(self, profile: Any, depth_scale: float) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "camera": "Intel RealSense D435i",
            "stream": "color+depth_aligned_to_color",
            "color_width": self.color_width,
            "color_height": self.color_height,
            "depth_width": self.depth_width,
            "depth_height": self.depth_height,
            "fps": self.fps,
            "depth_scale_m_per_unit": depth_scale,
            "video_codec": self.video_codec,
            "save_color_video": True,
            "save_depth_video": False,
            "yolo_model_path": self._yolo_model_path,
            "yolo_confidence": self._yolo_conf,
            "yolo_iou": self._yolo_iou,
            "yolo_imgsz": self._yolo_imgsz,
            "depth_estimator": "median of valid depth in center ROI of bbox",
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

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------
    def start_record(self, video_dir: str, record_stem: str, record_start_time: float) -> bool:
        """开始保存 RGB 视频、相机帧时间戳和 YOLO 检测结果。"""
        if not self.enable:
            return False
        if not self.start_camera():
            return False

        os.makedirs(video_dir, exist_ok=True)
        record_stem = os.path.splitext(os.path.basename(str(record_stem)))[0]

        color_video_path = os.path.join(video_dir, f"{record_stem}_color.mp4")
        timestamp_csv_path = os.path.join(video_dir, f"{record_stem}_camera_timestamps.csv")
        meta_json_path = os.path.join(video_dir, f"{record_stem}_camera_meta.json")
        detection_csv_path = os.path.join(video_dir, f"{record_stem}_object_detections.csv")

        with self._lock:
            self._close_record_outputs_locked()
            self._video_dir = video_dir
            self._record_stem = record_stem
            self._record_start_time = float(record_start_time)
            self._frame_idx = 0
            self.color_video_path = color_video_path
            self.timestamp_csv_path = timestamp_csv_path
            self.meta_json_path = meta_json_path
            self.detection_csv_path = detection_csv_path

            self._timestamp_file = open(timestamp_csv_path, "w", newline="", encoding="utf-8-sig")
            timestamp_fields = [
                "frame_idx",
                "camera_time_s",
                "system_time_s",
                "perf_counter_s",
                "color_realsense_timestamp_ms",
                "color_frame_number",
                "color_video",
            ]
            self._timestamp_writer = csv.DictWriter(self._timestamp_file, fieldnames=timestamp_fields)
            self._timestamp_writer.writeheader()

            self._detection_file = open(detection_csv_path, "w", newline="", encoding="utf-8-sig")
            detection_fields = [
                "camera_time_s",
                "system_time_s",
                "detection_time_s",
                "color_frame_number",
                "detection_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_u",
                "center_v",
                "depth_z_m",
                "x_m",
                "y_m",
                "z_m",
                "range_m",
                "inference_ms",
            ]
            self._detection_writer = csv.DictWriter(self._detection_file, fieldnames=detection_fields)
            self._detection_writer.writeheader()

            meta = dict(self.meta)
            meta.update(
                {
                    "record_stem": record_stem,
                    "record_start_time_system_s": float(record_start_time),
                    "video_dir": os.path.abspath(video_dir),
                    "color_video_path": color_video_path,
                    "timestamp_csv_path": timestamp_csv_path,
                    "detection_csv_path": detection_csv_path,
                    "created_time_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }
            )
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            self._recording = True

        print(f"[D435i] 已开始同步录制 RGB 视频和 YOLO 检测结果，保存目录: {video_dir}")
        print(f"[D435i] RGB视频: {color_video_path}")
        print(f"[D435i] 帧时间戳: {timestamp_csv_path}")
        print(f"[YOLO] 检测结果: {detection_csv_path}")
        return True

    def stop_record(self) -> None:
        """停止文件保存，但保持 RealSense 与 YOLO 预览线程运行。"""
        with self._lock:
            has_outputs = any(
                item is not None
                for item in (self._timestamp_file, self._color_writer, self._detection_file)
            )
            if not self._recording and not has_outputs:
                return
            self._recording = False
            frame_count = self._frame_idx
            color_path = self.color_video_path
            timestamp_path = self.timestamp_csv_path
            detection_path = self.detection_csv_path
            self._close_record_outputs_locked()

        print(f"[D435i] 已停止 RGB/检测结果录制，共保存约 {frame_count} 个相机帧时间戳")
        if color_path:
            print(f"[D435i] RGB视频已保存: {color_path}")
        if timestamp_path:
            print(f"[D435i] 帧时间戳已保存: {timestamp_path}")
        if detection_path:
            print(f"[YOLO] 检测结果已保存: {detection_path}")

    # ------------------------------------------------------------------
    # Preview getters
    # ------------------------------------------------------------------
    def get_latest_preview(self) -> Optional[np.ndarray]:
        """YOLO开启时优先返回标注帧；否则返回原始 RGB。"""
        with self._lock:
            if self._enable_yolo and self._latest_annotated_bgr is not None:
                return self._latest_annotated_bgr.copy()
            if self._latest_color_bgr is None:
                return None
            return self._latest_color_bgr.copy()

    def get_latest_preview_info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "system_time_s": self._latest_system_time_s,
                "frame_number": self._latest_frame_number,
                "realsense_timestamp_ms": self._latest_realsense_timestamp_ms,
                "detection_frame_number": self._latest_detection_frame_number,
                "inference_ms": self._latest_inference_ms,
            }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def close(self) -> None:
        self.stop_record()
        self._stop_event.set()

        with self._lock:
            capture_th = self._capture_thread
            yolo_th = self._yolo_thread

        for th in (capture_th, yolo_th):
            if th is not None and th.is_alive():
                th.join(timeout=2.0)

        with self._lock:
            pipeline = self.pipeline
            self.pipeline = None
            self.profile = None
            self.align = None
            self.color_intrinsics = None
            self._capture_thread = None
            self._yolo_thread = None
            self._pipeline_started = False
            self._latest_color_bgr = None
            self._latest_depth_m = None
            self._latest_annotated_bgr = None
            self._latest_detections = []

        if pipeline is not None:
            try:
                pipeline.stop()
                print("[D435i] RGB-D 相机 pipeline 已释放")
            except Exception as exc:
                print(f"[D435i][WARN] 释放 RGB-D pipeline 异常: {exc}")

    # ------------------------------------------------------------------
    # Internal file helpers
    # ------------------------------------------------------------------
    def _close_record_outputs_locked(self) -> None:
        for f in (self._timestamp_file, self._detection_file):
            if f is not None:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
        self._timestamp_file = None
        self._timestamp_writer = None
        self._detection_file = None
        self._detection_writer = None

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
        self._color_writer = cv2.VideoWriter(
            self.color_video_path,
            fourcc,
            float(self.fps),
            (int(w), int(h)),
            True,
        )
        if not self._color_writer.isOpened():
            print(f"[D435i][WARN] RGB VideoWriter 打开失败: {self.color_video_path}")
            try:
                self._color_writer.release()
            except Exception:
                pass
            self._color_writer = None

    # ------------------------------------------------------------------
    # Camera thread
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                pipeline = self.pipeline
                align = self.align
                depth_scale = self.depth_scale

            if pipeline is None or align is None:
                time.sleep(0.02)
                continue

            try:
                frames = pipeline.wait_for_frames(1000)
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_m = depth_raw.astype(np.float32) * float(depth_scale)

                system_time_s = time.time()
                perf_counter_s = time.perf_counter()
                frame_number = int(color_frame.get_frame_number())
                rs_timestamp_ms = float(color_frame.get_timestamp())

                with self._lock:
                    self._latest_color_bgr = color_image.copy()
                    self._latest_depth_m = depth_m.copy()
                    self._latest_system_time_s = system_time_s
                    self._latest_perf_counter_s = perf_counter_s
                    self._latest_frame_number = frame_number
                    self._latest_realsense_timestamp_ms = rs_timestamp_ms
                    self._latest_frame_seq += 1

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
                print(f"[D435i][WARN] RGB-D 采集线程异常: {exc}")
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # YOLO thread and depth/XYZ estimation
    # ------------------------------------------------------------------
    def _load_yolo_model(self, model_path: str):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "未安装 ultralytics。请执行: pip install ultralytics"
            ) from exc
        return YOLO(model_path)

    def _estimate_depth_m(self, depth_m: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[float]:
        h, w = depth_m.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(1, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(1, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1
        roi_ratio = self._depth_roi_ratio
        roi_w = max(1, int(round(bw * roi_ratio)))
        roi_h = max(1, int(round(bh * roi_ratio)))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx1 = max(x1, cx - roi_w // 2)
        rx2 = min(x2, rx1 + roi_w)
        ry1 = max(y1, cy - roi_h // 2)
        ry2 = min(y2, ry1 + roi_h)

        roi = depth_m[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        valid = roi[
            np.isfinite(roi)
            & (roi >= self._min_depth_m)
            & (roi <= self._max_depth_m)
        ]
        if valid.size == 0:
            return None

        # 中位数比单点深度更抗空洞和少量背景像素。
        return float(np.median(valid))

    def _deproject_xyz(self, u: int, v: int, depth_z_m: float) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        with self._lock:
            intr = self.color_intrinsics
        if intr is None or not math.isfinite(depth_z_m) or depth_z_m <= 0.0:
            return None, None, None, None
        try:
            point = rs.rs2_deproject_pixel_to_point(
                intr,
                [float(u), float(v)],
                float(depth_z_m),
            )
            x_m, y_m, z_m = [float(x) for x in point]
            range_m = math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
            return x_m, y_m, z_m, range_m
        except Exception:
            return None, None, None, None

    @staticmethod
    def _draw_text_lines(
        image: np.ndarray,
        x: int,
        y: int,
        lines: list[str],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1
        line_h = 20
        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        box_w = max((s[0] for s in sizes), default=0) + 10
        box_h = line_h * len(lines) + 6
        x = max(0, int(x))
        y_top = max(0, int(y) - box_h)
        x2 = min(image.shape[1] - 1, x + box_w)
        y2 = min(image.shape[0] - 1, y_top + box_h)
        cv2.rectangle(image, (x, y_top), (x2, y2), (0, 0, 0), -1)
        for idx, line in enumerate(lines):
            ty = y_top + 18 + idx * line_h
            cv2.putText(image, line, (x + 5, ty), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def _infer_and_annotate(
        self,
        model: Any,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        *,
        frame_number: int,
        frame_system_time_s: float,
        conf: float,
        iou: float,
        imgsz: int,
        device: Optional[str],
        show_xyz: bool,
    ) -> tuple[np.ndarray, list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        predict_kwargs: dict[str, Any] = {
            "source": color_bgr,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "classes": CLASSES,
            "verbose": False,
        }
        if device not in (None, "", "auto"):
            predict_kwargs["device"] = device

        results = model.predict(**predict_kwargs)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        annotated = color_bgr.copy()
        detections: list[dict[str, Any]] = []

        if not results:
            return annotated, detections, inference_ms

        result = results[0]
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return annotated, detections, inference_ms

        for det_idx, box in enumerate(boxes):
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                cls_id = int(box.cls[0].detach().cpu().item())
                confidence = float(box.conf[0].detach().cpu().item())
            except Exception:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1 = max(0, min(color_bgr.shape[1] - 1, x1))
            x2 = max(0, min(color_bgr.shape[1] - 1, x2))
            y1 = max(0, min(color_bgr.shape[0] - 1, y1))
            y2 = max(0, min(color_bgr.shape[0] - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = str(names.get(cls_id, cls_id) if isinstance(names, dict) else names[cls_id])
            center_u = int((x1 + x2) // 2)
            center_v = int((y1 + y2) // 2)
            depth_z_m = self._estimate_depth_m(depth_m, (x1, y1, x2, y2))

            x_m = y_m = z_m = range_m = None
            if depth_z_m is not None:
                x_m, y_m, z_m, range_m = self._deproject_xyz(center_u, center_v, depth_z_m)

            det = {
                "camera_time_s": None,
                "system_time_s": float(frame_system_time_s),
                "detection_time_s": float(time.time()),
                "color_frame_number": int(frame_number),
                "detection_index": int(det_idx),
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_u": center_u,
                "center_v": center_v,
                "depth_z_m": depth_z_m,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m,
                "range_m": range_m,
                "inference_ms": inference_ms,
            }
            detections.append(det)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (center_u, center_v), 3, (0, 0, 255), -1)
            lines = [f"{class_name} {confidence:.2f}"]
            if depth_z_m is not None:
                lines.append(f"Z={depth_z_m:.3f}m")
            else:
                lines.append("Z=N/A")
            if show_xyz and x_m is not None and y_m is not None and z_m is not None:
                lines.append(f"XYZ=({x_m:.2f},{y_m:.2f},{z_m:.2f})m")
                if range_m is not None:
                    lines.append(f"Range={range_m:.3f}m")
            self._draw_text_lines(annotated, x1, y1, lines)

        return annotated, detections, inference_ms

    @staticmethod
    def _fmt_optional(value: Any, ndigits: int = 8) -> str:
        if value is None:
            return ""
        try:
            value = float(value)
        except Exception:
            return ""
        if not math.isfinite(value):
            return ""
        return f"{value:.{ndigits}f}"

    def _write_detections_if_recording(
        self,
        detections: list[dict[str, Any]],
        frame_system_time_s: float,
    ) -> None:
        with self._lock:
            if not self._recording or self._record_start_time is None or self._detection_writer is None:
                return
            camera_time_s = frame_system_time_s - self._record_start_time
            # 推理线程可能在刚切入记录模式时处理到记录开始前的一帧，跳过负时间结果。
            if camera_time_s < 0.0:
                return
            for det in detections:
                detection_time_abs = det.get("detection_time_s")
                detection_time_rel = None if detection_time_abs is None else float(detection_time_abs) - self._record_start_time
                self._detection_writer.writerow(
                    {
                        "camera_time_s": self._fmt_optional(camera_time_s),
                        "system_time_s": self._fmt_optional(det.get("system_time_s")),
                        "detection_time_s": self._fmt_optional(detection_time_rel),
                        "color_frame_number": det.get("color_frame_number", ""),
                        "detection_index": det.get("detection_index", ""),
                        "class_id": det.get("class_id", ""),
                        "class_name": det.get("class_name", ""),
                        "confidence": self._fmt_optional(det.get("confidence"), 6),
                        "x1": det.get("x1", ""),
                        "y1": det.get("y1", ""),
                        "x2": det.get("x2", ""),
                        "y2": det.get("y2", ""),
                        "center_u": det.get("center_u", ""),
                        "center_v": det.get("center_v", ""),
                        "depth_z_m": self._fmt_optional(det.get("depth_z_m"), 6),
                        "x_m": self._fmt_optional(det.get("x_m"), 6),
                        "y_m": self._fmt_optional(det.get("y_m"), 6),
                        "z_m": self._fmt_optional(det.get("z_m"), 6),
                        "range_m": self._fmt_optional(det.get("range_m"), 6),
                        "inference_ms": self._fmt_optional(det.get("inference_ms"), 3),
                    }
                )
            if self._detection_file is not None:
                self._detection_file.flush()

    def _yolo_loop(self) -> None:
        model = None
        loaded_generation = -1
        processed_seq = -1
        last_infer_time = 0.0

        while not self._stop_event.is_set():
            with self._lock:
                enabled = self._enable_yolo
                model_path = self._yolo_model_path
                generation = self._model_generation
                conf = self._yolo_conf
                iou = self._yolo_iou
                imgsz = self._yolo_imgsz
                device = self._yolo_device
                show_xyz = self._show_xyz
                max_hz = self._yolo_max_hz
                frame_seq = self._latest_frame_seq

            if not enabled:
                time.sleep(0.05)
                continue

            if model is None or generation != loaded_generation:
                try:
                    with self._lock:
                        self._yolo_status = f"正在加载 {model_path}"
                    print(f"[YOLO] 正在加载模型: {model_path}")
                    model = self._load_yolo_model(model_path)
                    loaded_generation = generation
                    with self._lock:
                        self._yolo_status = "运行中"
                    print(f"[YOLO] 模型加载完成: {model_path}")
                except Exception as exc:
                    model = None
                    loaded_generation = generation
                    msg = f"YOLO模型加载失败: {exc}"
                    with self._lock:
                        self._yolo_status = msg
                    print(f"[YOLO][WARN] {msg}")
                    time.sleep(1.0)
                    continue

            if frame_seq == processed_seq:
                time.sleep(0.002)
                continue

            min_interval = 1.0 / max_hz
            now = time.perf_counter()
            if now - last_infer_time < min_interval:
                time.sleep(min(0.005, min_interval - (now - last_infer_time)))
                continue

            with self._lock:
                if self._latest_color_bgr is None or self._latest_depth_m is None:
                    color = depth = None
                    frame_number = None
                    frame_system_time_s = None
                else:
                    color = self._latest_color_bgr.copy()
                    depth = self._latest_depth_m.copy()
                    frame_number = self._latest_frame_number
                    frame_system_time_s = self._latest_system_time_s
                    processed_seq = self._latest_frame_seq

            if color is None or depth is None or frame_number is None or frame_system_time_s is None:
                time.sleep(0.01)
                continue

            try:
                annotated, detections, inference_ms = self._infer_and_annotate(
                    model,
                    color,
                    depth,
                    frame_number=int(frame_number),
                    frame_system_time_s=float(frame_system_time_s),
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    show_xyz=show_xyz,
                )
                last_infer_time = time.perf_counter()
                with self._lock:
                    self._latest_annotated_bgr = annotated
                    self._latest_detections = detections
                    self._latest_detection_frame_number = int(frame_number)
                    self._latest_inference_ms = float(inference_ms)
                    self._yolo_status = "运行中"
                self._write_detections_if_recording(detections, float(frame_system_time_s))
            except Exception as exc:
                with self._lock:
                    self._yolo_status = f"推理异常: {exc}"
                print(f"[YOLO][WARN] 推理线程异常: {exc}")
                time.sleep(0.05)


# 为尽量兼容旧代码中 from ... import D435iRecorder 的写法。
D435iRecorder = D435iYoloRecorder
