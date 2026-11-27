"""Shared-device perception runtime with latest-result caches."""

import copy
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import psutil

from config import audio_config as audio_cfg
from config import system_config as cfg
from src.acquisition.camera import Camera
from src.audio.audio_analyzer import AudioAnalyzer
from src.audio.audio_detector import AudioDetector
from src.face.face_analyzer import FaceAnalyzer
from src.face.face_detector import FaceDetector
from src.face.face_visualizer import draw_face
from src.pose.pose_analyzer import PoseAnalyzer
from src.pose.pose_detector import PoseDetector
from src.pose.pose_visualizer import draw_pose


class RateMeter:
    def __init__(self, window_seconds=2.0):
        self.window_seconds = window_seconds
        self.times = deque()
        self.lock = threading.Lock()

    def tick(self, timestamp=None):
        timestamp = time.perf_counter() if timestamp is None else timestamp
        with self.lock:
            self.times.append(float(timestamp))
            self._trim(timestamp)

    def _trim(self, timestamp):
        while self.times and timestamp - self.times[0] > self.window_seconds:
            self.times.popleft()

    def value(self, timestamp=None):
        timestamp = time.perf_counter() if timestamp is None else timestamp
        with self.lock:
            self._trim(timestamp)
            if len(self.times) < 2:
                return 0.0
            duration = self.times[-1] - self.times[0]
            return float((len(self.times) - 1) / duration) if duration > 0 else 0.0


class PerceptionSystem:
    """Load each model once and share cached results between all clients."""

    def __init__(
        self,
        camera_factory=Camera,
        pose_detector_factory=PoseDetector,
        face_detector_factory=FaceDetector,
        audio_detector_factory=AudioDetector,
        pose_analyzer_factory=PoseAnalyzer,
        face_analyzer_factory=FaceAnalyzer,
        audio_analyzer_factory=AudioAnalyzer,
    ):
        self.camera_factory = camera_factory
        self.pose_detector_factory = pose_detector_factory
        self.face_detector_factory = face_detector_factory
        self.audio_detector_factory = audio_detector_factory
        self.pose_analyzer_factory = pose_analyzer_factory
        self.face_analyzer_factory = face_analyzer_factory
        self.audio_analyzer_factory = audio_analyzer_factory

        self.lock = threading.RLock()
        self.frame_condition = threading.Condition(threading.Lock())
        self.stop_event = threading.Event()
        self.threads = []
        self.running = False
        self.status = "stopped"
        self.start_time = None
        self.latest_frame = None
        self.frame_sequence = 0

        self.camera = None
        self.pose_detector = None
        self.face_detector = None
        self.audio_detector = None
        self.pose_analyzer = None
        self.face_analyzer = None
        self.audio_analyzer = None

        self.records = {"pose": None, "face": None, "audio": None}
        self.placeholder_jpeg = {
            "pose": self._placeholder_jpeg("POSE UNAVAILABLE"),
            "face": self._placeholder_jpeg("FACE UNAVAILABLE"),
        }
        self.jpeg = {
            "pose": {"data": self.placeholder_jpeg["pose"], "time": 0.0, "version": 0},
            "face": {"data": self.placeholder_jpeg["face"], "time": 0.0, "version": 0},
        }
        self.waveform = np.empty(0, dtype=np.float32)
        self.compressed_waveform = []
        self.waveform_end_time = None
        self.waveform_limit = int(cfg.WAVEFORM_SECONDS * audio_cfg.SAMPLE_RATE)
        self.history = deque(maxlen=cfg.HISTORY_LIMIT)
        self.last_event_key = None
        self.errors = {}
        self.lighting = {
            "mode": cfg.DEFAULT_LIGHTING_MODE,
            "current_lux": cfg.DEFAULT_SIMULATED_LUX,
            "target_lux": cfg.DEFAULT_SIMULATED_LUX,
            "reason": "手动仿真初始设定",
            "status": "confirmed",
            "updated_at": None,
            "is_simulated": True,
        }
        self.camera_rate = RateMeter()
        self.pose_rate = RateMeter()
        self.face_rate = RateMeter()
        self.process = psutil.Process()
        self.process.cpu_percent(None)
        self.performance = {
            "host_cpu_percent": 0.0,
            "host_memory_percent": float(psutil.virtual_memory().percent),
            "process_cpu_percent": 0.0,
            "process_memory_mb": float(self.process.memory_info().rss / (1024 * 1024)),
            "camera_fps": 0.0,
            "pose_fps": 0.0,
            "face_fps": 0.0,
        }

    @staticmethod
    def _placeholder_jpeg(text):
        image = np.full((cfg.CAMERA_HEIGHT, cfg.CAMERA_WIDTH, 3), 235, dtype=np.uint8)
        cv2.putText(
            image, text, (80, cfg.CAMERA_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2,
        )
        ok, encoded = cv2.imencode(".jpg", image)
        return encoded.tobytes() if ok else b""

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return [PerceptionSystem._json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return PerceptionSystem._json_safe(value.item())
        if isinstance(value, dict):
            return {str(key): PerceptionSystem._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [PerceptionSystem._json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def add_event(self, event_type, message, target_lux=None):
        key = (event_type, message, target_lux)
        with self.lock:
            if key == self.last_event_key:
                return False
            now = datetime.now()
            self.history.appendleft({
                "timestamp": now.isoformat(timespec="seconds"),
                "time": now.strftime("%H:%M:%S"),
                "type": event_type,
                "event": message,
                "target_lux": int(target_lux) if target_lux is not None else None,
            })
            self.last_event_key = key
        return True

    def _record_error(self, module, error):
        message = f"{type(error).__name__}: {error}"
        with self.lock:
            self.errors[module] = message
            if self.running:
                self.status = "degraded"
        self.add_event("system_error", f"{module}不可用：{message}")

    def start(self):
        with self.lock:
            if self.running:
                return
            self.status = "starting"
            self.start_time = time.perf_counter()
            self.stop_event.clear()
            self.errors.clear()

        try:
            self.camera = self.camera_factory(
                camera_id=cfg.CAMERA_ID,
                width=cfg.CAMERA_WIDTH,
                height=cfg.CAMERA_HEIGHT,
            )
        except Exception as error:
            self._record_error("camera", error)
            self.camera = None

        if self.camera is not None:
            pose_detector = None
            try:
                pose_detector = self.pose_detector_factory()
                self.pose_analyzer = self.pose_analyzer_factory()
                self.pose_detector = pose_detector
            except Exception as error:
                self._record_error("pose", error)
                if pose_detector is not None:
                    pose_detector.close()
                self.pose_detector = None
                self.pose_analyzer = None
            face_detector = None
            try:
                face_detector = self.face_detector_factory()
                self.face_analyzer = self.face_analyzer_factory()
                self.face_detector = face_detector
            except Exception as error:
                self._record_error("face", error)
                if face_detector is not None:
                    face_detector.close()
                self.face_detector = None
                self.face_analyzer = None

        try:
            self.audio_detector = self.audio_detector_factory()
            self.audio_analyzer = self.audio_analyzer_factory()
        except Exception as error:
            self._record_error("audio", error)
            self.audio_detector = None
            self.audio_analyzer = None

        workers = [("system-monitor", self._monitor_loop)]
        if self.camera is not None:
            workers.append(("camera-capture", self._camera_loop))
        if self.pose_detector is not None:
            workers.append(("pose-inference", self._pose_loop))
        if self.face_detector is not None:
            workers.append(("face-inference", self._face_loop))
        if self.audio_detector is not None:
            workers.append(("audio-perception", self._audio_loop))
        self.threads = [
            threading.Thread(target=target, name=name, daemon=True)
            for name, target in workers
        ]
        with self.lock:
            self.running = True
            self.status = "running" if workers and not self.errors else "degraded"
            self.lighting["updated_at"] = datetime.now().isoformat(timespec="seconds")
        for thread in self.threads:
            thread.start()
        self.add_event("system", "统一感知系统已启动")

    def _camera_loop(self):
        interval = 1.0 / cfg.CAMERA_TARGET_HZ
        next_time = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                frame = self.camera.read()
                captured_at = time.perf_counter()
                with self.frame_condition:
                    self.frame_sequence += 1
                    self.latest_frame = (self.frame_sequence, captured_at, frame)
                    self.frame_condition.notify_all()
                self.camera_rate.tick(captured_at)
                next_time += interval
                delay = next_time - time.perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                else:
                    next_time = time.perf_counter()
        except Exception as error:
            self._record_error("camera", error)

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            now = time.perf_counter()
            memory = psutil.virtual_memory()
            process_memory = self.process.memory_info().rss
            performance = {
                "host_cpu_percent": float(psutil.cpu_percent(None)),
                "host_memory_percent": float(memory.percent),
                "process_cpu_percent": float(self.process.cpu_percent(None)),
                "process_memory_mb": float(process_memory / (1024 * 1024)),
                "camera_fps": self.camera_rate.value(now),
                "pose_fps": self.pose_rate.value(now),
                "face_fps": self.face_rate.value(now),
            }
            with self.lock:
                self.performance = performance
            self.stop_event.wait(1.0)

    def _get_latest_frame(self, last_sequence):
        with self.frame_condition:
            self.frame_condition.wait_for(
                lambda: self.stop_event.is_set()
                or (self.latest_frame is not None and self.latest_frame[0] != last_sequence),
                timeout=0.2,
            )
            if self.stop_event.is_set() or self.latest_frame is None:
                return None
            sequence, captured_at, frame = self.latest_frame
            return sequence, captured_at, frame.copy()

    def _video_timestamp_ms(self, captured_at, detector):
        value = int((captured_at - self.start_time) * 1000.0)
        return max(value, detector.last_timestamp_ms + 1)

    def _pose_loop(self):
        self._video_loop(
            "pose", cfg.POSE_TARGET_HZ, self.pose_detector,
            self.pose_analyzer, self.pose_rate,
        )

    def _face_loop(self):
        self._video_loop(
            "face", cfg.FACE_TARGET_HZ, self.face_detector,
            self.face_analyzer, self.face_rate,
        )

    def _video_loop(self, module, target_hz, detector, analyzer, rate_meter):
        last_sequence = -1
        next_time = time.perf_counter()
        interval = 1.0 / target_hz
        try:
            while not self.stop_event.is_set():
                delay = next_time - time.perf_counter()
                if delay > 0 and self.stop_event.wait(delay):
                    break
                item = self._get_latest_frame(last_sequence)
                if item is None:
                    continue
                sequence, captured_at, frame = item
                last_sequence = sequence
                timestamp_ms = self._video_timestamp_ms(captured_at, detector)
                inference_start = time.perf_counter()
                result, timestamp_ms = detector.detect(frame, timestamp_ms)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
                if module == "pose":
                    converted = analyzer.update(result, timestamp_ms, frame.shape)
                    analysis = analyzer.analyze(converted)
                    visual = draw_pose(frame.copy(), result)
                else:
                    converted = analyzer.update(result, timestamp_ms, frame.shape)
                    analysis = analyzer.analyze(converted)
                    visual = draw_face(frame.copy(), converted)
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", visual,
                    [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY],
                )
                generated_at = time.perf_counter()
                rate_meter.tick(generated_at)
                with self.lock:
                    self.records[module] = {
                        "analysis": self._json_safe(analysis),
                        "captured_at": float(captured_at),
                        "generated_at": float(generated_at),
                        "inference_ms": float(inference_ms),
                    }
                    if encoded_ok:
                        self.jpeg[module] = {
                            "data": encoded.tobytes(),
                            "time": float(generated_at),
                            "version": self.jpeg[module]["version"] + 1,
                        }
                next_time += interval
                if next_time < time.perf_counter() - interval:
                    next_time = time.perf_counter()
        except Exception as error:
            self._record_error(module, error)

    def _audio_loop(self):
        try:
            while not self.stop_event.is_set():
                detection = self.audio_detector.read()
                frame = self.audio_analyzer.update(detection)
                analysis = self.audio_analyzer.analyze(frame)
                generated_at = time.perf_counter()
                samples = detection.get("samples")
                with self.lock:
                    if detection.get("discontinuity"):
                        self.waveform = np.empty(0, dtype=np.float32)
                        self.compressed_waveform = []
                    if samples is not None:
                        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
                        self.waveform = np.concatenate((self.waveform, samples))[
                            -self.waveform_limit:
                        ]
                        self.waveform_end_time = detection.get("block_end_time")
                        self.compressed_waveform = self.compress_waveform(self.waveform)
                    self.records["audio"] = {
                        "analysis": self._json_safe(analysis),
                        "captured_at": detection.get("block_end_time"),
                        "generated_at": float(generated_at),
                        "inference_ms": self._json_safe(detection.get("inference_ms", {})),
                    }
        except Exception as error:
            self._record_error("audio", error)

    @staticmethod
    def compress_waveform(samples, max_points=cfg.WAVEFORM_POINTS):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(samples) == 0:
            return []
        if len(samples) <= max_points:
            return [float(value) for value in samples]
        bins = max(1, max_points // 2)
        boundaries = np.linspace(0, len(samples), bins + 1, dtype=int)
        output = []
        for index in range(bins):
            segment = samples[boundaries[index]:boundaries[index + 1]]
            if len(segment):
                output.extend((float(np.min(segment)), float(np.max(segment))))
        return output[:max_points]

    def _module_snapshot(self, module, now, ttl):
        record = self.records[module]
        if record is None:
            return {
                "available": False, "stale": True, "age_seconds": None,
                "captured_at": None, "generated_at": None,
                "inference_ms": None, "analysis": None,
            }
        age = max(0.0, now - record["generated_at"])
        fresh = age <= ttl
        analysis = record["analysis"]
        observable = not (
            module == "audio"
            and isinstance(analysis, dict)
            and not analysis.get("audio_available", False)
        )
        return {
            "available": bool(fresh and observable),
            "stale": not fresh,
            "age_seconds": float(age),
            "captured_at": record["captured_at"],
            "generated_at": record["generated_at"],
            "inference_ms": record["inference_ms"],
            "analysis": copy.deepcopy(analysis) if fresh else None,
        }

    def _fusion_snapshot(self):
        weights = copy.deepcopy(cfg.FUSION_BASE_WEIGHTS)
        return {
            "available": False,
            "status": "pending",
            "message": "多模态融合算法待接入",
            "valence": None,
            "arousal": None,
            "state_candidate": "unknown",
            "trajectory": [],
            "base_weights": weights,
            "effective_weights": {
                "valence": {name: None for name in weights},
                "arousal": {name: None for name in weights},
            },
            "reliability": None,
        }

    def snapshot(self):
        now = time.perf_counter()
        with self.lock:
            pose = self._module_snapshot("pose", now, cfg.POSE_RESULT_TTL_SECONDS)
            face = self._module_snapshot("face", now, cfg.FACE_RESULT_TTL_SECONDS)
            audio = self._module_snapshot("audio", now, cfg.AUDIO_RESULT_TTL_SECONDS)
            waveform_values = copy.deepcopy(self.compressed_waveform)
            waveform_age = (
                max(0.0, now - self.waveform_end_time)
                if self.waveform_end_time is not None else None
            )
            waveform_available = bool(
                waveform_values and waveform_age is not None
                and waveform_age <= cfg.AUDIO_RESULT_TTL_SECONDS
            )
            lighting = copy.deepcopy(self.lighting)
            history = copy.deepcopy(list(self.history))
            errors = copy.deepcopy(self.errors)
            status = self.status
            uptime = now - self.start_time if self.start_time is not None else 0.0
            performance = copy.deepcopy(self.performance)
        data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "monotonic_time": float(now),
            "system": {
                "status": status,
                "uptime_seconds": float(max(0.0, uptime)),
                "errors": errors,
                "performance": performance,
            },
            "pose": pose,
            "face": face,
            "audio": audio,
            "waveform": {
                "available": waveform_available,
                "stale": not waveform_available,
                "age_seconds": waveform_age,
                "sample_rate": audio_cfg.SAMPLE_RATE,
                "duration_seconds": cfg.WAVEFORM_SECONDS,
                "values": waveform_values if waveform_available else [],
            },
            "fusion": self._fusion_snapshot(),
            "lighting": lighting,
            "history": history,
        }
        return self._json_safe(data)

    def get_jpeg(self, module):
        if module not in self.jpeg:
            raise ValueError("Unknown video module")
        now = time.perf_counter()
        with self.lock:
            item = self.jpeg[module]
            if now - item["time"] > cfg.VIDEO_FRAME_TTL_SECONDS:
                return self.placeholder_jpeg[module], item["version"]
            return item["data"], item["version"]

    def set_lighting_mode(self, mode):
        if mode not in ("manual", "automatic"):
            raise ValueError("mode must be manual or automatic")
        with self.lock:
            if self.lighting["mode"] == mode:
                return copy.deepcopy(self.lighting)
            self.lighting["mode"] = mode
            self.lighting["reason"] = (
                "自动策略待接入，保持当前仿真照度"
                if mode == "automatic" else "手动仿真控制"
            )
            self.lighting["status"] = "pending" if mode == "automatic" else "confirmed"
            self.lighting["updated_at"] = datetime.now().isoformat(timespec="seconds")
            target = self.lighting["target_lux"]
        label = "自动控制模式已启用（策略待接入）" if mode == "automatic" else "手动控制模式已启用"
        self.add_event("lighting_mode", label, target)
        return self.lighting_snapshot()

    def set_manual_lux(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("lux must be numeric")
        value = int(round(value))
        if not cfg.MIN_SIMULATED_LUX <= value <= cfg.MAX_SIMULATED_LUX:
            raise ValueError("lux is outside the allowed range")
        with self.lock:
            if self.lighting["mode"] != "manual":
                raise RuntimeError("manual lux requires manual mode")
            if self.lighting["target_lux"] == value:
                return copy.deepcopy(self.lighting)
            self.lighting["target_lux"] = value
            self.lighting["current_lux"] = value
            self.lighting["reason"] = "用户手动调整仿真照度"
            self.lighting["status"] = "confirmed"
            self.lighting["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.add_event("lighting_manual", "手动仿真照度已调整", value)
        return self.lighting_snapshot()

    def lighting_snapshot(self):
        with self.lock:
            return copy.deepcopy(self.lighting)

    def stop(self):
        with self.lock:
            if not self.running and self.status == "stopped":
                return
            self.status = "stopping"
        self.stop_event.set()
        with self.frame_condition:
            self.frame_condition.notify_all()
        deadline = time.perf_counter() + cfg.THREAD_JOIN_TIMEOUT_SECONDS
        for thread in self.threads:
            thread.join(max(0.0, deadline - time.perf_counter()))
        for resource, method in (
            (self.pose_detector, "close"),
            (self.face_detector, "close"),
            (self.audio_detector, "close"),
            (self.camera, "release"),
        ):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as error:
                    self._record_error("shutdown", error)
        self.threads = []
        self.camera = self.pose_detector = self.face_detector = None
        self.audio_detector = None
        self.pose_analyzer = self.face_analyzer = self.audio_analyzer = None
        with self.lock:
            self.running = False
            self.status = "stopped"
        self.add_event("system", "统一感知系统已停止")
