"""Clocked image/video source with optional FFmpeg audio extraction."""

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class MediaFileSource:
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, path, sample_rate=16000, audio_block_size=512):
        self.path = Path(path).resolve()
        suffix = self.path.suffix.lower()
        self.kind = "image" if suffix in self.IMAGE_EXTENSIONS else "video"
        if suffix not in self.IMAGE_EXTENSIONS | self.VIDEO_EXTENSIONS:
            raise ValueError("unsupported media type")
        self.sample_rate, self.audio_block_size = int(sample_rate), int(audio_block_size)
        self.lock = threading.RLock()
        self.paused, self.stopped, self.finished = False, False, False
        self.position_seconds, self.duration_seconds = 0.0, 0.0
        self.timeline_origin, self.pause_started = time.perf_counter(), None
        self.frame_index, self.audio_index = 0, 0
        self.audio_samples, self.has_audio = np.empty(0, dtype=np.float32), False
        self.audio_status = "not_applicable"
        self.capture, self.image, self.fps = None, None, 0.0
        if self.kind == "image":
            self.image = cv2.imread(str(self.path))
            if self.image is None:
                raise RuntimeError("unable to decode image")
        else:
            self.capture = cv2.VideoCapture(str(self.path))
            if not self.capture.isOpened():
                raise RuntimeError("unable to open video")
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
            frames = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            self.duration_seconds = frames / self.fps if frames > 0 else 0.0
            self._load_audio()

    @staticmethod
    def _ffmpeg_path():
        project = Path(__file__).resolve().parents[2]
        portable = project / ".tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
        return str(portable) if portable.exists() else shutil.which("ffmpeg")

    def _load_audio(self):
        executable = self._ffmpeg_path()
        if not executable:
            self.audio_status = "ffmpeg_unavailable"
            return
        temp_dir = Path(__file__).resolve().parents[2] / ".temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(suffix=".f32", delete=False, dir=temp_dir)
        temporary.close()
        try:
            result = subprocess.run(
                [executable, "-v", "error", "-i", str(self.path), "-map", "0:a:0", "-ac", "1", "-ar", str(self.sample_rate), "-f", "f32le", "-y", temporary.name],
                capture_output=True, timeout=max(30.0, self.duration_seconds * 2.0), check=False,
            )
            if result.returncode == 0:
                self.audio_samples = np.fromfile(temporary.name, dtype="<f4")
                self.has_audio = len(self.audio_samples) > 0
                self.audio_status = "available" if self.has_audio else "no_audio_track"
                if self.duration_seconds <= 0 and self.has_audio:
                    self.duration_seconds = len(self.audio_samples) / self.sample_rate
            else:
                error = result.stderr.decode("utf-8", errors="ignore").lower()
                self.audio_status = "no_audio_track" if "matches no streams" in error else "extraction_failed"
        finally:
            Path(temporary.name).unlink(missing_ok=True)

    def _clock_position(self):
        if self.paused:
            return self.position_seconds
        return max(0.0, time.perf_counter() - self.timeline_origin)

    def read(self):
        with self.lock:
            if self.stopped or self.paused or self.finished:
                return None
            if self.kind == "image":
                if self.frame_index:
                    return None
                self.frame_index = 1
                return self.image.copy(), time.perf_counter()
            target_index = int(self._clock_position() * self.fps)
            if target_index < self.frame_index:
                return None
            while self.frame_index <= target_index:
                ok, frame = self.capture.read()
                if not ok:
                    self.finished = True
                    return None
                self.frame_index += 1
            self.position_seconds = min(self.duration_seconds, self.frame_index / self.fps)
            return frame, time.perf_counter()

    def read_audio_block(self):
        with self.lock:
            if not self.has_audio or self.stopped or self.paused or self.finished:
                return None
            target = min(len(self.audio_samples), int(self._clock_position() * self.sample_rate))
            if target - self.audio_index < self.audio_block_size:
                return None
            end = min(target, self.audio_index + self.audio_block_size)
            samples = self.audio_samples[self.audio_index:end].copy()
            start = self.audio_index / self.sample_rate
            self.audio_index = end
            return samples, start, end / self.sample_rate

    def play(self):
        with self.lock:
            if self.finished:
                self.replay()
                return
            if self.paused:
                self.timeline_origin = time.perf_counter() - self.position_seconds
                self.paused, self.pause_started = False, None

    def pause(self):
        with self.lock:
            if not self.paused and not self.stopped:
                self.position_seconds = self._clock_position()
                self.paused, self.pause_started = True, time.perf_counter()

    def replay(self):
        with self.lock:
            if self.capture is not None:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = self.audio_index = 0
            self.position_seconds, self.timeline_origin = 0.0, time.perf_counter()
            self.paused, self.stopped, self.finished = False, False, False

    def stop(self):
        with self.lock:
            self.stopped, self.paused = True, False
            self.position_seconds = 0.0

    def snapshot(self):
        with self.lock:
            position = self._clock_position() if not self.stopped else self.position_seconds
            if self.kind == "image":
                state = "analyzed" if self.frame_index else "ready"
            else:
                state = "stopped" if self.stopped else ("paused" if self.paused else ("finished" if self.finished else "playing"))
            return {"type": self.kind, "name": self.path.name, "state": state, "position_seconds": min(position, self.duration_seconds) if self.duration_seconds else position, "duration_seconds": self.duration_seconds, "progress": min(1.0, position / self.duration_seconds) if self.duration_seconds else None, "has_audio": self.has_audio, "audio_status": self.audio_status}

    def release(self):
        if self.capture is not None:
            self.capture.release()
