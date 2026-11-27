"""Lightweight NumPy acoustic, prosodic and audio-state analysis."""

import re
from collections import deque
from dataclasses import dataclass

import numpy as np

from config import audio_config as cfg


@dataclass
class AudioFrame:
    timestamp: float
    start_time: float | None
    end_time: float | None
    samples: np.ndarray | None
    data_available: bool
    speech_detected: bool
    discontinuity: bool
    rms: float | None
    dbfs: float | None
    smoothed_rms: float | None
    f0: float | None
    detection: dict


class AudioAnalyzer:
    def __init__(self, window_seconds=cfg.ANALYSIS_WINDOW_SECONDS):
        self.window_seconds = float(window_seconds)
        self.frames = deque()
        self.f0_history = deque()
        self.f0_pending = np.empty(0, dtype=np.float32)
        self.f0_pending_start = None
        self.smoothed_rms = None
        self.last_timestamp = None
        self.last_transcription = None
        self.last_sound_result = None
        self.last_frame = None

    @staticmethod
    def rms(samples):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.size == 0 or not np.isfinite(samples).all():
            return None
        return float(np.sqrt(np.mean(np.square(samples))))

    @staticmethod
    def dbfs(rms_value):
        if rms_value is None:
            return None
        if rms_value <= 0.0:
            return float(cfg.DBFS_FLOOR)
        return float(max(20.0 * np.log10(rms_value), cfg.DBFS_FLOOR))

    @staticmethod
    def _frame_f0(frame, sample_rate=cfg.SAMPLE_RATE):
        frame = np.asarray(frame, dtype=np.float64)
        if len(frame) < cfg.F0_FRAME_SAMPLES or not np.isfinite(frame).all():
            return None
        frame = frame[:cfg.F0_FRAME_SAMPLES]
        frame = frame - np.mean(frame)
        frame_rms = np.sqrt(np.mean(np.square(frame)))
        if frame_rms < cfg.F0_MIN_RMS:
            return None
        windowed = frame * np.hanning(len(frame))
        min_lag = max(1, int(sample_rate / cfg.F0_MAX_HZ))
        max_lag = min(len(frame) - 2, int(sample_rate / cfg.F0_MIN_HZ))
        correlations = np.empty(max_lag - min_lag + 1, dtype=float)
        for offset, lag in enumerate(range(min_lag, max_lag + 1)):
            left = windowed[:-lag]
            right = windowed[lag:]
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            correlations[offset] = (
                np.dot(left, right) / denominator if denominator > 1e-12 else 0.0
            )
        peak_offset = int(np.argmax(correlations))
        periodicity = float(correlations[peak_offset])
        if periodicity < cfg.F0_MIN_PERIODICITY:
            return None
        lag = float(min_lag + peak_offset)
        if 0 < peak_offset < len(correlations) - 1:
            left, center, right = correlations[peak_offset - 1:peak_offset + 2]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                lag += float(0.5 * (left - right) / denominator)
        if lag <= 0.0:
            return None
        f0 = sample_rate / lag
        if not cfg.F0_MIN_HZ <= f0 <= cfg.F0_MAX_HZ:
            return None
        return float(f0)

    @classmethod
    def estimate_f0(cls, samples, sample_rate=cfg.SAMPLE_RATE):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        values = []
        for start in range(
            0,
            max(0, len(samples) - cfg.F0_FRAME_SAMPLES + 1),
            cfg.F0_HOP_SAMPLES,
        ):
            value = cls._frame_f0(
                samples[start:start + cfg.F0_FRAME_SAMPLES], sample_rate
            )
            if value is not None:
                values.append(value)
        return float(np.median(values)) if values else None

    def reset_temporal(self):
        self.frames.clear()
        self.f0_history.clear()
        self.f0_pending = np.empty(0, dtype=np.float32)
        self.f0_pending_start = None
        self.smoothed_rms = None
        self.last_timestamp = None

    def reset(self):
        self.reset_temporal()
        self.last_transcription = None
        self.last_sound_result = None
        self.last_frame = None

    def _process_f0(self, samples, start_time, speech_detected):
        if not speech_detected:
            self.f0_pending = np.empty(0, dtype=np.float32)
            self.f0_pending_start = None
            return None
        if self.f0_pending_start is None:
            self.f0_pending_start = float(start_time)
        self.f0_pending = np.concatenate((self.f0_pending, samples))
        new_values = []
        while len(self.f0_pending) >= cfg.F0_FRAME_SAMPLES:
            value = self._frame_f0(self.f0_pending[:cfg.F0_FRAME_SAMPLES])
            center_time = (
                self.f0_pending_start
                + cfg.F0_FRAME_SAMPLES / (2.0 * cfg.SAMPLE_RATE)
            )
            if value is not None:
                self.f0_history.append((float(center_time), float(value)))
                new_values.append(value)
            self.f0_pending = self.f0_pending[cfg.F0_HOP_SAMPLES:].copy()
            self.f0_pending_start += cfg.F0_HOP_SAMPLES / cfg.SAMPLE_RATE
        return float(np.median(new_values)) if new_values else None

    def update(self, detection):
        timestamp = float(detection.get("timestamp", 0.0))
        available = bool(detection.get("data_available", False))
        discontinuity = bool(detection.get("discontinuity", False))
        if not available:
            self._ingest_results(detection)
            self.reset_temporal()
            frame = AudioFrame(
                timestamp, None, None, None, False, False, True,
                None, None, None, None, detection,
            )
            self.last_frame = frame
            return frame
        if discontinuity:
            self.reset_temporal()

        samples = np.asarray(detection["samples"], dtype=np.float32).reshape(-1)
        start_time = float(detection["block_start_time"])
        end_time = float(detection["block_end_time"])
        speech_detected = bool(detection.get("speech_detected", False))
        rms_value = self.rms(samples)
        dbfs_value = self.dbfs(rms_value)
        if self.smoothed_rms is None:
            self.smoothed_rms = rms_value
        else:
            dt = max(0.0, start_time - self.last_timestamp)
            alpha = 1.0 - np.exp(-dt / max(cfg.ACOUSTIC_EMA_TIME_CONSTANT, 1e-6))
            self.smoothed_rms = (
                alpha * rms_value + (1.0 - alpha) * self.smoothed_rms
            )
        f0 = self._process_f0(samples, start_time, speech_detected)
        self.frames.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "speech": speech_detected,
        })
        self.last_timestamp = start_time
        self._trim(end_time)
        self._ingest_results(detection)
        frame = AudioFrame(
            timestamp=timestamp,
            start_time=start_time,
            end_time=end_time,
            samples=samples,
            data_available=True,
            speech_detected=speech_detected,
            discontinuity=discontinuity,
            rms=rms_value,
            dbfs=dbfs_value,
            smoothed_rms=float(self.smoothed_rms),
            f0=f0,
            detection=detection,
        )
        self.last_frame = frame
        return frame

    def _trim(self, end_time):
        cutoff = end_time - self.window_seconds
        while self.frames and self.frames[0]["end_time"] < cutoff:
            self.frames.popleft()
        while self.f0_history and self.f0_history[0][0] < cutoff:
            self.f0_history.popleft()

    @staticmethod
    def content_unit_count(text):
        if not text:
            return 0
        clean = re.sub(r"<\|.*?\|>", "", str(text))
        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", clean))

    @classmethod
    def speech_rate(cls, text, duration):
        count = cls.content_unit_count(text)
        if count == 0 or duration is None or duration <= 0.0:
            return None, count
        return float(count * 60.0 / duration), count

    def _ingest_results(self, detection):
        transcriptions = detection.get("transcriptions") or []
        if transcriptions:
            item = dict(transcriptions[-1])
            rate, count = self.speech_rate(item.get("text"), item.get("duration"))
            item["speech_rate"] = rate
            item["content_unit_count"] = int(count)
            item["speech_rate_unit"] = "content_units_per_minute"
            self.last_transcription = item
        sound_results = detection.get("sound_results") or []
        if sound_results:
            self.last_sound_result = dict(sound_results[-1])

    def _ratios(self):
        if not self.frames:
            return 0.0, None, None
        valid_duration = sum(item["duration"] for item in self.frames)
        span = self.frames[-1]["end_time"] - self.frames[0]["start_time"]
        coverage = float(np.clip(valid_duration / span, 0.0, 1.0)) if span > 0 else 1.0
        if valid_duration <= 0.0:
            return coverage, None, None
        speech_duration = sum(
            item["duration"] for item in self.frames if item["speech"]
        )
        speech_ratio = float(np.clip(speech_duration / valid_duration, 0.0, 1.0))
        return coverage, speech_ratio, float(1.0 - speech_ratio)

    def _pitch_statistics(self):
        if len(self.f0_history) < cfg.F0_MIN_VALID_COUNT:
            return None, None, 0.0
        values = np.asarray([value for _, value in self.f0_history], dtype=float)
        median = float(np.median(values))
        pitch_range = float(
            np.percentile(values, cfg.PITCH_RANGE_HIGH_PERCENTILE)
            - np.percentile(values, cfg.PITCH_RANGE_LOW_PERCENTILE)
        )
        voiced_coverage = float(min(1.0, len(values) / cfg.F0_MIN_VALID_COUNT))
        return median, pitch_range, voiced_coverage

    @staticmethod
    def _history_item(item, current_time, ttl):
        if item is None:
            return None
        output = dict(item)
        observed_time = output.get("observed_time", output.get("timestamp"))
        age = max(0.0, current_time - observed_time) if observed_time is not None else None
        output["age_seconds"] = float(age) if age is not None else None
        output["is_current"] = bool(age is not None and age <= ttl)
        return output

    @staticmethod
    def _safe(value):
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    @classmethod
    def _json_value(cls, value):
        if isinstance(value, np.ndarray):
            return [cls._json_value(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return cls._json_value(value.item())
        if isinstance(value, dict):
            return {str(key): cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def _intensity(self, frame, speech_ratio, pitch_range):
        level_dbfs = self.dbfs(frame.smoothed_rms)
        if (
            not frame.data_available or not frame.speech_detected
            or speech_ratio is None
            or speech_ratio < cfg.INTENSITY_MIN_SPEECH_RATIO
            or level_dbfs is None or pitch_range is None
        ):
            return "unknown", None
        if (
            level_dbfs >= cfg.INTENSITY_HIGH_DBFS
            and pitch_range >= cfg.INTENSITY_HIGH_PITCH_RANGE_HZ
        ):
            candidate = "high_intensity"
        elif (
            level_dbfs <= cfg.INTENSITY_LOW_DBFS
            and pitch_range <= cfg.INTENSITY_LOW_PITCH_RANGE_HZ
        ):
            candidate = "low_intensity"
        else:
            candidate = "normal_intensity"
        level = np.clip(
            (level_dbfs - cfg.INTENSITY_DBFS_MIN)
            / (cfg.INTENSITY_DBFS_MAX - cfg.INTENSITY_DBFS_MIN),
            0.0,
            1.0,
        )
        pitch = np.clip(pitch_range / cfg.INTENSITY_PITCH_RANGE_MAX_HZ, 0.0, 1.0)
        return candidate, float(
            cfg.INTENSITY_LEVEL_WEIGHT * level
            + cfg.INTENSITY_PITCH_WEIGHT * pitch
        )

    def analyze(self, frame=None):
        frame = frame or self.last_frame
        if frame is None:
            raise RuntimeError("No audio frame is available")
        coverage, speech_ratio, pause_ratio = self._ratios()
        median_f0, pitch_range, voiced_coverage = self._pitch_statistics()
        detection = frame.detection
        current_time = float(detection.get("processed_time", frame.timestamp))
        transcription = self._history_item(
            self.last_transcription, current_time, cfg.TRANSCRIPTION_TTL_SECONDS
        )
        sound = self._history_item(
            self.last_sound_result, current_time, cfg.SOUND_EVENT_TTL_SECONDS
        )

        if not frame.data_available:
            speech_state = "unavailable"
        elif detection.get("processing"):
            speech_state = "processing"
        elif detection.get("transcriptions"):
            speech_state = detection["transcriptions"][-1]["speech_state"]
        elif frame.speech_detected:
            speech_state = "speaking"
        else:
            speech_state = "no_speech"
        intensity, intensity_score = self._intensity(
            frame, speech_ratio, pitch_range
        )

        availability = 1.0 if frame.data_available else 0.0
        continuity = 1.0 if frame.data_available and not frame.discontinuity else 0.0
        if frame.end_time is None:
            freshness = 0.0
        else:
            age = max(0.0, current_time - frame.end_time)
            freshness = float(np.clip(
                1.0 - age / cfg.RESULT_FRESHNESS_SECONDS, 0.0, 1.0
            ))
        base_features = int(frame.rms is not None) + int(frame.dbfs is not None)
        if frame.speech_detected:
            feature_coverage = (base_features + int(median_f0 is not None)) / 3.0
        else:
            feature_coverage = base_features / 2.0
        components = {
            "availability": availability,
            "continuity": continuity,
            "freshness": freshness,
            "temporal_coverage": coverage,
            "feature_coverage": float(feature_coverage),
        }
        reliability_score = float(sum(
            cfg.RELIABILITY_WEIGHTS[name] * value
            for name, value in components.items()
        ))
        reliable = bool(
            frame.data_available
            and reliability_score >= cfg.RELIABILITY_THRESHOLD
            and coverage >= cfg.MIN_TEMPORAL_COVERAGE
        )
        latest_segment = (
            detection.get("speech_segments", [])[-1]
            if detection.get("speech_segments") else None
        )
        segment_duration = (
            latest_segment.get("duration") if latest_segment
            else (
                self.last_transcription.get("duration")
                if self.last_transcription else None
            )
        )
        sound_output = None
        if sound is not None:
            sound_output = {
                "timestamp": self._safe(sound.get("timestamp")),
                "start_time": self._safe(sound.get("start_time")),
                "end_time": self._safe(sound.get("end_time")),
                "age_seconds": self._safe(sound.get("age_seconds")),
                "is_current": bool(sound.get("is_current")),
                "class_count": int(sound.get("class_count", 0)),
                "inference_ms": self._safe(sound.get("inference_ms")),
                "events": sound.get("sound_events", []),
                "raw_top": sound.get("raw_top", []),
            }
        output = {
            "timestamp": float(frame.timestamp),
            "observation_start_time": self._safe(frame.start_time),
            "observation_end_time": self._safe(frame.end_time),
            "processed_time": float(current_time),
            "audio_available": bool(frame.data_available),
            "speech_state": speech_state,
            "speech_active": bool(frame.speech_detected) if frame.data_available else None,
            "rms": self._safe(frame.rms),
            "smoothed_rms": self._safe(frame.smoothed_rms),
            "dbfs": self._safe(frame.dbfs),
            "smoothed_dbfs": self._safe(self.dbfs(frame.smoothed_rms)),
            "dbfs_unit": "dBFS",
            "f0": self._safe(frame.f0),
            "median_f0": self._safe(median_f0),
            "pitch_range": self._safe(pitch_range),
            "periodic_voicing": (
                bool(median_f0 is not None)
                if frame.data_available and frame.speech_detected else None
            ),
            "frequency_unit": "Hz",
            "voiced_feature_coverage": float(voiced_coverage),
            "speech_segment_duration": (
                self._safe(segment_duration)
            ),
            "speech_activity_ratio": self._safe(speech_ratio),
            "pause_ratio": self._safe(pause_ratio),
            "expression_intensity": intensity,
            "expression_intensity_score": self._safe(intensity_score),
            "transcription": transcription,
            "sound_event_result": sound_output,
            "reliability": {
                "score": reliability_score,
                "reliable": reliable,
                **components,
            },
            "temporal_coverage": float(coverage),
            "discontinuity": bool(frame.discontinuity),
            "discontinuity_reasons": list(
                detection.get("discontinuity_reasons", [])
            ),
            "reset_count": int(detection.get("reset_count", 0)),
            "inference_ms": detection.get("inference_ms", {}),
            "buffer_status": detection.get("buffer_status", {}),
        }
        return self._json_value(output)
