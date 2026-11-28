"""Synchronous VAD, ASR and environmental sound detector orchestration."""

import time
from concurrent.futures import ThreadPoolExecutor

import mediapipe as mp
import numpy as np
import sherpa_onnx

from config import audio_config as cfg
from src.acquisition.microphone import Microphone


class SileroVadDetector:
    def __init__(self):
        self.sample_rate = cfg.SAMPLE_RATE
        self.window_size = cfg.BLOCK_SIZE
        if not cfg.SILERO_VAD_MODEL.is_file():
            raise FileNotFoundError(cfg.SILERO_VAD_MODEL)

        model_config = sherpa_onnx.VadModelConfig()
        model_config.silero_vad.model = str(cfg.SILERO_VAD_MODEL)
        model_config.silero_vad.threshold = cfg.VAD_THRESHOLD
        model_config.silero_vad.min_speech_duration = cfg.VAD_MIN_SPEECH_SECONDS
        model_config.silero_vad.min_silence_duration = cfg.VAD_MIN_SILENCE_SECONDS
        model_config.silero_vad.max_speech_duration = cfg.VAD_MAX_SPEECH_SECONDS
        model_config.silero_vad.window_size = self.window_size
        model_config.sample_rate = self.sample_rate
        model_config.num_threads = cfg.VAD_NUM_THREADS
        model_config.provider = "cpu"
        model_config.debug = False
        self.vad = sherpa_onnx.VoiceActivityDetector(
            model_config,
            buffer_size_in_seconds=cfg.VAD_BUFFER_SECONDS,
        )
        self.audio_start_time = None

    def detect(self, samples, timestamp):
        if self.audio_start_time is None:
            self.audio_start_time = float(timestamp)
        self.vad.accept_waveform(samples)
        speech_detected = bool(self.vad.is_speech_detected())
        segments = []
        while not self.vad.empty():
            segment = self.vad.front
            speech_samples = np.asarray(segment.samples, dtype=np.float32).copy()
            start_time = (
                self.audio_start_time + int(segment.start) / self.sample_rate
            )
            duration = len(speech_samples) / self.sample_rate
            segments.append({
                "samples": speech_samples,
                "start_time": float(start_time),
                "end_time": float(start_time + duration),
                "duration": float(duration),
            })
            self.vad.pop()
        return speech_detected, segments

    def reset(self):
        self.vad.reset()
        self.audio_start_time = None


class SenseVoiceDetector:
    def __init__(self):
        self.sample_rate = cfg.SAMPLE_RATE
        model = cfg.SENSEVOICE_MODEL_DIR / "model.int8.onnx"
        tokens = cfg.SENSEVOICE_MODEL_DIR / "tokens.txt"
        if not model.is_file() or not tokens.is_file():
            raise FileNotFoundError(cfg.SENSEVOICE_MODEL_DIR)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(tokens),
            num_threads=cfg.ASR_NUM_THREADS,
            sample_rate=self.sample_rate,
            language=cfg.ASR_LANGUAGE,
            use_itn=cfg.ASR_USE_ITN,
            provider="cpu",
            debug=False,
        )

    @staticmethod
    def normalize_label(label):
        label = str(label or "").strip()
        if label.startswith("<|") and label.endswith("|>"):
            label = label[2:-2]
        return label.lower() if label else None

    def recognize(self, samples):
        start = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self.recognizer.decode_stream(stream)
        result = stream.result
        inference_time = time.perf_counter() - start
        duration = len(samples) / self.sample_rate
        text = str(result.text or "").strip()
        return {
            "text": text if text else None,
            "language": self.normalize_label(result.lang),
            "emotion": self.normalize_label(result.emotion),
            "event": self.normalize_label(result.event),
            "tokens": [str(token) for token in result.tokens],
            "timestamps": [float(value) for value in result.timestamps],
            "confidence": None,
            "speech_state": "transcribed" if text else "untranscribed",
            "inference_ms": float(inference_time * 1000.0),
            "rtf": float(inference_time / duration) if duration > 0 else None,
        }


class YAMNetDetector:
    def __init__(self):
        self.sample_rate = cfg.SAMPLE_RATE
        self.window_size = cfg.YAMNET_WINDOW_SAMPLES
        self.hop_size = cfg.YAMNET_HOP_SAMPLES
        self.buffer = np.empty(0, dtype=np.float32)
        self.buffer_start_time = None
        if not cfg.YAMNET_MODEL.is_file():
            raise FileNotFoundError(cfg.YAMNET_MODEL)
        options = mp.tasks.audio.AudioClassifierOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(cfg.YAMNET_MODEL)
            ),
            running_mode=mp.tasks.audio.RunningMode.AUDIO_CLIPS,
            max_results=-1,
        )
        self.classifier = mp.tasks.audio.AudioClassifier.create_from_options(options)

    def detect(self, samples, timestamp):
        if self.buffer_start_time is None:
            self.buffer_start_time = float(timestamp)
        self.buffer = np.concatenate((self.buffer, samples))
        results = []
        while len(self.buffer) >= self.window_size:
            window = self.buffer[:self.window_size]
            start_time = self.buffer_start_time
            end_time = start_time + self.window_size / self.sample_rate
            inference_start = time.perf_counter()
            audio_data = mp.tasks.components.containers.AudioData.create_from_array(
                window, self.sample_rate
            )
            classifications = self.classifier.classify(audio_data)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            all_events = []
            for result in classifications:
                for classification in result.classifications:
                    for category in classification.categories:
                        all_events.append({
                            "index": int(category.index),
                            "label": str(category.category_name),
                            "score": float(category.score),
                        })
            all_events.sort(key=lambda event: event["score"], reverse=True)
            by_index = {event["index"]: event for event in all_events}
            selected = []
            for index, expected_label in cfg.YAMNET_EVENT_CLASSES.items():
                event = by_index.get(index)
                if event is None or event["label"] != expected_label:
                    continue
                threshold = cfg.YAMNET_EVENT_THRESHOLDS[expected_label]
                if event["score"] >= threshold:
                    selected.append({
                        **event,
                        "display_name": cfg.YAMNET_EVENT_LABELS_ZH.get(
                            expected_label, expected_label
                        ),
                        "threshold": float(threshold),
                    })
            selected.sort(key=lambda event: event["score"], reverse=True)
            results.append({
                "start_time": float(start_time),
                "end_time": float(end_time),
                "timestamp": float(end_time),
                "class_count": int(len(all_events)),
                "sound_events": selected,
                "raw_top": all_events[:cfg.YAMNET_RAW_TOP_COUNT],
                "inference_ms": float(inference_ms),
            })
            self.buffer = self.buffer[self.hop_size:].copy()
            self.buffer_start_time += self.hop_size / self.sample_rate
        return results

    def reset(self):
        self.buffer = np.empty(0, dtype=np.float32)
        self.buffer_start_time = None

    def close(self):
        self.classifier.close()


class AudioDetector:
    """Own one microphone and schedule VAD, ASR and YAMNet processing."""

    def __init__(
        self,
        microphone=None,
        vad_detector=None,
        asr_detector=None,
        yamnet_detector=None,
        start_microphone=True,
    ):
        self.vad_detector = vad_detector or SileroVadDetector()
        self.asr_detector = asr_detector or SenseVoiceDetector()
        self.yamnet_detector = yamnet_detector or YAMNetDetector()
        self.microphone = microphone
        if self.microphone is None and start_microphone:
            try:
                self.microphone = Microphone(
                    device=cfg.MICROPHONE_DEVICE,
                    sample_rate=cfg.SAMPLE_RATE,
                    block_size=cfg.BLOCK_SIZE,
                    queue_size=cfg.MICROPHONE_QUEUE_SIZE,
                )
            except Exception:
                if hasattr(self.yamnet_detector, "close"):
                    self.yamnet_detector.close()
                raise
        self.last_timestamp = None
        self.last_sample_count = None
        self.last_dropped_blocks = 0
        self.last_input_overflows = 0
        self.reset_count = 0
        self.stream_available = False
        self.pending_segments = []
        self.asr_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sensevoice-asr"
        )
        self.asr_future = None
        self.asr_future_generation = None
        self.stream_generation = 0
        self.asr_failure_count = 0
        self.last_asr_error = None
        self.dropped_asr_segments = 0
        self.closed = False

    def _buffer_status(self):
        if self.microphone is None:
            return {
                "queued_blocks": 0,
                "dropped_blocks": 0,
                "input_overflows": 0,
            }
        return self.microphone.status()

    def _reset_stream(self):
        if hasattr(self.vad_detector, "reset"):
            self.vad_detector.reset()
        if hasattr(self.yamnet_detector, "reset"):
            self.yamnet_detector.reset()
        self.pending_segments.clear()
        self.stream_generation += 1
        self.last_timestamp = None
        self.last_sample_count = None
        self.reset_count += 1
        self.stream_available = False

    def reset(self):
        self._reset_stream()

    def read(self, timeout=cfg.READ_TIMEOUT_SECONDS):
        if self.microphone is None:
            raise RuntimeError("AudioDetector has no microphone")
        completed = self._poll_async_asr()
        samples, timestamp = self.microphone.read(timeout=timeout)
        status = self._buffer_status()
        status["dropped_asr_segments"] = int(self.dropped_asr_segments)
        status["asr_failure_count"] = int(self.asr_failure_count)
        status["last_asr_error"] = self.last_asr_error
        if samples is None:
            if self.stream_available:
                self._reset_stream()
                for item in completed:
                    item["valid_for_current_stream"] = False
            return self._unavailable_result(
                time.perf_counter(), status, "read_timeout", completed
            )
        result = self.process_block(samples, timestamp, status, completed)
        self._schedule_async_asr()
        result["buffer_status"]["dropped_asr_segments"] = int(
            self.dropped_asr_segments
        )
        result["buffer_status"]["asr_failure_count"] = int(
            self.asr_failure_count
        )
        result["buffer_status"]["last_asr_error"] = self.last_asr_error
        processing = bool(self.asr_future is not None or self.pending_segments)
        result["processing"] = processing
        if processing and not result["transcriptions"]:
            result["speech_state"] = "processing"
        return result

    def process_external_block(self, samples, timestamp):
        """Process a clocked media block without opening or reading a microphone."""
        completed = self._poll_async_asr()
        status = self._buffer_status()
        status.update({
            "dropped_asr_segments": int(self.dropped_asr_segments),
            "asr_failure_count": int(self.asr_failure_count),
            "last_asr_error": self.last_asr_error,
        })
        result = self.process_block(samples, timestamp, status, completed)
        self._schedule_async_asr()
        processing = bool(self.asr_future is not None or self.pending_segments)
        result["processing"] = processing
        if processing and not result["transcriptions"]:
            result["speech_state"] = "processing"
        return result

    def _unavailable_result(self, timestamp, status, reason, transcriptions=None):
        return {
            "data_available": False,
            "samples": None,
            "timestamp": float(timestamp),
            "block_start_time": None,
            "block_end_time": None,
            "speech_detected": False,
            "speech_segments": [],
            "transcriptions": transcriptions or [],
            "sound_results": [],
            "speech_state": "unavailable",
            "processing": False,
            "discontinuity": True,
            "discontinuity_reasons": [reason],
            "reset_count": int(self.reset_count),
            "inference_ms": {"vad": None, "asr": None, "yamnet": None},
            "buffer_status": status,
            "processed_time": float(time.perf_counter()),
        }

    def _recognize_pending(self):
        completed = []
        pending, self.pending_segments = self.pending_segments, []
        for segment in pending:
            recognition = self.asr_detector.recognize(segment["samples"])
            recognition.update({
                "start_time": float(segment["start_time"]),
                "end_time": float(segment["end_time"]),
                "duration": float(segment["duration"]),
                "observed_time": float(segment["end_time"]),
                "produced_time": float(time.perf_counter()),
                "stream_generation": int(self.stream_generation),
                "valid_for_current_stream": True,
            })
            completed.append(recognition)
        return completed

    def _recognize_segment(self, segment, stream_generation):
        recognition = self.asr_detector.recognize(segment["samples"])
        recognition.update({
            "start_time": float(segment["start_time"]),
            "end_time": float(segment["end_time"]),
            "duration": float(segment["duration"]),
            "observed_time": float(segment["end_time"]),
            "produced_time": float(time.perf_counter()),
            "stream_generation": int(stream_generation),
        })
        return recognition

    def _schedule_async_asr(self):
        if self.asr_future is None and self.pending_segments:
            segment = self.pending_segments.pop(0)
            self.asr_future_generation = self.stream_generation
            self.asr_future = self.asr_executor.submit(
                self._recognize_segment, segment, self.asr_future_generation
            )

    def _poll_async_asr(self):
        completed = []
        if self.asr_future is not None and self.asr_future.done():
            try:
                recognition = self.asr_future.result()
                recognition["valid_for_current_stream"] = bool(
                    self.asr_future_generation == self.stream_generation
                )
                completed.append(recognition)
            except Exception as error:
                self.asr_failure_count += 1
                self.last_asr_error = f"{type(error).__name__}: {error}"
            self.asr_future = None
            self.asr_future_generation = None
        self._schedule_async_asr()
        return completed

    def process_block(
        self, samples, timestamp, buffer_status=None,
        completed_transcriptions=None,
    ):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(samples) == 0 or not np.isfinite(samples).all():
            self._reset_stream()
            return self._unavailable_result(
                time.perf_counter(), buffer_status or self._buffer_status(),
                "invalid_samples",
            )
        timestamp = float(timestamp)
        status = buffer_status or self._buffer_status()
        reasons = []
        if self.last_timestamp is not None:
            expected = self.last_timestamp + self.last_sample_count / cfg.SAMPLE_RATE
            if abs(timestamp - expected) > cfg.TIMESTAMP_TOLERANCE_SECONDS:
                reasons.append("timestamp_gap")
            if status["dropped_blocks"] > self.last_dropped_blocks:
                reasons.append("dropped_blocks")
            if status["input_overflows"] > self.last_input_overflows:
                reasons.append("input_overflow")
        if reasons:
            self._reset_stream()
            for item in completed_transcriptions or []:
                item["valid_for_current_stream"] = False

        vad_start = time.perf_counter()
        speech_detected, segments = self.vad_detector.detect(samples, timestamp)
        vad_ms = (time.perf_counter() - vad_start) * 1000.0
        sound_results = self.yamnet_detector.detect(samples, timestamp)
        transcriptions = list(completed_transcriptions or [])
        asr_total_ms = sum(
            item.get("inference_ms", 0.0) for item in transcriptions
        )
        segment_metadata = []
        for segment in segments:
            if len(self.pending_segments) < cfg.ASR_PENDING_SEGMENTS_MAX:
                self.pending_segments.append(segment)
            else:
                self.dropped_asr_segments += 1
            segment_metadata.append({
                "start_time": float(segment["start_time"]),
                "end_time": float(segment["end_time"]),
                "duration": float(segment["duration"]),
            })

        if self.pending_segments:
            speech_state = "processing"
        elif transcriptions:
            speech_state = transcriptions[-1]["speech_state"]
        elif speech_detected:
            speech_state = "speaking"
        else:
            speech_state = "no_speech"
        self.last_timestamp = timestamp
        self.last_sample_count = len(samples)
        self.last_dropped_blocks = int(status["dropped_blocks"])
        self.last_input_overflows = int(status["input_overflows"])
        self.stream_available = True
        return {
            "data_available": True,
            "samples": samples,
            "sample_rate": cfg.SAMPLE_RATE,
            "timestamp": timestamp,
            "block_start_time": timestamp,
            "block_end_time": float(timestamp + len(samples) / cfg.SAMPLE_RATE),
            "speech_detected": bool(speech_detected),
            "speech_segments": segment_metadata,
            "transcriptions": transcriptions,
            "sound_results": sound_results,
            "speech_state": speech_state,
            "processing": bool(self.pending_segments),
            "processing_occurred": bool(segments),
            "discontinuity": bool(reasons),
            "discontinuity_reasons": reasons,
            "reset_count": int(self.reset_count),
            "inference_ms": {
                "vad": float(vad_ms),
                "asr": float(asr_total_ms) if transcriptions else None,
                "yamnet": (
                    float(sum(item["inference_ms"] for item in sound_results))
                    if sound_results else None
                ),
            },
            "buffer_status": status,
            "processed_time": float(time.perf_counter()),
        }

    def close(self):
        if self.closed:
            return
        self.asr_executor.shutdown(wait=True, cancel_futures=True)
        if hasattr(self.yamnet_detector, "close"):
            self.yamnet_detector.close()
        if self.microphone is not None:
            self.microphone.release()
        self.closed = True
