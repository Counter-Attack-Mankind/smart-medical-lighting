import json
import threading
import unittest

import numpy as np

from config import audio_config as cfg
from src.audio.audio_analyzer import AudioAnalyzer
from src.audio.audio_detector import AudioDetector


def detection(
    samples=None,
    timestamp=0.0,
    speech=False,
    transcriptions=None,
    sound_results=None,
    available=True,
    discontinuity=False,
    processing=False,
):
    samples = (
        np.zeros(cfg.BLOCK_SIZE, dtype=np.float32)
        if samples is None and available else samples
    )
    duration = len(samples) / cfg.SAMPLE_RATE if samples is not None else 0.0
    return {
        "data_available": available,
        "samples": samples,
        "sample_rate": cfg.SAMPLE_RATE,
        "timestamp": timestamp,
        "block_start_time": timestamp if available else None,
        "block_end_time": timestamp + duration if available else None,
        "speech_detected": speech,
        "speech_segments": [],
        "transcriptions": transcriptions or [],
        "sound_results": sound_results or [],
        "speech_state": "speaking" if speech else "no_speech",
        "processing": processing,
        "discontinuity": discontinuity,
        "discontinuity_reasons": ["test_gap"] if discontinuity else [],
        "reset_count": int(discontinuity),
        "inference_ms": {"vad": 0.1, "asr": None, "yamnet": None},
        "buffer_status": {
            "queued_blocks": 0,
            "dropped_blocks": 0,
            "input_overflows": 0,
        },
        "processed_time": timestamp + duration,
    }


class FakeVad:
    def __init__(self):
        self.reset_calls = 0

    def detect(self, samples, timestamp):
        return False, []

    def reset(self):
        self.reset_calls += 1


class FakeAsr:
    def recognize(self, samples):
        raise AssertionError("No segment should reach ASR in this test")


class SegmentVad(FakeVad):
    def __init__(self):
        super().__init__()
        self.emitted = False

    def detect(self, samples, timestamp):
        if self.emitted:
            return False, []
        self.emitted = True
        return False, [{
            "samples": samples.copy(),
            "start_time": timestamp,
            "end_time": timestamp + len(samples) / cfg.SAMPLE_RATE,
            "duration": len(samples) / cfg.SAMPLE_RATE,
        }]


class WorkingFakeAsr:
    def recognize(self, samples):
        return {
            "text": "测试",
            "language": "zh",
            "emotion": "neutral",
            "event": "speech",
            "tokens": ["测", "试"],
            "timestamps": [0.0, 0.1],
            "confidence": None,
            "speech_state": "transcribed",
            "inference_ms": 1.0,
            "rtf": 0.01,
        }


class FakeYamnet:
    def __init__(self):
        self.buffered = 0
        self.reset_calls = 0

    def detect(self, samples, timestamp):
        self.buffered += len(samples)
        return []

    def reset(self):
        self.buffered = 0
        self.reset_calls += 1

    def close(self):
        pass


class FakeMicrophone:
    def __init__(self, block_count=3):
        duration = cfg.BLOCK_SIZE / cfg.SAMPLE_RATE
        self.blocks = [
            (np.zeros(cfg.BLOCK_SIZE, dtype=np.float32), index * duration)
            for index in range(block_count)
        ]

    def read(self, timeout=None):
        return self.blocks.pop(0) if self.blocks else (None, None)

    def status(self):
        return {"queued_blocks": len(self.blocks), "dropped_blocks": 0, "input_overflows": 0}

    def release(self):
        pass


class BlockingFakeAsr(WorkingFakeAsr):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def recognize(self, samples):
        self.started.set()
        self.release.wait(1.0)
        result = super().recognize(samples)
        self.finished.set()
        return result


class TestAudioAnalyzer(unittest.TestCase):
    def test_sine_rms_dbfs_and_f0(self):
        time_axis = np.arange(cfg.SAMPLE_RATE) / cfg.SAMPLE_RATE
        samples = 0.1 * np.sin(2.0 * np.pi * 200.0 * time_axis)
        rms = AudioAnalyzer.rms(samples)
        self.assertAlmostEqual(rms, 0.1 / np.sqrt(2.0), places=5)
        self.assertAlmostEqual(AudioAnalyzer.dbfs(rms), -23.0103, places=3)
        self.assertAlmostEqual(
            AudioAnalyzer.estimate_f0(samples), 200.0, delta=2.0
        )

    def test_silence_aperiodic_and_missing_data(self):
        silence = np.zeros(cfg.SAMPLE_RATE, dtype=np.float32)
        noise = np.random.default_rng(4).normal(
            0.0, 0.1, cfg.SAMPLE_RATE
        ).astype(np.float32)
        self.assertEqual(AudioAnalyzer.rms(silence), 0.0)
        self.assertEqual(AudioAnalyzer.dbfs(0.0), cfg.DBFS_FLOOR)
        self.assertIsNone(AudioAnalyzer.estimate_f0(silence))
        self.assertIsNone(AudioAnalyzer.estimate_f0(noise))

        analyzer = AudioAnalyzer()
        output = analyzer.analyze(analyzer.update(detection(
            samples=None, timestamp=1.0, available=False
        )))
        self.assertEqual(output["speech_state"], "unavailable")
        self.assertIsNone(output["rms"])
        self.assertIsNone(output["f0"])

    def test_speech_pause_ratios_and_speech_rate(self):
        analyzer = AudioAnalyzer()
        block_duration = cfg.BLOCK_SIZE / cfg.SAMPLE_RATE
        for index in range(10):
            analyzer.update(detection(
                timestamp=index * block_duration,
                speech=index < 5,
            ))
        output = analyzer.analyze()
        self.assertAlmostEqual(output["speech_activity_ratio"], 0.5, places=6)
        self.assertAlmostEqual(output["pause_ratio"], 0.5, places=6)
        rate, count = AudioAnalyzer.speech_rate("你好，世界！abc test", 2.0)
        self.assertEqual(count, 6)
        self.assertAlmostEqual(rate, 180.0)

    def test_speech_states_and_history_freshness(self):
        analyzer = AudioAnalyzer()
        transcript = {
            "text": "你好世界",
            "language": "zh",
            "emotion": "neutral",
            "event": "speech",
            "tokens": ["你", "好", "世", "界"],
            "timestamps": [0.0, 0.2, 0.4, 0.6],
            "confidence": None,
            "speech_state": "transcribed",
            "inference_ms": 10.0,
            "rtf": 0.01,
            "start_time": 0.0,
            "end_time": 2.0,
            "observed_time": 2.0,
            "produced_time": 2.01,
            "duration": 2.0,
        }
        output = analyzer.analyze(analyzer.update(detection(
            timestamp=2.0, speech=False, transcriptions=[transcript]
        )))
        self.assertEqual(output["speech_state"], "transcribed")
        self.assertEqual(output["transcription"]["content_unit_count"], 4)
        self.assertAlmostEqual(output["transcription"]["speech_rate"], 120.0)
        self.assertTrue(output["transcription"]["is_current"])

        output = analyzer.analyze(analyzer.update(detection(
            timestamp=2.1, processing=True
        )))
        self.assertEqual(output["speech_state"], "processing")

        output = analyzer.analyze(analyzer.update(detection(
            timestamp=2.2, speech=False
        )))
        self.assertEqual(output["speech_state"], "no_speech")
        self.assertEqual(output["transcription"]["text"], "你好世界")

        output = analyzer.analyze(analyzer.update(detection(
            timestamp=20.0, speech=False, discontinuity=True
        )))
        self.assertFalse(output["transcription"]["is_current"])

    def test_detector_gap_resets_model_caches(self):
        vad, yamnet = FakeVad(), FakeYamnet()
        detector = AudioDetector(
            vad_detector=vad,
            asr_detector=FakeAsr(),
            yamnet_detector=yamnet,
            start_microphone=False,
        )
        samples = np.zeros(cfg.BLOCK_SIZE, dtype=np.float32)
        detector.process_block(samples, 0.0)
        detector.process_block(samples, cfg.BLOCK_SIZE / cfg.SAMPLE_RATE)
        result = detector.process_block(samples, 0.20)
        self.assertTrue(result["discontinuity"])
        self.assertIn("timestamp_gap", result["discontinuity_reasons"])
        self.assertEqual(vad.reset_calls, 1)
        self.assertEqual(yamnet.reset_calls, 1)
        self.assertEqual(yamnet.buffered, cfg.BLOCK_SIZE)
        detector.close()

    def test_detector_processing_then_transcribed(self):
        detector = AudioDetector(
            vad_detector=SegmentVad(),
            asr_detector=WorkingFakeAsr(),
            yamnet_detector=FakeYamnet(),
            start_microphone=False,
        )
        samples = np.zeros(cfg.BLOCK_SIZE, dtype=np.float32)
        first = detector.process_block(samples, 0.0)
        self.assertTrue(first["processing"])
        self.assertEqual(first["speech_state"], "processing")
        completed = detector._recognize_pending()
        second = detector.process_block(
            samples,
            cfg.BLOCK_SIZE / cfg.SAMPLE_RATE,
            completed_transcriptions=completed,
        )
        self.assertFalse(second["processing"])
        self.assertEqual(second["speech_state"], "transcribed")
        self.assertEqual(second["transcriptions"][0]["text"], "测试")
        detector.close()

    def test_detector_read_keeps_processing_while_asr_runs(self):
        asr = BlockingFakeAsr()
        detector = AudioDetector(
            microphone=FakeMicrophone(),
            vad_detector=SegmentVad(),
            asr_detector=asr,
            yamnet_detector=FakeYamnet(),
            start_microphone=False,
        )
        try:
            first = detector.read()
            self.assertTrue(first["processing"])
            self.assertTrue(asr.started.wait(0.5))

            second = detector.read()
            self.assertTrue(second["processing"])
            self.assertFalse(second["transcriptions"])

            asr.release.set()
            self.assertTrue(asr.finished.wait(0.5))
            third = detector.read()
            self.assertEqual(third["transcriptions"][0]["text"], "测试")
        finally:
            asr.release.set()
            detector.close()

    def test_strict_json_output(self):
        analyzer = AudioAnalyzer()
        block_duration = cfg.BLOCK_SIZE / cfg.SAMPLE_RATE
        for index in range(25):
            offset = index * cfg.BLOCK_SIZE
            time_axis = (offset + np.arange(cfg.BLOCK_SIZE)) / cfg.SAMPLE_RATE
            samples = (0.05 * np.sin(2 * np.pi * 180.0 * time_axis)).astype(np.float32)
            frame = analyzer.update(detection(
                samples=samples,
                timestamp=index * block_duration,
                speech=True,
            ))
        output = analyzer.analyze(frame)
        self.assertIsNotNone(output["median_f0"])
        json.dumps(output, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
