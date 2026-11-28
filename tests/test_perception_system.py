import asyncio
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import system_config as cfg
from src.system.perception_system import PerceptionSystem, RateMeter
from web.app import create_app


class ClosedResource:
    def __init__(self):
        self.closed = False
        self.released = False

    def close(self):
        self.closed = True

    def release(self):
        self.released = True


class PerceptionSystemTests(unittest.TestCase):
    def test_pose_head_roi_uses_head_points_and_shoulders(self):
        points = [SimpleNamespace(
            x=0.5, y=0.3, visibility=0.95, presence=0.95
        ) for _ in range(33)]
        for index, x, y in (
            (0, .50, .28), (1, .47, .26), (2, .46, .26),
            (4, .53, .26), (5, .54, .26), (7, .43, .28),
            (8, .57, .28), (9, .48, .33), (10, .52, .33),
            (11, .38, .48), (12, .62, .48),
        ):
            points[index].x, points[index].y = x, y
        result = SimpleNamespace(pose_landmarks=[points])
        roi = PerceptionSystem._pose_head_roi(result, (480, 640, 3), 7, 1.25)
        self.assertIsNotNone(roi)
        self.assertEqual(roi["sequence"], 7)
        self.assertGreaterEqual(roi["reliability"], 0.9)
        x1, y1, x2, y2 = roi["bbox_pixels"]
        self.assertLess(x1, 320)
        self.assertGreater(x2, 320)
        self.assertLess(y1, 160)
        self.assertGreater(y2, 160)

    def test_waveform_compression_preserves_extrema_and_limit(self):
        samples = np.sin(np.linspace(0.0, 20.0 * np.pi, 4000)).astype(np.float32)
        compressed = PerceptionSystem.compress_waveform(samples, max_points=100)
        self.assertLessEqual(len(compressed), 100)
        self.assertGreater(max(compressed), 0.95)
        self.assertLess(min(compressed), -0.95)

    def test_snapshot_is_json_safe_and_discards_stale_analysis(self):
        system = PerceptionSystem()
        now = time.perf_counter()
        system.records["pose"] = {
            "analysis": {"score": np.float32(0.5), "bad": np.nan},
            "captured_at": now,
            "generated_at": now,
            "inference_ms": np.float64(2.5),
        }
        fresh = system.snapshot()
        self.assertTrue(fresh["pose"]["available"])
        self.assertEqual(fresh["pose"]["analysis"]["score"], 0.5)
        self.assertIsNone(fresh["pose"]["analysis"]["bad"])
        json.dumps(fresh, allow_nan=False)

        system.records["pose"]["generated_at"] = now - cfg.POSE_RESULT_TTL_SECONDS - 1.0
        stale = system.snapshot()
        self.assertFalse(stale["pose"]["available"])
        self.assertTrue(stale["pose"]["stale"])
        self.assertIsNone(stale["pose"]["analysis"])

    def test_lighting_validation_history_deduplication_and_bound(self):
        system = PerceptionSystem()
        state = system.set_manual_lux(420)
        self.assertEqual(state["current_lux"], 420)
        original_count = len(system.history)
        system.set_manual_lux(420)
        self.assertEqual(len(system.history), original_count)

        system.set_lighting_mode("automatic")
        with self.assertRaises(RuntimeError):
            system.set_manual_lux(300)
        with self.assertRaises(ValueError):
            system.set_lighting_mode("invalid")
        with self.assertRaises(ValueError):
            system.set_manual_lux(cfg.MAX_SIMULATED_LUX + 1)

        for index in range(cfg.HISTORY_LIMIT + 10):
            system.add_event("test", f"event-{index}")
        self.assertEqual(len(system.history), cfg.HISTORY_LIMIT)

    def test_shutdown_releases_each_owned_resource(self):
        system = PerceptionSystem()
        camera = ClosedResource()
        pose = ClosedResource()
        face = ClosedResource()
        audio = ClosedResource()
        system.camera = camera
        system.pose_detector = pose
        system.face_detector = face
        system.audio_detector = audio
        system.running = True
        system.status = "running"

        system.stop()

        self.assertTrue(camera.released)
        self.assertTrue(pose.closed)
        self.assertTrue(face.closed)
        self.assertTrue(audio.closed)
        self.assertEqual(system.status, "stopped")

    def test_hardware_initialization_failure_enters_degraded_mode(self):
        calls = {"camera": 0, "audio": 0}

        def camera_factory(**_kwargs):
            calls["camera"] += 1
            raise RuntimeError("camera unavailable")

        def audio_factory():
            calls["audio"] += 1
            raise RuntimeError("audio unavailable")

        system = PerceptionSystem(
            camera_factory=camera_factory,
            audio_detector_factory=audio_factory,
        )
        system.start()
        try:
            self.assertEqual(system.status, "degraded")
            self.assertEqual(calls, {"camera": 1, "audio": 1})
            snapshot = system.snapshot()
            self.assertFalse(snapshot["pose"]["available"])
            self.assertFalse(snapshot["face"]["available"])
            self.assertFalse(snapshot["audio"]["available"])
            self.assertIn("camera", snapshot["system"]["errors"])
            json.dumps(snapshot, allow_nan=False)
        finally:
            system.stop()

    def test_rate_meter_reports_measured_rate(self):
        meter = RateMeter(window_seconds=2.0)
        for timestamp in (1.0, 1.1, 1.2, 1.3):
            meter.tick(timestamp)
        self.assertAlmostEqual(meter.value(1.3), 10.0, places=6)

    def test_web_routes_share_the_injected_system(self):
        system = PerceptionSystem()
        app = create_app(system, manage_lifespan=False)
        self.assertIs(app.state.perception_system, system)
        status_route = next(
            route for route in app.routes if getattr(route, "path", None) == "/api/status"
        )
        first = asyncio.run(status_route.endpoint())
        second = asyncio.run(status_route.endpoint())
        self.assertEqual(first["system"]["status"], "stopped")
        self.assertEqual(second["lighting"], first["lighting"])

    def test_start_creates_one_camera_and_one_of_each_video_model(self):
        calls = {"camera": 0, "pose": 0, "face": 0}

        class FailingCamera(ClosedResource):
            def read(self):
                raise RuntimeError("end test capture")

        class Detector(ClosedResource):
            last_timestamp_ms = -1

        def camera_factory(**_kwargs):
            calls["camera"] += 1
            return FailingCamera()

        def pose_factory():
            calls["pose"] += 1
            return Detector()

        def face_factory():
            calls["face"] += 1
            return Detector()

        def audio_factory():
            raise RuntimeError("audio disabled for test")

        system = PerceptionSystem(
            camera_factory=camera_factory,
            pose_detector_factory=pose_factory,
            face_detector_factory=face_factory,
            audio_detector_factory=audio_factory,
            pose_analyzer_factory=lambda: object(),
            face_analyzer_factory=lambda: object(),
        )
        system.start()
        try:
            time.sleep(0.02)
            self.assertEqual(calls, {"camera": 1, "pose": 1, "face": 1})
        finally:
            system.stop()


    def test_observation_time_controls_freshness_and_reports_ages(self):
        system = PerceptionSystem()
        now = time.perf_counter()
        system.records["face"] = {
            "analysis": {"face_detected": True, "reliability": {"reliable": True}},
            "captured_at": now - cfg.FACE_RESULT_TTL_SECONDS - 0.2,
            "generated_at": now,
            "inference_ms": 3.0,
        }
        result = system.snapshot()["face"]
        self.assertFalse(result["available"])
        self.assertTrue(result["stale"])
        self.assertGreater(result["observation_age"], cfg.FACE_RESULT_TTL_SECONDS)
        self.assertLess(result["result_age"], 0.1)
        self.assertGreater(result["processing_latency"], cfg.FACE_RESULT_TTL_SECONDS)
        self.assertIsNone(result["analysis"])

    def test_missing_target_is_not_an_available_observation(self):
        system = PerceptionSystem()
        now = time.perf_counter()
        system.records["pose"] = {
            "analysis": {"person_detected": False, "reliability": {"overall": 0.0}},
            "captured_at": now,
            "generated_at": now,
            "inference_ms": 2.0,
        }
        result = system.snapshot()["pose"]
        self.assertTrue(result["available"])
        self.assertFalse(result["target_detected"])
        self.assertFalse(result["observation_available"])

    def test_waveform_envelope_frequency_and_actual_duration(self):
        samples = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        envelope = PerceptionSystem.compress_waveform_envelope(samples, max_bins=25)
        self.assertEqual(len(envelope), 25)
        self.assertTrue(all(len(pair) == 2 and pair[0] <= pair[1] for pair in envelope))

        system = PerceptionSystem()
        system.last_waveform_update = 10.0
        self.assertFalse(system._waveform_update_due(10.05))
        self.assertTrue(system._waveform_update_due(10.101))
        self.assertTrue(system._waveform_update_due(10.01, discontinuity=True))
        system.compressed_waveform = envelope
        system.waveform_duration = 0.5
        system.waveform_end_time = time.perf_counter()
        waveform = system.snapshot()["waveform"]
        self.assertAlmostEqual(waveform["duration_seconds"], 0.5)
        self.assertEqual(waveform["envelope"], envelope)

    def test_same_video_sequence_is_not_returned_twice(self):
        system = PerceptionSystem()
        system.latest_frame = (
            7, time.perf_counter(), np.zeros((2, 2, 3), dtype=np.uint8)
        )
        self.assertIsNotNone(system._get_latest_frame(6))
        self.assertIsNone(system._get_latest_frame(7))

    def test_audio_partial_initialization_is_cleaned_up(self):
        audio = ClosedResource()

        def unavailable_camera(**_kwargs):
            raise RuntimeError("camera disabled")

        def failing_analyzer():
            raise RuntimeError("analyzer failed")

        system = PerceptionSystem(
            camera_factory=unavailable_camera,
            audio_detector_factory=lambda: audio,
            audio_analyzer_factory=failing_analyzer,
        )
        system.start()
        try:
            self.assertTrue(audio.closed)
            self.assertIsNone(system.audio_detector)
            self.assertIn("audio", system.errors)
        finally:
            system.stop()

    def test_stop_timeout_does_not_close_live_thread_resources(self):
        system = PerceptionSystem()
        release = threading.Event()
        worker = threading.Thread(target=release.wait, name="blocked-test-worker")
        worker.start()
        camera = ClosedResource()
        system.threads = [worker]
        system.camera = camera
        system.running = True
        system.status = "running"
        try:
            with patch.object(cfg, "THREAD_JOIN_TIMEOUT_SECONDS", 0.01):
                self.assertFalse(system.stop())
            self.assertEqual(system.status, "stop_timeout")
            self.assertFalse(camera.released)
        finally:
            release.set()
            worker.join(1.0)
            with patch.object(cfg, "THREAD_JOIN_TIMEOUT_SECONDS", 0.1):
                system.stop()


if __name__ == "__main__":
    unittest.main()
