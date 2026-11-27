import asyncio
import json
import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
