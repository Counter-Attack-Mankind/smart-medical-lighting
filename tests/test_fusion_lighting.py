import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.system.fusion_analyzer import FusionAnalyzer
from src.system.lighting_controller import LightingController
from src.system.media_source import MediaFileSource


def module(captured_at, analysis):
    return {
        "available": True,
        "captured_at": captured_at,
        "observation_age": 0.0,
        "analysis": {**analysis, "reliability": {"score": 1.0, "reliable": True}},
    }


class FusionLightingTests(unittest.TestCase):
    def test_dimension_specific_weighting_and_json(self):
        analyzer = FusionAnalyzer()
        pose = module(1.0, {"motion_state": "high_activity"})
        face = module(1.0, {"stable_expression": "happy", "expression_scores": {"happy": 1.0}})
        output = analyzer.analyze(pose, face, None, 1.0)
        self.assertAlmostEqual(output["valence"], 0.8)
        self.assertAlmostEqual(output["arousal"], (0.35 * 0.7 + 0.25 * 0.3) / 0.60)
        self.assertEqual(output["effective_weights"]["valence"]["face"], 1.0)
        json.dumps(output, allow_nan=False)

    def test_duplicate_and_snapshot_do_not_confirm(self):
        analyzer = FusionAnalyzer()
        pose = module(1.0, {"motion_state": "high_activity"})
        face = module(1.0, {"stable_expression": "happy", "expression_scores": {"happy": 1.0}})
        first = analyzer.analyze(pose, face, None, 1.0, temporal_active=False)
        repeated = analyzer.analyze(pose, face, None, 20.0, temporal_active=False)
        self.assertTrue(first["new_evidence"])
        self.assertFalse(repeated["new_evidence"])
        self.assertEqual(repeated["confirmed_state"], "unknown")

    def test_temporal_confirmation_uses_new_observations(self):
        analyzer = FusionAnalyzer()
        output = None
        for second in range(5):
            pose = module(float(second), {"motion_state": "high_activity"})
            face = module(float(second), {"stable_expression": "happy", "expression_scores": {"happy": 1.0}})
            output = analyzer.analyze(pose, face, None, float(second))
        self.assertEqual(output["state_candidate"], "pleasant_active")
        self.assertEqual(output["confirmed_state"], "pleasant_active")
        self.assertEqual(output["status"], "confirmed")

    def test_semantic_request_and_rate_limited_lighting(self):
        analyzer = FusionAnalyzer()
        audio = module(1.0, {"transcription": {"text": "\u6211\u60f3\u4f11\u606f\uff0c\u8bf7\u6697\u4e00\u70b9"}, "expression_intensity": "normal_intensity"})
        fusion = analyzer.analyze(None, None, audio, 1.0)
        self.assertTrue(fusion["semantic"]["rest_intent"])
        self.assertEqual(fusion["semantic"]["lighting_request"], "dimmer")
        controller = LightingController()
        controller.set_mode("automatic")
        controller.update(fusion, 1.0)
        state = controller.update(fusion, 2.0)
        self.assertEqual(state["target_lux"], 200.0)
        self.assertEqual(state["current_lux"], 285.0)

    def test_sensitive_semantics_and_negation(self):
        sensitive = module(1.0, {"transcription": {"text": "我不想活了", "observed_time": 1.0}})
        _score, semantic = FusionAnalyzer._content(sensitive)
        self.assertEqual(semantic["alerts"][0]["category"], "self_harm_suicide")
        self.assertFalse(semantic["alerts"][0]["affects_lighting"])
        self.assertIn("不想活了", semantic["matched_phrases"])

        negated = module(2.0, {"transcription": {"text": "我不想死，也不会自杀", "observed_time": 2.0}})
        _score, semantic = FusionAnalyzer._content(negated)
        self.assertEqual(semantic["alerts"], [])

        reported = module(3.0, {"transcription": {"text": "新闻里他说想死", "observed_time": 3.0}})
        _score, semantic = FusionAnalyzer._content(reported)
        self.assertEqual(semantic["alerts"], [])

        threat = module(4.0, {"transcription": {"text": "我要跟你拼了", "observed_time": 4.0}})
        score, semantic = FusionAnalyzer._content(threat)
        self.assertEqual(semantic["alerts"][0]["category"], "threat_conflict")
        self.assertEqual(score, (None, None))

        no_light = module(5.0, {"transcription": {"text": "不要关灯", "observed_time": 5.0}})
        _score, semantic = FusionAnalyzer._content(no_light)
        self.assertIsNone(semantic["lighting_request"])

    def test_image_source_emits_only_one_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.png"
            cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
            source = MediaFileSource(path)
            first = source.read()
            self.assertEqual(source.kind, "image")
            self.assertIsNotNone(first)
            self.assertIsNone(source.read())
            self.assertFalse(source.has_audio)


if __name__ == "__main__":
    unittest.main()
