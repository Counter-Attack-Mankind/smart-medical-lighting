import json
import math
import unittest
from types import SimpleNamespace

import numpy as np

from src.pose.pose_analyzer import PoseAnalyzer, TimedState


BASE = np.full((33, 3), (0.5, 0.5, 0.0), dtype=float)
BASE[0] = (0.50, 0.10, 0.0)
for index, x, y in (
    (1, .48, .10), (2, .47, .11), (3, .46, .11), (4, .52, .10),
    (5, .53, .11), (6, .54, .11), (7, .45, .13), (8, .55, .13),
    (9, .48, .15), (10, .52, .15), (11, .42, .25), (12, .58, .25),
    (13, .38, .40), (14, .62, .40), (15, .36, .55), (16, .64, .55),
    (17, .35, .57), (18, .65, .57), (19, .34, .56), (20, .66, .56),
    (21, .36, .54), (22, .64, .54), (23, .44, .52), (24, .56, .52),
    (25, .44, .72), (26, .56, .72), (27, .44, .92), (28, .56, .92),
    (29, .43, .94), (30, .57, .94), (31, .42, .96), (32, .58, .96),
):
    BASE[index] = (x, y, 0.0)


def result_from(points=None, valid=None, detected=True):
    if not detected:
        return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
    points = BASE.copy() if points is None else np.asarray(points, dtype=float)
    valid = np.ones(33, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    image = []
    world = []
    for index, (x, y, z) in enumerate(points):
        confidence = 0.95 if valid[index] else 0.05
        image.append(SimpleNamespace(
            x=float(x), y=float(y), z=float(z),
            visibility=confidence, presence=confidence,
        ))
        # World coordinates are synthetic but preserve all joint geometry.
        world.append(SimpleNamespace(
            x=float(x), y=float(y), z=float(z),
            visibility=confidence, presence=confidence,
        ))
    return SimpleNamespace(pose_landmarks=[image], pose_world_landmarks=[world])


def transformed(shift_x=0.0, scale=1.0):
    points = BASE.copy()
    center = np.array([0.5, 0.5])
    points[:, :2] = center + (points[:, :2] - center) * scale
    points[:, 0] += shift_x
    return points


def sitting_points():
    points = BASE.copy()
    for index, xyz in {
        11: (.30, .20, 0), 12: (.40, .20, 0),
        23: (.30, .50, 0), 24: (.40, .50, 0),
        25: (.50, .50, 0), 26: (.60, .50, 0),
        27: (.50, .80, 0), 28: (.60, .80, 0),
    }.items():
        points[index] = xyz
    return points


class TestTemporalPoseAnalysis(unittest.TestCase):
    shape = (480, 640, 3)

    def test_irregular_time_uses_real_delta(self):
        analyzer = PoseAnalyzer()
        samples = []
        for time_s in (0.0, 0.10, 0.27, 0.49, 0.70, 0.91, 1.14, 1.37):
            points = transformed(shift_x=0.02 * time_s)
            analyzer.update(result_from(points), round(time_s * 1000), self.shape)
            if analyzer.motion_samples:
                samples.append(analyzer.motion_samples[-1]["global"])
        self.assertGreater(min(samples), 0.05)
        # Ignore the intentional EMA warm-up, then velocity is independent of
        # the irregular sample spacing.
        steady = samples[3:]
        self.assertLess(max(steady) / min(steady), 1.20)

    def test_translation_is_separate_from_local_motion(self):
        analyzer = PoseAnalyzer()
        for index in range(12):
            analyzer.update(
                result_from(transformed(shift_x=index * 0.004)), index * 100, self.shape
            )
        output = analyzer.analyze()
        self.assertGreater(output["global_motion"], 0.12)
        self.assertLess(output["upper_body_activity"], 0.03)
        self.assertNotEqual(output["motion_state"], "high_activity")

    def test_local_arm_motion_and_scale_normalization(self):
        arm = PoseAnalyzer()
        for index in range(12):
            points = BASE.copy()
            points[15, 1] += index * 0.003
            arm.update(result_from(points), index * 100, self.shape)
        arm_output = arm.analyze()
        self.assertGreater(arm_output["upper_body_activity"], 0.02)
        self.assertLess(arm_output["global_motion"], 0.01)

        scale = PoseAnalyzer()
        for index in range(12):
            scale.update(
                result_from(transformed(scale=1.0 + index * 0.008)),
                index * 100,
                self.shape,
            )
        scale_output = scale.analyze()
        self.assertLess(scale_output["activity_raw"], 0.03)

    def test_periodic_upper_motion(self):
        analyzer = PoseAnalyzer()
        output = None
        for index in range(101):
            time_s = index * 0.05
            points = BASE.copy()
            points[15, 1] += 0.055 * math.sin(2.0 * math.pi * time_s)
            output = analyzer.analyze(
                analyzer.update(result_from(points), round(time_s * 1000), self.shape)
            )
        self.assertTrue(output["behaviors"]["repetitive_motion"])
        self.assertEqual(output["repetitive_motion"]["region"], "upper_body")
        self.assertAlmostEqual(output["repetitive_motion"]["period"], 1.0, delta=0.15)

    def test_periodicity_does_not_bridge_detection_gap(self):
        analyzer = PoseAnalyzer()
        output = None
        for index in range(40):
            time_s = index * 0.05
            points = BASE.copy()
            points[15, 1] += 0.06 * math.sin(2 * math.pi * time_s)
            output = analyzer.analyze(analyzer.update(
                result_from(points), round(time_s * 1000), self.shape
            ))
        analyzer.analyze(analyzer.update(result_from(detected=False), 2000, self.shape))
        for index in range(41, 81):
            time_s = index * 0.05
            points = BASE.copy()
            points[15, 1] += 0.06 * math.sin(2 * math.pi * time_s)
            output = analyzer.analyze(analyzer.update(
                result_from(points), round(time_s * 1000), self.shape
            ))
        self.assertFalse(output["behaviors"]["repetitive_motion"])

    def test_walking_and_pacing_candidates(self):
        walking = PoseAnalyzer()
        output = None
        for index in range(31):
            time_s = index * 0.1
            points = transformed(shift_x=index * 0.005)
            phase = 0.040 * math.sin(2 * math.pi * time_s)
            points[27, 0] += phase
            points[28, 0] -= phase
            output = walking.analyze(
                walking.update(result_from(points), index * 100, self.shape)
            )
        self.assertTrue(output["behaviors"]["walking"])

        pacing = PoseAnalyzer()
        for index in range(61):
            time_s = index * 0.1
            # Three directional legs over six seconds.
            segment = int(time_s // 1.5)
            within = (time_s % 1.5) / 1.5
            start = -0.10 if segment % 2 == 0 else 0.10
            end = -start
            shift = start + (end - start) * within
            output = pacing.analyze(
                pacing.update(result_from(transformed(shift_x=shift)), index * 100, self.shape)
            )
        self.assertTrue(output["behaviors"]["pacing_candidate"])
        self.assertGreaterEqual(output["pacing"]["reversals"], 2)

    def test_posture_transition_event_and_fsm_stability(self):
        analyzer = PoseAnalyzer()
        outputs = []
        for index in range(4):
            outputs.append(analyzer.analyze(
                analyzer.update(result_from(BASE), index * 150, self.shape)
            ))
        for index in range(4, 9):
            outputs.append(analyzer.analyze(
                analyzer.update(result_from(sitting_points()), index * 150, self.shape)
            ))
        events = [item["behavior_event"] for item in outputs if item["behavior_event"]]
        self.assertTrue(any(event["type"] == "sitting_down" for event in events))

        fsm = TimedState(0.4)
        for timestamp, state in ((0.0, "standing"), (.2, "standing"), (.4, "standing")):
            fsm.update(state, timestamp)
        self.assertEqual(fsm.state, "standing")
        fsm.update("sitting", .5)
        fsm.update("standing", .65)
        self.assertEqual(fsm.state, "standing")

    def test_occlusion_loss_recovery_and_json(self):
        analyzer = PoseAnalyzer()
        for index in range(8):
            analyzer.analyze(analyzer.update(result_from(), index * 100, self.shape))
        head_shoulders = np.zeros(33, dtype=bool)
        head_shoulders[:13] = True
        partial = None
        for index in range(8, 14):
            partial = analyzer.analyze(analyzer.update(
                result_from(valid=head_shoulders), index * 100, self.shape
            ))
        self.assertEqual(partial["visible_region"], "head_shoulders")
        self.assertEqual(partial["posture"], "unknown")
        self.assertIsNone(partial["lower_body_activity"])

        missing = analyzer.analyze(analyzer.update(
            result_from(detected=False), 1400, self.shape
        ))
        self.assertFalse(missing["person_detected"])
        self.assertEqual(missing["motion_state"], "unknown")
        self.assertIsNone(missing["activity_raw"])

        recovered = analyzer.analyze(analyzer.update(result_from(), 1500, self.shape))
        self.assertIsNone(recovered["activity_raw"])
        json.dumps(recovered, allow_nan=False)

    def test_upper_body_scope_without_visible_hips(self):
        analyzer = PoseAnalyzer()
        upper_only = np.zeros(33, dtype=bool)
        upper_only[:23] = True
        output = None
        for index in range(8):
            points = BASE.copy()
            points[15, 1] += index * 0.002
            output = analyzer.analyze(analyzer.update(
                result_from(points, valid=upper_only), index * 100, self.shape
            ))
        self.assertEqual(output["visible_region"], "upper_body")
        self.assertEqual(output["activity_scope"], "upper_body")
        self.assertEqual(output["posture"], "unknown")
        self.assertIsNotNone(output["upper_body_activity"])
        self.assertIsNone(output["lower_body_activity"])


if __name__ == "__main__":
    unittest.main()
