import json
import math
import unittest
from types import SimpleNamespace

import numpy as np

from config import face_config as cfg
from src.face.face_analyzer import FaceAnalyzer


def rotation_matrix(yaw=0.0, pitch=0.0, roll=0.0):
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)],
    ])
    ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)],
    ])
    rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rz @ ry @ rx
    return matrix


def face_landmarks():
    landmarks = []
    for index in range(cfg.NUM_LANDMARKS):
        x = 0.35 + 0.30 * (index % 22) / 21.0
        y = 0.25 + 0.50 * (index // 22) / 21.0
        landmarks.append(SimpleNamespace(x=x, y=y, z=0.0))
    landmarks[33] = SimpleNamespace(x=0.40, y=0.40, z=0.0)
    landmarks[263] = SimpleNamespace(x=0.60, y=0.40, z=0.0)
    return landmarks


def create_result(overrides=None, detected=True, matrix=None, omitted=()):
    if not detected:
        return SimpleNamespace(
            face_landmarks=[],
            face_blendshapes=[],
            facial_transformation_matrixes=[],
        )
    values = {name: 0.0 for name in cfg.BLENDSHAPE_NAMES}
    values["_neutral"] = 1.0
    values.update(overrides or {})
    categories = [
        SimpleNamespace(category_name=name, score=value)
        for name, value in values.items() if name not in omitted
    ]
    return SimpleNamespace(
        face_landmarks=[face_landmarks()],
        face_blendshapes=[categories],
        facial_transformation_matrixes=[
            np.eye(4) if matrix is None else matrix
        ],
    )


HAPPY = {
    "mouthSmileLeft": 0.80,
    "mouthSmileRight": 0.70,
    "cheekSquintLeft": 0.45,
    "cheekSquintRight": 0.40,
    "eyeSquintLeft": 0.25,
    "eyeSquintRight": 0.20,
    "mouthUpperUpLeft": 0.20,
    "mouthUpperUpRight": 0.18,
}


class TestFaceAnalyzer(unittest.TestCase):
    shape = (480, 640, 3)

    def test_combination_rule_and_asymmetry(self):
        analyzer = FaceAnalyzer()
        frame = analyzer.update(create_result(HAPPY), 0, self.shape)
        output = analyzer.analyze(frame)
        self.assertEqual(frame.landmarks.shape, (478, 3))
        self.assertEqual(len(output["blendshapes"]), 52)
        self.assertEqual(output["raw_expression"], "happy")
        self.assertGreater(output["expression_scores"]["happy"], 0.4)
        self.assertAlmostEqual(
            output["action_features"]["mouth_smile_asymmetry"], 0.10, places=5
        )

        # Opening the jaw alone is speaking/yawning evidence, not surprise.
        other = FaceAnalyzer()
        output = other.analyze(other.update(
            create_result({"jawOpen": 0.85}), 0, self.shape
        ))
        self.assertEqual(output["raw_expression"], "unknown")
        self.assertEqual(output["expression_scores"]["surprised"], 0.0)

        incomplete = FaceAnalyzer()
        output = incomplete.analyze(incomplete.update(
            create_result(HAPPY, omitted=("mouthSmileLeft",)), 0, self.shape
        ))
        self.assertEqual(output["raw_expression"], "unknown")

    def test_actual_time_confirmation_and_blink(self):
        analyzer = FaceAnalyzer()
        outputs = []
        for timestamp in (0, 80, 190, 330, 520, 710):
            outputs.append(analyzer.analyze(analyzer.update(
                create_result(HAPPY), timestamp, self.shape
            )))
        self.assertEqual(outputs[2]["stable_expression"], "unknown")
        self.assertEqual(outputs[-1]["stable_expression"], "happy")
        self.assertGreater(outputs[-1]["state_duration"], 0.0)

        blink = FaceAnalyzer()
        blink.analyze(blink.update(create_result(), 0, self.shape))
        blink_output = blink.analyze(blink.update(create_result({
            "eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.82,
        }), 100, self.shape))
        self.assertTrue(blink_output["facial_actions"]["blink_candidate"])

    def test_detection_loss_recovery_and_json(self):
        analyzer = FaceAnalyzer()
        for timestamp in (0, 100, 200, 300, 400, 500, 600):
            output = analyzer.analyze(analyzer.update(
                create_result(HAPPY), timestamp, self.shape
            ))
        self.assertEqual(output["stable_expression"], "happy")

        lost = analyzer.analyze(analyzer.update(
            create_result(detected=False), 700, self.shape
        ))
        self.assertEqual(lost["stable_expression"], "unknown")
        self.assertIsNone(lost["face_activity"])
        self.assertFalse(lost["reliability"]["reliable"])

        recovered = analyzer.analyze(analyzer.update(
            create_result(HAPPY), 800, self.shape
        ))
        self.assertEqual(recovered["stable_expression"], "unknown")
        self.assertIsNone(recovered["face_activity"])
        json.dumps(recovered, allow_nan=False)

    def test_head_pose_and_side_face_reliability(self):
        analyzer = FaceAnalyzer()
        expected = {"yaw": 12.0, "pitch": -6.0, "roll": 8.0}
        frame = analyzer.update(create_result(
            matrix=rotation_matrix(**expected)
        ), 0, self.shape)
        for axis, value in expected.items():
            self.assertAlmostEqual(frame.raw_head_pose[axis], value, places=5)

        side = FaceAnalyzer()
        output = side.analyze(side.update(create_result(
            HAPPY, matrix=rotation_matrix(yaw=70.0)
        ), 0, self.shape))
        self.assertFalse(output["reliability"]["reliable"])
        self.assertEqual(output["raw_expression"], "unknown")
        self.assertAlmostEqual(output["head_pose"]["yaw"], 70.0, places=5)

    def test_head_and_local_face_motion_are_separate(self):
        local = FaceAnalyzer()
        local.analyze(local.update(create_result(), 0, self.shape))
        output = local.analyze(local.update(create_result(HAPPY), 100, self.shape))
        self.assertGreater(output["face_activity"], 0.0)
        self.assertAlmostEqual(output["head_motion"], 0.0, places=6)

        head = FaceAnalyzer()
        head.analyze(head.update(create_result(), 0, self.shape))
        output = head.analyze(head.update(create_result(
            matrix=rotation_matrix(yaw=10.0)
        ), 100, self.shape))
        self.assertAlmostEqual(output["face_activity"], 0.0, places=6)
        self.assertGreater(output["head_motion"], 0.0)


if __name__ == "__main__":
    unittest.main()
