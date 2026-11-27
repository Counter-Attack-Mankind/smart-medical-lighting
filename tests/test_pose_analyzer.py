
import sys
import unittest

from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from pose_analyzer import PoseAnalyzer


def create_result(detected=True):
    """构造模拟的MediaPipe检测结果。"""

    if not detected:
        return SimpleNamespace(
            pose_landmarks=[],
            pose_world_landmarks=[],
        )

    landmarks = [
        SimpleNamespace(
            x=0.5,
            y=0.5,
            z=0.0,
            visibility=0.9,
            presence=0.9,
        )
        for _ in range(33)
    ]

    return SimpleNamespace(
        pose_landmarks=[landmarks],
        pose_world_landmarks=[landmarks],
    )


class TestPoseAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = PoseAnalyzer(window_seconds=5.0)
        self.frame_shape = (480, 640, 3)

    def test_landmark_shape(self):
        pose_frame = self.analyzer.update(
            create_result(),
            1000,
            self.frame_shape,
        )

        self.assertEqual(pose_frame.landmarks.shape, (33, 3))
        self.assertEqual(pose_frame.world_landmarks.shape, (33, 3))
        self.assertEqual(pose_frame.valid_mask.shape, (33,))
        self.assertEqual(pose_frame.timestamp, 1.0)
        self.assertEqual(
            self.analyzer.get_valid_point_count(), 33
        )

    def test_time_window(self):
        timestamps = [0, 100, 700, 1800, 3500, 5100, 6200]

        for timestamp in timestamps:
            self.analyzer.update(
                create_result(),
                timestamp,
                self.frame_shape,
            )

        self.assertLessEqual(
            self.analyzer.get_window_duration(), 5.0
        )

        self.assertEqual(
            self.analyzer.get_frame_count(), 4
        )

    def test_missing_person(self):
        pose_frame = self.analyzer.update(
            create_result(detected=False),
            1000,
            self.frame_shape,
        )

        self.assertFalse(pose_frame.person_detected)
        self.assertTrue(np.isnan(pose_frame.landmarks).all())
        self.assertEqual(
            self.analyzer.get_valid_point_count(), 0
        )
        self.assertEqual(
            self.analyzer.get_frame_count(), 1
        )

    def test_invalid_landmarks(self):
        result = create_result()

        result.pose_landmarks[0][0].visibility = 0.1
        result.pose_landmarks[0][1].presence = 0.2
        result.pose_landmarks[0][2].x = np.nan
        result.pose_landmarks[0][3].y = 1.2

        self.analyzer.update(
            result,
            1000,
            self.frame_shape,
        )

        self.assertEqual(
            self.analyzer.get_valid_point_count(), 29
        )

    def test_timestamp_order(self):
        self.analyzer.update(
            create_result(),
            1000,
            self.frame_shape,
        )

        with self.assertRaises(ValueError):
            self.analyzer.update(
                create_result(),
                900,
                self.frame_shape,
            )

    def test_angle_90(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])

        angle = self.analyzer.calculate_angle(a, b, c)

        self.assertAlmostEqual(angle, 90.0, places=5)

    def test_angle_180(self):
        a = np.array([-1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])

        angle = self.analyzer.calculate_angle(a, b, c)

        self.assertAlmostEqual(angle, 180.0, places=5)

    def test_degenerate_angle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])

        angle = self.analyzer.calculate_angle(a, b, c)

        self.assertIsNone(angle)

    def test_aspect_ratio(self):
        pose_frame = self.analyzer.update(
            create_result(),
            1000,
            self.frame_shape,
        )

        points = self.analyzer.get_image_points(pose_frame)

        # 640x480：y坐标乘以480/640
        self.assertAlmostEqual(points[0, 0], 0.5)
        self.assertAlmostEqual(points[0, 1], 0.375)

    def test_unknown_posture(self):
        pose_frame = self.analyzer.update(
            create_result(detected=False),
            1000,
            self.frame_shape,
        )

        analysis = self.analyzer.analyze_posture(pose_frame)

        self.assertEqual(analysis["posture"], "unknown")
        self.assertEqual(analysis["visible_region"], "none")
        self.assertIsNone(
            analysis["features"]["left_knee"]
        )

    def test_standing_posture(self):
        result = create_result()

        positions = {
            11: (0.4, 0.2, 0.0),
            12: (0.6, 0.2, 0.0),
            23: (0.4, 0.5, 0.0),
            24: (0.6, 0.5, 0.0),
            25: (0.4, 0.7, 0.0),
            26: (0.6, 0.7, 0.0),
            27: (0.4, 0.9, 0.0),
            28: (0.6, 0.9, 0.0),
        }

        for index, (x, y, z) in positions.items():
            point = result.pose_landmarks[0][index]
            point.x = x
            point.y = y
            point.z = z

        pose_frame = self.analyzer.update(
            result,
            1000,
            self.frame_shape,
        )

        analysis = self.analyzer.analyze_posture(pose_frame)

        self.assertEqual(analysis["posture"], "standing")

    def test_sitting_posture(self):
        result = create_result()

        positions = {
            11: (0.3, 0.2, 0.0),
            12: (0.4, 0.2, 0.0),
            23: (0.3, 0.5, 0.0),
            24: (0.4, 0.5, 0.0),
            25: (0.5, 0.5, 0.0),
            26: (0.6, 0.5, 0.0),
            27: (0.5, 0.8, 0.0),
            28: (0.6, 0.8, 0.0),
        }

        for index, (x, y, z) in positions.items():
            point = result.pose_landmarks[0][index]
            point.x = x
            point.y = y
            point.z = z

        pose_frame = self.analyzer.update(
            result,
            1000,
            self.frame_shape,
        )

        analysis = self.analyzer.analyze_posture(pose_frame)

        self.assertEqual(analysis["posture"], "sitting")

    def test_occluded_joint(self):
        result = create_result()

        # 左膝不可见
        result.pose_landmarks[0][25].visibility = 0.1

        pose_frame = self.analyzer.update(
            result,
            1000,
            self.frame_shape,
        )

        angle = self.analyzer.get_joint_angle(
            pose_frame, 23, 25, 27
        )

        self.assertIsNone(angle)

    def test_lying_candidate(self):
        result = create_result()
        positions = {
            11: (.20, .40, 0), 12: (.20, .50, 0),
            23: (.50, .40, 0), 24: (.50, .50, 0),
            25: (.70, .40, 0), 26: (.70, .50, 0),
            27: (.90, .40, 0), 28: (.90, .50, 0),
        }
        for index, (x, y, z) in positions.items():
            point = result.pose_landmarks[0][index]
            point.x, point.y, point.z = x, y, z
        frame = self.analyzer.update(result, 1000, self.frame_shape)
        self.assertEqual(
            self.analyzer.analyze_posture(frame)["posture"],
            "lying_candidate",
        )

    def test_bending_posture(self):
        result = create_result()
        positions = {
            11: (.25, .30, 0), 12: (.35, .30, 0),
            23: (.45, .50, 0), 24: (.55, .50, 0),
            25: (.45, .70, 0), 26: (.55, .70, 0),
            27: (.45, .90, 0), 28: (.55, .90, 0),
        }
        for index, (x, y, z) in positions.items():
            point = result.pose_landmarks[0][index]
            point.x, point.y, point.z = x, y, z
        frame = self.analyzer.update(result, 1000, self.frame_shape)
        self.assertEqual(
            self.analyzer.analyze_posture(frame)["posture"],
            "bending",
        )


if __name__ == "__main__":
    unittest.main()
