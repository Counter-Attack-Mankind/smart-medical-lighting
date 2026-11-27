
from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]
        model_path = (
            project_root
            / "models"
            / "pose"
            / "pose_landmarker_lite.task"
        )

        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.start_time = time.perf_counter()
        self.last_timestamp_ms = -1

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_frame.copy()

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.perf_counter() - self.start_time) * 1000
        )
        timestamp_ms = max(
            timestamp_ms, self.last_timestamp_ms + 1
        )
        self.last_timestamp_ms = timestamp_ms

        result = self.detector.detect_for_video(
            mp_image, timestamp_ms
        )

        return result, timestamp_ms

    def close(self):
        self.detector.close()
