
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import face_config as cfg


class FaceDetector:
    def __init__(self):
        if not cfg.MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found: {cfg.MODEL_PATH}"
            )

        # MediaPipe Face Landmarker 配置
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(cfg.MODEL_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=cfg.NUM_FACES,
            min_face_detection_confidence=(
                cfg.MIN_FACE_DETECTION_CONFIDENCE
            ),
            min_face_presence_confidence=(
                cfg.MIN_FACE_PRESENCE_CONFIDENCE
            ),
            min_tracking_confidence=(
                cfg.MIN_TRACKING_CONFIDENCE
            ),
            output_face_blendshapes=(
                cfg.OUTPUT_FACE_BLENDSHAPES
            ),
            output_facial_transformation_matrixes=(
                cfg.OUTPUT_FACIAL_TRANSFORMATION_MATRIXES
            ),
        )

        self.detector = (
            vision.FaceLandmarker.create_from_options(options)
        )

        self.start_time = time.perf_counter()
        self.last_timestamp_ms = -1

    def detect(self, frame, timestamp_ms=None):
        # OpenCV BGR 转换为 MediaPipe RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # 独立运行时生成时间戳
        if timestamp_ms is None:
            timestamp_ms = int(
                (time.perf_counter() - self.start_time) * 1000
            )
            timestamp_ms = max(
                timestamp_ms,
                self.last_timestamp_ms + 1
            )
        else:
            # 外部时间戳必须严格递增
            timestamp_ms = int(timestamp_ms)

            if timestamp_ms <= self.last_timestamp_ms:
                raise ValueError(
                    "Frame timestamps must be strictly increasing."
                )

        self.last_timestamp_ms = timestamp_ms

        # VIDEO 模式推理
        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result, timestamp_ms

    def close(self):
        self.detector.close()
