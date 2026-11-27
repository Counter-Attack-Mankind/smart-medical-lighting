
import time

import cv2

from camera import Camera
from pose_detector import PoseDetector
from pose_visualizer import draw_pose


def main():
    camera = None
    detector = None

    try:
        camera = Camera(camera_id=0, width=640, height=480)
        detector = PoseDetector()

        previous_time = time.perf_counter()
        fps = 0.0

        print("Pose detection started. Press Q to exit.")

        while True:
            frame = camera.read()

            result = detector.detect(frame)
            frame = draw_pose(frame, result)

            current_time = time.perf_counter()
            dt = current_time - previous_time
            previous_time = current_time

            if dt > 0:
                instant_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant_fps

            pose_count = len(result.pose_landmarks)

            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Persons: {pose_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame, "Status: RUNNING",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.imshow("MediaPipe Pose Landmarker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except (RuntimeError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")

    finally:
        if detector is not None:
            detector.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
