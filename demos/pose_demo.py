
import time

import cv2

from src.acquisition.camera import Camera
from src.pose.pose_analyzer import PoseAnalyzer
from src.pose.pose_detector import PoseDetector
from src.pose.pose_visualizer import draw_dashboard, draw_pose


def main():
    camera = None
    detector = None

    try:
        camera = Camera(camera_id=0, width=640, height=480)
        detector = PoseDetector()

        analyzer = PoseAnalyzer()
        previous_time = time.perf_counter()
        fps = 0.0

        print("Pose analysis started. Press Q to exit.")

        while True:
            frame = camera.read()

            # MediaPipe人体关键点检测
            result, timestamp_ms = detector.detect(frame)

            # NumPy数据转换与连续帧缓存
            pose_frame = analyzer.update(
                result,
                timestamp_ms,
                frame.shape,
            )

            # 姿态、连续运动、行为及可靠性统一分析
            analysis = analyzer.analyze(pose_frame)

            # 绘制人体骨架
            frame = draw_pose(frame, result)

            # FPS计算
            current_time = time.perf_counter()
            dt = current_time - previous_time
            previous_time = current_time

            if dt > 0:
                instant_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant_fps

            frame = draw_dashboard(frame, analysis, fps)

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
