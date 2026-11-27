
import time

import cv2

from config import face_config as cfg
from src.acquisition.camera import Camera
from src.face.face_analyzer import FaceAnalyzer
from src.face.face_detector import FaceDetector
from src.face.face_visualizer import draw_face, draw_dashboard


def main():
    camera = None
    detector = None
    analyzer = None

    try:
        camera = Camera(
            camera_id=cfg.CAMERA_ID,
            width=cfg.CAMERA_WIDTH,
            height=cfg.CAMERA_HEIGHT
        )

        detector = FaceDetector()
        analyzer = FaceAnalyzer()

        previous_time = time.perf_counter()
        fps = 0.0

        first_detection = True

        print("人脸特征检测已启动，按 Q 退出。")

        while True:

            frame = camera.read()

            # 模型推理
            start_time = time.perf_counter()

            result, timestamp_ms = detector.detect(frame)

            inference_ms = (
                time.perf_counter() - start_time
            ) * 1000

            # 帧率计算
            current_time = time.perf_counter()

            dt = current_time - previous_time
            previous_time = current_time

            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            face_frame = analyzer.update(result, timestamp_ms, frame.shape)
            analysis = analyzer.analyze(face_frame)

            if face_frame.face_detected and first_detection:
                print("面部关键点形状:", face_frame.landmarks.shape)
                print("Blendshape数量:", analysis["blendshape_count"])
                matrix = face_frame.transformation_matrix
                if matrix is not None:
                    print("面部变换矩阵形状:", matrix.shape)
                first_detection = False

            # 绘制面部网格
            display = draw_face(frame.copy(), face_frame)

            # 绘制中文信息面板
            display = draw_dashboard(
                display,
                analysis,
                fps,
                inference_ms
            )

            cv2.imshow(
                "MediaPipe Face Landmarker",
                display
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n收到退出指令。")

    except (RuntimeError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")

    finally:

        if detector is not None:
            detector.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print("人脸检测程序已退出。")


if __name__ == "__main__":
    main()
