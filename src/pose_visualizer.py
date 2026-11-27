
import cv2
from mediapipe.tasks.python import vision


POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS


def draw_pose(frame, result, visibility_threshold=0.5):
    if not result.pose_landmarks:
        return frame

    height, width = frame.shape[:2]

    for landmarks in result.pose_landmarks:
        # 绘制骨架连接线
        for connection in POSE_CONNECTIONS:
            start = connection.start
            end = connection.end

            p1 = landmarks[start]
            p2 = landmarks[end]

            if (p1.visibility < visibility_threshold or
                    p2.visibility < visibility_threshold):
                continue

            x1, y1 = int(p1.x * width), int(p1.y * height)
            x2, y2 = int(p2.x * width), int(p2.y * height)

            cv2.line(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

        # 绘制关键点
        for landmark in landmarks:
            if landmark.visibility < visibility_threshold:
                continue

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame, (x, y), 4,
                (0, 0, 255), -1
            )

    return frame
