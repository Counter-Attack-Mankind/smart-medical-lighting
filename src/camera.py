
import cv2


class Camera:
    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        success, frame = self.cap.read()
        if not success:
            raise RuntimeError("摄像头读取画面失败")
        return frame

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    camera = None

    try:
        camera = Camera(camera_id=0)

        while True:
            frame = camera.read()
            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as e:
        print(f"[ERROR] {e}")

    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
