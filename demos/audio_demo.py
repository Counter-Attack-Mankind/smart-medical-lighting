"""Standalone microphone, speech, sound-event and prosody demo."""

import cv2

from config import audio_config as cfg
from src.audio.audio_analyzer import AudioAnalyzer
from src.audio.audio_detector import AudioDetector
from src.audio.audio_visualizer import AudioVisualizer


def main():
    detector = None
    try:
        print("正在加载音频模型，请稍候……")
        detector = AudioDetector()
        analyzer = AudioAnalyzer()
        visualizer = AudioVisualizer()
        print("音频感知已启动，按 Q 或 Ctrl+C 退出。")

        while True:
            detection = detector.read(cfg.READ_TIMEOUT_SECONDS)
            audio_frame = analyzer.update(detection)
            analysis = analyzer.analyze(audio_frame)
            display = visualizer.render(analysis, detection.get("samples"))
            cv2.imshow("Audio Perception", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n收到退出指令。")
    except (RuntimeError, FileNotFoundError) as error:
        print(f"[ERROR] {error}")
    finally:
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()
        print("音频感知程序已退出。")


if __name__ == "__main__":
    main()
