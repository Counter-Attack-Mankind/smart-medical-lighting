
import time
from queue import Queue, Empty, Full

import numpy as np
import sounddevice as sd

from config import audio_config as cfg


class Microphone:
    def __init__(
        self,
        device=cfg.MICROPHONE_DEVICE,
        sample_rate=cfg.SAMPLE_RATE,
        block_size=cfg.BLOCK_SIZE,
        queue_size=cfg.MICROPHONE_QUEUE_SIZE
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.buffer = Queue(maxsize=queue_size)
        self.dropped_blocks = 0
        self.input_overflows = 0
        self._released = False

        # 创建麦克风音频流
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback
        )

        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        # 记录音频块首个采样点的近似单调时间戳
        timestamp = (
            time.perf_counter()
            + time_info.inputBufferAdcTime
            - time_info.currentTime
        )

        # 从 (512, 1) 转换为 (512,)
        samples = indata[:, 0].copy()

        if status.input_overflow:
            self.input_overflows += 1

        # 非阻塞入队
        try:
            self.buffer.put_nowait((samples, timestamp))

        except Full:
            # 缓冲区满时丢弃最旧的音频块
            try:
                self.buffer.get_nowait()
                self.dropped_blocks += 1
            except Empty:
                pass

            self.buffer.put_nowait((samples, timestamp))

    def read(self, timeout=1.0):
        try:
            return self.buffer.get(timeout=timeout)
        except Empty:
            return None, None

    def status(self):
        """Return a consistent snapshot for discontinuity detection/UI."""
        return {
            "queued_blocks": int(self.buffer.qsize()),
            "dropped_blocks": int(self.dropped_blocks),
            "input_overflows": int(self.input_overflows),
        }

    def release(self):
        if not self._released:
            self.stream.stop()
            self.stream.close()
            self._released = True


if __name__ == "__main__":
    microphone = None

    try:
        microphone = Microphone()

        print("Microphone started.")
        print("Sample rate:", microphone.sample_rate)
        print("Block size:", microphone.block_size)
        print("Press Ctrl+C to exit.\n")

        previous_time = time.perf_counter()
        first_block = True

        while True:
            samples, timestamp = microphone.read()

            if samples is None:
                print("\nNo audio data received.")
                continue

            if first_block:
                print("Audio shape:", samples.shape)
                print("Audio dtype:", samples.dtype)
                first_block = False

            # 计算均方根能量
            rms = float(np.sqrt(np.mean(samples ** 2)))

            # 转换为 dBFS，避免 log10(0)
            dbfs = 20.0 * np.log10(max(rms, 1e-8))

            current_time = time.perf_counter()

            # 限制终端刷新频率为约 10 Hz
            if current_time - previous_time >= 0.1:
                previous_time = current_time

                level = int(np.clip(
                    (dbfs + 60.0) / 60.0, 0.0, 1.0
                ) * 30)

                bar = "#" * level + "-" * (30 - level)

                print(
                    f"\r[{bar}] "
                    f"RMS: {rms:.5f} | "
                    f"dBFS: {dbfs:7.2f} | "
                    f"Dropped: {microphone.dropped_blocks} | "
                    f"Overflow: {microphone.input_overflows}",
                    end="",
                    flush=True
                )

    except KeyboardInterrupt:
        print("\nMicrophone test stopped.")

    finally:
        if microphone is not None:
            microphone.release()

        print("Microphone released.")
