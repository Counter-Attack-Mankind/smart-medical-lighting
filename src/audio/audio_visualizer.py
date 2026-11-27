"""OpenCV/Pillow Chinese dashboard for the standalone audio demo."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import audio_config as cfg


STATE_ZH = {
    "unavailable": "音频不可用",
    "no_speech": "当前无语音",
    "speaking": "正在说话",
    "processing": "正在转写",
    "transcribed": "已获得转写",
    "untranscribed": "语音未能转写",
}
INTENSITY_ZH = {
    "normal_intensity": "正常强度候选",
    "high_intensity": "高强度候选",
    "low_intensity": "低强度候选",
    "unknown": "未知",
}


@lru_cache(maxsize=4)
def _font(size):
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _number(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def _clip_text(text, length=24):
    text = str(text or "--")
    return text if len(text) <= length else text[:length - 1] + "…"


class AudioVisualizer:
    def __init__(self):
        self.waveform = np.empty(0, dtype=np.float32)
        self.max_samples = int(cfg.WAVEFORM_SECONDS * cfg.SAMPLE_RATE)

    def _append(self, samples):
        if samples is None:
            return
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.waveform = np.concatenate((self.waveform, samples))[-self.max_samples:]

    def render(self, analysis, samples=None):
        self._append(samples)
        width, height = cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT
        image = Image.new("RGB", (width, height), (244, 246, 248))
        draw = ImageDraw.Draw(image)
        title_font, text_font, small_font = _font(22), _font(16), _font(14)
        draw.text((20, 14), "音频感知与韵律分析", fill=(25, 25, 25), font=title_font)

        left_x, left_width = 20, 650
        wave_top, wave_height = 65, 250
        draw.rectangle(
            (left_x, wave_top, left_x + left_width, wave_top + wave_height),
            fill=(24, 29, 34), outline=(120, 125, 130),
        )
        center_y = wave_top + wave_height // 2
        draw.line((left_x, center_y, left_x + left_width, center_y), fill=(70, 78, 84))
        if len(self.waveform):
            indices = np.linspace(0, len(self.waveform) - 1, left_width).astype(int)
            values = np.clip(self.waveform[indices], -1.0, 1.0)
            points = [
                (left_x + index, int(center_y - value * (wave_height * 0.44)))
                for index, value in enumerate(values)
            ]
            draw.line(points, fill=(70, 210, 135), width=1)
        draw.text((left_x, wave_top - 24), "最近两秒音频波形", fill=(35, 35, 35), font=text_font)

        dbfs = analysis["dbfs"]
        bar_y = wave_top + wave_height + 50
        draw.text((left_x, bar_y - 28), "数字音频电平（dBFS，非声压级）", fill=(35, 35, 35), font=text_font)
        bar_width = left_width - 110
        draw.rectangle((left_x, bar_y, left_x + bar_width, bar_y + 22), outline=(120, 120, 120))
        level = 0.0 if dbfs is None else float(np.clip(
            (dbfs - cfg.DISPLAY_DBFS_MIN)
            / (cfg.DISPLAY_DBFS_MAX - cfg.DISPLAY_DBFS_MIN),
            0.0,
            1.0,
        ))
        if level > 0:
            color = (65, 165, 95) if level < 0.72 else (225, 135, 55)
            draw.rectangle(
                (left_x + 1, bar_y + 1, left_x + int(bar_width * level), bar_y + 21),
                fill=color,
            )
        draw.text((left_x + bar_width + 12, bar_y), _number(dbfs, 1), fill=(35, 35, 35), font=text_font)

        y = bar_y + 50
        draw.text((left_x, y), f"语音状态：{STATE_ZH[analysis['speech_state']]}", fill=(35, 35, 35), font=text_font)
        y += 26
        draw.text(
            (left_x, y),
            f"表达强度：{INTENSITY_ZH[analysis['expression_intensity']]}    可靠性：{analysis['reliability']['score']:.2f}",
            fill=(35, 35, 35), font=text_font,
        )
        y += 26
        draw.text(
            (left_x, y),
            f"语音比例：{_number(analysis['speech_activity_ratio'])}    停顿比例：{_number(analysis['pause_ratio'])}    覆盖率：{analysis['temporal_coverage']:.0%}",
            fill=(35, 35, 35), font=text_font,
        )

        panel_x = 700
        draw.line((685, 55, 685, height - 20), fill=(190, 190, 190))
        y = 60
        draw.text((panel_x, y), "声学与韵律特征", fill=(25, 25, 25), font=text_font)
        y += 30
        for line in (
            f"RMS：{_number(analysis['rms'], 5)}",
            f"中位基频：{_number(analysis['median_f0'], 1)} Hz",
            f"音高变化：{_number(analysis['pitch_range'], 1)} Hz",
            f"片段时长：{_number(analysis['speech_segment_duration'], 2)} s",
        ):
            draw.text((panel_x, y), line, fill=(35, 35, 35), font=small_font)
            y += 23

        transcription = analysis.get("transcription")
        y += 5
        draw.text((panel_x, y), "最近转写", fill=(25, 25, 25), font=text_font)
        y += 27
        if transcription:
            current = "当前有效" if transcription["is_current"] else "历史记录"
            draw.text((panel_x, y), f"{current}，观测于 {transcription['observed_time']:.2f}", fill=(70, 70, 70), font=small_font)
            y += 22
            draw.text((panel_x, y), _clip_text(transcription.get("text"), 27), fill=(25, 25, 25), font=text_font)
            y += 25
            draw.text(
                (panel_x, y),
                f"语言：{transcription.get('language') or '--'}  情感标签：{transcription.get('emotion') or '--'}",
                fill=(35, 35, 35), font=small_font,
            )
            y += 21
            draw.text(
                (panel_x, y),
                f"事件标签：{transcription.get('event') or '--'}  语速：{_number(transcription.get('speech_rate'), 1)} 单位/分",
                fill=(35, 35, 35), font=small_font,
            )
            y += 23
        else:
            draw.text((panel_x, y), "--", fill=(80, 80, 80), font=small_font)
            y += 24

        y += 5
        draw.text((panel_x, y), "环境声音事件（不归因于个人）", fill=(25, 25, 25), font=text_font)
        y += 27
        sound = analysis.get("sound_event_result")
        events = sound["events"] if sound and sound["is_current"] else []
        if events:
            for event in events[:4]:
                draw.text(
                    (panel_x, y),
                    f"{event['label']}  {event['score']:.2f}",
                    fill=(35, 35, 35), font=small_font,
                )
                y += 21
        else:
            draw.text((panel_x, y), "当前无超过阈值的白名单事件", fill=(80, 80, 80), font=small_font)
            y += 22

        timings = analysis.get("inference_ms", {})
        buffer_status = analysis.get("buffer_status", {})
        y = height - 104
        draw.text(
            (panel_x, y),
            f"耗时 VAD/ASR/YAMNet：{_number(timings.get('vad'))} / {_number(timings.get('asr'))} / {_number(timings.get('yamnet'))} ms",
            fill=(65, 65, 65), font=small_font,
        )
        y += 22
        draw.text(
            (panel_x, y),
            f"缓冲：{buffer_status.get('queued_blocks', 0)}  丢块：{buffer_status.get('dropped_blocks', 0)}  溢出：{buffer_status.get('input_overflows', 0)}",
            fill=(65, 65, 65), font=small_font,
        )
        y += 22
        draw.text(
            (panel_x, y),
            "仅表示客观声音与表达特征，不用于医学诊断",
            fill=(110, 65, 65), font=small_font,
        )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
