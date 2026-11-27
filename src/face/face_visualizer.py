"""OpenCV face mesh and Pillow Chinese analysis panel."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks.python import vision


FACE_CONNECTIONS = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
EXPRESSION_ZH = {
    "happy": "开心候选",
    "sad": "悲伤候选",
    "angry": "愤怒候选",
    "surprised": "惊讶候选",
    "neutral": "中性候选",
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


def draw_face(frame, face_frame):
    """Draw the mesh from the analyzer-converted landmarks."""
    if not face_frame.face_detected:
        return frame
    height, width = frame.shape[:2]
    points = face_frame.landmarks[:, :2]
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return frame
    pixels = np.zeros((len(points), 2), dtype=np.int32)
    pixels[finite] = (
        points[finite] * np.asarray([width, height])
    ).astype(np.int32)

    for connection in FACE_CONNECTIONS:
        if not finite[connection.start] or not finite[connection.end]:
            continue
        cv2.line(
            frame,
            tuple(map(int, pixels[connection.start])),
            tuple(map(int, pixels[connection.end])),
            (100, 180, 100),
            1,
        )
    for x, y in pixels[finite]:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

    x_min, y_min = pixels[finite].min(axis=0)
    x_max, y_max = pixels[finite].max(axis=0)
    cv2.rectangle(
        frame,
        (int(x_min), int(y_min)),
        (int(x_max), int(y_max)),
        (255, 180, 0),
        2,
    )
    return frame


def _value(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def draw_dashboard(frame, analysis, fps, inference_ms, panel_width=440):
    """Compose the camera image and analysis without recomputing features."""
    height, width = frame.shape[:2]
    canvas = np.full((height, width + panel_width, 3), 245, dtype=np.uint8)
    canvas[:, :width] = frame
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    title_font, text_font, small_font = _font(21), _font(15), _font(13)
    x, y = width + 18, 12
    draw.text((x, y), "面部动作与表情候选分析", fill=(25, 25, 25), font=title_font)
    y += 34

    reliability = analysis["reliability"]
    actions = analysis["facial_actions"]
    pose = analysis["head_pose"]
    detected = "已检测" if analysis["face_detected"] else "未检测"
    blink = actions["blink_candidate"]
    blink_text = "候选" if blink else ("否" if blink is False else "--")
    closed = actions["eyes_closed"]
    closed_text = "是" if closed else ("否" if closed is False else "--")
    lines = [
        f"帧率：{fps:.1f} FPS    推理：{inference_ms:.2f} ms",
        f"人脸：{detected}    关键点：{analysis['landmark_count']}/478",
        f"动作系数：{analysis['blendshape_count']}/52",
        f"原始候选：{EXPRESSION_ZH[analysis['raw_expression']]}",
        f"稳定候选：{EXPRESSION_ZH[analysis['stable_expression']]}",
        (
            f"可靠性：{reliability['score']:.2f} "
            f"({'有效' if reliability['reliable'] else '不足'})    "
            f"覆盖：{analysis['temporal_coverage']:.0%}"
        ),
        f"面部活动：{_value(analysis['face_activity'])}    头部运动：{_value(analysis['head_motion'])}",
        (
            f"头姿 Y/P/R：{_value(pose['yaw'], 1) if pose else '--'} / "
            f"{_value(pose['pitch'], 1) if pose else '--'} / "
            f"{_value(pose['roll'], 1) if pose else '--'}°"
        ),
        f"闭眼：{closed_text}    眨眼：{blink_text}    张嘴：{_value(actions['mouth_open_score'])}",
        f"状态持续：{analysis['state_duration']:.1f} s",
    ]
    for line in lines:
        draw.text((x, y), line, fill=(35, 35, 35), font=text_font)
        y += 21

    y += 2
    draw.line((x, y, width + panel_width - 18, y), fill=(190, 190, 190))
    y += 7
    draw.text((x, y), "独立表情规则评分（非概率）", fill=(35, 35, 35), font=text_font)
    y += 24
    bar_x = x + 92
    bar_width = panel_width - 165
    scores = analysis["expression_scores"]
    for key, label in (
        ("happy", "开心"), ("sad", "悲伤"), ("angry", "愤怒"),
        ("surprised", "惊讶"), ("neutral", "中性"),
    ):
        score = scores[key]
        draw.text((x, y), label, fill=(35, 35, 35), font=small_font)
        draw.rectangle(
            (bar_x, y + 3, bar_x + bar_width, y + 14),
            outline=(145, 145, 145),
        )
        if score is not None:
            fill_width = int(bar_width * np.clip(score, 0.0, 1.0))
            if fill_width > 0:
                draw.rectangle(
                    (bar_x + 1, y + 4, bar_x + fill_width, y + 13),
                    fill=(65, 145, 190),
                )
        draw.text(
            (bar_x + bar_width + 7, y),
            _value(score),
            fill=(35, 35, 35),
            font=small_font,
        )
        y += 21

    draw.text(
        (x, height - 22),
        "仅表示外观表情候选，不用于医学诊断",
        fill=(105, 70, 70),
        font=small_font,
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
