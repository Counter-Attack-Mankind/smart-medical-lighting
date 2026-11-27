
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks.python import vision


POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS

POSTURE_ZH = {
    "standing": "站立", "sitting": "坐姿", "lying_candidate": "卧姿候选",
    "bending": "弯腰", "unknown": "未知",
}
MOTION_ZH = {
    "still": "持续静止", "low_activity": "低活动", "normal_activity": "正常活动",
    "high_activity": "高活动", "unknown": "未知",
}
REGION_ZH = {
    "full_body": "全身", "upper_body": "上半身", "head_shoulders": "头肩",
    "head_only": "仅头部", "none": "未检测", "unknown": "未知",
}


def _font(size):
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


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


def _number(value):
    return "不可用" if value is None else f"{value:.3f}"


def draw_dashboard(frame, analysis, fps, panel_width=430):
    """Compose camera view and a compact Chinese information panel."""
    height, width = frame.shape[:2]
    canvas = np.full((height, width + panel_width, 3), 245, dtype=np.uint8)
    canvas[:, :width] = frame
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    title_font, text_font, small_font = _font(22), _font(16), _font(14)
    x, y = width + 20, 16
    draw.text((x, y), "人体姿态与动作分析", fill=(25, 25, 25), font=title_font)
    y += 40
    detected = "已检测" if analysis["person_detected"] else "未检测"
    behaviors = analysis["behaviors"]
    behavior_names = []
    if behaviors.get("walking"):
        behavior_names.append("行走")
    if behaviors.get("pacing_candidate"):
        behavior_names.append("踱步候选")
    if behaviors.get("repetitive_motion"):
        behavior_names.append("重复运动")
    event = analysis.get("behavior_event")
    if event:
        behavior_names.append("起身" if event["type"] == "standing_up" else "坐下")
    behavior_text = "、".join(behavior_names) or "未发现"
    reliability = analysis["reliability"]
    duration = analysis["state_duration"]
    repetitive = analysis.get("repetitive_motion")
    lines = [
        f"运行帧率：{fps:.1f} FPS",
        f"人体状态：{detected}  有效点：{analysis['valid_landmarks']}/33",
        f"可见区域：{REGION_ZH.get(analysis['visible_region'], analysis['visible_region'])}",
        f"基础姿态：{POSTURE_ZH.get(analysis['posture'], analysis['posture'])}",
        f"运动状态：{MOTION_ZH.get(analysis['motion_state'], analysis['motion_state'])}",
        f"具体行为：{behavior_text}",
        f"头部活动：{_number(analysis['head_activity'])}",
        f"上肢活动：{_number(analysis['upper_body_activity'])}",
        f"下肢活动：{_number(analysis['lower_body_activity'])}",
        f"整体移动：{_number(analysis['global_motion'])}",
        f"活动范围：{REGION_ZH.get(analysis['activity_scope'], analysis['activity_scope'])}",
        f"重复区域：{REGION_ZH.get(repetitive['region'], repetitive['region']) if repetitive else '无'}",
        f"姿态持续：{duration['posture']:.1f}s  运动持续：{duration['motion']:.1f}s",
        f"可靠性：整体 {reliability['overall']:.0%} / 躯干 {reliability['torso']:.0%}",
        f"有效时间覆盖率：{analysis['temporal_coverage']:.0%}",
    ]
    for line in lines:
        draw.text((x, y), line, fill=(35, 35, 35), font=text_font)
        y += 23

    score = analysis["activity_score"]
    bar_y = min(height - 25, y + 27)
    draw.text((x, bar_y - 23), "综合活动评分", fill=(35, 35, 35), font=small_font)
    bar_x, bar_width, bar_height = x + 115, panel_width - 160, 14
    draw.rectangle((bar_x, bar_y - 20, bar_x + bar_width, bar_y - 20 + bar_height), outline=(130, 130, 130))
    if score is not None:
        fill = int(bar_width * score)
        color = (55, 170, 80) if score < 0.5 else (230, 150, 45)
        if fill > 0:
            draw.rectangle((bar_x + 1, bar_y - 19, bar_x + fill, bar_y - 7), fill=color)
        draw.text((bar_x + bar_width + 8, bar_y - 24), f"{score:.2f}", fill=(35, 35, 35), font=small_font)
    else:
        draw.text((bar_x + bar_width + 8, bar_y - 24), "--", fill=(90, 90, 90), font=small_font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
