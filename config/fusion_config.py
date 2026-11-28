"""Initial Stage 05.4 engineering parameters; calibration is required."""

FUSION_HZ = 5.0
BASE_WEIGHTS = {"pose": 0.35, "face": 0.25, "speech_content": 0.20, "speech_prosody": 0.20}
MIN_EFFECTIVE_WEIGHT = 0.20
SMOOTHING_TIME_CONSTANT_SECONDS = 1.0
TRAJECTORY_SECONDS = 30.0
POSE_AROUSAL = {"still": -0.5, "low_activity": -0.3, "normal_activity": 0.0, "high_activity": 0.7}
FACE_SCORES = {"happy": (0.8, 0.3), "sad": (-0.7, -0.4), "angry": (-0.8, 0.8), "surprised": (None, 0.7), "neutral": (0.0, None)}
PROSODY_AROUSAL = {"low_intensity": -0.5, "normal_intensity": 0.0, "high_intensity": 0.7}
SENSEVOICE_EMOTION = {"happy": (0.5, 0.2), "sad": (-0.5, -0.3), "angry": (-0.6, 0.6), "neutral": (0.0, 0.0)}
YAMNET_AROUSAL_BONUS = {
    "Shout": 0.15, "Yell": 0.15, "Screaming": 0.20,
    "Crying, sobbing": 0.10, "Alarm": 0.15, "Fire alarm": 0.20,
    "Smash, crash": 0.15, "Bang": 0.12, "Silence": -0.10,
}
SEMANTIC_CATEGORIES = {
    "negative_emotion": {
        "phrases": ("难过", "伤心", "害怕", "紧张", "不舒服", "痛苦", "很糟糕"),
        "valence": -0.25,
    },
    "depressive_expression": {
        "phrases": ("没有希望", "什么都不想做", "没有意义", "撑不下去", "很压抑", "活着很累"),
        "valence": -0.35,
    },
    "agitation": {
        "phrases": ("烦躁", "狂躁", "控制不住", "冷静不下来", "要爆炸了", "烦死了", "砸东西"),
        "valence": -0.30,
    },
    "self_harm_suicide": {
        "phrases": ("不想活了", "不想活下去", "想死", "自杀", "轻生", "结束生命", "伤害自己", "活着没意思", "绝望"),
        "valence": None,
        "sensitive": True,
    },
    "threat_conflict": {
        "phrases": ("杀了你", "打死你", "伤害别人", "报复", "同归于尽", "威胁你", "跟你拼了"),
        "valence": None,
        "sensitive": True,
    },
    "positive_emotion": {
        "phrases": ("开心", "高兴", "舒服", "轻松", "很好", "满意", "谢谢", "喜欢", "有希望"),
        "valence": 0.35,
    },
    "rest_intent": {
        "phrases": ("想休息", "我要休息", "准备睡觉", "想睡觉", "关灯休息", "安静休息"),
        "valence": None,
    },
}
SEMANTIC_NEGATIONS = ("不", "没", "没有", "并非", "不是", "不会", "别", "不要", "从未")
SEMANTIC_NEGATION_WINDOW = 5
SEMANTIC_NEGATION_EXEMPT_PHRASES = ("不想活了", "不想活下去", "控制不住", "太亮了", "太暗了")
SEMANTIC_REPORTED_CONTEXT = ("新闻里", "电影里", "书里", "研究中", "研究显示", "举例", "别人说", "他说", "她说")
SEMANTIC_REPORTED_CONTEXT_WINDOW = 10
SEMANTIC_CONTEXT_EXCLUSIONS = {
    "agitation": ("控制不住笑", "控制不住开心"),
}
LIGHTING_REQUESTS = {
    "off": ("\u5173\u706f", "\u628a\u706f\u5173\u4e86", "\u5173\u95ed\u706f\u5149"),
    "on": ("\u5f00\u706f", "\u628a\u706f\u6253\u5f00", "\u6253\u5f00\u706f\u5149"),
    "brighter": ("\u4eae\u4e00\u70b9", "\u8c03\u4eae", "\u592a\u6697\u4e86"),
    "dimmer": ("\u6697\u4e00\u70b9", "\u8c03\u6697", "\u592a\u4eae\u4e86"),
}
CONTENT_MAX_MATCH_MULTIPLIER = 2
YAMNET_CONTEXT_SCALE = 0.5
NEUTRAL_THRESHOLD = 0.15
STATE_CONFIRM_SECONDS = 3.0
MIN_STATE_HOLD_SECONDS = 5.0
HYSTERESIS = 0.05
INSUFFICIENT_TIMEOUT_SECONDS = 3.0
