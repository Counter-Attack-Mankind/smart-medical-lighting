"""Audio acquisition, model and analysis engineering parameters.

Thresholds are initial values that require calibration with the final
microphone, enclosure, room acoustics and deployment hardware. dBFS is a
digital full-scale level and must not be interpreted as physical SPL.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "audio"
SILERO_VAD_MODEL = MODEL_DIR / "silero_vad.onnx"
SENSEVOICE_MODEL_DIR = MODEL_DIR / "sensevoice"
YAMNET_MODEL = MODEL_DIR / "yamnet.tflite"

# Acquisition and continuity
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
MICROPHONE_QUEUE_SIZE = 64
MICROPHONE_DEVICE = None
READ_TIMEOUT_SECONDS = 0.25
TIMESTAMP_TOLERANCE_SECONDS = 0.020

# Silero VAD
VAD_THRESHOLD = 0.50
VAD_MIN_SPEECH_SECONDS = 0.25
VAD_MIN_SILENCE_SECONDS = 0.50
VAD_MAX_SPEECH_SECONDS = 20.0
VAD_BUFFER_SECONDS = 30
VAD_NUM_THREADS = 1

# SenseVoiceSmall
ASR_NUM_THREADS = 2
ASR_LANGUAGE = "zh"
ASR_USE_ITN = True
ASR_PENDING_SEGMENTS_MAX = 4
TRANSCRIPTION_TTL_SECONDS = 15.0

# YAMNet. The model consumes 975 ms windows every 500 ms.
YAMNET_WINDOW_SAMPLES = 15600
YAMNET_HOP_SAMPLES = 8000
YAMNET_CLASS_COUNT = 521
YAMNET_RAW_TOP_COUNT = 10
SOUND_EVENT_TTL_SECONDS = 2.0

# Exact indices and names were verified against the local YAMNet TFLite
# metadata. Events describe the captured environment, not a specific person.
YAMNET_EVENT_CLASSES = {
    0: "Speech", 6: "Shout", 9: "Yell", 10: "Children shouting",
    11: "Screaming", 12: "Whispering", 13: "Laughter",
    19: "Crying, sobbing", 23: "Sigh", 42: "Cough", 58: "Clapping",
    352: "Slam", 353: "Knock", 354: "Tap", 382: "Alarm",
    389: "Alarm clock", 393: "Smoke detector, smoke alarm",
    394: "Fire alarm", 435: "Glass", 437: "Shatter",
    454: "Thump, thud", 460: "Bang", 461: "Slap, smack",
    463: "Smash, crash", 464: "Breaking", 494: "Silence",
}
YAMNET_EVENT_THRESHOLDS = {
    "Speech": 0.35, "Whispering": 0.25, "Shout": 0.30,
    "Yell": 0.30, "Children shouting": 0.35, "Screaming": 0.30,
    "Crying, sobbing": 0.30, "Laughter": 0.30, "Sigh": 0.25,
    "Cough": 0.30, "Clapping": 0.30, "Slam": 0.30,
    "Knock": 0.30, "Tap": 0.30, "Alarm": 0.30,
    "Alarm clock": 0.30, "Smoke detector, smoke alarm": 0.30,
    "Fire alarm": 0.30, "Glass": 0.30, "Shatter": 0.30,
    "Thump, thud": 0.30, "Bang": 0.30, "Slap, smack": 0.30,
    "Smash, crash": 0.30, "Breaking": 0.30, "Silence": 0.50,
}
YAMNET_EVENT_LABELS_ZH = {
    "Speech": "说话声", "Whispering": "耳语", "Shout": "喊叫",
    "Yell": "大喊", "Children shouting": "儿童喊叫", "Screaming": "尖叫",
    "Crying, sobbing": "哭泣/抽泣", "Laughter": "笑声", "Sigh": "叹气",
    "Cough": "咳嗽", "Clapping": "拍手", "Slam": "猛关/撞击",
    "Knock": "敲击", "Tap": "轻敲", "Alarm": "警报",
    "Alarm clock": "闹钟", "Smoke detector, smoke alarm": "烟雾报警器",
    "Fire alarm": "火警", "Glass": "玻璃声", "Shatter": "碎裂声",
    "Thump, thud": "沉闷撞击", "Bang": "砰响", "Slap, smack": "拍打声",
    "Smash, crash": "猛烈撞击", "Breaking": "物品破裂", "Silence": "安静",
}

# Acoustic and prosodic analysis
ANALYSIS_WINDOW_SECONDS = 10.0
ACOUSTIC_EMA_TIME_CONSTANT = 0.20
DBFS_FLOOR = -120.0
F0_FRAME_SAMPLES = 640
F0_HOP_SAMPLES = 320
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
F0_MIN_RMS = 0.003
F0_MIN_PERIODICITY = 0.45
F0_MIN_VALID_COUNT = 3
PITCH_RANGE_LOW_PERCENTILE = 10.0
PITCH_RANGE_HIGH_PERCENTILE = 90.0

# Expression-intensity candidates. These are objective engineering rules, not
# mappings to anger, sadness or any diagnosis.
INTENSITY_MIN_SPEECH_RATIO = 0.10
INTENSITY_HIGH_DBFS = -20.0
INTENSITY_HIGH_PITCH_RANGE_HZ = 55.0
INTENSITY_LOW_DBFS = -38.0
INTENSITY_LOW_PITCH_RANGE_HZ = 30.0
INTENSITY_DBFS_MIN = -55.0
INTENSITY_DBFS_MAX = -12.0
INTENSITY_PITCH_RANGE_MAX_HZ = 120.0
INTENSITY_LEVEL_WEIGHT = 0.60
INTENSITY_PITCH_WEIGHT = 0.40

# Reliability
RESULT_FRESHNESS_SECONDS = 2.0
MIN_TEMPORAL_COVERAGE = 0.60
RELIABILITY_THRESHOLD = 0.60
RELIABILITY_WEIGHTS = {
    "availability": 0.25,
    "continuity": 0.20,
    "freshness": 0.15,
    "temporal_coverage": 0.25,
    "feature_coverage": 0.15,
}

# Visualization
DISPLAY_WIDTH = 1120
DISPLAY_HEIGHT = 680
WAVEFORM_SECONDS = 2.0
DISPLAY_DBFS_MIN = -60.0
DISPLAY_DBFS_MAX = 0.0
