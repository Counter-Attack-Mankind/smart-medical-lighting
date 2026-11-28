"""Face Landmarker configuration."""

from pathlib import Path


# Project and model paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face"
    / "face_landmarker.task"
)


# Face Landmarker parameters
NUM_FACES = 1

MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

OUTPUT_FACE_BLENDSHAPES = True
OUTPUT_FACIAL_TRANSFORMATION_MATRIXES = True

# Pose-guided head ROI. The ROI is used only when it belongs to the exact same
# captured frame; otherwise Face Landmarker falls back to the full image.
POSE_HEAD_ROI_ENABLED = True
POSE_HEAD_ROI_WAIT_SECONDS = 0.045
POSE_HEAD_ROI_CACHE_SIZE = 16
POSE_HEAD_VISIBILITY_MIN = 0.45
POSE_HEAD_PRESENCE_MIN = 0.45
POSE_HEAD_MIN_VALID_POINTS = 3
POSE_HEAD_MIN_SIZE_PX = 72
POSE_HEAD_FACE_SPAN_SCALE = 2.4
POSE_HEAD_SHOULDER_SCALE = 0.78
POSE_HEAD_HEIGHT_TO_WIDTH = 1.20
POSE_HEAD_EXPANSION = 1.22
POSE_HEAD_CENTER_Y_OFFSET = 0.08
POSE_HEAD_ROI_MIN_RELIABILITY = 0.50
POSE_HEAD_ROI_RELIABILITY_WEIGHT = 0.20


# Expected model outputs
NUM_LANDMARKS = 478
NUM_BLENDSHAPES = 52


# Camera parameters
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# The names follow the 52-output MediaPipe Blendshape model. The coefficients
# are action intensities in [0, 1], not emotion probabilities.
BLENDSHAPE_NAMES = (
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)

# Temporal analysis. All values below are engineering starting points and
# require calibration with the final camera, distance and patient population.
WINDOW_SECONDS = 3.0
EMA_TIME_CONSTANT = 0.12
MAX_OBSERVATION_GAP = 0.35
EXPRESSION_CONFIRM_SECONDS = {
    "happy": 0.40,
    "sad": 0.25,
    "angry": 0.25,
    "surprised": 0.22,
    "neutral": 0.32,
    "unknown": 0.25,
}
MIN_TEMPORAL_COVERAGE = 0.55

# Reliability
MIN_FACE_SCALE = 0.08
GOOD_FACE_SCALE = 0.18
MIN_LANDMARK_RANGE_RATIO = 0.95
MIN_BLENDSHAPE_COVERAGE = 0.90
MAX_RELIABLE_YAW_DEG = 55.0
MAX_RELIABLE_PITCH_DEG = 40.0
RELIABILITY_THRESHOLD = 0.60
RELIABILITY_WEIGHTS = {
    "landmarks": 0.25,
    "face_scale": 0.20,
    "blendshapes": 0.25,
    "head_pose": 0.15,
    "geometry": 0.15,
}

# Pair means and asymmetry use abs(left - right). Rule scores are independent
# weighted action scores; they are never passed through Softmax.
EXPRESSION_WEIGHTS = {
    "happy": {
        "mouth_smile": 0.55,
        "cheek_squint": 0.20,
        "eye_squint": 0.15,
        "mouth_upper_up": 0.10,
    },
    "sad": {
        "mouth_frown": 0.40,
        "brow_inner_up": 0.30,
        "mouth_lower_down": 0.18,
        "mouth_press": 0.12,
    },
    "angry": {
        "brow_down": 0.38,
        "eye_squint": 0.18,
        "nose_sneer": 0.18,
        "mouth_press": 0.14,
        "mouth_frown": 0.12,
    },
    "surprised": {
        "eye_wide": 0.30,
        "brow_inner_up": 0.20,
        "brow_outer_up": 0.15,
        "jaw_open": 0.25,
        "mouth_stretch": 0.10,
    },
}
EXPRESSION_CONFLICTS = {
    "happy": {"mouth_frown": 0.25, "brow_down": 0.12},
    "sad": {"mouth_smile": 0.30},
    "angry": {"mouth_smile": 0.25, "eye_wide": 0.10},
    "surprised": {"eye_blink": 0.20, "eye_squint": 0.15},
}

# Sadness has two explainable paths. A mild path combines inner-brow lift with
# another mouth/brow cue. A crying-like path accepts inner-brow lift or strong
# brow tension, then requires eye/lower-face support; mouthFrown is supportive
# but is not mandatory. These are appearance rules, not an inference of emotion.
SAD_MILD_BROW_INNER_MIN = 0.08
SAD_MILD_SUPPORT_MIN = 0.08
CRY_BROW_INNER_MIN = 0.10
CRY_BROW_DOWN_MIN = 0.22
CRY_EYE_CLOSURE_MIN = 0.16
CRY_EYE_SQUINT_MIN = 0.10
CRY_MOUTH_DISTRESS_MIN = 0.10
CRY_REQUIRED_COMPONENTS = 3
CRY_SCORE_WEIGHTS = {
    "brow_inner_up": 0.24,
    "brow_down": 0.18,
    "eye_blink": 0.16,
    "eye_squint": 0.10,
    "mouth_lower_down": 0.12,
    "jaw_open": 0.10,
    "mouth_frown": 0.06,
    "mouth_press": 0.04,
}
CRY_SAD_SCORE_GAIN = 1.18
CRY_HAPPY_SUPPRESSION = 0.82
CRY_SUPPRESSION_FULL_SCALE = 0.42
EXPRESSION_MIN_SCORE = {
    "happy": 0.32,
    "sad": 0.22,
    "angry": 0.21,
    "surprised": 0.22,
}
EXPRESSION_MIN_MARGIN = {
    "happy": 0.04,
    "sad": 0.02,
    "angry": 0.02,
    "surprised": 0.02,
}
EXPRESSION_SWITCH_MARGIN = {
    "happy": 0.08,
    "sad": 0.04,
    "angry": 0.04,
    "surprised": 0.04,
    "neutral": 0.05,
    "unknown": 0.04,
}
EXPRESSION_EXIT_SCORE = {
    "happy": 0.24,
    "sad": 0.18,
    "angry": 0.18,
    "surprised": 0.18,
}
NEUTRAL_EXIT_SCORE = 0.62

# Necessary multi-action evidence. These prevent a single mouth/eye action
# from being mapped directly to an expression candidate.
HAPPY_SMILE_MIN = 0.25
HAPPY_SUPPORT_MIN = 0.08
# Legacy names remain for calibration notes; sadness gating now uses the two
# paths above instead of requiring mouthFrown.
SAD_FROWN_MIN = 0.12
SAD_SUPPORT_MIN = 0.08
ANGRY_BROW_MIN = 0.14
ANGRY_SUPPORT_MIN = 0.08
SURPRISE_EYE_WIDE_MIN = 0.13
SURPRISE_BROW_MIN = 0.10
SURPRISE_JAW_MIN = 0.12

# Neutral is inferred from low expressive activation; the model's `_neutral`
# coefficient is deliberately not used.
NEUTRAL_FEATURE_WEIGHTS = {
    "mouth_smile": 0.18,
    "mouth_frown": 0.14,
    "brow_down": 0.14,
    "brow_inner_up": 0.10,
    "eye_wide": 0.10,
    "eye_squint": 0.08,
    "nose_sneer": 0.08,
    "jaw_open": 0.10,
    "mouth_press": 0.08,
}
NEUTRAL_MAX_ACTION_WEIGHT = 0.55
NEUTRAL_MEAN_ACTION_WEIGHT = 0.45
NEUTRAL_MIN_SCORE = 0.68
NEUTRAL_ACTION_MAX = 0.24
NEUTRAL_EXPRESSION_MAX = 0.32

# Objective facial actions and motion
EYE_CLOSED_THRESHOLD = 0.62
BLINK_ARM_THRESHOLD = 0.22
BLINK_TRIGGER_THRESHOLD = 0.62
BLINK_EVENT_HOLD_SECONDS = 0.22
MOUTH_OPEN_THRESHOLD = 0.30
FACE_ACTIVITY_FULL_SCALE = 2.0
HEAD_MOTION_FULL_SCALE_DEG_PER_SEC = 120.0
ACTIVITY_BLENDSHAPES = tuple(
    name for name in BLENDSHAPE_NAMES
    if name != "_neutral" and not name.startswith("eyeLook")
)
