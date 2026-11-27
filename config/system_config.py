"""Unified perception runtime and local dashboard parameters."""

# Scheduling targets. Displayed rates always come from runtime measurements.
CAMERA_TARGET_HZ = 30.0
POSE_TARGET_HZ = 15.0
FACE_TARGET_HZ = 15.0
WEBSOCKET_TARGET_HZ = 10.0
MJPEG_TARGET_HZ = 15.0

CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
JPEG_QUALITY = 82

# Cache freshness
POSE_RESULT_TTL_SECONDS = 1.0
FACE_RESULT_TTL_SECONDS = 1.0
AUDIO_RESULT_TTL_SECONDS = 1.0
VIDEO_FRAME_TTL_SECONDS = 2.0

# Audio waveform sent to the browser: five seconds are retained, but only a
# peak-preserving compressed envelope is transmitted.
WAVEFORM_SECONDS = 5.0
WAVEFORM_POINTS = 500
WAVEFORM_UPDATE_HZ = 10.0

# Runtime and UI state
HISTORY_LIMIT = 50
THREAD_JOIN_TIMEOUT_SECONDS = 10.0
DEFAULT_LIGHTING_MODE = "manual"
DEFAULT_SIMULATED_LUX = 300
MIN_SIMULATED_LUX = 0
MAX_SIMULATED_LUX = 600

FUSION_BASE_WEIGHTS = {
    "pose": 0.35,
    "face": 0.25,
    "speech_content": 0.20,
    "speech_prosody": 0.20,
}
