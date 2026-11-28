"""Initial Stage 05.5 demo settings, not clinical prescriptions."""

CONTROL_HZ = 10.0
DEFAULT_LUX = 300.0
MIN_LUX = 0.0
MAX_LUX = 600.0
RISE_RATE_LUX_PER_SECOND = 30.0
FALL_RATE_LUX_PER_SECOND = 30.0

STATE_TARGET_LUX = {
    "tense_active": 200.0,
    "pleasant_active": 300.0,
    "low_negative": 500.0,
    "calm_relaxed": 150.0,
    "neutral": 300.0,
}
REST_INTENT_LUX = 100.0
LIGHTING_REQUEST_LUX = {"off": 0.0, "on": 300.0}
RELATIVE_REQUEST_STEP_LUX = 100.0
