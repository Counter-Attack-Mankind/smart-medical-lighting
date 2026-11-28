"""Rate-limited virtual lighting controller with manual priority."""

import time
from config import lighting_config as cfg


class LightingController:
    def __init__(self):
        self.mode, self.current_lux, self.target_lux = "manual", cfg.DEFAULT_LUX, cfg.DEFAULT_LUX
        self.reason, self.status, self.last_update, self.last_request_signature = "manual default", "confirmed", None, None

    def reset_timebase(self):
        self.last_update = None

    def set_mode(self, mode):
        if mode not in ("manual", "automatic"):
            raise ValueError("mode must be manual or automatic")
        self.mode, self.reason, self.status = mode, ("automatic fusion control" if mode == "automatic" else "manual control"), "confirmed"

    def set_manual_lux(self, value):
        if self.mode != "manual":
            raise RuntimeError("manual lux requires manual mode")
        value = float(value)
        if not cfg.MIN_LUX <= value <= cfg.MAX_LUX:
            raise ValueError("lux is outside the allowed range")
        self.target_lux = self.current_lux = value
        self.reason, self.status = "manual target", "confirmed"

    def _automatic_target(self, fusion):
        semantic = (fusion or {}).get("semantic") or {}
        request = semantic.get("lighting_request")
        observation_id = semantic.get("observation_id")
        if request:
            if request in cfg.LIGHTING_REQUEST_LUX:
                self.last_request_signature = (observation_id, request)
                return cfg.LIGHTING_REQUEST_LUX[request], f"explicit request: {request}"
            if (observation_id, request) != self.last_request_signature:
                self.last_request_signature = (observation_id, request)
                delta = cfg.RELATIVE_REQUEST_STEP_LUX if request == "brighter" else -cfg.RELATIVE_REQUEST_STEP_LUX
                self.target_lux += delta
            return self.target_lux, f"explicit request: {request}"
        if semantic.get("rest_intent"):
            return cfg.REST_INTENT_LUX, "explicit rest intent"
        state = (fusion or {}).get("confirmed_state")
        if state in cfg.STATE_TARGET_LUX:
            return cfg.STATE_TARGET_LUX[state], f"confirmed state: {state}"
        return None, "insufficient information; holding illuminance"

    def update(self, fusion=None, timestamp=None):
        timestamp = time.perf_counter() if timestamp is None else float(timestamp)
        dt, self.last_update = (0.0 if self.last_update is None else max(0.0, timestamp - self.last_update)), timestamp
        if self.mode == "automatic":
            target, self.reason = self._automatic_target(fusion)
            if target is not None:
                self.target_lux = max(cfg.MIN_LUX, min(cfg.MAX_LUX, float(target)))
            self.status = "confirmed" if target is not None else "holding"
        difference = self.target_lux - self.current_lux
        rate = cfg.RISE_RATE_LUX_PER_SECOND if difference >= 0 else cfg.FALL_RATE_LUX_PER_SECOND
        step = min(abs(difference), rate * min(dt, 0.5))
        self.current_lux += step if difference >= 0 else -step
        return self.snapshot()

    def snapshot(self):
        return {"mode": self.mode, "current_lux": round(self.current_lux, 2), "target_lux": round(self.target_lux, 2), "reason": self.reason, "status": self.status, "is_simulated": True}
