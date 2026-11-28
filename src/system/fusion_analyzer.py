"""Reliability-weighted Russell valence/arousal fusion and state FSM."""

import math
from collections import deque

from config import fusion_config as cfg


class FusionAnalyzer:
    def __init__(self):
        self.smoothed = {"valence": None, "arousal": None}
        self.last_evidence_time = None
        self.last_signature = None
        self.trajectory = deque()
        self.confirmed_state = "unknown"
        self.candidate_state = "unknown"
        self.candidate_since = None
        self.confirmed_since = None
        self.insufficient_since = None

    def reset(self):
        self.__init__()

    @staticmethod
    def _clamp(value):
        return None if value is None else float(max(-1.0, min(1.0, value)))

    @staticmethod
    def _reliability(module):
        if not module or not module.get("available"):
            return 0.0
        analysis = module.get("analysis") or {}
        reliability = analysis.get("reliability")
        if isinstance(reliability, dict):
            score = reliability.get("score")
            if score is None:
                values = [v for v in reliability.values() if isinstance(v, (int, float))]
                score = sum(values) / len(values) if values else None
        else:
            score = reliability
        score = 1.0 if score is None else float(score)
        age = module.get("observation_age")
        freshness = 1.0 if age is None else max(0.0, 1.0 - float(age) / 2.0)
        return max(0.0, min(1.0, score * freshness))

    @staticmethod
    def _pose(module):
        return None, cfg.POSE_AROUSAL.get(((module or {}).get("analysis") or {}).get("motion_state"))

    @staticmethod
    def _face(module):
        analysis = (module or {}).get("analysis") or {}
        expression = analysis.get("stable_expression")
        if expression not in cfg.FACE_SCORES:
            expression = analysis.get("raw_expression")
        score = cfg.FACE_SCORES.get(expression)
        if not score:
            return None, None
        confidence = (analysis.get("expression_scores") or {}).get(expression, 1.0)
        confidence = 1.0 if confidence is None else confidence
        return tuple(None if item is None else float(item) * float(confidence) for item in score)

    @staticmethod
    def _phrase_matches(text, phrases, category=None):
        matches = []
        exclusions = cfg.SEMANTIC_CONTEXT_EXCLUSIONS.get(category, ())
        for phrase in sorted(phrases, key=len, reverse=True):
            start = text.find(phrase)
            while start >= 0:
                prefix = text[max(0, start - cfg.SEMANTIC_NEGATION_WINDOW):start]
                reported = text[max(0, start - cfg.SEMANTIC_REPORTED_CONTEXT_WINDOW):start]
                negated = (
                    phrase not in cfg.SEMANTIC_NEGATION_EXEMPT_PHRASES
                    and any(prefix.endswith(term) for term in cfg.SEMANTIC_NEGATIONS)
                )
                excluded = any(term in text for term in exclusions)
                if not negated and not excluded and not any(term in reported for term in cfg.SEMANTIC_REPORTED_CONTEXT):
                    matches.append(phrase)
                    break
                start = text.find(phrase, start + len(phrase))
        return matches

    @classmethod
    def _content(cls, module):
        analysis = (module or {}).get("analysis") or {}
        transcription = analysis.get("transcription") or {}
        text = str(transcription.get("text") or "").strip()
        empty = {
            "categories": [], "alerts": [], "matched_phrases": [],
            "rest_intent": False, "lighting_request": None,
            "lighting_request_matches": [],
            "observation_id": None,
        }
        if not text:
            return (None, None), empty
        categories, alerts, valence_terms = [], [], []
        for name, settings in cfg.SEMANTIC_CATEGORIES.items():
            matches = cls._phrase_matches(text, settings["phrases"], name)
            if not matches:
                continue
            item = {"category": name, "matched_phrases": matches}
            categories.append(item)
            if settings.get("sensitive"):
                alerts.append({
                    **item, "requires_human_review": True,
                    "affects_fusion": False, "affects_lighting": False,
                })
            elif settings.get("valence") is not None:
                multiplier = min(len(matches), cfg.CONTENT_MAX_MATCH_MULTIPLIER)
                valence_terms.append(settings["valence"] * multiplier)
        request = None
        request_matches = []
        for name, phrases in cfg.LIGHTING_REQUESTS.items():
            matches = cls._phrase_matches(text, phrases, "lighting_request")
            if matches:
                request, request_matches = name, matches
                categories.append({"category": "lighting_request", "matched_phrases": matches, "request": name})
                break
        all_matches = list(dict.fromkeys(
            phrase for item in categories for phrase in item["matched_phrases"]
        ))
        valence = cls._clamp(sum(valence_terms)) if valence_terms else None
        semantic = {
            "categories": categories,
            "alerts": alerts,
            "matched_phrases": all_matches,
            "rest_intent": any(item["category"] == "rest_intent" for item in categories),
            "lighting_request": request,
            "lighting_request_matches": request_matches,
            "observation_id": transcription.get("observed_time") or transcription.get("end_time"),
        }
        return (valence, None), semantic

    @staticmethod
    def _prosody(module):
        analysis = (module or {}).get("analysis") or {}
        arousal = cfg.PROSODY_AROUSAL.get(analysis.get("expression_intensity"))
        valence = None
        emotion = str(((analysis.get("transcription") or {}).get("emotion") or "")).lower()
        for name, score in cfg.SENSEVOICE_EMOTION.items():
            if name in emotion:
                valence, bonus = score
                arousal = bonus if arousal is None else max(-1.0, min(1.0, arousal + bonus * 0.25))
                break
        sound = analysis.get("sound_event_result") or {}
        if sound.get("is_current"):
            context = 0.0
            for event in sound.get("events") or []:
                name = event.get("class_name") or event.get("name") or event.get("label")
                context += cfg.YAMNET_AROUSAL_BONUS.get(name, 0.0) * float(event.get("score", 0.0))
            if context:
                arousal = (0.0 if arousal is None else arousal) + context * cfg.YAMNET_CONTEXT_SCALE
                arousal = max(-1.0, min(1.0, arousal))
        return valence, arousal

    @staticmethod
    def _state(valence, arousal, previous="unknown"):
        if valence is None or arousal is None:
            return "unknown"
        limit = cfg.NEUTRAL_THRESHOLD + (cfg.HYSTERESIS if previous == "neutral" else 0.0)
        if abs(valence) <= limit and abs(arousal) <= limit:
            return "neutral"
        if valence >= cfg.NEUTRAL_THRESHOLD:
            return "pleasant_active" if arousal >= 0.0 else "calm_relaxed"
        if valence <= -cfg.NEUTRAL_THRESHOLD:
            return "tense_active" if arousal >= 0.0 else "low_negative"
        return "pleasant_active" if arousal > cfg.NEUTRAL_THRESHOLD else ("calm_relaxed" if arousal < -cfg.NEUTRAL_THRESHOLD else "neutral")

    def _fsm(self, candidate, timestamp, temporal_active, new_evidence):
        if not temporal_active:
            return self.confirmed_state, "snapshot"
        if candidate == "unknown":
            self.insufficient_since = timestamp if self.insufficient_since is None else self.insufficient_since
            if timestamp - self.insufficient_since >= cfg.INSUFFICIENT_TIMEOUT_SECONDS:
                self.confirmed_state, self.confirmed_since = "unknown", timestamp
            return self.confirmed_state, "insufficient"
        self.insufficient_since = None
        if not new_evidence:
            return self.confirmed_state, "confirmed" if self.confirmed_state == candidate else "pending"
        if candidate != self.candidate_state:
            self.candidate_state, self.candidate_since = candidate, timestamp
        held = timestamp - self.confirmed_since if self.confirmed_since is not None else math.inf
        ready = timestamp - self.candidate_since >= cfg.STATE_CONFIRM_SECONDS
        if candidate != self.confirmed_state and ready and held >= cfg.MIN_STATE_HOLD_SECONDS:
            self.confirmed_state, self.confirmed_since = candidate, timestamp
        return self.confirmed_state, "confirmed" if self.confirmed_state == candidate else "pending"

    def analyze(self, pose, face, audio, timestamp, temporal_active=True):
        values = {"pose": self._pose(pose), "face": self._face(face), "speech_prosody": self._prosody(audio)}
        values["speech_content"], semantic = self._content(audio)
        reliability = {"pose": self._reliability(pose), "face": self._reliability(face), "speech_content": self._reliability(audio), "speech_prosody": self._reliability(audio)}
        effective, fused, totals = {}, {}, {}
        for index, dimension in enumerate(("valence", "arousal")):
            raw = {name: cfg.BASE_WEIGHTS[name] * reliability[name] for name in cfg.BASE_WEIGHTS if values[name][index] is not None}
            total = sum(raw.values())
            totals[dimension] = total
            if total >= cfg.MIN_EFFECTIVE_WEIGHT:
                effective[dimension] = {name: raw.get(name, 0.0) / total for name in cfg.BASE_WEIGHTS}
                fused[dimension] = sum(effective[dimension][name] * values[name][index] for name in raw)
            else:
                effective[dimension] = {name: None for name in cfg.BASE_WEIGHTS}
                fused[dimension] = None
        signature = tuple((m or {}).get("captured_at") for m in (pose, face, audio))
        new_evidence = signature != self.last_signature and any(item is not None for item in signature)
        if new_evidence:
            dt = None if self.last_evidence_time is None else max(0.0, timestamp - self.last_evidence_time)
            for dimension in ("valence", "arousal"):
                raw = fused[dimension]
                if raw is None:
                    self.smoothed[dimension] = None
                elif self.smoothed[dimension] is None or dt is None:
                    self.smoothed[dimension] = raw
                else:
                    alpha = 1.0 - math.exp(-min(dt, 0.5) / cfg.SMOOTHING_TIME_CONSTANT_SECONDS)
                    self.smoothed[dimension] += alpha * (raw - self.smoothed[dimension])
            self.last_signature, self.last_evidence_time = signature, timestamp
        candidate = self._state(self.smoothed["valence"], self.smoothed["arousal"], self.confirmed_state)
        confirmed, status = self._fsm(candidate, timestamp, temporal_active, new_evidence)
        if temporal_active and new_evidence and all(self.smoothed[d] is not None for d in self.smoothed):
            self.trajectory.append({"timestamp": float(timestamp), **self.smoothed})
        while self.trajectory and timestamp - self.trajectory[0]["timestamp"] > cfg.TRAJECTORY_SECONDS:
            self.trajectory.popleft()
        return {"available": any(v is not None for v in self.smoothed.values()), "status": status, "message": "fusion active" if new_evidence else "waiting for new observation", "valence": self.smoothed["valence"], "arousal": self.smoothed["arousal"], "raw_valence": fused["valence"], "raw_arousal": fused["arousal"], "state_candidate": candidate, "confirmed_state": confirmed, "state_duration": 0.0 if self.confirmed_since is None else max(0.0, timestamp - self.confirmed_since), "base_weights": dict(cfg.BASE_WEIGHTS), "effective_weights": effective, "effective_weight_totals": totals, "modality_scores": values, "modality_reliability": reliability, "semantic": semantic, "trajectory": list(self.trajectory), "new_evidence": new_evidence}
