"""NumPy facial-action, expression-candidate and temporal analysis."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import face_config as cfg


BLENDSHAPE_NAMES = cfg.BLENDSHAPE_NAMES
BLENDSHAPE_INDEX = {name: index for index, name in enumerate(BLENDSHAPE_NAMES)}
ACTIVITY_INDICES = np.asarray(
    [BLENDSHAPE_INDEX[name] for name in cfg.ACTIVITY_BLENDSHAPES], dtype=int
)
EXPRESSIONS = ("happy", "sad", "angry", "surprised", "neutral")


@dataclass
class FaceFrame:
    """One converted Face Landmarker observation."""

    timestamp: float
    face_detected: bool
    landmarks: np.ndarray
    raw_blendshapes: np.ndarray
    smoothed_blendshapes: np.ndarray
    transformation_matrix: np.ndarray | None
    raw_head_pose: dict | None
    smoothed_head_pose: dict | None
    landmark_range_ratio: float
    blendshape_coverage: float
    face_scale: float | None
    geometry_valid: bool
    reliability_score: float
    reliable: bool
    blink_candidate: bool | None


class ExpressionState:
    """Actual-time confirmation with score-based exit and switch hysteresis."""

    def __init__(self):
        self.state = "unknown"
        self.since = None
        self.candidate = None
        self.candidate_since = None

    def reset(self, timestamp):
        self.state = "unknown"
        self.since = timestamp
        self.candidate = None
        self.candidate_since = None

    def update(self, candidate, scores, timestamp, reliable):
        if not reliable or scores is None:
            self.reset(timestamp)
            return self.state

        if self.state != "unknown":
            current_score = scores.get(self.state, 0.0)
            exit_score = (
                cfg.NEUTRAL_EXIT_SCORE
                if self.state == "neutral"
                else cfg.EXPRESSION_EXIT_SCORE
            )
            challenger = scores.get(candidate, 0.0) if candidate != "unknown" else 0.0
            if (
                current_score >= exit_score
                and (
                    candidate == self.state
                    or challenger < current_score + cfg.EXPRESSION_SWITCH_MARGIN
                )
            ):
                self.candidate = None
                self.candidate_since = None
                return self.state

        if candidate == self.state:
            self.candidate = None
            self.candidate_since = None
            return self.state

        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = timestamp
            return self.state

        if timestamp - self.candidate_since >= cfg.EXPRESSION_CONFIRM_SECONDS:
            self.state = candidate
            self.since = timestamp
            self.candidate = None
            self.candidate_since = None
        return self.state

    def duration(self, timestamp):
        if self.state == "unknown" or self.since is None:
            return 0.0
        return max(0.0, timestamp - self.since)


class FaceAnalyzer:
    """Convert Face Landmarker output once and expose JSON-safe analysis."""

    def __init__(self, window_seconds=cfg.WINDOW_SECONDS):
        self.window_seconds = float(window_seconds)
        self.frames = deque()
        self.motion_samples = deque()
        self.expression_state = ExpressionState()
        self.blink_armed = False
        self.last_blink_time = None

    def update(self, result, timestamp_ms, frame_shape):
        timestamp = float(timestamp_ms) / 1000.0
        if self.frames and timestamp <= self.frames[-1].timestamp:
            raise ValueError("Face timestamps must be increasing")

        detected = bool(result.face_landmarks)
        landmarks = np.full((cfg.NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
        raw = np.full(cfg.NUM_BLENDSHAPES, np.nan, dtype=np.float32)
        matrix = None

        if detected and len(result.face_landmarks[0]) == cfg.NUM_LANDMARKS:
            landmarks = np.asarray(
                [[point.x, point.y, point.z] for point in result.face_landmarks[0]],
                dtype=np.float32,
            )

        if detected and result.face_blendshapes:
            for item in result.face_blendshapes[0]:
                index = BLENDSHAPE_INDEX.get(item.category_name)
                if index is not None and np.isfinite(item.score):
                    raw[index] = np.clip(float(item.score), 0.0, 1.0)

        if detected and result.facial_transformation_matrixes:
            candidate = np.asarray(
                result.facial_transformation_matrixes[0], dtype=float
            )
            if candidate.size == 16 and np.isfinite(candidate).all():
                matrix = candidate.reshape(4, 4)

        range_ratio, face_scale, geometry_valid = self._landmark_quality(landmarks)
        blendshape_coverage = float(np.count_nonzero(np.isfinite(raw)) / len(raw))
        raw_head_pose = self.matrix_to_euler(matrix)
        reliability_score, reliable = self._reliability(
            detected,
            range_ratio,
            face_scale,
            geometry_valid,
            blendshape_coverage,
            raw_head_pose,
        )

        smoothed = raw.copy()
        smoothed_head = dict(raw_head_pose) if raw_head_pose is not None else None
        previous = self.frames[-1] if self.frames else None
        if previous is not None and reliable and previous.reliable:
            dt = timestamp - previous.timestamp
            if dt <= cfg.MAX_OBSERVATION_GAP:
                alpha = 1.0 - np.exp(-dt / max(cfg.EMA_TIME_CONSTANT, 1e-6))
                common = np.isfinite(raw) & np.isfinite(previous.smoothed_blendshapes)
                smoothed[common] = (
                    alpha * raw[common]
                    + (1.0 - alpha) * previous.smoothed_blendshapes[common]
                )
                if smoothed_head is not None and previous.smoothed_head_pose is not None:
                    for axis in ("yaw", "pitch", "roll"):
                        delta = self._angle_delta(
                            smoothed_head[axis], previous.smoothed_head_pose[axis]
                        )
                        smoothed_head[axis] = (
                            previous.smoothed_head_pose[axis] + alpha * delta
                        )

        blink_candidate = self._update_blink(raw, reliable, timestamp)
        frame = FaceFrame(
            timestamp=timestamp,
            face_detected=detected,
            landmarks=landmarks,
            raw_blendshapes=raw,
            smoothed_blendshapes=smoothed,
            transformation_matrix=matrix,
            raw_head_pose=raw_head_pose,
            smoothed_head_pose=smoothed_head,
            landmark_range_ratio=range_ratio,
            blendshape_coverage=blendshape_coverage,
            face_scale=face_scale,
            geometry_valid=geometry_valid,
            reliability_score=reliability_score,
            reliable=reliable,
            blink_candidate=blink_candidate,
        )

        # An unreliable observation terminates the current temporal segment.
        # Keeping only that boundary frame makes recovery coverage start fresh
        # and prevents pre-loss motion from leaking into the new segment.
        if not reliable:
            self.frames.clear()
            self.motion_samples.clear()
            previous = None
        self.frames.append(frame)
        if previous is not None:
            motion = self._motion_between(previous, frame)
            if motion is not None:
                self.motion_samples.append(motion)
        self._trim(timestamp)
        return frame

    def _trim(self, timestamp):
        while self.frames and timestamp - self.frames[0].timestamp > self.window_seconds:
            self.frames.popleft()
        while (
            self.motion_samples
            and timestamp - self.motion_samples[0]["timestamp"] > self.window_seconds
        ):
            self.motion_samples.popleft()

    @staticmethod
    def _landmark_quality(landmarks):
        finite = np.isfinite(landmarks).all(axis=1)
        if not np.any(finite):
            return 0.0, None, False
        xy = landmarks[:, :2]
        in_range = (
            finite
            & (xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0)
            & (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0)
        )
        ratio = float(np.count_nonzero(in_range) / cfg.NUM_LANDMARKS)
        valid_xy = xy[in_range]
        if len(valid_xy) < 2:
            return ratio, None, False
        extent = np.ptp(valid_xy, axis=0)
        scale = float(np.sqrt(max(extent[0] * extent[1], 0.0)))
        eye_distance = None
        if finite[33] and finite[263]:
            eye_distance = float(np.linalg.norm(xy[33] - xy[263]))
        geometry_valid = bool(
            np.count_nonzero(finite) == cfg.NUM_LANDMARKS
            and min(extent) > 1e-4
            and eye_distance is not None
            and eye_distance > 0.01
        )
        return ratio, scale, geometry_valid

    @staticmethod
    def matrix_to_euler(matrix):
        """Extract Rz(roll) @ Ry(yaw) @ Rx(pitch), in right-handed degrees.

        The returned transform maps the canonical face into MediaPipe's metric
        face frame. Positive angles follow the right-hand rule around +Y, +X
        and +Z respectively. A mirrored preview does not change these signs.
        """
        if matrix is None:
            return None
        rotation = np.asarray(matrix[:3, :3], dtype=float)
        try:
            u, _, vh = np.linalg.svd(rotation)
        except np.linalg.LinAlgError:
            return None
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        yaw = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
        cosine = np.cos(yaw)
        if abs(cosine) > 1e-6:
            pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
            roll = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
            roll = 0.0
        return {
            "yaw": float(np.degrees(yaw)),
            "pitch": float(np.degrees(pitch)),
            "roll": float(np.degrees(roll)),
        }

    @staticmethod
    def _angle_delta(current, previous):
        return float((current - previous + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _reliability(
        detected, range_ratio, face_scale, geometry_valid, blendshape_coverage,
        head_pose,
    ):
        if not detected:
            return 0.0, False
        landmark_score = float(
            np.clip(range_ratio / cfg.MIN_LANDMARK_RANGE_RATIO, 0.0, 1.0)
        )
        if face_scale is None:
            scale_score = 0.0
        else:
            denominator = max(cfg.GOOD_FACE_SCALE - cfg.MIN_FACE_SCALE, 1e-6)
            scale_score = float(
                np.clip((face_scale - cfg.MIN_FACE_SCALE) / denominator, 0.0, 1.0)
            )
        blendshape_score = float(np.clip(blendshape_coverage, 0.0, 1.0))
        if head_pose is None:
            head_score = 0.0
            direction_ok = False
        else:
            yaw_ratio = abs(head_pose["yaw"]) / cfg.MAX_RELIABLE_YAW_DEG
            pitch_ratio = abs(head_pose["pitch"]) / cfg.MAX_RELIABLE_PITCH_DEG
            head_score = float(np.clip(1.0 - 0.5 * (yaw_ratio + pitch_ratio), 0.0, 1.0))
            direction_ok = yaw_ratio <= 1.0 and pitch_ratio <= 1.0
        components = {
            "landmarks": landmark_score,
            "face_scale": scale_score,
            "blendshapes": blendshape_score,
            "head_pose": head_score,
            "geometry": 1.0 if geometry_valid else 0.0,
        }
        score = float(sum(
            cfg.RELIABILITY_WEIGHTS[name] * value
            for name, value in components.items()
        ))
        reliable = bool(
            score >= cfg.RELIABILITY_THRESHOLD
            and range_ratio >= cfg.MIN_LANDMARK_RANGE_RATIO
            and face_scale is not None and face_scale >= cfg.MIN_FACE_SCALE
            and blendshape_coverage >= cfg.MIN_BLENDSHAPE_COVERAGE
            and geometry_valid and direction_ok
        )
        return score, reliable

    def _update_blink(self, raw, reliable, timestamp):
        blink = self._pair(raw, "eyeBlink")
        if not reliable or blink[0] is None:
            self.blink_armed = False
            self.last_blink_time = None
            return None
        value = blink[0]
        if value <= cfg.BLINK_ARM_THRESHOLD:
            self.blink_armed = True
        elif value >= cfg.BLINK_TRIGGER_THRESHOLD and self.blink_armed:
            self.last_blink_time = timestamp
            self.blink_armed = False
        return bool(
            self.last_blink_time is not None
            and timestamp - self.last_blink_time <= cfg.BLINK_EVENT_HOLD_SECONDS
        )

    def _motion_between(self, previous, current):
        dt = current.timestamp - previous.timestamp
        if (
            dt <= 0.0 or dt > cfg.MAX_OBSERVATION_GAP
            or not previous.reliable or not current.reliable
        ):
            return None
        common = (
            np.isfinite(previous.smoothed_blendshapes[ACTIVITY_INDICES])
            & np.isfinite(current.smoothed_blendshapes[ACTIVITY_INDICES])
        )
        indices = ACTIVITY_INDICES[common]
        face_activity = None
        if len(indices):
            velocity = (
                current.smoothed_blendshapes[indices]
                - previous.smoothed_blendshapes[indices]
            ) / dt
            face_activity = float(np.sqrt(np.mean(np.square(velocity))))
        head_motion = None
        if previous.smoothed_head_pose is not None and current.smoothed_head_pose is not None:
            angular_velocity = [
                self._angle_delta(
                    current.smoothed_head_pose[axis],
                    previous.smoothed_head_pose[axis],
                ) / dt
                for axis in ("yaw", "pitch", "roll")
            ]
            head_motion = float(np.linalg.norm(angular_velocity))
        return {
            "timestamp": current.timestamp,
            "dt": dt,
            "face_activity": face_activity,
            "head_motion": head_motion,
        }

    @staticmethod
    def _value(values, name):
        value = values[BLENDSHAPE_INDEX[name]]
        return None if not np.isfinite(value) else float(value)

    @classmethod
    def _pair(cls, values, prefix):
        left = cls._value(values, prefix + "Left")
        right = cls._value(values, prefix + "Right")
        if left is None or right is None:
            return None, None
        return (left + right) / 2.0, abs(left - right)

    @classmethod
    def extract_action_features(cls, values):
        pairs = {}
        for output_name, blendshape_name in (
            ("mouth_smile", "mouthSmile"),
            ("mouth_frown", "mouthFrown"),
            ("brow_down", "browDown"),
            ("brow_outer_up", "browOuterUp"),
            ("cheek_squint", "cheekSquint"),
            ("eye_blink", "eyeBlink"),
            ("eye_squint", "eyeSquint"),
            ("eye_wide", "eyeWide"),
            ("mouth_lower_down", "mouthLowerDown"),
            ("mouth_press", "mouthPress"),
            ("mouth_stretch", "mouthStretch"),
            ("mouth_upper_up", "mouthUpperUp"),
            ("nose_sneer", "noseSneer"),
        ):
            mean, asymmetry = cls._pair(values, blendshape_name)
            pairs[output_name] = mean
            pairs[output_name + "_asymmetry"] = asymmetry
        pairs.update({
            "brow_inner_up": cls._value(values, "browInnerUp"),
            "jaw_open": cls._value(values, "jawOpen"),
            "mouth_close": cls._value(values, "mouthClose"),
            "mouth_funnel": cls._value(values, "mouthFunnel"),
            "mouth_pucker": cls._value(values, "mouthPucker"),
        })
        return pairs

    @staticmethod
    def expression_scores(features):
        required = set(cfg.NEUTRAL_FEATURE_WEIGHTS)
        if any(features.get(name) is None for name in required):
            return None
        scores = {}
        for expression in ("happy", "sad", "angry", "surprised"):
            positive = sum(
                weight * features[name]
                for name, weight in cfg.EXPRESSION_WEIGHTS[expression].items()
            )
            conflict = sum(
                weight * features[name]
                for name, weight in cfg.EXPRESSION_CONFLICTS[expression].items()
            )
            scores[expression] = float(np.clip(positive - conflict, 0.0, 1.0))

        if not (
            features["mouth_smile"] >= cfg.HAPPY_SMILE_MIN
            and max(
                features["cheek_squint"], features["eye_squint"],
                features["mouth_upper_up"],
            ) >= cfg.HAPPY_SUPPORT_MIN
        ):
            scores["happy"] = 0.0
        if not (
            features["mouth_frown"] >= cfg.SAD_FROWN_MIN
            and max(features["brow_inner_up"], features["mouth_lower_down"])
            >= cfg.SAD_SUPPORT_MIN
        ):
            scores["sad"] = 0.0
        if not (
            features["brow_down"] >= cfg.ANGRY_BROW_MIN
            and max(
                features["eye_squint"], features["nose_sneer"],
                features["mouth_press"], features["mouth_frown"],
            ) >= cfg.ANGRY_SUPPORT_MIN
        ):
            scores["angry"] = 0.0
        if not (
            features["eye_wide"] >= cfg.SURPRISE_EYE_WIDE_MIN
            and max(features["brow_inner_up"], features["brow_outer_up"])
            >= cfg.SURPRISE_BROW_MIN
            and features["jaw_open"] >= cfg.SURPRISE_JAW_MIN
        ):
            scores["surprised"] = 0.0

        weighted_activation = sum(
            weight * features[name]
            for name, weight in cfg.NEUTRAL_FEATURE_WEIGHTS.items()
        )
        maximum_activation = max(features[name] for name in required)
        scores["neutral"] = float(np.clip(
            1.0
            - cfg.NEUTRAL_MAX_ACTION_WEIGHT * maximum_activation
            - cfg.NEUTRAL_MEAN_ACTION_WEIGHT * weighted_activation,
            0.0,
            1.0,
        ))
        return scores

    @staticmethod
    def select_candidate(scores, features):
        if scores is None:
            return "unknown"
        ranked = sorted(
            ((scores[name], name) for name in ("happy", "sad", "angry", "surprised")),
            reverse=True,
        )
        best_score, best_name = ranked[0]
        second_score = ranked[1][0]
        if (
            best_score >= cfg.EXPRESSION_MIN_SCORE[best_name]
            and best_score - second_score >= cfg.EXPRESSION_MIN_MARGIN
        ):
            return best_name
        expressive_max = max(scores[name] for name in ("happy", "sad", "angry", "surprised"))
        action_max = max(features[name] for name in cfg.NEUTRAL_FEATURE_WEIGHTS)
        if (
            scores["neutral"] >= cfg.NEUTRAL_MIN_SCORE
            and action_max <= cfg.NEUTRAL_ACTION_MAX
            and expressive_max <= cfg.NEUTRAL_EXPRESSION_MAX
        ):
            return "neutral"
        return "unknown"

    def _temporal_coverage(self):
        if len(self.frames) < 2:
            return 0.0
        duration = self.frames[-1].timestamp - self.frames[0].timestamp
        if duration <= 0.0:
            return 0.0
        covered = sum(sample["dt"] for sample in self.motion_samples)
        return float(np.clip(covered / duration, 0.0, 1.0))

    def _weighted_motion(self, key):
        samples = [
            (sample[key], sample["dt"])
            for sample in self.motion_samples if sample[key] is not None
        ]
        if not samples:
            return None
        total = sum(dt for _, dt in samples)
        return float(sum(value * dt for value, dt in samples) / total)

    @staticmethod
    def _json_float(value):
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    def analyze(self, frame=None):
        if frame is None:
            if not self.frames:
                raise RuntimeError("No face frame is available")
            frame = self.frames[-1]

        raw_features = self.extract_action_features(frame.raw_blendshapes)
        features = self.extract_action_features(frame.smoothed_blendshapes)
        raw_scores = self.expression_scores(raw_features)
        scores = self.expression_scores(features)
        raw_candidate = (
            self.select_candidate(raw_scores, raw_features)
            if frame.reliable else "unknown"
        )
        smoothed_candidate = (
            self.select_candidate(scores, features)
            if frame.reliable else "unknown"
        )
        coverage = self._temporal_coverage()
        current_motion = bool(
            self.motion_samples
            and abs(self.motion_samples[-1]["timestamp"] - frame.timestamp) < 1e-9
        )
        temporally_reliable = bool(
            frame.reliable and current_motion
            and coverage >= cfg.MIN_TEMPORAL_COVERAGE
        )
        stable = self.expression_state.update(
            smoothed_candidate, scores, frame.timestamp, temporally_reliable
        )
        face_activity = self._weighted_motion("face_activity") if current_motion else None
        head_motion = self._weighted_motion("head_motion") if current_motion else None

        left_closed = self._value(frame.smoothed_blendshapes, "eyeBlinkLeft")
        right_closed = self._value(frame.smoothed_blendshapes, "eyeBlinkRight")
        actions_available = frame.reliable and left_closed is not None and right_closed is not None
        action_output = {
            "eye_closed_left": (
                left_closed >= cfg.EYE_CLOSED_THRESHOLD if actions_available else None
            ),
            "eye_closed_right": (
                right_closed >= cfg.EYE_CLOSED_THRESHOLD if actions_available else None
            ),
            "eyes_closed": (
                min(left_closed, right_closed) >= cfg.EYE_CLOSED_THRESHOLD
                if actions_available else None
            ),
            "blink_candidate": frame.blink_candidate if actions_available else None,
            "mouth_open": (
                features["jaw_open"] >= cfg.MOUTH_OPEN_THRESHOLD
                if frame.reliable and features["jaw_open"] is not None else None
            ),
            "mouth_open_score": (
                self._json_float(features["jaw_open"]) if frame.reliable else None
            ),
            "smile": self._json_float(features["mouth_smile"]) if frame.reliable else None,
            "eye_blink": self._json_float(features["eye_blink"]) if frame.reliable else None,
        }

        coefficients = {
            name: self._json_float(frame.raw_blendshapes[index])
            for index, name in enumerate(BLENDSHAPE_NAMES)
        }
        score_output = {
            name: self._json_float(scores[name]) if scores is not None else None
            for name in EXPRESSIONS
        }
        raw_score_output = {
            name: self._json_float(raw_scores[name]) if raw_scores is not None else None
            for name in EXPRESSIONS
        }
        head_pose = None
        if frame.smoothed_head_pose is not None and frame.face_detected:
            head_pose = {
                **{axis: float(frame.smoothed_head_pose[axis]) for axis in ("yaw", "pitch", "roll")},
                "unit": "degree",
                "rotation_order": "Rz(roll)@Ry(yaw)@Rx(pitch)",
                "sign_convention": "right_handed_model_coordinates",
            }

        return {
            "timestamp": float(frame.timestamp),
            "face_detected": bool(frame.face_detected),
            "landmark_count": int(np.count_nonzero(
                np.isfinite(frame.landmarks).all(axis=1)
            )),
            "blendshape_count": int(np.count_nonzero(
                np.isfinite(frame.raw_blendshapes)
            )),
            "raw_expression": raw_candidate,
            "stable_expression": stable,
            "expression_scores": score_output,
            "raw_expression_scores": raw_score_output,
            "blendshapes": coefficients,
            "action_features": {
                name: self._json_float(value) for name, value in features.items()
            },
            "facial_actions": action_output,
            "head_pose": head_pose,
            "face_activity": self._json_float(face_activity),
            "face_activity_score": self._json_float(
                np.clip(face_activity / cfg.FACE_ACTIVITY_FULL_SCALE, 0.0, 1.0)
                if face_activity is not None else None
            ),
            "head_motion": self._json_float(head_motion),
            "head_motion_score": self._json_float(
                np.clip(head_motion / cfg.HEAD_MOTION_FULL_SCALE_DEG_PER_SEC, 0.0, 1.0)
                if head_motion is not None else None
            ),
            "reliability": {
                "score": float(frame.reliability_score),
                "reliable": bool(frame.reliable),
                "landmark_range_ratio": float(frame.landmark_range_ratio),
                "blendshape_coverage": float(frame.blendshape_coverage),
                "face_scale": self._json_float(frame.face_scale),
                "geometry_valid": bool(frame.geometry_valid),
                "head_pose_available": frame.raw_head_pose is not None,
            },
            "temporal_coverage": float(coverage),
            "state_duration": self.expression_state.duration(frame.timestamp),
        }
