"""Lightweight pose, motion and behaviour analysis for MediaPipe landmarks."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import pose_config as cfg


HEAD = np.arange(0, 11)
UPPER = np.arange(11, 23)
LOWER = np.arange(23, 33)
TORSO = np.array([11, 12, 23, 24])


@dataclass
class PoseFrame:
    """One timestamped pose observation."""

    timestamp: float
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    valid_mask: np.ndarray
    frame_width: int
    frame_height: int
    person_detected: bool
    smoothed_landmarks: np.ndarray
    smoothed_world_landmarks: np.ndarray


class TimedState:
    """Small actual-time FSM; invalid observations reset confirmation."""

    def __init__(self, confirm_seconds):
        self.confirm_seconds = confirm_seconds
        self.state = "unknown"
        self.since = None
        self.candidate = None
        self.candidate_since = None
        self.last_timestamp = None

    def update(self, candidate, timestamp, reliable=True):
        if not reliable or candidate is None or candidate == "unknown":
            self.state = "unknown"
            self.since = timestamp
            self.candidate = None
            self.candidate_since = None
            self.last_timestamp = timestamp
            return self.state

        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return self.state
        self.last_timestamp = timestamp

        if candidate == self.state:
            self.candidate = None
            self.candidate_since = None
            return self.state

        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = timestamp
            return self.state

        if timestamp - self.candidate_since >= self.confirm_seconds:
            self.state = candidate
            self.since = timestamp
            self.candidate = None
            self.candidate_since = None
        return self.state

    def duration(self, timestamp):
        if self.since is None or self.state == "unknown":
            return 0.0
        return max(0.0, timestamp - self.since)


class PoseAnalyzer:
    """Convert landmarks and produce a JSON-safe, explainable analysis."""

    def __init__(
        self,
        window_seconds=cfg.WINDOW_SECONDS,
        visibility_threshold=cfg.VISIBILITY_THRESHOLD,
        presence_threshold=cfg.PRESENCE_THRESHOLD,
    ):
        self.window_seconds = float(window_seconds)
        self.visibility_threshold = visibility_threshold
        self.presence_threshold = presence_threshold
        self.frames = deque()
        self.motion_samples = deque()
        self.behavior_samples = deque()
        self.posture_fsm = TimedState(cfg.POSTURE_CONFIRM_SECONDS)
        self.motion_fsm = TimedState(cfg.MOTION_CONFIRM_SECONDS)
        self.walk_fsm = TimedState(cfg.BEHAVIOR_CONFIRM_SECONDS)
        self.pacing_fsm = TimedState(cfg.BEHAVIOR_CONFIRM_SECONDS)
        self.repetitive_fsm = TimedState(cfg.BEHAVIOR_CONFIRM_SECONDS)
        self.last_stable_posture = "unknown"
        self.last_posture_time = None
        self.posture_unknown_since = None
        self.last_event = None

    def update(self, result, timestamp_ms, frame_shape):
        """Convert a MediaPipe result and append it to the time window."""
        timestamp = float(timestamp_ms) / 1000.0
        height, width = frame_shape[:2]
        if self.frames and timestamp <= self.frames[-1].timestamp:
            raise ValueError("Pose timestamps must be increasing")

        landmarks = np.full((33, 3), np.nan, dtype=np.float32)
        world = np.full((33, 3), np.nan, dtype=np.float32)
        visibility = np.zeros(33, dtype=np.float32)
        presence = np.zeros(33, dtype=np.float32)
        valid = np.zeros(33, dtype=bool)
        person_detected = bool(result.pose_landmarks)

        if person_detected:
            pose = result.pose_landmarks[0]
            if len(pose) == 33:
                landmarks = np.asarray(
                    [[p.x, p.y, p.z] for p in pose], dtype=np.float32
                )
                visibility = np.asarray(
                    [p.visibility for p in pose], dtype=np.float32
                )
                presence = np.asarray(
                    [p.presence for p in pose], dtype=np.float32
                )
                if result.pose_world_landmarks and len(result.pose_world_landmarks[0]) == 33:
                    world = np.asarray(
                        [[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]],
                        dtype=np.float32,
                    )
                valid = (
                    (visibility >= self.visibility_threshold)
                    & (presence >= self.presence_threshold)
                    & np.isfinite(landmarks).all(axis=1)
                    & (landmarks[:, 0] >= 0.0)
                    & (landmarks[:, 0] <= 1.0)
                    & (landmarks[:, 1] >= 0.0)
                    & (landmarks[:, 1] <= 1.0)
                )
            else:
                person_detected = False

        smooth_landmarks = landmarks.copy()
        smooth_world = world.copy()
        if self.frames and person_detected:
            previous = self.frames[-1]
            dt = timestamp - previous.timestamp
            if previous.person_detected and dt <= cfg.MAX_MOTION_GAP:
                alpha = 1.0 - np.exp(-dt / max(cfg.EMA_TIME_CONSTANT, 1e-6))
                common = valid & previous.valid_mask
                smooth_landmarks[common] = (
                    alpha * landmarks[common]
                    + (1.0 - alpha) * previous.smoothed_landmarks[common]
                )
                world_common = common & np.isfinite(world).all(axis=1) & np.isfinite(
                    previous.smoothed_world_landmarks
                ).all(axis=1)
                smooth_world[world_common] = (
                    alpha * world[world_common]
                    + (1.0 - alpha) * previous.smoothed_world_landmarks[world_common]
                )

        frame = PoseFrame(
            timestamp, landmarks, world, visibility, presence, valid,
            width, height, person_detected, smooth_landmarks, smooth_world,
        )
        previous = self.frames[-1] if self.frames else None
        self.frames.append(frame)
        if previous is not None:
            sample = self._motion_between(previous, frame)
            if sample is not None:
                self.motion_samples.append(sample)
        self._append_behavior_sample(frame)
        self._trim(timestamp)
        return frame

    def _trim(self, timestamp):
        while self.frames and timestamp - self.frames[0].timestamp > self.window_seconds:
            self.frames.popleft()
        while self.motion_samples and timestamp - self.motion_samples[0]["timestamp"] > self.window_seconds:
            self.motion_samples.popleft()
        while self.behavior_samples and timestamp - self.behavior_samples[0]["timestamp"] > cfg.BEHAVIOR_WINDOW_SECONDS:
            self.behavior_samples.popleft()

    def get_frame_count(self):
        return len(self.frames)

    def get_window_duration(self):
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    def get_valid_point_count(self):
        return int(np.count_nonzero(self.frames[-1].valid_mask)) if self.frames else 0

    @staticmethod
    def calculate_angle(a, b, c):
        ba = np.asarray(a) - np.asarray(b)
        bc = np.asarray(c) - np.asarray(b)
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if not np.isfinite([norm_ba, norm_bc]).all() or min(norm_ba, norm_bc) < 1e-8:
            return None
        cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def get_image_points(frame, smoothed=False):
        source = frame.smoothed_landmarks if smoothed else frame.landmarks
        points = source[:, :2].copy().astype(float)
        points[:, 1] *= frame.frame_height / frame.frame_width
        return points

    def get_joint_angle(self, frame, a, b, c):
        indices = [a, b, c]
        if not np.all(frame.valid_mask[indices]):
            return None
        points = frame.smoothed_world_landmarks[indices]
        if not np.isfinite(points).all():
            points = self.get_image_points(frame, smoothed=True)[indices]
        return self.calculate_angle(points[0], points[1], points[2])

    def get_torso_angle(self, frame):
        if not np.all(frame.valid_mask[TORSO]):
            return None
        points = self.get_image_points(frame, smoothed=True)
        shoulder = (points[11] + points[12]) / 2.0
        hip = (points[23] + points[24]) / 2.0
        vector = shoulder - hip
        if np.linalg.norm(vector) < 1e-8:
            return None
        return float(np.degrees(np.arctan2(abs(vector[0]), abs(vector[1]))))

    def get_thigh_angle(self, frame, hip, knee):
        if not np.all(frame.valid_mask[[hip, knee]]):
            return None
        vector = self.get_image_points(frame, smoothed=True)[knee] - self.get_image_points(
            frame, smoothed=True
        )[hip]
        if np.linalg.norm(vector) < 1e-8:
            return None
        return float(np.degrees(np.arctan2(abs(vector[1]), abs(vector[0]))))

    def extract_posture_features(self, frame):
        return {
            "left_knee": self.get_joint_angle(frame, 23, 25, 27),
            "right_knee": self.get_joint_angle(frame, 24, 26, 28),
            "left_hip": self.get_joint_angle(frame, 11, 23, 25),
            "right_hip": self.get_joint_angle(frame, 12, 24, 26),
            "left_elbow": self.get_joint_angle(frame, 11, 13, 15),
            "right_elbow": self.get_joint_angle(frame, 12, 14, 16),
            "torso_angle": self.get_torso_angle(frame),
            "left_thigh_angle": self.get_thigh_angle(frame, 23, 25),
            "right_thigh_angle": self.get_thigh_angle(frame, 24, 26),
        }

    @staticmethod
    def _region_ratio(frame, indices):
        return float(np.count_nonzero(frame.valid_mask[indices]) / len(indices))

    def _region_reliability(self, frame):
        return {
            "head": self._region_ratio(frame, HEAD),
            "upper_body": self._region_ratio(frame, UPPER),
            "lower_body": self._region_ratio(frame, LOWER),
            "torso": self._region_ratio(frame, TORSO),
            "overall": float(np.count_nonzero(frame.valid_mask) / 33.0),
        }

    def get_visible_region(self, frame):
        if not frame.person_detected:
            return "none"
        reliability = self._region_reliability(frame)
        torso_ok = reliability["torso"] >= cfg.TORSO_VALID_RATIO
        shoulders = np.all(frame.valid_mask[[11, 12]])
        if (
            torso_ok
            and reliability["upper_body"] >= cfg.FULL_UPPER_VALID_RATIO
            and reliability["lower_body"] >= cfg.FULL_LOWER_VALID_RATIO
        ):
            return "full_body"
        if shoulders and reliability["upper_body"] >= cfg.UPPER_REGION_VALID_RATIO:
            return "upper_body"
        if reliability["head"] >= cfg.HEAD_REGION_VALID_RATIO and shoulders:
            return "head_shoulders"
        if reliability["head"] >= cfg.HEAD_REGION_VALID_RATIO:
            return "head_only"
        return "unknown"

    def classify_posture(self, frame, features):
        if not frame.person_detected or self.get_visible_region(frame) != "full_body":
            return "unknown"
        torso = features["torso_angle"]
        if torso is None:
            return "unknown"
        sides = []
        for side in ("left", "right"):
            knee, hip = features[f"{side}_knee"], features[f"{side}_hip"]
            thigh = features[f"{side}_thigh_angle"]
            if knee is not None and hip is not None:
                sides.append((knee, hip, thigh))
        if not sides:
            return "unknown"
        straight_knees = all(x[0] >= cfg.KNEE_STRAIGHT_MIN for x in sides)
        straight_hips = all(x[1] >= cfg.HIP_STRAIGHT_MIN for x in sides)
        bent_knees = all(cfg.KNEE_BENT_MIN <= x[0] <= cfg.KNEE_BENT_MAX for x in sides)
        bent_hips = all(cfg.HIP_BENT_MIN <= x[1] <= cfg.HIP_BENT_MAX for x in sides)
        horizontal_thighs = all(
            x[2] is not None and x[2] <= cfg.THIGH_HORIZONTAL_MAX for x in sides
        )
        points = self.get_image_points(frame, smoothed=True)
        shoulder = (points[11] + points[12]) / 2.0
        hip = (points[23] + points[24]) / 2.0
        ankles = [points[i] for i in (27, 28) if frame.valid_mask[i]]
        vertical = bool(ankles) and shoulder[1] < hip[1] and all(p[1] > hip[1] for p in ankles)
        if torso <= cfg.TORSO_UPRIGHT_MAX and straight_knees and straight_hips and vertical:
            return "standing"
        if torso <= cfg.TORSO_UPRIGHT_MAX and bent_knees and bent_hips and horizontal_thighs and vertical:
            return "sitting"
        if torso >= cfg.TORSO_HORIZONTAL_MIN and straight_hips and horizontal_thighs:
            return "lying_candidate"
        if torso >= cfg.TORSO_BENDING_MIN and straight_knees and bent_hips and vertical:
            return "bending"
        return "unknown"

    def analyze_posture(self, frame):
        features = self.extract_posture_features(frame)
        return {
            "timestamp": frame.timestamp,
            "person_detected": frame.person_detected,
            "visible_region": self.get_visible_region(frame),
            "posture": self.classify_posture(frame, features),
            "features": features,
        }

    def _reference(self, frame):
        points = self.get_image_points(frame, smoothed=True)
        mask = frame.valid_mask
        if np.all(mask[TORSO]):
            shoulder = (points[11] + points[12]) / 2.0
            hip = (points[23] + points[24]) / 2.0
            center = (shoulder + hip) / 2.0
            torso_length = np.linalg.norm(shoulder - hip)
            shoulder_width = np.linalg.norm(points[11] - points[12])
            scale = max(torso_length, shoulder_width)
            signature = "torso"
        elif np.all(mask[[11, 12]]):
            center = (points[11] + points[12]) / 2.0
            scale = np.linalg.norm(points[11] - points[12])
            signature = "shoulders"
        else:
            return None
        if not np.isfinite(scale) or scale < cfg.MIN_BODY_SCALE:
            return None
        return center, float(scale), signature

    @staticmethod
    def _rms_speed(current, previous, indices, common, dt):
        selected = indices[common[indices]]
        if len(selected) == 0:
            return None
        speed = np.linalg.norm(current[selected] - previous[selected], axis=1) / dt
        return float(np.sqrt(np.mean(np.square(speed))))

    def _motion_between(self, previous, current):
        dt = current.timestamp - previous.timestamp
        if (
            dt <= 0.0 or dt > cfg.MAX_MOTION_GAP
            or not previous.person_detected or not current.person_detected
        ):
            return None
        ref_a, ref_b = self._reference(previous), self._reference(current)
        if ref_a is None or ref_b is None or ref_a[2] != ref_b[2]:
            return None
        ratio = ref_b[1] / ref_a[1]
        if ratio > cfg.MAX_SCALE_RATIO or ratio < 1.0 / cfg.MAX_SCALE_RATIO:
            return None
        scale = (ref_a[1] + ref_b[1]) / 2.0
        center_delta = ref_b[0] - ref_a[0]
        global_motion = float(np.linalg.norm(center_delta) / (scale * dt))
        p0 = self.get_image_points(previous, smoothed=True)
        p1 = self.get_image_points(current, smoothed=True)
        # Comparing body-centred, independently scaled coordinates prevents
        # camera approach/retreat from appearing as local limb motion.
        p0 = (p0 - ref_a[0]) / ref_a[1]
        p1 = (p1 - ref_b[0]) / ref_b[1]
        common = previous.valid_mask & current.valid_mask
        metrics = {
            "head": self._rms_speed(p1, p0, HEAD, common, dt),
            "upper": self._rms_speed(p1, p0, UPPER, common, dt),
            "lower": self._rms_speed(p1, p0, LOWER, common, dt),
        }
        scope = self.get_visible_region(current)
        allowed = {
            "full_body": ("head", "upper", "lower"),
            "upper_body": ("head", "upper"),
            "head_shoulders": ("head",),
        }.get(scope, ())
        for name in metrics:
            if name not in allowed:
                metrics[name] = None
        weights = {
            "head": cfg.ACTIVITY_HEAD_WEIGHT,
            "upper": cfg.ACTIVITY_UPPER_WEIGHT,
            "lower": cfg.ACTIVITY_LOWER_WEIGHT,
        }
        available = [(metrics[k], weights[k]) for k in allowed if metrics[k] is not None]
        local = None
        if available:
            local = sum(value * weight for value, weight in available) / sum(
                weight for _, weight in available
            )
        activity = None
        if local is not None:
            activity = (1.0 - cfg.ACTIVITY_GLOBAL_WEIGHT) * local + cfg.ACTIVITY_GLOBAL_WEIGHT * global_motion
        return {
            "timestamp": current.timestamp,
            "dt": dt,
            "head": metrics["head"],
            "upper": metrics["upper"],
            "lower": metrics["lower"],
            "global": global_motion,
            "activity": activity,
            "scope": scope,
        }

    def _temporal_coverage(self, key=None):
        duration = self.get_window_duration()
        if duration <= 0.0:
            return 0.0
        covered = 0.0
        for sample in self.motion_samples:
            if key is None or sample.get(key) is not None:
                covered += sample["dt"]
        return float(np.clip(covered / duration, 0.0, 1.0))

    def _weighted_motion(self, key):
        values = [(s[key], s["dt"]) for s in self.motion_samples if s.get(key) is not None]
        if not values:
            return None
        total = sum(weight for _, weight in values)
        if total <= 0:
            return None
        return float(sum(value * weight for value, weight in values) / total)

    def _motion_candidate(self, activity):
        if activity is None:
            return "unknown"
        current = self.motion_fsm.state
        if current == "still" and activity <= cfg.ACTIVITY_STILL_EXIT:
            return "still"
        if (
            current == "low_activity"
            and cfg.ACTIVITY_STILL_ENTER < activity <= cfg.ACTIVITY_LOW_EXIT
        ):
            return "low_activity"
        if current == "high_activity" and activity >= cfg.ACTIVITY_HIGH_EXIT:
            return "high_activity"
        if activity <= cfg.ACTIVITY_STILL_ENTER:
            return "still"
        if activity < cfg.ACTIVITY_LOW_ENTER:
            return "low_activity"
        if activity >= cfg.ACTIVITY_HIGH_ENTER:
            return "high_activity"
        return "normal_activity"

    def _append_behavior_sample(self, frame):
        reference = self._reference(frame) if frame.person_detected else None
        points = self.get_image_points(frame, smoothed=True)
        sample = {
            "timestamp": frame.timestamp,
            "center": reference[0].copy() if reference else None,
            "scale": reference[1] if reference else None,
            "signature": reference[2] if reference else None,
            "left_ankle": points[27].copy() if frame.valid_mask[27] and reference else None,
            "right_ankle": points[28].copy() if frame.valid_mask[28] and reference else None,
            "head_signal": None,
            "left_upper_signal": None,
            "right_upper_signal": None,
            "left_lower_signal": None,
            "right_lower_signal": None,
        }
        if reference:
            center, scale, _ = reference
            if frame.valid_mask[0]:
                sample["head_signal"] = float((points[0, 0] - center[0]) / scale)
            if frame.valid_mask[15]:
                sample["left_upper_signal"] = float((points[15, 1] - center[1]) / scale)
            if frame.valid_mask[16]:
                sample["right_upper_signal"] = float((points[16, 1] - center[1]) / scale)
            if frame.valid_mask[27]:
                sample["left_lower_signal"] = float((points[27, 0] - center[0]) / scale)
            if frame.valid_mask[28]:
                sample["right_lower_signal"] = float((points[28, 0] - center[0]) / scale)
        self.behavior_samples.append(sample)

    def _walking_candidate(self, global_motion, lower_motion):
        if global_motion is None or lower_motion is None:
            return None
        samples = self._latest_contiguous_samples(
            ("left_ankle", "right_ankle", "center", "scale")
        )
        if len(samples) < cfg.WALK_MIN_SAMPLES:
            return False
        left = np.asarray([(s["left_ankle"] - s["center"]) / s["scale"] for s in samples[-30:]])
        right = np.asarray([(s["right_ankle"] - s["center"]) / s["scale"] for s in samples[-30:]])
        axis = int(np.argmax(np.std(left, axis=0) + np.std(right, axis=0)))
        left = left[:, axis]
        right = right[:, axis]
        if np.std(left) < 1e-4 or np.std(right) < 1e-4:
            alternating = False
        else:
            alternating = np.corrcoef(left, right)[0, 1] <= cfg.WALK_ANKLE_ANTICORRELATION
        return bool(
            global_motion >= cfg.WALK_GLOBAL_MIN
            and lower_motion >= cfg.WALK_LOWER_MIN
            and alternating
        )

    def _pacing_details(self):
        samples = self._latest_contiguous_samples(("center", "scale"))
        if len(samples) < 4:
            return False, 0, 0.0
        times = np.asarray([s["timestamp"] for s in samples])
        duration = times[-1] - times[0]
        if duration < cfg.PACING_MIN_DURATION:
            return False, 0, duration
        median_scale = float(np.median([s["scale"] for s in samples]))
        centers = np.asarray([s["center"] / median_scale for s in samples])
        # Select the dominant camera-plane direction and suppress jitter.
        axis = int(np.argmax(np.ptp(centers, axis=0)))
        displacement = np.diff(centers[:, axis])
        dt = np.diff(times)
        speeds = displacement / np.maximum(dt, 1e-6)
        signs = np.sign(np.where(np.abs(speeds) >= cfg.PACING_DIRECTION_DEADBAND, speeds, 0.0))
        signs = signs[signs != 0]
        reversals = int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        mean_speed = float(np.sum(np.abs(displacement)) / duration)
        detected = (
            reversals >= cfg.PACING_MIN_REVERSALS
            and mean_speed >= cfg.PACING_MIN_SPEED
        )
        return bool(detected), reversals, float(duration)

    @staticmethod
    def _periodicity(times, values):
        if len(times) < cfg.REPETITIVE_MIN_SAMPLES or times[-1] - times[0] < cfg.REPETITIVE_MIN_DURATION:
            return None
        step = 1.0 / cfg.REPETITIVE_RESAMPLE_HZ
        grid = np.arange(times[0], times[-1] + step * 0.5, step)
        signal = np.interp(grid, times, values)
        signal -= np.mean(signal)
        amplitude = float(np.percentile(signal, 95) - np.percentile(signal, 5))
        if amplitude < cfg.REPETITIVE_MIN_AMPLITUDE or np.std(signal) < 1e-6:
            return None
        best = None
        min_lag = max(1, int(cfg.REPETITIVE_MIN_PERIOD / step))
        max_lag = min(len(signal) // 2, int(cfg.REPETITIVE_MAX_PERIOD / step))
        for lag in range(min_lag, max_lag + 1):
            a, b = signal[:-lag], signal[lag:]
            if len(a) < 4 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
                continue
            correlation = float(np.corrcoef(a, b)[0, 1])
            if best is None or correlation > best[0]:
                best = (correlation, lag * step)
        if best is None or best[0] < cfg.REPETITIVE_MIN_CORRELATION:
            return None
        cycles = (times[-1] - times[0]) / best[1]
        if cycles < 2.0:
            return None
        return {"correlation": best[0], "period": best[1], "amplitude": amplitude}

    def _repetitive_details(self, walking):
        candidates = []
        for region, key, side in (
            ("head", "head_signal", None),
            ("upper_body", "left_upper_signal", "left"),
            ("upper_body", "right_upper_signal", "right"),
            ("lower_body", "left_lower_signal", "left"),
            ("lower_body", "right_lower_signal", "right"),
        ):
            if walking and region in ("upper_body", "lower_body"):
                continue
            samples = self._latest_contiguous_samples((key,))
            if not samples:
                continue
            details = self._periodicity(
                np.asarray([s["timestamp"] for s in samples]),
                np.asarray([s[key] for s in samples]),
            )
            if details:
                details["region"] = region
                details["side"] = side
                details["duration"] = float(samples[-1]["timestamp"] - samples[0]["timestamp"])
                candidates.append(details)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item["correlation"] * item["amplitude"])

    def _latest_contiguous_samples(self, required):
        """Return only the newest uninterrupted, same-reference observations."""
        selected = []
        newer_time = None
        signature = None
        for sample in reversed(self.behavior_samples):
            if any(sample.get(key) is None for key in required):
                break
            if newer_time is not None and newer_time - sample["timestamp"] > cfg.MAX_MOTION_GAP:
                break
            if signature is None:
                signature = sample.get("signature")
            elif sample.get("signature") != signature:
                break
            selected.append(sample)
            newer_time = sample["timestamp"]
        selected.reverse()
        return selected

    def _transition_event(self, posture, timestamp):
        event = None
        if posture == "unknown":
            if self.posture_unknown_since is None:
                self.posture_unknown_since = timestamp
        else:
            gap_ok = (
                self.posture_unknown_since is None
                or timestamp - self.posture_unknown_since <= cfg.TRANSITION_MAX_SECONDS
            )
            self.posture_unknown_since = None
        if posture != "unknown" and posture != self.last_stable_posture:
            if self.last_posture_time is not None and gap_ok:
                if self.last_stable_posture == "sitting" and posture == "standing":
                    event = {"type": "standing_up", "timestamp": timestamp}
                elif self.last_stable_posture == "standing" and posture == "sitting":
                    event = {"type": "sitting_down", "timestamp": timestamp}
            self.last_stable_posture = posture
            self.last_posture_time = timestamp
        if event:
            self.last_event = event
        if self.last_event and timestamp - self.last_event["timestamp"] <= cfg.EVENT_HOLD_SECONDS:
            return dict(self.last_event)
        return None

    @staticmethod
    def _safe(value):
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    def analyze(self, frame=None):
        """Return the current unified analysis; all fields are JSON serializable."""
        if frame is None:
            if not self.frames:
                raise RuntimeError("No pose frame is available")
            frame = self.frames[-1]
        posture_data = self.analyze_posture(frame)
        reliability = self._region_reliability(frame)
        visible_region = posture_data["visible_region"]
        posture_reliable = visible_region == "full_body"
        posture = self.posture_fsm.update(
            posture_data["posture"], frame.timestamp, posture_reliable
        )

        coverage = self._temporal_coverage("activity")
        activity = self._weighted_motion("activity")
        head = self._weighted_motion("head")
        upper = self._weighted_motion("upper")
        lower = self._weighted_motion("lower")
        global_motion = self._weighted_motion("global")
        current_observed = bool(
            self.motion_samples
            and abs(self.motion_samples[-1]["timestamp"] - frame.timestamp) < 1e-9
        )
        motion_reliable = (
            frame.person_detected and visible_region not in ("none", "unknown", "head_only")
            and coverage >= cfg.MIN_TEMPORAL_COVERAGE and current_observed
        )
        motion_state = self.motion_fsm.update(
            self._motion_candidate(activity), frame.timestamp, motion_reliable
        )
        if not motion_reliable:
            activity = None
            head = upper = lower = global_motion = None

        walk_candidate = self._walking_candidate(global_motion, lower)
        walking = self.walk_fsm.update(
            "walking" if walk_candidate else "not_walking",
            frame.timestamp,
            walk_candidate is not None and motion_reliable and visible_region == "full_body",
        ) == "walking"
        pacing_candidate, reversals, pacing_duration = self._pacing_details()
        pacing = self.pacing_fsm.update(
            "pacing_candidate" if pacing_candidate else "not_pacing",
            frame.timestamp,
            motion_reliable and visible_region == "full_body",
        ) == "pacing_candidate"
        repetitive = self._repetitive_details(walking)
        repetitive_active = self.repetitive_fsm.update(
            "repetitive_motion" if repetitive else "not_repetitive",
            frame.timestamp,
            frame.person_detected and visible_region not in ("none", "unknown"),
        ) == "repetitive_motion"
        if not repetitive_active:
            repetitive = None

        event = self._transition_event(posture, frame.timestamp)
        activity_scope = {
            "full_body": "full_body",
            "upper_body": "upper_body",
            "head_shoulders": "head",
        }.get(visible_region, "none")
        behavior_observable = motion_reliable and visible_region == "full_body"
        behaviors = {
            "walking": walking if behavior_observable else None,
            "pacing_candidate": pacing if behavior_observable else None,
            "repetitive_motion": repetitive_active if frame.person_detected else None,
            "standing_up": bool(event and event["type"] == "standing_up") if posture_reliable else None,
            "sitting_down": bool(event and event["type"] == "sitting_down") if posture_reliable else None,
        }
        return {
            "timestamp": float(frame.timestamp),
            "person_detected": bool(frame.person_detected),
            "visible_region": visible_region,
            "valid_landmarks": int(np.count_nonzero(frame.valid_mask)),
            "posture": posture,
            "motion_state": motion_state,
            "activity_raw": self._safe(activity),
            "activity_score": self._safe(
                np.clip(activity / cfg.ACTIVITY_SCORE_FULL_SCALE, 0.0, 1.0)
                if activity is not None else None
            ),
            "head_activity": self._safe(head) if activity_scope != "none" else None,
            "upper_body_activity": self._safe(upper) if activity_scope in ("upper_body", "full_body") else None,
            "lower_body_activity": self._safe(lower) if activity_scope == "full_body" else None,
            "global_motion": self._safe(global_motion) if activity_scope != "none" else None,
            "behaviors": behaviors,
            "behavior_event": event,
            "repetitive_motion": repetitive,
            "pacing": {
                "reversals": reversals,
                "observed_duration": pacing_duration,
            } if behavior_observable else None,
            "reliability": {key: float(value) for key, value in reliability.items()},
            "temporal_coverage": float(coverage),
            "activity_scope": activity_scope,
            "state_duration": {
                "posture": self.posture_fsm.duration(frame.timestamp),
                "motion": self.motion_fsm.duration(frame.timestamp),
            },
            "features": {
                key: self._safe(value) for key, value in posture_data["features"].items()
            },
        }
