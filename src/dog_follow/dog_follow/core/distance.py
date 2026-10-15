import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .types import Track

# COCO-17 keypoint indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def bearing(self, u: float) -> float:
        """Horizontal angle to pixel column u; positive = left (REP-103)."""
        return math.atan2(self.cx - u, self.fx)


@dataclass
class MonoParams:
    camera_height: float = 0.45  # m, optical centre above ground
    camera_pitch: float = 0.0  # rad, positive = tilted down
    person_height: float = 1.70  # m, prior; refined online from the ground-plane estimate
    torso_length: float = 0.50  # m, shoulder-centre to hip-centre
    edge_margin: float = 4.0  # px, box touching the border is considered truncated
    min_ground_angle: float = math.radians(2.0)  # below this the ground estimate is too sensitive
    kp_min_conf: float = 0.5
    height_adapt_rate: float = 0.05
    process_accel: float = 1.0  # m/s^2 for the range filter
    max_range: float = 15.0


class RangeFilter:
    """1D constant-velocity Kalman filter on range."""

    def __init__(self, accel: float):
        self.accel = accel
        self.reset()

    def reset(self):
        self.x: Optional[np.ndarray] = None
        self.P = np.eye(2)
        self.t: Optional[float] = None

    def predict(self, t: float):
        if self.x is None:
            return
        dt = max(0.0, t - self.t)
        self.t = t
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([[0.5 * dt * dt], [dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (G @ G.T) * self.accel ** 2

    def update(self, z: float, var: float, t: float):
        if self.x is None:
            self.x = np.array([z, 0.0])
            self.P = np.diag([var, 1.0])
            self.t = t
            return
        self.predict(t)
        H = np.array([[1.0, 0.0]])
        S = float((H @ self.P @ H.T)[0, 0]) + var
        K = (self.P @ H.T / S).ravel()
        self.x = self.x + K * (z - self.x[0])
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P

    @property
    def value(self) -> Optional[float]:
        return None if self.x is None else float(self.x[0])


class RangeEstimator:
    """Turns per-frame range measurements of the target into a filtered (bearing, range)."""

    cam: CameraModel
    filter: RangeFilter
    max_range: float = 15.0
    coast_time: float = 0.7  # s without a range measurement before range is reported unknown

    _last_meas: float = -math.inf

    def reset(self):
        self.filter.reset()
        self._last_meas = -math.inf

    def measure(self, track: Track) -> Tuple[Optional[float], float]:
        raise NotImplementedError

    def update(self, track: Optional[Track], stamp: float) -> Tuple[float, float, bool]:
        """Returns (bearing rad, horizontal range m, range_valid)."""
        bearing = 0.0
        if track is not None:
            bearing = self.cam.bearing(0.5 * (track.box[0] + track.box[2]))
            z, var = self.measure(track)
            if z is not None and 0.2 < z < self.max_range:
                # Depth along the optical axis -> horizontal range to the person.
                self.filter.update(z / max(math.cos(bearing), 0.2), var, stamp)
                self._last_meas = stamp
                return bearing, self.filter.value, True
        self.filter.predict(stamp)
        r = self.filter.value
        # Coasting on a prediction for long is unsafe (e.g. person too close for any cue).
        if r is None or stamp - self._last_meas > self.coast_time:
            return bearing, math.nan, False
        return bearing, r, False


class MonoGeometryEstimator(RangeEstimator):
    """Range to a person from a single calibrated RGB camera."""

    def __init__(self, cam: CameraModel, params: MonoParams):
        self.cam = cam
        self.p = params
        self.filter = RangeFilter(params.process_accel)
        self.max_range = params.max_range
        self.person_height = params.person_height

    def reset(self):
        super().reset()
        self.person_height = self.p.person_height

    def measure(self, track: Track) -> Tuple[Optional[float], float]:
        """Fused depth along the optical axis (m) and its variance, or (None, inf)."""
        x1, y1, x2, y2 = track.box
        m = self.p.edge_margin
        top_cut = y1 <= m
        bottom_cut = y2 >= self.cam.height - m
        estimates = []

        ground = None
        if not bottom_cut:
            angle = math.atan2(y2 - self.cam.cy, self.cam.fy) + self.p.camera_pitch
            if angle > self.p.min_ground_angle:
                ground = self.p.camera_height / math.tan(angle)
                # A 2 px foot-line error scales with range^2; plus terrain/pitch slop.
                var = (ground ** 2 * 2.0 / (self.cam.fy * self.p.camera_height)) ** 2 + 0.05 ** 2
                estimates.append((ground, var))

        if not top_cut and not bottom_cut:
            h_px = y2 - y1
            if ground is not None and track.score > 0.6:
                observed = h_px * ground / self.cam.fy
                if 1.0 < observed < 2.2:
                    a = self.p.height_adapt_rate
                    self.person_height = (1 - a) * self.person_height + a * observed
            z = self.cam.fy * self.person_height / max(h_px, 1.0)
            estimates.append((z, (0.12 * z) ** 2))

        torso = self._torso_px(track.keypoints)
        if torso is not None:
            z = self.cam.fy * self.p.torso_length / torso
            estimates.append((z, (0.15 * z) ** 2))

        if not estimates:
            return None, math.inf
        w = np.array([1.0 / v for _, v in estimates])
        z = float(np.dot(w, [e for e, _ in estimates]) / w.sum())
        return z, float(1.0 / w.sum())

    def _torso_px(self, kps: Optional[np.ndarray]) -> Optional[float]:
        if kps is None:
            return None
        idx = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
        if np.any(kps[idx, 2] < self.p.kp_min_conf):
            return None
        shoulder = kps[[L_SHOULDER, R_SHOULDER], :2].mean(axis=0)
        hip = kps[[L_HIP, R_HIP], :2].mean(axis=0)
        d = float(np.linalg.norm(shoulder - hip))
        return d if d > 5.0 else None


class DepthImageEstimator(RangeEstimator):
    """Range from a depth image registered to the colour image (D435i align_depth, ZED depth)."""

    def __init__(self, cam: CameraModel, process_accel=1.0, depth_scale=0.001, inner=0.4):
        self.cam = cam
        self.depth_scale = depth_scale  # uint16 mm -> m; float32 images are already metres
        self.inner = inner  # fraction of the box (centred, upper body) sampled
        self.filter = RangeFilter(process_accel)
        self.depth: Optional[np.ndarray] = None

    def set_depth(self, depth: np.ndarray):
        self.depth = depth

    def measure(self, track: Track) -> Tuple[Optional[float], float]:
        if self.depth is None:
            return None, math.inf
        x1, y1, x2, y2 = track.box
        cx, w, h = 0.5 * (x1 + x2), x2 - x1, y2 - y1
        u0, u1 = int(cx - 0.5 * self.inner * w), int(cx + 0.5 * self.inner * w)
        v0, v1 = int(y1 + 0.2 * h), int(y1 + (0.2 + self.inner) * h)
        H, W = self.depth.shape[:2]
        roi = self.depth[max(v0, 0):min(v1, H), max(u0, 0):min(u1, W)]
        vals = roi.astype(float).ravel()
        if self.depth.dtype == np.uint16:
            vals = vals * self.depth_scale
        vals = vals[np.isfinite(vals) & (vals > 0.1)]
        if vals.size < 20:
            return None, math.inf
        # Lower percentile: the person is the nearest surface inside their own box.
        z = float(np.percentile(vals, 30))
        return z, (0.02 + 0.01 * z * z) ** 2

