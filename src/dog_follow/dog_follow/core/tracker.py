"""ByteTrack-style multi-object tracker (Zhang et al., ECCV 2022), constant-velocity Kalman filter."""
from typing import List

import numpy as np

from .types import Detection, Track


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(cost: np.ndarray, max_cost: float):
    """Greedy assignment; with few people in view it matches Hungarian in practice."""
    matches, used_r, used_c = [], set(), set()
    if cost.size:
        for flat in np.argsort(cost, axis=None):
            r, c = np.unravel_index(flat, cost.shape)
            if cost[r, c] > max_cost:
                break
            if r in used_r or c in used_c:
                continue
            matches.append((int(r), int(c)))
            used_r.add(r)
            used_c.add(c)
    un_r = [i for i in range(cost.shape[0]) if i not in used_r]
    un_c = [j for j in range(cost.shape[1]) if j not in used_c]
    return matches, un_r, un_c


class _KalmanBox:
    """State: cx, cy, w, h and their velocities."""

    def __init__(self, box: np.ndarray):
        cx, cy, w, h = self._to_xywh(box)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        self.P = np.diag([w, h, w, h, 4 * w, 4 * h, 4 * w, 4 * h]) ** 2 * 0.05
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    @staticmethod
    def _to_xywh(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self):
        h = max(self.x[3], 1.0)
        q = np.array([h / 20, h / 20, h / 20, h / 20, h / 160, h / 160, h / 160, h / 160]) ** 2
        self.x = self.F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = self.F @ self.P @ self.F.T + np.diag(q)

    def update(self, box: np.ndarray):
        z = np.array(self._to_xywh(box))
        h = max(self.x[3], 1.0)
        R = np.diag([h / 20, h / 20, h / 20, h / 20]) ** 2
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P


class _TrackState:
    def __init__(self, track_id: int, det: Detection):
        self.track_id = track_id
        self.kf = _KalmanBox(det.box)
        self.score = det.score
        self.keypoints = det.keypoints
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection):
        self.kf.update(det.box)
        self.score = det.score
        self.keypoints = det.keypoints
        self.hits += 1
        self.time_since_update = 0

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            box=self.kf.box(),
            score=self.score,
            keypoints=self.keypoints if self.time_since_update == 0 else None,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
        )


class ByteTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.1, new_track_thresh=0.6,
                 match_iou=0.2, max_lost_frames=30, min_hits=3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_iou = match_iou
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        self._tracks: List[_TrackState] = []
        self._next_id = 1

    def reset(self):
        self._tracks = []

    def update(self, detections: List[Detection]) -> List[Track]:
        for t in self._tracks:
            t.predict()

        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        pool = list(range(len(self._tracks)))
        matched_tracks = set()

        def associate(dets, track_idx):
            if not dets or not track_idx:
                return list(range(len(dets))), track_idx
            t_boxes = np.array([self._tracks[i].kf.box() for i in track_idx])
            d_boxes = np.array([d.box for d in dets])
            cost = 1.0 - iou_matrix(t_boxes, d_boxes)
            matches, un_t, un_d = greedy_match(cost, 1.0 - self.match_iou)
            for ti, di in matches:
                self._tracks[track_idx[ti]].update(dets[di])
                matched_tracks.add(track_idx[ti])
            return un_d, [track_idx[i] for i in un_t]

        unmatched_high, remaining = associate(high, pool)
        associate(low, remaining)

        for di in unmatched_high:
            if high[di].score >= self.new_track_thresh:
                self._tracks.append(_TrackState(self._next_id, high[di]))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_lost_frames]
        return [t.to_track() for t in self._tracks
                if t.time_since_update == 0 and t.hits >= self.min_hits]
