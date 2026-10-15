from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .reid_gallery import TargetGallery
from .tracker import iou_matrix
from .types import FollowState, Track


@dataclass
class SelectorParams:
    reid_accept: float = 0.75  # similarity to re-acquire a lost target
    reid_reject: float = 0.45  # below this the followed track is treated as an ID switch
    reject_frames: int = 5
    confirm_frames: int = 3  # consecutive matches needed to re-acquire
    lost_timeout: float = 10.0  # s in LOST before giving up -> IDLE
    gallery_min_score: float = 0.6
    gallery_max_overlap: float = 0.1  # skip gallery updates when another person overlaps
    gallery_every_n: int = 5


class TargetSelector:
    """Keeps a single externally chosen person locked across occlusions.

    `embed(track)` is called lazily and must return an appearance feature or None.
    """

    def __init__(self, params: SelectorParams, embed: Callable[[Track], Optional[np.ndarray]]):
        self.p = params
        self.embed = embed
        self.gallery = TargetGallery()
        self.release()

    def release(self):
        self.state = FollowState.IDLE
        self.target_id = -1
        self.gallery.clear()
        self._lock_request: Optional[dict] = None
        self._lost_since = 0.0
        self._now = 0.0
        self._candidates: Dict[int, int] = {}
        self._mismatch = 0
        self._frames = 0
        self.last_track: Optional[Track] = None

    def request_lock_id(self, track_id: int):
        self._lock_request = {"id": int(track_id)}

    def request_lock_pixel(self, u: float, v: float):
        self._lock_request = {"uv": (float(u), float(v))}

    def update(self, tracks: List[Track], stamp: float) -> Optional[Track]:
        self._frames += 1
        self._now = stamp
        if self._lock_request is not None:
            self._try_lock(tracks)

        if self.state == FollowState.IDLE:
            return None

        current = next((t for t in tracks if t.track_id == self.target_id), None)
        if self.state == FollowState.TRACKING:
            if current is not None and self._verify(current, tracks):
                self.last_track = current
                return current
            self._enter_lost(stamp)

        # LOST
        if stamp - self._lost_since > self.p.lost_timeout:
            self.release()
            return None
        found = self._reacquire(tracks)
        if found is not None:
            self.target_id = found.track_id
            self.state = FollowState.TRACKING
            self._mismatch = 0
            self.last_track = found
        return found

    @property
    def lost_duration(self) -> float:
        return self._now - self._lost_since if self.state == FollowState.LOST else 0.0

    def _try_lock(self, tracks: List[Track]):
        req, chosen = self._lock_request, None
        if "id" in req:
            chosen = next((t for t in tracks if t.track_id == req["id"]), None)
        else:
            u, v = req["uv"]
            inside = [t for t in tracks if t.box[0] <= u <= t.box[2] and t.box[1] <= v <= t.box[3]]
            # Prefer the box whose centre is closest to the click (handles overlapping people).
            if inside:
                chosen = min(inside, key=lambda t: ((t.box[0] + t.box[2]) / 2 - u) ** 2
                             + ((t.box[1] + t.box[3]) / 2 - v) ** 2)
        if chosen is None:
            return  # keep the request until the person shows up as a confirmed track
        self._lock_request = None
        feat = self.embed(chosen)
        if feat is None:
            return
        self.release()
        self.gallery.add(feat)
        self.target_id = chosen.track_id
        self.state = FollowState.TRACKING
        self.last_track = chosen

    def _verify(self, track: Track, tracks: List[Track]) -> bool:
        isolated = self._isolated(track, tracks)
        # Check appearance whenever another person is close (where ID switches happen)
        # and periodically otherwise; refresh the gallery only on clean views.
        if not isolated or self._frames % self.p.gallery_every_n == 0:
            feat = self.embed(track)
            if feat is None:
                return True
            sim = self.gallery.similarity(feat)
            if sim < self.p.reid_reject:
                self._mismatch += 1
                if self._mismatch >= self.p.reject_frames:
                    return False
                return True
            self._mismatch = 0
            if isolated and track.score >= self.p.gallery_min_score and sim >= self.p.reid_accept:
                self.gallery.add(feat)
        return True

    def _isolated(self, track: Track, tracks: List[Track]) -> bool:
        others = [t.box for t in tracks if t.track_id != track.track_id]
        if not others:
            return True
        return float(iou_matrix(track.box[None], np.array(others)).max()) <= self.p.gallery_max_overlap

    def _enter_lost(self, stamp: float):
        self.state = FollowState.LOST
        self._lost_since = stamp
        self._candidates = {}
        self._mismatch = 0

    def _reacquire(self, tracks: List[Track]) -> Optional[Track]:
        best, best_sim = None, self.p.reid_accept
        seen = set()
        for t in tracks:
            # The original ID coming back still has to pass the appearance check.
            feat = self.embed(t)
            if feat is None:
                continue
            sim = self.gallery.similarity(feat)
            if sim < self.p.reid_accept:
                continue
            seen.add(t.track_id)
            self._candidates[t.track_id] = self._candidates.get(t.track_id, 0) + 1
            if self._candidates[t.track_id] >= self.p.confirm_frames and sim >= best_sim:
                best, best_sim = t, sim
        self._candidates = {k: v for k, v in self._candidates.items() if k in seen}
        return best
