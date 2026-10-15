from typing import List, Optional

import numpy as np

from .distance import RangeEstimator
from .target_selector import TargetSelector
from .tracker import ByteTracker
from .types import Detection, FollowState, TargetEstimate, Track


class FollowPipeline:
    """detections -> tracks -> locked target -> (bearing, range). Framework-free."""

    def __init__(self, tracker: ByteTracker, selector: TargetSelector, ranger: RangeEstimator):
        self.tracker = tracker
        self.selector = selector
        self.ranger = ranger
        self.tracks: List[Track] = []
        self._last_bearing = 0.0
        self._was_idle = True

    def step(self, detections: List[Detection], stamp: float) -> TargetEstimate:
        self.tracks = self.tracker.update(detections)
        target = self.selector.update(self.tracks, stamp)
        state = self.selector.state

        if state == FollowState.IDLE:
            if not self._was_idle:
                self.ranger.reset()
            self._was_idle = True
            return TargetEstimate(state=state)
        if self._was_idle:
            self.ranger.reset()
            self._was_idle = False

        bearing, distance, valid = self.ranger.update(target, stamp)
        if target is not None:
            self._last_bearing = bearing
        return TargetEstimate(
            state=state,
            track_id=self.selector.target_id,
            bearing=bearing if target is not None else self._last_bearing,
            distance=distance,
            distance_valid=valid,
            box=None if target is None else np.asarray(target.box),
            lost_duration=self.selector.lost_duration,
            last_bearing=self._last_bearing,
        )

    def select_id(self, track_id: int):
        self.selector.request_lock_id(track_id)

    def select_pixel(self, u: float, v: float):
        self.selector.request_lock_pixel(u, v)

    def release(self):
        self.selector.release()
        self.ranger.reset()
        self._was_idle = True

    @staticmethod
    def detections_from_array(boxes: np.ndarray, scores: np.ndarray,
                              keypoints: Optional[np.ndarray] = None) -> List[Detection]:
        return [Detection(box=boxes[i], score=float(scores[i]),
                          keypoints=None if keypoints is None else keypoints[i])
                for i in range(len(boxes))]
