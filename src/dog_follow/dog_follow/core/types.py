from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class Detection:
    box: np.ndarray  # x1, y1, x2, y2 in pixels
    score: float
    keypoints: Optional[np.ndarray] = None  # (K, 3): x, y, conf, COCO-17 order


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    keypoints: Optional[np.ndarray] = None
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    feature: Optional[np.ndarray] = field(default=None, repr=False)


class FollowState(str, Enum):
    IDLE = "idle"
    TRACKING = "tracking"
    LOST = "lost"


@dataclass
class TargetEstimate:
    state: FollowState
    track_id: int = -1
    bearing: float = 0.0  # rad, positive = target to the left (ROS REP-103)
    distance: float = float("nan")  # m, nan when unknown
    distance_valid: bool = False
    box: Optional[np.ndarray] = None
    lost_duration: float = 0.0
    last_bearing: float = 0.0
