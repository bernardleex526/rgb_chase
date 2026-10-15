import math
from dataclasses import dataclass
from typing import Tuple

from .types import FollowState, TargetEstimate


@dataclass
class ControllerParams:
    follow_distance: float = 1.5  # m
    distance_deadband: float = 0.15
    min_distance: float = 0.8  # closer than this -> back off
    kp_dist: float = 0.6
    kd_dist: float = 0.1
    kp_yaw: float = 1.2
    kd_yaw: float = 0.05
    bearing_deadband: float = math.radians(3.0)
    max_vx: float = 0.6
    max_vx_reverse: float = 0.2
    max_wz: float = 0.7
    max_ax: float = 0.6  # m/s^2
    max_aw: float = 1.5  # rad/s^2
    turn_slowdown_angle: float = math.radians(35.0)  # forward speed -> 0 at this bearing
    lost_coast_time: float = 1.0  # keep turning toward the last bearing this long
    search_after: float = 2.0  # then rotate in place toward last seen side
    search_wz: float = 0.35
    search_timeout: float = 8.0


class FollowController:
    """Distance + heading PD with heading-gated forward speed and rate limits."""

    def __init__(self, params: ControllerParams):
        self.p = params
        self.reset()

    def reset(self):
        self._vx = 0.0
        self._wz = 0.0
        self._prev_t = None
        self._prev_ed = None
        self._prev_eb = None

    def compute(self, target: TargetEstimate, stamp: float) -> Tuple[float, float]:
        dt = 0.0 if self._prev_t is None else max(0.0, min(stamp - self._prev_t, 0.5))
        self._prev_t = stamp
        vx, wz = self._desired(target, dt)
        return self._rate_limit(vx, wz, dt)

    def _desired(self, t: TargetEstimate, dt: float) -> Tuple[float, float]:
        p = self.p
        if t.state == FollowState.TRACKING:
            eb = t.bearing
            wz = 0.0
            if abs(eb) > p.bearing_deadband:
                deb = 0.0 if self._prev_eb is None or dt == 0 else (eb - self._prev_eb) / dt
                wz = p.kp_yaw * eb + p.kd_yaw * deb
            self._prev_eb = eb

            vx = 0.0
            if not math.isnan(t.distance):
                ed = t.distance - p.follow_distance
                if t.distance < p.min_distance:
                    vx = -p.max_vx_reverse
                elif abs(ed) > p.distance_deadband:
                    ded = 0.0 if self._prev_ed is None or dt == 0 else (ed - self._prev_ed) / dt
                    vx = p.kp_dist * ed + p.kd_dist * ded
                self._prev_ed = ed
                # Don't walk forward while the person is far off to the side.
                if vx > 0:
                    vx *= max(0.0, 1.0 - abs(eb) / p.turn_slowdown_angle)
            return (_clamp(vx, -p.max_vx_reverse, p.max_vx), _clamp(wz, -p.max_wz, p.max_wz))

        self._prev_ed = None
        self._prev_eb = None
        if t.state == FollowState.LOST:
            side = 1.0 if t.last_bearing >= 0 else -1.0
            if t.lost_duration < p.lost_coast_time:
                return 0.0, _clamp(p.kp_yaw * t.last_bearing, -p.max_wz, p.max_wz)
            if p.search_after <= t.lost_duration < p.search_timeout:
                return 0.0, side * p.search_wz
        return 0.0, 0.0

    def _rate_limit(self, vx: float, wz: float, dt: float) -> Tuple[float, float]:
        if dt > 0:
            self._vx += _clamp(vx - self._vx, -self.p.max_ax * dt, self.p.max_ax * dt)
            self._wz += _clamp(wz - self._wz, -self.p.max_aw * dt, self.p.max_aw * dt)
        else:
            self._vx, self._wz = 0.0, 0.0
        return self._vx, self._wz


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
