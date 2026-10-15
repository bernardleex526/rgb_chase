import math

import numpy as np
import pytest

from dog_follow.core.controller import ControllerParams, FollowController
from dog_follow.core.distance import CameraModel, MonoGeometryEstimator, MonoParams, DepthImageEstimator
from dog_follow.core.pipeline import FollowPipeline
from dog_follow.core.target_selector import SelectorParams, TargetSelector
from dog_follow.core.tracker import ByteTracker
from dog_follow.core.types import Detection, FollowState, TargetEstimate, Track

CAM = CameraModel(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
CAM_H = 0.45


def person_box(x_lat: float, z: float, height=1.7, width=0.5, clip=True):
    """Project a standing person at lateral x (m, +right in image) and depth z onto CAM.

    Clipped to the image like a real detector output.
    """
    u = CAM.cx + CAM.fx * x_lat / z
    v_foot = CAM.cy + CAM.fy * CAM_H / z
    v_head = CAM.cy + CAM.fy * (CAM_H - height) / z
    half_w = 0.5 * CAM.fx * width / z
    box = np.array([u - half_w, v_head, u + half_w, v_foot])
    if clip:
        box = np.clip(box, 0, [CAM.width, CAM.height, CAM.width, CAM.height])
    return box


class FakeEmbedder:
    """Appearance = identity colour; the pipeline stores the true identity per track."""

    def __init__(self):
        self.identity = {}
        rng = np.random.default_rng(0)
        self.colours = {k: rng.normal(size=64) for k in "ABC"}

    def __call__(self, track: Track):
        who = self.identity.get(track.track_id)
        if who is None:
            return None
        return self.colours[who] + np.random.default_rng(track.age).normal(scale=0.2, size=64)


def make_pipeline(embedder):
    sel = TargetSelector(SelectorParams(lost_timeout=3.0), embedder)
    return FollowPipeline(ByteTracker(min_hits=2), sel, MonoGeometryEstimator(CAM, MonoParams(camera_height=CAM_H)))


def run_scene(frames, embedder, pipe, lock_at=3, lock_uv=None, dt=1 / 15):
    """frames: list of {identity: box}. Track ids are bound to identities by overlap."""
    out = []
    for i, people in enumerate(frames):
        dets = [Detection(box=b, score=0.9) for b in people.values()]
        if i == lock_at and lock_uv is not None:
            pipe.select_pixel(*lock_uv)
        est = pipe.step(dets, i * dt)
        for t in pipe.tracks:
            best = max(people.items(), key=lambda kv: _iou(kv[1], t.box))
            embedder.identity[t.track_id] = best[0]
        out.append(est)
    return out


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_mono_range_accuracy():
    est = MonoGeometryEstimator(CAM, MonoParams(camera_height=CAM_H))
    for z in [2.0, 4.0, 7.0]:
        t = Track(track_id=1, box=person_box(0.0, z), score=0.9)
        zm, _ = est.measure(t)
        assert zm == pytest.approx(z, rel=0.05)
    # At 1 m both head and feet leave a 480 px image: no geometric cue without keypoints.
    assert est.measure(Track(track_id=1, box=person_box(0.0, 1.0), score=0.9))[0] is None


def test_mono_range_truncated_box_uses_ground_plane():
    est = MonoGeometryEstimator(CAM, MonoParams(camera_height=CAM_H, person_height=1.5))
    box = person_box(0.0, 1.5)
    assert box[1] == 0.0  # head cut at 1.5 m, feet visible
    zm, _ = est.measure(Track(track_id=1, box=box, score=0.9))
    assert zm == pytest.approx(1.5, rel=0.05)


def test_mono_torso_keypoints_when_feet_cut():
    est = MonoGeometryEstimator(CAM, MonoParams(camera_height=CAM_H))
    z = 2.0
    box = person_box(0.0, z)
    box[3] = CAM.height  # feet cut off
    kps = np.zeros((17, 3))
    sh_v = CAM.cy + CAM.fy * (CAM_H - 1.45) / z
    hip_v = CAM.cy + CAM.fy * (CAM_H - 0.95) / z
    kps[[5, 6]] = [[310, sh_v, 1], [330, sh_v, 1]]
    kps[[11, 12]] = [[312, hip_v, 1], [328, hip_v, 1]]
    zm, _ = est.measure(Track(track_id=1, box=box, score=0.9, keypoints=kps))
    assert zm == pytest.approx(z, rel=0.1)


def test_depth_estimator_uses_person_not_background():
    est = DepthImageEstimator(CAM)
    depth = np.full((480, 640), 6000, dtype=np.uint16)
    box = person_box(0.0, 2.0)
    x1, y1, x2, y2 = box.astype(int)
    depth[y1:y2, x1 + 10:x2 - 10] = 2000
    est.set_depth(depth)
    zm, _ = est.measure(Track(track_id=1, box=box, score=0.9))
    assert zm == pytest.approx(2.0, abs=0.05)


def test_lock_by_pixel_and_follow():
    emb = FakeEmbedder()
    pipe = make_pipeline(emb)
    frames = [{"A": person_box(-0.5, 3.0 - 0.02 * i), "B": person_box(0.8, 4.0)} for i in range(40)]
    a0 = person_box(-0.5, 3.0 - 0.02 * 3)
    est = run_scene(frames, emb, pipe, lock_uv=((a0[0] + a0[2]) / 2, (a0[1] + a0[3]) / 2))
    assert est[2].state == FollowState.IDLE
    assert est[-1].state == FollowState.TRACKING
    assert emb.identity[est[-1].track_id] == "A"
    assert est[-1].bearing > 0  # A is left of centre
    assert est[-1].distance == pytest.approx(math.hypot(0.5, 3.0 - 0.02 * 39), rel=0.08)


def test_reacquire_after_full_occlusion():
    emb = FakeEmbedder()
    pipe = make_pipeline(emb)
    frames = []
    for i in range(80):
        people = {"B": person_box(0.6, 4.0)}
        if not 20 <= i < 35:  # A hidden for 1 s; tracker drops it -> new id later
            people["A"] = person_box(-0.4, 2.5)
        frames.append(people)
    a = frames[0]["A"]
    est = run_scene(frames, emb, pipe, lock_uv=((a[0] + a[2]) / 2, (a[1] + a[3]) / 2))
    states = [e.state for e in est]
    assert FollowState.LOST in states[20:36]
    assert est[-1].state == FollowState.TRACKING
    assert emb.identity[est[-1].track_id] == "A"


def test_never_switches_to_other_person_while_lost():
    emb = FakeEmbedder()
    pipe = make_pipeline(emb)
    frames = [{"A": person_box(-0.4, 2.5), "B": person_box(0.6, 3.0)} for _ in range(10)]
    frames += [{"B": person_box(0.6, 3.0)} for _ in range(30)]
    a = frames[0]["A"]
    est = run_scene(frames, emb, pipe, lock_uv=((a[0] + a[2]) / 2, (a[1] + a[3]) / 2))
    assert all(e.state != FollowState.TRACKING for e in est[15:])


def test_lost_timeout_goes_idle():
    emb = FakeEmbedder()
    pipe = make_pipeline(emb)
    frames = [{"A": person_box(0, 2.0)} for _ in range(10)] + [{} for _ in range(60)]
    a = frames[0]["A"]
    est = run_scene(frames, emb, pipe, lock_uv=((a[0] + a[2]) / 2, (a[1] + a[3]) / 2))
    assert est[-1].state == FollowState.IDLE


def test_id_switch_rejected_by_appearance():
    """Tracker hands the target's ID to another person; the selector must notice."""
    emb = FakeEmbedder()
    sel = TargetSelector(SelectorParams(), emb)
    box = person_box(0, 2.0)
    t = Track(track_id=7, box=box, score=0.9)
    emb.identity[7] = "A"
    sel.request_lock_id(7)
    for i in range(5):
        sel.update([t], i * 0.1)
    assert sel.state == FollowState.TRACKING
    emb.identity[7] = "C"  # same id, different person
    other = Track(track_id=8, box=box + 5, score=0.9)  # forces appearance checks every frame
    emb.identity[8] = "B"
    for i in range(5, 15):
        sel.update([t, other], i * 0.1)
    assert sel.state == FollowState.LOST


def _target(**kw):
    base = dict(state=FollowState.TRACKING, bearing=0.0, distance=1.5, distance_valid=True)
    base.update(kw)
    return TargetEstimate(**base)


def test_controller_behaviour():
    p = ControllerParams()
    c = FollowController(p)
    for i in range(60):
        vx, wz = c.compute(_target(distance=4.0, bearing=0.0), i * 0.05)
    assert vx == pytest.approx(p.max_vx)
    assert wz == pytest.approx(0.0)

    c.reset()
    for i in range(60):
        vx, wz = c.compute(_target(distance=1.5, bearing=0.0), i * 0.05)
    assert vx == pytest.approx(0.0) and wz == pytest.approx(0.0)

    c.reset()
    for i in range(60):
        vx, wz = c.compute(_target(distance=4.0, bearing=math.radians(40)), i * 0.05)
    assert vx == pytest.approx(0.0, abs=1e-6)  # turn first
    assert wz == pytest.approx(p.max_wz)

    c.reset()
    for i in range(60):
        vx, wz = c.compute(_target(distance=0.5), i * 0.05)
    assert vx == pytest.approx(-p.max_vx_reverse)


def test_controller_rate_limit_and_search():
    p = ControllerParams()
    c = FollowController(p)
    c.compute(_target(distance=5.0), 0.0)
    vx, _ = c.compute(_target(distance=5.0), 0.1)
    assert vx == pytest.approx(p.max_ax * 0.1)

    c.reset()
    lost = TargetEstimate(state=FollowState.LOST, last_bearing=-0.3, lost_duration=3.0)
    for i in range(40):
        vx, wz = c.compute(lost, i * 0.05)
    assert vx == 0.0 and wz == pytest.approx(-p.search_wz)

    idle = TargetEstimate(state=FollowState.IDLE)
    for i in range(40, 80):
        vx, wz = c.compute(idle, i * 0.05)
    assert vx == 0.0 and wz == pytest.approx(0.0)


def test_range_goes_unknown_when_measurements_stop():
    est = MonoGeometryEstimator(CAM, MonoParams(camera_height=CAM_H))
    far = Track(track_id=1, box=person_box(0.0, 3.0), score=0.9)
    for i in range(10):
        _, r, ok = est.update(far, i * 0.1)
    assert ok and r == pytest.approx(3.0, rel=0.05)
    close = Track(track_id=1, box=person_box(0.0, 1.0), score=0.9)  # no cue at 1 m
    _, r, ok = est.update(close, 1.1)
    assert not ok and not math.isnan(r)  # short coast on the filter
    _, r, ok = est.update(close, 2.0)
    assert math.isnan(r)  # then unknown -> controller stops advancing
