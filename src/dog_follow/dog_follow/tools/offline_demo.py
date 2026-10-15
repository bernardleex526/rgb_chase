"""Run the follow pipeline on a video file or webcam without ROS. Click a person to lock, r = release, q = quit.

python -m dog_follow.tools.offline_demo --video walk.mp4 --det yolox_s.onnx --reid osnet_x0_25.onnx \
    --fx 600 --camera-height 0.45 [--save out.mp4]
"""
import argparse
import math
import time

import numpy as np

from ..backends.models import OsnetReid, YoloxPersonDetector
from ..core.controller import ControllerParams, FollowController
from ..core.distance import CameraModel, MonoGeometryEstimator, MonoParams
from ..core.pipeline import FollowPipeline
from ..core.target_selector import SelectorParams, TargetSelector
from ..core.tracker import ByteTracker
from ..core.types import FollowState


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="0", help="file path or camera index")
    ap.add_argument("--det", required=True)
    ap.add_argument("--reid", required=True)
    ap.add_argument("--fx", type=float, help="focal length px (default: assume 70 deg HFOV)")
    ap.add_argument("--camera-height", type=float, default=0.45)
    ap.add_argument("--camera-pitch-deg", type=float, default=0.0)
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    cap = cv2.VideoCapture(int(a.video) if a.video.isdigit() else a.video)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {a.video}")
    h, w = frame.shape[:2]
    fx = a.fx or (w / 2) / math.tan(math.radians(35))
    cam = CameraModel(fx=fx, fy=fx, cx=w / 2, cy=h / 2, width=w, height=h)
    det = YoloxPersonDetector(a.det)
    reid = OsnetReid(a.reid)
    state = {"frame": frame, "cache": {}}

    def embed(track):
        if track.track_id not in state["cache"]:
            state["cache"][track.track_id] = reid(state["frame"], track.box)
        return state["cache"][track.track_id]

    pipe = FollowPipeline(ByteTracker(), TargetSelector(SelectorParams(), embed),
                          MonoGeometryEstimator(cam, MonoParams(camera_height=a.camera_height,
                                                                camera_pitch=math.radians(a.camera_pitch_deg))))
    ctrl = FollowController(ControllerParams())
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if a.save else None

    cv2.namedWindow("follow")
    cv2.setMouseCallback("follow", lambda ev, x, y, *_: ev == cv2.EVENT_LBUTTONDOWN and pipe.select_pixel(x, y))
    idx, timings = 0, []
    while ok:
        t0 = time.perf_counter()
        stamp = idx / fps
        state["frame"], state["cache"] = frame, {}
        est = pipe.step(det(frame), stamp)
        vx, wz = ctrl.compute(est, stamp)
        timings.append(time.perf_counter() - t0)

        for t in pipe.tracks:
            x1, y1, x2, y2 = t.box.astype(int)
            is_t = t.track_id == est.track_id and est.state == FollowState.TRACKING
            c = (0, 0, 255) if is_t else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3 if is_t else 1)
            cv2.putText(frame, str(t.track_id), (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        dist = "--" if math.isnan(est.distance) else f"{est.distance:.2f}m"
        hud = (f"{est.state.value} id={est.track_id} {dist} {math.degrees(est.bearing):+.0f}deg "
               f"vx={vx:+.2f} wz={wz:+.2f} {1 / np.mean(timings[-30:]):.0f}fps")
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if writer:
            writer.write(frame)
        cv2.imshow("follow", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            pipe.release()
        ok, frame = cap.read()
        idx += 1
    if writer:
        writer.release()


if __name__ == "__main__":
    main()
