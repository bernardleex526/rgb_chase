"""Camera image -> detection/tracking/target lock -> /follow/target."""
import json
import math
import threading

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

from ..backends.models import OsnetReid, RtmPose, YoloxPersonDetector
from ..core.distance import CameraModel, DepthImageEstimator, MonoGeometryEstimator, MonoParams
from ..core.pipeline import FollowPipeline
from ..core.target_selector import SelectorParams, TargetSelector
from ..core.tracker import ByteTracker
from ..core.types import FollowState


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("follow_perception")
        p = self.declare_parameter
        p("detector_model", "")
        p("detector_input_size", 640)
        p("reid_model", "")
        p("pose_model", "")
        p("range_source", "mono")  # mono | depth
        p("depth_scale", 0.001)
        p("camera_height", 0.45)
        p("camera_pitch_deg", 0.0)
        p("person_height", 1.70)
        p("reid_accept", 0.75)
        p("reid_reject", 0.45)
        p("lost_timeout", 10.0)
        p("publish_debug_image", True)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.bridge = CvBridge()
        self.detector = YoloxPersonDetector(g("detector_model"), g("detector_input_size"))
        self.reid = OsnetReid(g("reid_model"))
        self.pose = RtmPose(g("pose_model")) if g("pose_model") else None
        self._frame = None
        self._feat_cache = {}
        self.selector = TargetSelector(
            SelectorParams(reid_accept=g("reid_accept"), reid_reject=g("reid_reject"),
                           lost_timeout=g("lost_timeout")),
            self._embed)
        self.pipeline = None  # built on first CameraInfo
        self.range_source = g("range_source")
        self.mono_params = MonoParams(camera_height=g("camera_height"),
                                      camera_pitch=math.radians(g("camera_pitch_deg")),
                                      person_height=g("person_height"))
        self.depth_scale = g("depth_scale")
        self.debug = g("publish_debug_image")
        self._lock = threading.Lock()

        self.create_subscription(CameraInfo, "camera_info", self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, "image", self._on_image, qos_profile_sensor_data)
        if self.range_source == "depth":
            self.create_subscription(Image, "depth", self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Int32, "~/select_id", lambda m: self._select(lambda pl: pl.select_id(m.data)), 10)
        self.create_subscription(Point, "~/select_pixel",
                                 lambda m: self._select(lambda pl: pl.select_pixel(m.x, m.y)), 10)
        self.create_service(Trigger, "~/release", self._on_release)

        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub_target = self.create_publisher(String, "/follow/target", reliable)
        self.pub_point = self.create_publisher(PointStamped, "/follow/target_point", 10)
        self.pub_debug = self.create_publisher(Image, "/follow/debug_image", 1)
        self.get_logger().info(f"perception ready, range_source={self.range_source}")

    def _on_info(self, msg: CameraInfo):
        if self.pipeline is not None:
            return
        k = msg.k
        cam = CameraModel(fx=k[0], fy=k[4], cx=k[2], cy=k[5], width=msg.width, height=msg.height)
        if self.range_source == "depth":
            ranger = DepthImageEstimator(cam, depth_scale=self.depth_scale)
        else:
            ranger = MonoGeometryEstimator(cam, self.mono_params)
        self.pipeline = FollowPipeline(ByteTracker(), self.selector, ranger)
        self.get_logger().info(f"camera intrinsics fx={k[0]:.1f} cx={k[2]:.1f} {msg.width}x{msg.height}")

    def _on_depth(self, msg: Image):
        if self.pipeline is not None:
            self.pipeline.ranger.set_depth(self.bridge.imgmsg_to_cv2(msg, "passthrough"))

    def _select(self, fn):
        with self._lock:
            if self.pipeline is None:
                self.get_logger().warn("no camera_info yet, selection ignored")
                return
            fn(self.pipeline)
        self.get_logger().info("target selection requested")

    def _on_release(self, _req, resp):
        with self._lock:
            if self.pipeline is not None:
                self.pipeline.release()
        resp.success = True
        resp.message = "released"
        return resp

    def _embed(self, track):
        if track.track_id not in self._feat_cache:
            self._feat_cache[track.track_id] = self.reid(self._frame, track.box)
        return self._feat_cache[track.track_id]

    def _on_image(self, msg: Image):
        if self.pipeline is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dets = self.detector(frame)
        if self.pose is not None and dets:
            for d, k in zip(dets, self.pose(frame, [d.box for d in dets])):
                d.keypoints = k
        with self._lock:
            self._frame = frame
            self._feat_cache = {}
            est = self.pipeline.step(dets, stamp)
            tracks = list(self.pipeline.tracks)

        out = {
            "stamp": stamp,
            "state": est.state.value,
            "track_id": est.track_id,
            "bearing": est.bearing,
            "distance": None if math.isnan(est.distance) else est.distance,
            "distance_valid": est.distance_valid,
            "lost_duration": est.lost_duration,
            "last_bearing": est.last_bearing,
            "tracks": [{"id": t.track_id, "box": [round(float(v), 1) for v in t.box]} for t in tracks],
        }
        self.pub_target.publish(String(data=json.dumps(out)))
        if est.state == FollowState.TRACKING and not math.isnan(est.distance):
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = est.distance * math.cos(est.bearing)
            pt.point.y = est.distance * math.sin(est.bearing)
            self.pub_point.publish(pt)
        if self.debug and self.pub_debug.get_subscription_count() > 0:
            self.pub_debug.publish(self._draw(frame, tracks, est, msg.header))

    def _draw(self, frame, tracks, est, header):
        import cv2

        img = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.box.astype(int)
            target = t.track_id == est.track_id and est.state == FollowState.TRACKING
            colour = (0, 0, 255) if target else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3 if target else 1)
            cv2.putText(img, str(t.track_id), (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        dist = "--" if math.isnan(est.distance) else f"{est.distance:.2f}m"
        cv2.putText(img, f"{est.state.value} id={est.track_id} {dist} {math.degrees(est.bearing):+.0f}deg",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        out.header = header
        return out


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
