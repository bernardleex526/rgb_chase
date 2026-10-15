"""/follow/target -> geometry_msgs/Twist on cmd_vel, with stale-data watchdog and enable switch."""
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool

from ..core.controller import ControllerParams, FollowController
from ..core.types import FollowState, TargetEstimate


class ControllerNode(Node):
    def __init__(self):
        super().__init__("follow_controller")
        defaults = ControllerParams()
        for name, val in vars(defaults).items():
            self.declare_parameter(name, val)
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("target_timeout", 0.5)
        self.declare_parameter("enabled_on_start", False)
        self.params = ControllerParams(**{n: self.get_parameter(n).value for n in vars(defaults)})
        self.ctrl = FollowController(self.params)
        self.target_timeout = self.get_parameter("target_timeout").value
        self.enabled = self.get_parameter("enabled_on_start").value
        self._target = None
        self._target_rx = 0.0

        self.pub = self.create_publisher(Twist, "cmd_vel", 1)
        self.create_subscription(String, "/follow/target", self._on_target, 1)
        self.create_service(SetBool, "~/enable", self._on_enable)
        self.create_timer(1.0 / self.get_parameter("rate").value, self._tick)
        self.get_logger().info(f"controller ready, enabled={self.enabled}")

    def _on_target(self, msg: String):
        d = json.loads(msg.data)
        self._target = TargetEstimate(
            state=FollowState(d["state"]),
            track_id=d["track_id"],
            bearing=d["bearing"],
            distance=math.nan if d["distance"] is None else d["distance"],
            distance_valid=d["distance_valid"],
            lost_duration=d["lost_duration"],
            last_bearing=d["last_bearing"],
        )
        self._target_rx = time.monotonic()

    def _on_enable(self, req, resp):
        self.enabled = req.data
        self.ctrl.reset()
        if not self.enabled:
            self.pub.publish(Twist())
        resp.success = True
        resp.message = f"enabled={self.enabled}"
        self.get_logger().info(resp.message)
        return resp

    def _tick(self):
        now = time.monotonic()
        if not self.enabled:
            return
        fresh = self._target is not None and now - self._target_rx < self.target_timeout
        target = self._target if fresh else TargetEstimate(state=FollowState.IDLE)
        if not fresh and self._target is not None:
            self.get_logger().warn("perception stale, stopping", throttle_duration_sec=2.0)
        vx, wz = self.ctrl.compute(target, now)
        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        self.pub.publish(cmd)

    def stop(self):
        self.pub.publish(Twist())


def main():
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
