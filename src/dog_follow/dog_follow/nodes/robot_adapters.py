"""Robot adapters: the only robot-specific code. Input is always `cmd_vel_in` (geometry_msgs/Twist).

generic_twist: republish to any Twist topic with clamping and a watchdog (Unitree go2_ros2_sdk,
               Clearpath, Nav2-ready bases ...).
m20:           DEEP Robotics M20 / M20 Pro via sdk_deploy built with RemoteCommandType::kRos2.
               Talks to /rl_deploy/cmd_vel (needs >=50 Hz, 0.25 s timeout on the robot) and
               /rl_deploy/command_state; reads /rl_deploy/status (drdds/msg/RobotStatus).
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


class TwistAdapter(Node):
    def __init__(self, name="robot_adapter"):
        super().__init__(name)
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("input_timeout", 0.3)
        self.declare_parameter("max_vx", 0.6)
        self.declare_parameter("max_vx_reverse", 0.2)
        self.declare_parameter("max_vy", 0.0)
        self.declare_parameter("max_wz", 0.7)
        self.declare_parameter("output_best_effort", False)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.timeout = g("input_timeout")
        self.lim = (g("max_vx"), g("max_vx_reverse"), g("max_vy"), g("max_wz"))
        self._cmd = Twist()
        self._rx = 0.0
        self._estop = False

        qos = QoSProfile(depth=1)
        if g("output_best_effort"):
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            qos.durability = DurabilityPolicy.VOLATILE
        self.out = self.create_publisher(Twist, g("output_topic"), qos)
        self.create_subscription(Twist, "cmd_vel_in", self._on_cmd, 1)
        self.create_service(Trigger, "~/estop", self._on_estop)
        self.create_service(Trigger, "~/clear_estop", self._on_clear)
        self.create_timer(1.0 / g("rate"), self._tick)

    def _on_cmd(self, msg: Twist):
        self._cmd = msg
        self._rx = time.monotonic()

    def _on_estop(self, _req, resp):
        self._estop = True
        self.out.publish(Twist())
        resp.success, resp.message = True, "estop latched"
        self.get_logger().warn(resp.message)
        return resp

    def _on_clear(self, _req, resp):
        self._estop = False
        resp.success, resp.message = True, "estop cleared"
        self.get_logger().info(resp.message)
        return resp

    def limited(self) -> Twist:
        out = Twist()
        if self._estop or time.monotonic() - self._rx > self.timeout:
            return out
        c = self._cmd
        if not all(math.isfinite(v) for v in (c.linear.x, c.linear.y, c.angular.z)):
            return out
        max_vx, max_rev, max_vy, max_wz = self.lim
        out.linear.x = min(max(c.linear.x, -max_rev), max_vx)
        out.linear.y = min(max(c.linear.y, -max_vy), max_vy)
        out.angular.z = min(max(c.angular.z, -max_wz), max_wz)
        return out

    def _tick(self):
        self.out.publish(self.limited())

    def stop(self):
        self.out.publish(Twist())


class M20Adapter(TwistAdapter):
    def __init__(self):
        super().__init__("m20_adapter")
        self.declare_parameter("sdk_ns", "/rl_deploy")
        self.declare_parameter("auto_stand", False)
        self.declare_parameter("stand_settle_time", 3.0)
        ns = self.get_parameter("sdk_ns").value
        # SDK subscriber is best_effort/volatile depth 1 on <ns>/cmd_vel.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.destroy_publisher(self.out)
        self.out = self.create_publisher(Twist, f"{ns}/cmd_vel", qos)
        self.state_pub = self.create_publisher(
            String, f"{ns}/command_state",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE))
        self.robot_state = "unknown"
        self._status_rx = 0.0
        try:
            from drdds.msg import RobotStatus

            self.create_subscription(RobotStatus, f"{ns}/status", self._on_status, 1)
        except ImportError:
            self.get_logger().warn("drdds not found: robot state unknown, build drdds from sdk_deploy for status")
        for cmd in ("stand", "rl_control", "lie_down", "damping"):
            self.create_service(Trigger, f"~/{cmd}", self._state_cb(cmd))
        if self.get_parameter("auto_stand").value:
            self._auto_stand()

    def _on_status(self, msg):
        self.robot_state = msg.current_state
        self._status_rx = time.monotonic()

    def _state_cb(self, cmd):
        def cb(_req, resp):
            if cmd in ("lie_down", "damping"):
                self.stop()
            self.state_pub.publish(String(data=cmd))
            resp.success, resp.message = True, f"sent {cmd} (current: {self.robot_state})"
            self.get_logger().info(resp.message)
            return resp
        return cb

    def _auto_stand(self):
        self.get_logger().info("auto_stand: stand -> rl_control")
        self.state_pub.publish(String(data="stand"))
        settle = self.get_parameter("stand_settle_time").value

        def go_rl():
            timer.cancel()
            self.state_pub.publish(String(data="rl_control"))
            self.get_logger().info("auto_stand: rl_control sent")
        timer = self.create_timer(settle, go_rl)

    def limited(self) -> Twist:
        out = super().limited()
        # Without fresh status we can't see the robot's mode; keep commanding (SDK ignores
        # velocity outside rl_control), but warn so the operator knows.
        if self._status_rx and time.monotonic() - self._status_rx > 1.0:
            self.get_logger().warn("no /status from M20 for >1 s", throttle_duration_sec=2.0)
        elif self._status_rx and self.robot_state != "rl_control":
            self.get_logger().info(f"robot in '{self.robot_state}', velocity ignored until rl_control",
                                   throttle_duration_sec=5.0)
        return out


def _run(node_cls):
    rclpy.init()
    node = node_cls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


def main_twist():
    _run(TwistAdapter)


def main_m20():
    _run(M20Adapter)
