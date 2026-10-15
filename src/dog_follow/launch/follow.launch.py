"""Full follow stack.

  ros2 launch dog_follow follow.launch.py robot:=m20 image:=/camera/color/image_raw \
      camera_info:=/camera/color/camera_info
  robot:=twist  -> generic adapter publishing to robots.yaml:robot_adapter.output_topic
  robot:=none   -> no adapter (inspect /follow/cmd_vel only)
  range:=depth depth:=/camera/aligned_depth_to_color/image_raw   (phase 2)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("dog_follow")
    follow_cfg = os.path.join(share, "config", "follow.yaml")
    robot_cfg = os.path.join(share, "config", "robots.yaml")

    args = [
        DeclareLaunchArgument("robot", default_value="none", choices=["m20", "twist", "none"]),
        DeclareLaunchArgument("image", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("camera_info", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("depth", default_value="/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("range", default_value="mono", choices=["mono", "depth"]),
    ]

    perception = Node(
        package="dog_follow", executable="perception", name="follow_perception", output="screen",
        parameters=[follow_cfg, {"range_source": LC("range")}],
        remappings=[("image", LC("image")), ("camera_info", LC("camera_info")), ("depth", LC("depth"))],
    )
    controller = Node(
        package="dog_follow", executable="controller", name="follow_controller", output="screen",
        parameters=[follow_cfg], remappings=[("cmd_vel", "/follow/cmd_vel")],
    )
    m20 = Node(
        package="dog_follow", executable="m20_adapter", name="m20_adapter", output="screen",
        parameters=[robot_cfg], remappings=[("cmd_vel_in", "/follow/cmd_vel")],
        condition=LaunchConfigurationEquals("robot", "m20"),
    )
    twist = Node(
        package="dog_follow", executable="twist_adapter", name="robot_adapter", output="screen",
        parameters=[robot_cfg], remappings=[("cmd_vel_in", "/follow/cmd_vel")],
        condition=LaunchConfigurationEquals("robot", "twist"),
    )
    return LaunchDescription(args + [perception, controller, m20, twist])
