"""The fleet node (one Active NTFields model for every agent) with config/fleet.yaml.
ros2 launch mantf_g1_bridge fleet.launch.py [use_sim_time:=true] [mode:=cooperative|greedy]"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("mantf_g1_bridge"), "config", "fleet.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("mode", default_value="cooperative"),
        Node(package="mantf_g1_bridge", executable="fleet", name="mantf_fleet", output="screen",
             parameters=[cfg, {"use_sim_time": LaunchConfiguration("use_sim_time"),
                               "mode": LaunchConfiguration("mode"),
                               "map_yaml": os.environ.get("MANTF_G1_MAP", "")}]),
    ])
