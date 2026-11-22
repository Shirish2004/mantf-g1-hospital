import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("mantf_g1_bridge")
    cfg = os.path.join(pkg, "config", "planner.yaml")
    map_yaml = LaunchConfiguration("map_yaml")
    model_path = LaunchConfiguration("model_path")
    device = LaunchConfiguration("device")

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", default_value=os.environ.get("MANTF_G1_MAP", "")),
        DeclareLaunchArgument("model_path", default_value=os.environ.get("MANTF_ROOT", "") + "/Experiments"),
        DeclareLaunchArgument("device", default_value="auto"),
        Node(
            package="mantf_g1_bridge",
            executable="mantf_nav",
            name="mantf_nav",
            output="screen",
            parameters=[cfg, {
                "map_yaml": map_yaml,
                "model_path": model_path,
                "device": device,
            }],
        ),
    ])
