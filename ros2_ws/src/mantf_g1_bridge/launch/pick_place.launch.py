import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("mantf_g1_bridge")
    cfg = os.path.join(pkg, "config", "pick_place.yaml")
    pick_location = LaunchConfiguration("pick_location")
    place_location = LaunchConfiguration("place_location")
    instrument = LaunchConfiguration("instrument")

    return LaunchDescription([
        DeclareLaunchArgument("pick_location", default_value="theatre_instrument_table"),
        DeclareLaunchArgument("place_location", default_value="scrub_room"),
        DeclareLaunchArgument("instrument", default_value="scalpel"),
        Node(
            package="mantf_g1_bridge",
            executable="pick_place",
            name="pick_place",
            output="screen",
            parameters=[cfg, {
                "pick_location": pick_location,
                "place_location": place_location,
                "instrument": instrument,
            }],
        ),
    ])
