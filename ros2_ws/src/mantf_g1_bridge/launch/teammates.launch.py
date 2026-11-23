"""robot_state_publisher per kinematic teammate (frame prefix <id>/) so RViz draws them as G1 models."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    urdf = open(os.path.join(get_package_share_directory("g1_description"), "urdf", "g1_29dof_brainco_ros.urdf")).read()
    sim = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    return [Node(package="robot_state_publisher", executable="robot_state_publisher", namespace=k, name="rsp",
                 parameters=[{"robot_description": urdf, "frame_prefix": f"{k}/", "use_sim_time": sim}])
            for k in LaunchConfiguration("agents").perform(context).split(",") if k]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("agents", default_value="k1,k2"),
                              DeclareLaunchArgument("use_sim_time", default_value="true"),
                              OpaqueFunction(function=_nodes)])
