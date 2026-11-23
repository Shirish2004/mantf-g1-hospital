#!/usr/bin/env bash
# Pixi activation: make the system ROS 2 Humble underlay, the WardZero overlay and
# this project's overlay visible to the Pixi Python 3.10 interpreter.

MANTF_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MANTF_G1_ROOT
source "$MANTF_G1_ROOT/scripts/paths.sh"
export MANTF_ROOT HG1_ROOT
export MANTF_G1_MAP="${MANTF_G1_MAP:-$HG1_ROOT/ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/hospital_gt.yaml}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1

# Conda's libstdc++/libssl shadow the system libraries rclpy's DDS vendors link
# against. Python and pip wheels find their own libs through RPATH, so dropping
# only the env lib directory is safe and keeps rclpy working.
if [ -n "${CONDA_PREFIX:-}" ]; then
    LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
        | grep -v -F "$CONDA_PREFIX/lib" | grep -v '^$' | paste -sd: -)"
    export LD_LIBRARY_PATH
fi

source /opt/ros/humble/setup.bash

# The simulator and WardZero nodes speak CycloneDDS. Set MANTF_RMW=rmw_fastrtps_cpp
# to use Fast DDS instead.
export RMW_IMPLEMENTATION="${MANTF_RMW:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-2}"
if [ "${MANTF_CYCLONE_LO:-1}" = 1 ] && [ -z "${CYCLONEDDS_URI:-}" ]; then
    export CYCLONEDDS_URI="file://$MANTF_G1_ROOT/config/cyclonedds.xml"
fi

if [ -f "$HG1_ROOT/ros2_ws/install/setup.bash" ]; then
    source "$HG1_ROOT/ros2_ws/install/setup.bash"
fi
# Source this last: it supplies the Python 3.10 g1_interfaces build and shadows
# the 3.13 copy that may be left in WardZero's install tree.
if [ -f "$MANTF_G1_ROOT/ros2_overlay/install/setup.bash" ]; then
    source "$MANTF_G1_ROOT/ros2_overlay/install/setup.bash"
fi

export PYTHONPATH="$MANTF_G1_ROOT/ros2_ws/src/mantf_g1_bridge${PYTHONPATH:+:$PYTHONPATH}"
