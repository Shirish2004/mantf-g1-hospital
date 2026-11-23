#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/paths.sh"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-2}" RMW_IMPLEMENTATION="${MANTF_RMW:-rmw_cyclonedds_cpp}"
[ "${MANTF_CYCLONE_LO:-1}" = 1 ] && export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://$ROOT/config/cyclonedds.xml}"
exec env -u PYTHONPATH -u LD_LIBRARY_PATH -u CONDA_PREFIX bash -c "source /opt/ros/humble/setup.bash && source $HG1_ROOT/ros2_ws/install/setup.bash && exec rviz2 -d '$ROOT/config/mantf.rviz' --ros-args -p use_sim_time:=${MANTF_SIM_TIME:-true} $*"
