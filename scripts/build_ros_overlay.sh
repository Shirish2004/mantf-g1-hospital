#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/paths.sh"
SRC="$HG1_ROOT/ros2_ws/src/hospital-sim-ros2"
CLEAN_PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -e '\.pixi' -e conda | paste -sd: -)"
exec env -i HOME="$HOME" USER="$USER" PATH="$CLEAN_PATH" TERM="${TERM:-xterm}" bash -c "
set -e
source /opt/ros/humble/setup.bash
cd '$ROOT/ros2_overlay' 2>/dev/null || { mkdir -p '$ROOT/ros2_overlay'; cd '$ROOT/ros2_overlay'; }
colcon build --symlink-install \
  --base-paths '$SRC/g1_interfaces' '$SRC/g1_mission' '$SRC/third_party/unitree_ros2' '$ROOT/ros2_ws/src/mantf_g1_bridge' \
  --packages-select g1_interfaces g1_mission unitree_api unitree_go unitree_hg mantf_g1_bridge \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=OFF $*
"
