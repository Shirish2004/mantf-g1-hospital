#!/usr/bin/env bash
# Stop every stack started by run_stack.sh (the fleet saves its maps, trajectories and videos first).
for p in $(pgrep -f "scripts/run_stack.sh"); do kill -TERM "$p"; done
for _ in $(seq 1 45); do pgrep -f "mantf_g1_bridge.fleet_node|run_stack.sh" >/dev/null || break; sleep 1; done
pkill -9 -f "mantf_g1_bridge\.|g1_bringup|g1_mujoco_sim|g1_controller|g1_teleop_manager|run_sim_62|rviz2 -d .*mantf.rviz|teammates.launch" 2>/dev/null
echo "[stop] done"
