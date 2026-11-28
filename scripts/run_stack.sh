#!/usr/bin/env bash
# run_stack.sh STAGE [options]
#   scene    hospital only, no robot (Isaac GUI with the full asset scene, or the MuJoCo viewer)
#   robot    + G1, controller, mode manager, RViz, OpenCV sensor viewer
#   nav      + people + the fleet: one Active NTFields model (mantf) for the G1 and its kinematic teammates (default)
#   mission  + pick/place mission node
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/paths.sh"
export HG1_ROOT
STAGE=nav
case "${1:-}" in scene|robot|nav|mission) STAGE="$1"; shift ;; esac
BACKEND=mujoco; ISAAC_MODE=--gui; PEOPLE=5; RVIZ=1; VIEWER=1; MJVIEW=true; SPAWN=""; EXTRA=()
ARR="$ROOT/config/arrangement.yaml"; MODE=cooperative; RUNS=""; PSPEED=0.0; DETECT=true; SEED=0; SPREAD=0.0
while [ $# -gt 0 ]; do
  case "$1" in
    --isaac) BACKEND=isaac ;;
    --isaac-headless) BACKEND=isaac; ISAAC_MODE=--headless ;;
    --mujoco) BACKEND=mujoco ;;
    --kinematic) BACKEND=kinematic ;;
    --domain) export ROS_DOMAIN_ID="$2"; shift ;;
    --spawn) SPAWN="$2"; shift ;;
    --arrangement) ARR="$(realpath "$2")"; shift ;;
    --mode) MODE="$2"; shift ;;
    --runs) RUNS="$2"; shift ;;
    --people) PEOPLE="$2"; shift ;;
    --no-walkers|--no-people) PEOPLE=0 ;;
    --people-speed) PSPEED="$2"; shift ;;
    --no-detect) DETECT=false ;;
    --seed) SEED="$2"; shift ;;
    --spread) SPREAD="$2"; shift ;;
    --no-rviz) RVIZ=0 ;;
    --no-viewer) VIEWER=0 ;;
    --no-mjviewer) MJVIEW=false ;;
    -h|--help) sed -n 2,6p "$0"; echo "options: --mujoco|--isaac|--isaac-headless|--kinematic --domain N --spawn NAME|x,y,yaw --arrangement FILE --mode cooperative|greedy --runs DIR --people N --people-speed V --no-detect --seed S --spread R --no-rviz --no-viewer --no-mjviewer"; exit 0 ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-2}"
SPAWN="${SPAWN:-$(python3 "$ROOT/scripts/arrangement.py" "$ARR" spawn)}"
RUNS="${RUNS:-$ROOT/runs/$BACKEND/$MODE}"
[ "${MANTF_CYCLONE_LO:-1}" = 1 ] && export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://$ROOT/config/cyclonedds.xml}"
export MANTF_G1_MAP="${MANTF_G1_MAP:-$HG1_ROOT/ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/hospital_gt.yaml}"
SIMT=true; [ "$BACKEND" = kinematic ] && SIMT=false
export MANTF_SIM_TIME=$SIMT
LOG="$ROOT/logs/${STAGE}_${BACKEND}_$(date +%m%d_%H%M%S)"; mkdir -p "$LOG"
PIDS=()
cleanup() {
  echo; echo "[stack] stopping"
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done
  for _ in $(seq 1 40); do pgrep -f mantf_g1_bridge.fleet_node >/dev/null || break; sleep 1; done
  for p in "${PIDS[@]}"; do kill -9 -- "-$p" 2>/dev/null || kill -9 "$p" 2>/dev/null; done
}
trap cleanup EXIT INT TERM
bg() { local name="$1"; shift; setsid "$@" >"$LOG/$name.log" 2>&1 & PIDS+=($!); echo "[stack] $name (log $LOG/$name.log)"; }
PIX=(pixi run --manifest-path "$ROOT/pixi.toml")
# WardZero's nodes run on the system Python and may need packages from the user site (~/.local, e.g. onnxruntime for
# the G1 controller): drop pixi's PYTHONNOUSERSITE / PYTHONPATH for them, or the controller dies and the G1 falls.
WZ="unset PYTHONNOUSERSITE PYTHONPATH PYTHONHOME; source $HG1_ROOT/env.sh >/dev/null && source $ROOT/ros2_overlay/install/setup.bash && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID CYCLONEDDS_URI=${CYCLONEDDS_URI:-} &&"

ensure_isaac_assets() {
  local a="$HG1_ROOT/hospital-sim-env/isaac/assets"
  if [ ! -e "$a/isaac/Isaac" ] || [ ! -e "$a/i4h/Props" ]; then
    echo "[stack] linking Isaac asset packs (hospital-sim-env/isaac/scene/select_assets.py full)"
    python3 "$HG1_ROOT/hospital-sim-env/isaac/scene/select_assets.py" full
  fi
}
spawn_xyz() { python3 - "$SPAWN" <<'PY'
import json, os, sys
s = sys.argv[1]
if "," in s:
    print(s); raise SystemExit
d = json.load(open(os.path.join(os.environ["HG1_ROOT"], "g1-mujoco-sim/worlds/spawns.json")))
v = d.get("hospital", d)[s]
print(f"{v['x']},{v['y']},{v.get('yaw', 0.0)}")
PY
}

start_manager() {
  local upper
  upper=$(python3 -c "import json;print(json.load(open('$HG1_ROOT/ros2_ws/src/hospital-sim-ros2/g1_controller/policies/unitree_g1_12dof_placeholder/policy.json'))['default_qpos'][12:29])")
  bg manager bash -c "$WZ ros2 run g1_teleop_manager manager_node --ros-args -r __node:=g1_teleop_manager -p use_sim_time:=true -p zones_file:=$ROOT/config/zones_sim.yaml -p 'upper_body_default:=$upper'"
}

if [ "$STAGE" = scene ]; then
  if [ "$BACKEND" = isaac ]; then
    ensure_isaac_assets
    bg scene env -u PYTHONPATH -u LD_LIBRARY_PATH -u CONDA_PREFIX "$HG1_ROOT/hospital-sim-env/isaac/launch_isaac.sh" --open
  else
    bg scene "$HG1_ROOT/.venv/bin/python" -m mujoco.viewer --mjcf="$HG1_ROOT/g1-mujoco-sim/worlds/hospital_world_tier2.xml"
  fi
  echo "[stack] scene up ($BACKEND). Ctrl-C closes it."; wait; exit 0
fi

case "$BACKEND" in
  isaac)
    ensure_isaac_assets
    bg isaac env -u PYTHONPATH "$ROOT/scripts/run_isaac_62.sh" $ISAAC_MODE --theatre --camera --spawn "$SPAWN"
    for i in $(seq 1 120); do grep -q "running (Ctrl-C" "$LOG/isaac.log" 2>/dev/null && break; sleep 2; done
    bg estop "${PIX[@]}" python -m mantf_g1_bridge.sim_estop
    bg wardzero bash -c "$WZ ros2 launch g1_bringup sim.launch.py backend:=isaac manager:=false ${EXTRA[*]}"
    start_manager ;;
  mujoco)
    bg wardzero bash -c "$WZ ros2 launch g1_bringup sim.launch.py backend:=mujoco manager:=false viewer:=$MJVIEW spawn:=$SPAWN ${EXTRA[*]}"
    start_manager ;;
  kinematic)
    bg sim "${PIX[@]}" python -m mantf_g1_bridge.kinematic_sim --ros-args -p "start:=[$(spawn_xyz)]" -p map_yaml:="$MANTF_G1_MAP" ;;
esac
if [ "$BACKEND" != kinematic ]; then
  # the G1 controller must load its walking policy, or the G1 has no motor commands and falls
  ok=0
  for _ in $(seq 1 120); do
    if grep -q "controller_node.*process has died" "$LOG/wardzero.log" 2>/dev/null; then
      echo "[stack] ERROR: WardZero's G1 controller died (see $LOG/wardzero.log):"
      grep -m3 -E "Error|error" "$LOG/wardzero.log" | sed 's/^/          /'
      echo "[stack] without it the G1 falls; fix the error above and start again"; exit 1
    fi
    grep -q "g1_controller\]: policy " "$LOG/wardzero.log" 2>/dev/null && { ok=1; break; }
    sleep 2
  done
  [ "$ok" = 1 ] && echo "[stack] G1 controller up (policy loaded)" || echo "[stack] WARNING: the G1 controller has not reported its policy yet"
fi
sleep 5
[ "$RVIZ" = 1 ] && bg rviz bash "$ROOT/scripts/rviz.sh"
[ "$VIEWER" = 1 ] && bg viewer "${PIX[@]}" python -m mantf_g1_bridge.sensor_viewer --ros-args -p use_sim_time:=$SIMT \
  -p title:="G1 sensors ($BACKEND, domain $ROS_DOMAIN_ID)"
if [ "$STAGE" = nav ] || [ "$STAGE" = mission ]; then
  [ "$PEOPLE" -gt 0 ] && bg people "${PIX[@]}" python -m mantf_g1_bridge.walkers_node --ros-args -p use_sim_time:=$SIMT -p count:=$PEOPLE -p speed:=$PSPEED
  ROBOTS="$(python3 "$ROOT/scripts/arrangement.py" "$ARR" robots)"
  AG="$(python3 "$ROOT/scripts/arrangement.py" "$ARR" teammates)"
  [ -n "$AG" ] && bg teammates bash -c "$WZ ros2 launch mantf_g1_bridge teammates.launch.py agents:=$AG use_sim_time:=$SIMT"
  bg fleet "${PIX[@]}" python -m mantf_g1_bridge.fleet_node --ros-args \
    --params-file "$ROOT/ros2_ws/src/mantf_g1_bridge/config/fleet.yaml" -p use_sim_time:=$SIMT \
    -p map_yaml:="$MANTF_G1_MAP" -p "robots:=[$ROBOTS]" -p mode:=$MODE -p explore:=false \
    -p runs_dir:="$RUNS" -p arrangement:="$ARR" -p people_speed:=$PSPEED -p people:=$PEOPLE -p detect:=$DETECT \
    -p seed:=$SEED -p spread:=$SPREAD -p "workspace:=[$(python3 "$ROOT/scripts/arrangement.py" "$ARR" workspace)]"
fi
if [ "$STAGE" = mission ]; then
  bg mission "${PIX[@]}" python -m mantf_g1_bridge.pick_place_node --ros-args \
    -p config:="$ROOT/ros2_ws/src/mantf_g1_bridge/config/pick_place.yaml" -p repeat:=false -p use_sim_time:=$SIMT
fi
cat <<MSG

[stack] $STAGE up ($BACKEND, domain $ROS_DOMAIN_ID, logs $LOG). In another terminal (export ROS_DOMAIN_ID=$ROS_DOMAIN_ID):
  mg1 goto pre_op | mg1 goto 20 -12.4 0 | mg1 goto --robot k1 pacu   one model plans for every robot (or RViz "2D Goal Pose")
  mg1 pickplace scissors pre_op_desk                          pick/place (stage mission)
  mg1 status                                                  fleet state, loss, math checks
Ctrl-C here stops everything; the fleet saves maps (2-D, 3-D), trajectories, videos and results.md on exit.
MSG
wait
