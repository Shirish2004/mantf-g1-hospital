#!/usr/bin/env bash
# run_experiments.sh [GROUP ...] [--sim isaac|offline] [--timeout S]
# The evaluation table, one run per row of each group, in Isaac (1 simulated G1 + kinematic teammates):
#   agents   n = 2..12 agents, 5 people                                (time to complete, time per agent to reach)
#   speed    4 agents, every person at 0.3/0.6/0.9/1.2/1.5 m/s          (robot vs obstacle speed, avoidance, next position)
#   spread   4 agents, teammates spawned within 2/4/8 m of the G1       (spawning agents nearer)
#   explore  4 agents, 8 people, SE(2) tracker on vs off                (exploration time with dynamic objects)
# The SE(2) tracking accuracy comes out of every run (tracking_log.csv). Then: analyze_experiments.py.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MANTF_ROOT="${MANTF_ROOT_EXP:-$ROOT/mantf}"
SIM=isaac; TIMEOUT=2400; SETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --sim) SIM="$2"; shift ;;
    --timeout) TIMEOUT="$2"; shift ;;
    *) SETS+=("$1") ;;
  esac
  shift
done
[ ${#SETS[@]} -eq 0 ] && SETS=(agents speed spread explore)
LOG="$ROOT/logs/experiments_$(date +%m%d_%H%M%S).log"
FAST=(--no-rviz --no-viewer)
[ "$SIM" = isaac ] && FAST=(--isaac-headless "${FAST[@]}")

run() {  # run TAG ARRANGEMENT [stack options]
  local tag="$1" arr="$2"; shift 2
  echo "[exp] $(date +%H:%M:%S) $tag ($arr $*)" | tee -a "$LOG"
  if [ "$SIM" = offline ]; then
    bash "$ROOT/scripts/run_case.sh" cooperative offline --arrangement "$arr" --timeout 600 --tag "exp/$tag" -- "$@" \
      >> "$LOG" 2>&1
  else
    bash "$ROOT/scripts/run_case.sh" cooperative "$SIM" --arrangement "$arr" --timeout "$TIMEOUT" --tag "exp/$tag" \
      -- "${FAST[@]}" "$@" >> "$LOG" 2>&1
  fi
  bash "$ROOT/scripts/stop_stack.sh" >/dev/null 2>&1; sleep 10
}

for g in "${SETS[@]}"; do
  case "$g" in
    agents)
      for n in 2 3 4 5 6 7 8 9 10 11 12; do run "agents/n$n" "$ROOT/config/arrangement_$n.yaml"; done ;;
    speed)
      for v in 0.3 0.6 0.9 1.2 1.5; do run "speed/v$v" "$ROOT/config/arrangement_4.yaml" --people-speed "$v"; done ;;
    spread)
      for r in 2 4 8; do
        f="$ROOT/config/spread/arrangement_4_r${r}_s0.yaml"
        [ -f "$f" ] || pixi run --manifest-path "$ROOT/pixi.toml" python "$ROOT/scripts/make_arrangement.py" spread --n 4 --spread "$r" --seed 0
        run "spread/r$r" "$f" --seed 0 --spread "$r"
      done ;;
    explore)
      run "explore/detect" "$ROOT/config/arrangement_4.yaml" --people 8
      run "explore/nodetect" "$ROOT/config/arrangement_4.yaml" --people 8 --no-detect ;;
    *) echo "unknown group $g" ;;
  esac
done
echo "[exp] done; log $LOG" | tee -a "$LOG"
