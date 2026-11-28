#!/usr/bin/env bash
# run_case.sh cooperative|greedy mujoco|isaac|offline [--arrangement FILE] [--timeout S] [--domain N] [--tag NAME] [-- stack options]
# Brings the stack up, sends every agent its goal from the arrangement, waits until all are done, stops (the fleet saves
# maps, 3-D map, trajectories, videos, results.md) and prints the results.
# Runs land in runs/<sim>/<case>/<arrangement>/<time>/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${1:-cooperative}"; SIM="${2:-mujoco}"; shift 2 || true
ARR="$ROOT/config/arrangement.yaml"; TIMEOUT=900; DOMAIN="${ROS_DOMAIN_ID:-2}"; MORE=(); TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --arrangement) ARR="$(realpath "$2")"; shift ;;
    --timeout) TIMEOUT="$2"; shift ;;
    --domain) DOMAIN="$2"; shift ;;
    --tag) TAG="$2"; shift ;;
    --) shift; MORE=("$@"); break ;;
    *) MORE+=("$1") ;;
  esac
  shift
done
RUNS="$ROOT/runs/$SIM/$CASE/${TAG:-$(basename "$ARR" .yaml)}"
PIX=(pixi run --manifest-path "$ROOT/pixi.toml")
if [ "$SIM" = offline ]; then
  exec "${PIX[@]}" python "$ROOT/scripts/fleet_offline.py" --mode "$CASE" --arrangement "$ARR" --max-time "$TIMEOUT" \
    --out "$RUNS" "${MORE[@]}"
fi
export ROS_DOMAIN_ID="$DOMAIN"
mkdir -p "$RUNS"; BEFORE="$(ls "$RUNS")"
LOG="$ROOT/logs/case_${SIM}_${CASE}_$(date +%m%d_%H%M%S)"; mkdir -p "$LOG"
setsid bash "$ROOT/scripts/run_stack.sh" nav "--$SIM" --mode "$CASE" --arrangement "$ARR" --runs "$RUNS" --domain "$DOMAIN" \
  "${MORE[@]}" > "$LOG/stack.out" 2>&1 < /dev/null &
STACK=$!
trap 'kill -TERM $STACK 2>/dev/null; bash "$ROOT/scripts/stop_stack.sh"; exit 130' INT TERM
echo "[case] $CASE on $SIM (domain $DOMAIN), logs $LOG; waiting for the fleet"
RUN=""
for _ in $(seq 1 600); do
  NEW=$(comm -13 <(echo "$BEFORE" | sort) <(ls "$RUNS" | sort) | tail -1)
  [ -n "$NEW" ] && { RUN="$RUNS/$NEW"; break; }
  kill -0 $STACK 2>/dev/null || { echo "[case] stack exited, see $LOG/stack.out"; exit 1; }
  sleep 2
done
[ -z "$RUN" ] && { echo "[case] the fleet did not start"; bash "$ROOT/scripts/stop_stack.sh"; exit 1; }
# the fleet is up; give the simulator and the G1 controller a moment, then send the goals (agents start making
# model calls once they have a goal)
F=$(ls -td "$ROOT"/logs/nav_${SIM}_* 2>/dev/null | head -1)
for _ in $(seq 1 60); do grep -q "one model" "$F/fleet.log" 2>/dev/null && break; sleep 2; done
sleep 15
echo "[case] run folder $RUN; sending goals"
GP=()
while read -r rid goal; do
  if [ "$rid" = g1 ] || [ "$rid" = "$(python3 "$ROOT/scripts/arrangement.py" "$ARR" robots | cut -d'"' -f2 | cut -d: -f1)" ]; then
    "${PIX[@]}" python -m mantf_g1_bridge.goto $goal > "$LOG/goto_$rid.log" 2>&1 &
  else
    "${PIX[@]}" python -m mantf_g1_bridge.goto --robot "$rid" $goal > "$LOG/goto_$rid.log" 2>&1 &
  fi
  GP+=($!)
done < <(python3 "$ROOT/scripts/arrangement.py" "$ARR" goals)
END=$((SECONDS + TIMEOUT))
for p in "${GP[@]}"; do
  while kill -0 "$p" 2>/dev/null && [ $SECONDS -lt $END ]; do sleep 5; done
done
for f in "$LOG"/goto_*.log; do echo "[case] $(basename "$f" .log): $(tail -n 1 "$f")"; done
sleep 5
bash "$ROOT/scripts/stop_stack.sh"
wait $STACK 2>/dev/null
cat "$RUN/results.md" 2>/dev/null
echo "[case] saved $RUN"
