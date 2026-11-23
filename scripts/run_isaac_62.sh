#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/paths.sh"
export HG1_ROOT ISAAC_DIR="${ISAAC_DIR:-$HOME/isaacsim}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-2}"
MODE=--headless
case "${1:-}" in --gui|--headless|--smoke) MODE="$1"; shift ;; esac
if [ "$MODE" = "--smoke" ]; then
  exec "$HG1_ROOT/g1-isaac-sim/run.sh" --python "$ROOT/isaac/run_sim_62.py" --smoke "$@"
fi
exec "$HG1_ROOT/g1-isaac-sim/run.sh" --python "$ROOT/isaac/run_sim_62.py" "$MODE" "$@"
