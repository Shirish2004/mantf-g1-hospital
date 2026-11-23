---
title: Troubleshooting
nav_order: 9
---

# Troubleshooting

| symptom | cause / fix |
|---|---|
| `WardZero/` or `mantf/` is empty | the submodules were not fetched: `git submodule update --init --recursive` |
| `cannot import mantf from MANTF_ROOT=...` | mantf not found: fetch the submodule, or set `MANTF_ROOT` to your mantf checkout |
| scripts use the wrong WardZero | `echo $HG1_ROOT`; the order is `$HG1_ROOT`, `./WardZero` (if built), `../WardZero` (`scripts/paths.sh`) |
| `mg1: command not found` | run `pixi run setup` once, then open a new shell |
| nodes do not see each other | the same `ROS_DOMAIN_ID` in every terminal; CycloneDDS is pinned to loopback (`MANTF_CYCLONE_LO=0` to undo) |
| the G1 does not move after `mg1 goto` | `mg1 goto` requests STAND then NAV from WardZero's mode manager; in Isaac the e-stop heartbeat is started by the stack |
| very slow replanning (`mg1 status`: many seconds between calls) | another GPU/CPU-heavy program; or run without windows: `-- --no-rviz --no-viewer --no-mjviewer` |
| `numpy` 2 / pinocchio import errors | use the pixi environment (`mg1 shell`), not the system Python |
| Isaac hospital shows only boxes | asset packs not linked: `python3 $HG1_ROOT/hospital-sim-env/isaac/scene/select_assets.py full` (the stack does this) |
| Isaac too dark or too bright | `MANTF_ISAAC_LIGHTS=0` turns the added ceiling lights off |
| leftover processes after a crash | `mg1 stop`; it stops every stack and node of this project |
| no video in `agents/<id>/` | videos are written only while the agent has a goal; install `ffmpeg` for H.264 |
| a run stopped early with `preempted` goals | another `mg1 case` or `mg1 goto` sent goals on the same DDS domain; use `--domain` |

Logs: `logs/<stage>_<sim>_<time>/<node>.log` for every process of a stack, and `logs/case_<sim>_<case>_<time>/`
for `mg1 case` (stack output and one log per goal).
