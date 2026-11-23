---
title: Commands
nav_order: 3
---

# Commands
{: .no_toc }

Every command is a pixi task: `mg1 <task> [args]` (after `pixi run setup`), or `pixi run <task> [args]` inside the
project.

1. TOC
{:toc}

## Overview

| command | what it does |
|---|---|
| `mg1 case CASE SIM` | one complete experiment: bring up, send every goal, wait, save, print `results.md` |
| `mg1 scene` | the hospital only (MuJoCo viewer or Isaac GUI) |
| `mg1 robot` | + the G1, RViz, OpenCV sensor window |
| `mg1 stack` | + people + the fleet (one model for all agents), interactive |
| `mg1 goto` | send a goal to one agent |
| `mg1 status` | watch every agent, the training and the math checks |
| `mg1 stop` | stop everything (the fleet saves first) |
| `mg1 offline` | the fleet without a simulator |
| `mg1 pickplace`, `mg1 mission` | pick-and-place mission (stage `mission`) |
| `mg1 fleet`, `mg1 people`, `mg1 viewer`, `mg1 rviz`, `mg1 ksim` | single nodes |
| `mg1 test`, `mg1 isaac-smoke` | tests |
| `mg1 build`, `mg1 setup` | build the ROS overlay; install the global environment |
| `mg1 isaac` | Isaac Sim 6.2 with WardZero's G1 backend, on its own |
| `mg1 cd`, `mg1 shell`, `mg1 wardzero` | go to the project; pixi shell; WardZero's environment |

## Experiments: `mg1 case`

```bash
mg1 case cooperative mujoco
mg1 case greedy mujoco
mg1 case cooperative isaac
mg1 case greedy isaac
mg1 case cooperative offline
mg1 case greedy offline
mg1 case cooperative mujoco --arrangement config/arrangement_1.yaml     # 1 agent
mg1 case cooperative isaac  --arrangement config/arrangement_4.yaml     # 4 agents (up to arrangement_5)
for c in cooperative greedy; do mg1 case $c isaac --arrangement config/arrangement_4.yaml --timeout 1500; done   # both cases
mg1 case greedy isaac --timeout 1200 --domain 22 -- --people 8 --no-rviz
```

`mg1 case CASE SIM [options] [-- stack options]` (`scripts/run_case.sh`)

| argument | values / default | |
|---|---|---|
| `CASE` | `cooperative` \| `greedy` | how agents choose viewpoints ([Method](method#next-viewpoint-paper-sec-iv-e3-goal-directed-multi-agent)) |
| `SIM` | `mujoco` \| `isaac` \| `offline` | simulator of the G1; offline = no simulator |
| `--arrangement FILE` | `config/arrangement.yaml` | starts and goals ([Arrangements](arrangements)) |
| `--timeout S` | 900 | wall seconds (offline: simulated seconds) before giving up |
| `--domain N` | `$ROS_DOMAIN_ID` | DDS domain of this run |
| `-- ...` | | passed to `mg1 stack` (below) |

Results: `runs/<SIM>/<CASE>/<arrangement>/<MMDD_HHMMSS>/` ([Results and outputs](results)).

## Interactive: `mg1 scene`, `mg1 robot`, `mg1 stack`

```bash
mg1 scene                       # MuJoCo hospital only
mg1 scene --isaac               # Isaac hospital only (full assets)
mg1 robot                       # + G1 (MuJoCo), RViz, OpenCV sensor window
mg1 robot --isaac
mg1 stack                       # + people + fleet (MuJoCo)
mg1 stack --isaac --domain 22   # Isaac on its own DDS domain (can run beside MuJoCo)
mg1 stack --mode greedy --arrangement config/arrangement_5.yaml --people 8
mg1 stack mission               # + pick/place mission node
```

`mg1 stack [STAGE] [options]` (`scripts/run_stack.sh`)

| option | default | |
|---|---|---|
| `STAGE` = `scene` / `robot` / `nav` / `mission` | `nav` | each includes the previous |
| `--mujoco` / `--isaac` / `--isaac-headless` / `--kinematic` | `--mujoco` | simulator of the G1 (`kinematic` = no physics) |
| `--arrangement FILE` | `config/arrangement.yaml` | agents and starts (the G1 spawns at the first start) |
| `--mode cooperative\|greedy` | `cooperative` | the case |
| `--people N` / `--no-people` | 5 | people on lanes cut from the map's free space |
| `--domain N` | `$ROS_DOMAIN_ID` (2) | DDS domain |
| `--runs DIR` | `runs/<sim>/<mode>` | where the run folder goes |
| `--spawn NAME\|x,y,yaw` | from the arrangement | override the G1 spawn |
| `--no-rviz` `--no-viewer` `--no-mjviewer` | off | leave windows out (faster planning) |

Ctrl-C or `mg1 stop` stops everything. The fleet first saves its maps, 3-D map, trajectories, videos and
`results.md`. Logs of every process: `logs/<stage>_<sim>_<time>/`.

## Goals and status

```bash
mg1 goto list                       # named places
mg1 goto pacu                       # the simulated G1 (requests STAND -> NAV from WardZero's mode manager)
mg1 goto 20.0 -12.4 0               # x y yaw in the map frame (= simulator world frame)
mg1 goto --robot k1 pre_op          # a kinematic teammate
mg1 goto --no-mode pacu             # do not touch the G1's mode
mg1 status                          # every agent: state, pose, plan text, (gx, gy), calls; loss, coverage, checks
mg1 status --once
```

In RViz, "2D Goal Pose" sends a goal to the G1 (topic `/goal_pose`); `/<id>/goal_pose` reaches teammate `<id>`.

## Offline: `mg1 offline`

```bash
mg1 offline                                              # config/arrangement.yaml, cooperative
mg1 offline --mode greedy --arrangement config/arrangement_5.yaml --max-time 600
mg1 offline --robots g1@8.93,-12.45,0>26.93,-7.75,0 k1@25.93,-12.45,3.14   # k1 without a goal explores
```

| option | default | |
|---|---|---|
| `--arrangement FILE` | `config/arrangement.yaml` | starts and goals |
| `--robots id@x,y,yaw[>x,y,yaw] ...` | from the arrangement | override; no goal = explore |
| `--mode` | `cooperative` | or `greedy` |
| `--people` | 5 | |
| `--workspace cx,cy,side` | fitted to the arrangement | field square |
| `--max-time` | 300 | simulated seconds |
| `--round` | 0.2 | simulated seconds per model call |
| `--max-v`, `--epochs` | 0.75, 5 | |
| `--fourier-scale` | 1.0 | x mantf's Fourier feature frequencies |
| `--out DIR` | `runs/offline/<mode>` | |
| `--no-video` | off | skip the videos |

## Pick and place

```bash
mg1 stack mission                       # stage with the mission node
mg1 pickplace scissors pre_op_desk      # instrument, place [, pick [, home]]
mg1 mission start | abort | pause | resume | goto:<location>
```

## Single nodes

```bash
mg1 fleet -p "robots:=[\"g1:sim\", \"k1:kin:25.9,-12.4,3.1\"]" -p mode:=greedy   # the fleet node alone
mg1 people -p count:=8 -p seed:=1        # people node (/mantf/agents, /mantf/people)
mg1 viewer                               # OpenCV window: RGB | depth | lidar BEV
mg1 rviz                                 # RViz with config/mantf.rviz
mg1 ksim                                 # a kinematic G1 standing in for the simulator
ros2 launch mantf_g1_bridge fleet.launch.py mode:=greedy      # the fleet node via ros2 launch
ros2 launch mantf_g1_bridge teammates.launch.py agents:=k1,k2 # robot_state_publisher per teammate (RViz models)
```

## Tests and setup

```bash
mg1 test           # unit tests (maps, guard, round-robin fleet, people, separation, escapes)
mg1 isaac-smoke    # Isaac 6.2 API check
mg1 build          # rebuild the ROS overlay (after changing messages, launch files or package.xml)
mg1 setup          # (re)install the global environment for bash and fish
tensorboard --logdir runs
```

## ROS interface

| name | type | |
|---|---|---|
| `/navigate_to_pose`, `/<id>/navigate_to_pose` | `nav2_msgs/NavigateToPose` | goal of the G1 / of teammate `<id>` |
| `/goal_pose`, `/<id>/goal_pose` | `geometry_msgs/PoseStamped` | the same as a topic (RViz) |
| `/mantf/fleet` | `std_msgs/String` (JSON) | every agent: pose, state, plan, (gx, gy), shared data, requests, events |
| `/mantf/status` | `std_msgs/String` (JSON) | the G1 (used by the sensor window) |
| `/mantf/path`, `/mantf/<id>/path` | `nav_msgs/Path` | Eq. 11 plan of each agent |
| `/mantf/<id>/trail` | `nav_msgs/Path` | trajectory so far |
| `/mantf/explored` | `nav_msgs/OccupancyGrid` | shared map learned from lidar |
| `/mantf/map` | `nav_msgs/OccupancyGrid` | hospital map (display only) |
| `/mantf/map3d` | `sensor_msgs/PointCloud2` | shared 3-D voxel map |
| `/mantf/fleet_markers` | `visualization_msgs/MarkerArray` | viewpoints, goals, plan text, requests |
| `/mantf/fleet_poses` | `sensor_msgs/PointCloud2` | robot positions (people avoid them) |
| `/mantf/agents`, `/mantf/people`, `/mantf/walkers` | `PointCloud2`, `PoseArray`, `PointCloud2` | people for the model, Isaac, the viewer |
| `/g1/cmd_vel_nav` | `geometry_msgs/Twist` | the fleet's command to the simulated G1 |
