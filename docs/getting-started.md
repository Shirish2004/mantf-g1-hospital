---
title: Getting started
nav_order: 2
---

# Getting started
{: .no_toc }

This page takes you from nothing to your first saved result. Each step says what it does and how to check that it
worked.

1. TOC
{:toc}

## What you need

| | |
|---|---|
| Computer | Ubuntu 22.04, an NVIDIA GPU with CUDA, 20 GB free disk for MuJoCo (about 200 GB more for Isaac Sim and its assets) |
| ROS 2 | Humble, installed from apt (step 1) |
| pixi | the Python environment manager: `curl -fsSL https://pixi.sh/install.sh \| bash` |
| Isaac Sim 6.2 | optional, only for Isaac runs, installed at `~/isaacsim` |

The project uses two other repositories as **git submodules**:

| submodule | what it is | used for |
|---|---|---|
| `mantf/` | [gatetub/mantf](https://github.com/gatetub/mantf): the multi-agent Active NTFields model | the neural network, its loss and its planner (never modified) |
| `WardZero/` | [Bkrakshith/hospital-sim](https://github.com/Bkrakshith/hospital-sim) (branch v3.1), with its own submodules | the hospital, the Unitree G1, the MuJoCo and Isaac simulators, the G1 controller |

## Install

### Step 1: ROS 2 Humble and system packages

```bash
# ROS 2 Humble: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html, then:
sudo apt install ros-humble-desktop ros-humble-navigation2 ros-humble-rmw-cyclonedds-cpp \
  python3-colcon-common-extensions python3-vcstool python3-venv git ffmpeg
```

Check: `ls /opt/ros/humble/setup.bash` prints the path.

### Step 2: get the code with its submodules

```bash
git clone --recurse-submodules https://github.com/Shirish2004/mantf-g1-hospital.git
cd mantf-g1-hospital
```

This downloads this project, `mantf/` and `WardZero/` (with all of WardZero's own submodules). If you cloned
without `--recurse-submodules`, run `git submodule update --init --recursive` now.

Check: `git submodule status` lists `mantf` and `WardZero` with no `-` in front.

{: .note }
> **Already have the two repositories next to this one?** If `../mantf` and `../WardZero` exist (as in the original
> development setup), you can skip the submodules. The scripts use them automatically. The order is: the
> environment variables `MANTF_ROOT` / `HG1_ROOT`, then `./mantf` / `./WardZero` if checked out, then `../mantf` /
> `../WardZero` (`scripts/paths.sh`).

### Step 3: build WardZero (the simulator side)

These are WardZero's own steps, run inside its folder:

```bash
cd WardZero
python3 -m venv --system-site-packages .venv
source env.sh
for r in g1-brainco-description g1-mujoco-sim ros2_ws/src/hospital-sim-ros2 g1-webcam-teleop hospital-sim-env; do
  [ -f "$r/requirements.txt" ] && .venv/bin/pip install -r "$r/requirements.txt"
done
(cd ros2_ws && colcon build --symlink-install)
cd ..
```

Check: `ls WardZero/ros2_ws/install/setup.bash` prints the path. Optionally run
`source WardZero/env.sh && ros2 launch g1_bringup sim.launch.py viewer:=true`; a MuJoCo window with the G1 in the
hospital opens (Ctrl-C to close).

### Step 4: set up this project

```bash
pixi install        # the Python 3.10 environment (torch with CUDA, numpy < 2, opencv, ...); a few minutes
pixi run build      # builds the ROS 2 messages and this project's package for Python 3.10 (ros2_overlay/)
pixi run setup      # adds the `mg1` command and the environment to bash and fish
```

Then **open a new terminal**, so that `mg1` exists.

Check: `mg1 test` ends with `10 passed`.

### Step 5 (optional): Isaac Sim 6.2

Install Isaac Sim 6.2 at `~/isaacsim` and the Isaac 6.0 asset pack at `~/isaacsim_assets/Assets/Isaac/6.0` (see
WardZero's README). WardZero targets Isaac 6.0; this project patches it to run on 6.2. The first Isaac run links the
asset packs into the hospital scene by itself.

Check: `mg1 isaac-smoke` prints `all APIs present`.

## Your first results

```bash
mg1 case cooperative offline --timeout 120   # 1. the whole pipeline without a simulator (about 3 minutes)
mg1 case cooperative mujoco                  # 2. three G1s in the MuJoCo hospital
mg1 case cooperative isaac                   # 3. the same in Isaac Sim (slower)
```

Each command:

1. starts everything it needs (simulator, RViz, sensor window, people, the fleet);
2. sends every robot its goal from `config/arrangement.yaml`;
3. waits until every robot is done (or `--timeout` seconds pass);
4. saves everything and prints a summary.

The summary, maps, videos and logs are in the folder it prints at the end, for example
`runs/mujoco/cooperative/arrangement/1123_053021/`. Start with its `results.md`. Then watch
`agents/<robot>/view_ep*_success.mp4` and look at `maps/final_fleet.png`. Every file is explained in
[Results and outputs](results).

## Words used in these pages

| word | meaning |
|---|---|
| agent / robot | one G1 humanoid. `g1` is simulated with physics (MuJoCo or Isaac); `k1`, `k2`, ... are *kinematic* G1s that move exactly as commanded |
| arrangement | the list of agents with their start and goal (`config/arrangement*.yaml`) |
| the model / the field | the one neural network that predicts the arrival time T between two points, given the nearest other agent |
| call | one turn of one agent: it shares x, y, lidar and plan; the model trains once and plans for it |
| viewpoint | where an agent heads next: a frontier cell, a cell of the mapped area, or its goal |
| frontier | the edge between the mapped free area and the unknown |
| (gx, gy) | the direction the model tells the agent to move in (`-grad T`) |
| case | `cooperative` or `greedy`: how agents choose viewpoints |
| guard | the safety layer that checks every motion command against the lidar and the other agents |
