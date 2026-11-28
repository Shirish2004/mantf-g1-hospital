# mantf-g1-hospital

**Multi-agent mapping and motion planning with one physics-informed neural network.**
Unitree G1 humanoids start where you choose in the WardZero hospital (MuJoCo or Isaac Sim 6.2) and must reach goals
you choose, in a building they have never seen. One model, Active Neural Time Fields
([paper](references/2410.09883v2.pdf), Alg. 1) with the multi-agent mantf network (`../mantf`, unchanged), learns the
arrival-time field online from all agents' lidar and plans for every agent. The agents share their x, y and plans
with it.

**Documentation: <https://shirish2004.github.io/mantf-g1-hospital/>** (source in [`docs/`](docs/))

```
call k -> agent (k mod N): shares x, y, lidar, plan  ->  one model trains on it  ->  returns its viewpoint, (gx, gy)
                                                                                    and its Eq. 11 plan
```

- One model, one shared 2-D and 3-D map, centralised round-robin calls (agent 1, 2, ..., N, 1, ...).
- Two cases with the same model and sharing: `cooperative` (arrival time + goal distance + others' plans)
  and `greedy` (straight for the agent's own goal).
- 1 to 12 agents (one simulated G1 + up to 11 kinematic G1 teammates), each with its own start and goal
  (`config/arrangement_1.yaml` ... `arrangement_12.yaml`). An agent that reaches its goal stops and leaves the calls
  (N agents -> N - 1 calls per cycle); the others still see it as an obstacle.
- People walk through the hospital; the model treats them as other agents. In Isaac they are animated characters,
  and ceiling lights are added where the stage has none.
- Collision avoidance on the G1's real footprint. Contacts and close calls are measured for every agent.
- No A\*, no prior map for planning, no checkpoints: every run learns from zero.
- Saved for every run: every model call, every agent's lidar, trajectory, contacts, coverage and own-view video of each
  goal, the G1 head-camera video, the 2-D and 3-D maps, training logs and math checks.

## Setup (once)

You need Ubuntu 22.04, ROS 2 Humble, an NVIDIA GPU and [pixi](https://pixi.sh). Isaac Sim 6.2 is only needed for
Isaac runs. The full walkthrough, with a check after every step, is in [Getting started](docs/getting-started.md).

```bash
# 1. code + submodules: mantf (the model) and WardZero (hospital, G1, MuJoCo/Isaac), with WardZero's own submodules
git clone --recurse-submodules https://github.com/Shirish2004/mantf-g1-hospital.git
cd mantf-g1-hospital                 # (already cloned? git submodule update --init --recursive)

# 2. build WardZero (its own steps)
cd WardZero && python3 -m venv --system-site-packages .venv && source env.sh
for r in g1-brainco-description g1-mujoco-sim ros2_ws/src/hospital-sim-ros2 g1-webcam-teleop hospital-sim-env; do
  [ -f "$r/requirements.txt" ] && .venv/bin/pip install -r "$r/requirements.txt"; done
(cd ros2_ws && colcon build --symlink-install) && cd ..

# 3. this project
pixi install && pixi run build && pixi run setup     # then open a new terminal
mg1 test                                             # should end with "10 passed"
```

| submodule | repository | pinned |
|---|---|---|
| `mantf/` | [gatetub/mantf](https://github.com/gatetub/mantf) | `fullstack` (nagents + aniANTF SE(2) tracker) |
| `WardZero/` | [Bkrakshith/hospital-sim](https://github.com/Bkrakshith/hospital-sim) | `v3.1` @ 05e4e3b |

Already have `../mantf` and `../WardZero` next to this repository? Then the submodules are optional. The scripts
look for each dependency in this order: `$MANTF_ROOT` / `$HG1_ROOT`, the submodule if checked out, then the sibling
folder (`scripts/paths.sh`).

## Produce results

One command per experiment. It starts everything, sends every robot its goal, waits until all are done, saves, and
prints a summary:

```bash
mg1 case cooperative mujoco      # -> runs/mujoco/cooperative/arrangement/<time>/
mg1 case greedy mujoco
mg1 case cooperative isaac       # -> runs/isaac/cooperative/arrangement/<time>/
mg1 case greedy isaac
mg1 case cooperative offline     # no simulator
mg1 case cooperative mujoco --arrangement config/arrangement_1.yaml    # 1 agent (the paper's setting)
mg1 case cooperative isaac  --arrangement config/arrangement_4.yaml    # 4 agents

# both cases for one arrangement, one after the other:
for c in cooperative greedy; do mg1 case $c isaac --arrangement config/arrangement_4.yaml --timeout 1500; done
mg1 case greedy isaac --timeout 1200 --domain 22 -- --people 8 --no-rviz --no-viewer
```

Where each robot starts and where it must go is set in an *arrangement* file:

```yaml
# config/arrangement.yaml
robots:
  g1: {start: corridor_west, goal: pacu}          # simulated G1 (MuJoCo / Isaac)
  k1: {start: corridor_theatre, goal: pre_op}     # kinematic G1 teammates (up to 4)
  k2: {start: scrub_room, goal: theatre_door}
```

## Evaluation (n agents + moving-object tracker)

The model is mantf's `fullstack` branch: the set of other agents (K = 6 nearest robots and people) pooled as the
network's input, and one SE(2) tracker per agent that finds moving people from its lidar and bends the field with
the anisotropic metric `A = I - lam u u^T` (see [How it works](docs/method.md#moving-people-se2-tracker)). 50 epochs
per call.

```bash
export MANTF_ROOT=$PWD/mantf                       # the in-repo fullstack checkout (overrides a global ../mantf)
bash scripts/run_experiments.sh                    # all groups in Isaac: agents (n=2..12), speed, spread, explore
bash scripts/run_experiments.sh speed --sim offline
pixi run python scripts/analyze_experiments.py     # charts + summary.md in runs/isaac/cooperative/exp/report/
```

Results: [docs/experiments.md](docs/experiments.md).

## All commands

| command | what it does |
|---|---|
| `mg1 case CASE SIM [--arrangement F] [--timeout S] [--domain N] [-- stack opts]` | one experiment: up, goals, wait, save, `results.md`; CASE `cooperative`\|`greedy`, SIM `mujoco`\|`isaac`\|`offline` |
| `mg1 scene [--isaac]` | the hospital only |
| `mg1 robot [--isaac]` | + G1, RViz, OpenCV sensor window (RGB \| depth \| lidar BEV) |
| `mg1 stack [STAGE] [opts]` | + people + the fleet (interactive); STAGE `scene`\|`robot`\|`nav`\|`mission` |
| `mg1 goto <place>` \| `<x> <y> [yaw]` \| `list` | goal for the G1 (requests STAND -> NAV) |
| `mg1 goto --robot k1 <place>` | goal for a teammate |
| `mg1 status [--once]` | every agent: plan, (gx, gy), calls; loss, coverage, math checks |
| `mg1 stop` | stop everything; the fleet saves first |
| `mg1 offline [opts]` | the fleet without a simulator ([options](docs/commands.md#offline-mg1-offline)) |
| `mg1 pickplace <instrument> <place> [pick [home]]` | pick and place (stage `mission`) |
| `mg1 mission start\|abort\|pause\|resume\|goto:<loc>` | mission control |
| `mg1 fleet`, `mg1 people`, `mg1 viewer`, `mg1 rviz`, `mg1 ksim` | single nodes |
| `mg1 isaac [--gui\|--headless] [run_sim args]` | Isaac Sim 6.2 with WardZero's G1 backend (lights, people) |
| `mg1 test` | unit tests |
| `mg1 isaac-smoke` | Isaac 6.2 API check |
| `mg1 build` | rebuild the ROS overlay |
| `mg1 setup` | (re)install the global environment |
| `mg1 cd`, `mg1 shell`, `mg1 wardzero` | go to the project; pixi shell; WardZero environment |
| `tensorboard --logdir runs` | training curves and math checks |

`mg1 stack` options:

| option | default | |
|---|---|---|
| `--mujoco` / `--isaac` / `--isaac-headless` / `--kinematic` | `--mujoco` | simulator of the G1 |
| `--arrangement FILE` | `config/arrangement.yaml` | agents and starts |
| `--mode cooperative\|greedy` | `cooperative` | the case |
| `--people N` / `--no-people` | 5 | people on lanes cut from the map's free space |
| `--domain N` | `$ROS_DOMAIN_ID` (2) | DDS domain |
| `--runs DIR` | `runs/<sim>/<mode>` | run folder |
| `--spawn NAME\|x,y,yaw` | from the arrangement | G1 spawn override |
| `--no-rviz` `--no-viewer` `--no-mjviewer` | off | leave windows out |

Full reference, including the ROS interface: [docs/commands.md](docs/commands.md).

## Outputs

`runs/<sim>/<case>/<arrangement>/<time>/`: `results.md`, `calls.csv`, `train_log.csv`, `math_log.csv`,
`longrange_log.csv`, `episodes.json`, `events.log`, `maps/`, `map3d/` (.ply), `trajectories/`, `fleet_bev.mp4`, `tb/`.
Every agent's record is in `agents/<id>/`: `calls.jsonl`, `scans.npz`, `trajectory.csv`, `contacts.csv`,
`coverage.png`, `view_epNN_success.mp4`, and `camera_epNN_success.mp4` for the G1.
See [docs/results.md](docs/results.md).

## Documentation

| page | |
|---|---|
| [Getting started](docs/getting-started.md) | requirements, install, first run |
| [Commands](docs/commands.md) | every command and option, ROS topics and actions |
| [Arrangements](docs/arrangements.md) | starts and goals, 1 to 5 agents |
| [How it works](docs/method.md) | the model, round-robin calls, viewpoints, two cases, collision avoidance, math checks |
| [Results and outputs](docs/results.md) | every output file |
| [Simulators](docs/simulators.md) | MuJoCo, Isaac 6.2 (lights, people), kinematic teammates |
| [Status](docs/status.md) | measured results and known limitations |
| [Troubleshooting](docs/troubleshooting.md) | common problems |

**Publishing the docs on GitHub Pages:** push this repository, then go to *Settings → Pages → Build and deployment*
and choose *Deploy from a branch*, branch `main`, folder `/docs`. The site builds with the Just the Docs theme
(`docs/_config.yml`) and appears at <https://shirish2004.github.io/mantf-g1-hospital/>.

To preview locally: `cd docs && bundle exec jekyll serve` (needs Ruby, Jekyll, and the `jekyll-remote-theme` and
`jekyll-seo-tag` gems).
