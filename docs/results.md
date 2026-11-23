---
title: Results and outputs
nav_order: 6
---

# Producing and reading results

## 1. Choose starts and goals

`config/arrangement.yaml`. The first robot is the simulated G1 (MuJoCo or Isaac); the others are kinematic G1s on
the same map. Use a named place (`mg1 goto list`) or `[x, y, yaw]` in the map frame (= simulator world frame).

```yaml
robots:
  g1: {start: corridor_west, goal: pacu}
  k1: {start: corridor_theatre, goal: pre_op}
  k2: {start: scrub_room, goal: theatre_door}
  # k3: {start: [20.0, -12.4, 0.0], goal: sterile_store}
```

Ready-made: `config/arrangement_1.yaml` ... `config/arrangement_5.yaml` (1 to 5 agents; at most 5).
Every agent needs its own start and goal; keep them at least ~1 m from walls and ~2.5 m from each other.

## 2. Run a case

```bash
mg1 case cooperative mujoco          # brings up MuJoCo + RViz + sensor window, sends every goal, waits, saves
mg1 case greedy mujoco
mg1 case cooperative isaac           # Isaac Sim 6.2 GUI, lights, walking people, teammates as G1 bodies
mg1 case greedy isaac
mg1 case cooperative offline         # no simulator (fast check)
mg1 case greedy offline
mg1 case cooperative isaac --arrangement config/arrangement_4.yaml
```

Options: `--arrangement FILE`, `--timeout S` (default 900), `--domain N` (DDS domain), and anything after `--` goes
to the stack (for example `-- --people 8 --no-rviz`).

While a run is going: `mg1 status` (every agent's plan and (gx, gy)), RViz (paths, viewpoints, plan text above each
robot), the OpenCV sensor window (G1 RGB | depth | lidar BEV with teammates and people).

## 3. Where results go

`runs/<mujoco|isaac|offline>/<cooperative|greedy>/<arrangement>/<MMDD_HHMMSS>/`

| file | content |
|---|---|
| `results.md` | summary: training, math checks, every episode, every agent |
| `arrangement.yaml` | starts and goals of this run |
| `config.json` | model and sampling parameters (workspace, dmin/dmax, N, m, j, batch, lr, memory) |
| `calls.csv` | every model call: t, call, agent, x, y, gx, gy, S, viewpoint, kind, plan, other agent, goal |
| `train_log.csv` | per call: agent, epoch, loss, eikonal diff, rows, people, min robot/people gap, train s |
| `math_log.csv` | local math checks every 5 calls (see docs/method.md) |
| `longrange_log.csv` | every 16 calls: T vs true shortest-path rank correlation, (gx, gy) towards walls |
| `episodes.json` | per goal: success, time, path, mean speed, min wall/person/robot distance, contacts |
| `events.log` | opportunities, tried viewpoints, contacts, proximity, goals reached or failed |
| `maps/<tag>_fleet.png` | shared explored map, cells tinted by the agent that saw them, trails, plans, (gx, gy) arrows |
| `maps/<tag>_fields.png` | per agent: learned speed S + T contours, arrival time T |
| `maps/<tag>.npz` | log-odds grid, seen_by, speed/time fields, trails |
| `map3d/map3d.{ply,npz,png}` | shared 3-D voxel map (open the .ply in MeshLab/CloudCompare) with trajectories |
| `trajectories/` | `<agent>.csv` (t, x, y, yaw), `trajectories.npz` (+ people), `trajectories.png` |
| `fleet_bev.mp4` | the shared map, one frame per model call |
| `tb/` | TensorBoard: `tensorboard --logdir runs` |

Every agent, in `agents/<id>/`:

| file | content |
|---|---|
| `calls.jsonl` | one line per call of this agent: pose, what it shared, what it received (other-agent input, others' viewpoints, positions, people, requests), viewpoint and kind, (gx, gy), S, full plan, plan and training times |
| `scans.npz` | the lidar of each call (origin, hit points) |
| `trajectory.csv` | t, x, y, yaw at 5 Hz |
| `contacts.csv` | wall contacts (G1 footprint touching the true map): t, x, y, yaw, duration, command |
| `agent_proximity.json` | moments it came closer than 0.60 m to a robot or 0.55 m to a person |
| `coverage.png` | cells this agent added to the shared map |
| `view_epNN_{success,fail}.mp4` | its own view during each goal: shared map around it, facing up, its lidar, plan, (gx, gy), plan text |
| `camera_epNN_{success,fail}.mp4` | the simulated G1 only: head camera RGB | depth with a HUD |
| `episodes.json`, `summary.json` | its goals and totals (calls, s between calls, contacts, min wall distance, path, map cells) |

Maps are saved every 20 s, at the end of each episode and on exit (`mg1 stop` or Ctrl-C).
