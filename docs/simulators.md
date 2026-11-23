---
title: Simulators
nav_order: 7
---

# Simulators

## MuJoCo (default)

WardZero's MuJoCo hospital with the G1, its placeholder walking policy, 2-D and 3-D lidar, and the head RGB-D camera.
`mg1 stack` launches it with `manager:=false`. The teleop manager is then started with `config/zones_sim.yaml`, which
lifts the clinical speed zones (0.4-0.6 m/s) for simulation. The policy loses about 0.13 m/s of every command and does
not move below about 0.34 m/s; the fleet commands up to 0.75 m/s.

MuJoCo simulates no people. The people exist for the model and the guard (`/mantf/agents`) and are drawn in RViz,
but the MuJoCo lidar and camera do not see them.

## Isaac Sim 6.2

WardZero targets Isaac Sim 6.0. `scripts/run_isaac_62.sh` runs WardZero's `run.sh --python isaac/run_sim_62.py`,
which patches `SimulationApp` (asset root -> the local 6.0 asset pack, extensions) and then runs WardZero's
`run_sim.py`. The first run links the asset packs (`hospital-sim-env/isaac/assets/{isaac,i4h}`).

`isaac/isaac_extras.py` runs inside Isaac:

| addition | how | switch |
|---|---|---|
| lights | 0.8 m ceiling panels every 3.5 m over free floor wherever the stage has no light within 3 m | `MANTF_ISAAC_LIGHTS=0` |
| people | one Isaac People character per `/mantf/people` entry, following the published pose while playing the walk cycle in place | `MANTF_ISAAC_PEOPLE=0` |
| teammates | coloured capsule robots (white block = front) at the kinematic teammates' poses from `/mantf/fleet` | with people |

The Isaac lidar ray-casts a static snapshot of the collision geometry, so it does not see people or teammates. The
G1 camera does see them, and the model and guard get their positions directly.

Isaac runs well below real time with the full hospital, GUI and camera (about 0.2x here). Everything uses sim time,
so the results are unaffected, but a run takes longer in wall time.

## Kinematic teammates

`k1, k2, ...` are simulated inside the fleet node: commands are integrated, the G1 footprint collides with the true
map (`hospital_gt`, rasterised from the simulator geometry), and a 360-beam 2-D lidar ray-casts that map plus the
people and the other robots. RViz draws them as G1 models (`robot_state_publisher` per teammate, frame prefix
`k<i>/`).

## DDS

CycloneDDS is pinned to loopback (`config/cyclonedds.xml`; `MANTF_CYCLONE_LO=0` to disable). Use `--domain N` to run
two stacks at once.
