---
title: Status and measured results
nav_order: 8
---

# Status and measured results
{: .no_toc }

Measured on 2026-11-23. The machine was shared with other GPU work, so planning in the simulators was slower than
offline. Run folders are under `runs/`.

1. TOC
{:toc}

## Offline (no simulator, kinematic G1s, 5 people)

Cooperative case, `--max-time 300`:

| agents | reached | wall contacts | close calls | times (s) |
|---|---|---|---|---|
| 1 | 1 / 1 | 0 | 0 | 59 |
| 3 | 3 / 3 | 0 | 0 | 39, 88, 102 |

With 3 agents, cooperative vs greedy, path lengths (m): g1 29 vs 41, k1 24 vs 40, k2 28 vs 56 (earlier code
version, same model).

## Simulators (3 agents: g1 simulated, k1 and k2 kinematic)

| run | reached | wall contacts | close calls | s between an agent's calls |
|---|---|---|---|---|
| MuJoCo cooperative | g1, k1 (k2 out of time) | 0 | 1 (g1, person, 0.548 m) | 7.1 |
| MuJoCo greedy | 3 / 3 | 0 | 1 (g1, person) | 4.4 |
| Isaac cooperative | 3 / 3 | g1: 5 (2 confirmed by Isaac's own lidar) | 0 | 1.5 |
| Isaac greedy | 3 / 3 | g1: 4 | g1: 4 | 1.5 |
| MuJoCo, 1 agent | 1 / 1 | g1: 1 (0.25 s) | 0 | 0.8 |

Close call: closer than 0.60 m between robot centres, or 0.55 m to a person.

## Is the math doing what the paper asks?

From `math_log.csv` and `longrange_log.csv`, on frames the model has not trained on yet:

| check | expected | measured |
|---|---|---|
| eikonal residual | small, steady | 0.08-0.40 |
| learned S vs target S* (correlation) | high | 0.6-0.93 |
| T(a, b) = T(b, a) | exact | exact (symmetric network) |
| speed near an agent / alone | < 1 | 0.03-0.10 |
| T vs true shortest path (rank correlation) | close to 1 | 0.80-0.95 |

## Fixed after these measurements

- **The simulated G1 fell at the start when launched with `mg1`.** WardZero's G1 controller could not import
  `onnxruntime`, because pixi hides the user site-packages, where it is installed. With no motor commands the G1
  collapsed. WardZero's nodes now start without pixi's Python settings, the stack checks that the controller loaded
  its policy, and the fleet detects a fall.
- **More than 5 agents** is no longer supported (the 10-agent runs blocked each other in the corridor). Arrangements
  now go from 1 to 5 agents.
- **Agents that reach their goal** now stop and leave the round-robin (N -> N - 1 calls), instead of helping others.
- **Teammates in Isaac** are shown as G1 bodies instead of capsules.

## Known limitations

{: .warning }
> **The physically simulated G1 still touches walls occasionally.** WardZero's placeholder walking policy executes
> commands loosely. Measured in MuJoCo: forward is about half the commanded speed, backward commands produce no
> backward motion, sideways is weak, and it keeps moving briefly after a stop. The guard models this, but brief
> contacts (0.25-1 s) remain, mostly in doorways and at the PACU goal, which is 0.65 m from furniture. The kinematic
> teammates have had **0** contacts in every run. A stricter mode (slower near obstacles, 0.25 m margin) is the next
> step.

{: .warning }
> **Planning is slower in the simulators.** Training competes with the simulator, RViz and the sensor window for the
> CPU, so each agent is replanned every 1.5-7 s instead of every 0.6 s. Run cases with `-- --no-rviz --no-viewer
> --no-mjviewer` for faster planning.

{: .note }
> In Isaac, contacts are measured against `hospital_gt`, which is rasterised from the MuJoCo geometry. Some reported
> Isaac contacts do not appear on the map built from Isaac's own lidar.

{: .note }
> MuJoCo has no people. Their positions reach the model and the guard, but the MuJoCo lidar and camera do not see them.
