---
title: Arrangements (starts and goals)
nav_order: 4
---

# Arrangements: every agent's start and goal

An arrangement file lists the agents. The **first** agent is the physically simulated G1 (MuJoCo or Isaac); the
others are kinematic G1s that share the same map, model and people. Each agent has its own start and its own goal.
Use a named place or `[x, y, yaw]` in the map frame (= simulator world frame, metres and radians).

```yaml
# config/arrangement.yaml (3 agents)
robots:
  g1: {start: corridor_west, goal: pacu}            # the simulated G1
  k1: {start: corridor_theatre, goal: pre_op}       # kinematic G1
  k2: {start: scrub_room, goal: theatre_door}
  # k3: {start: [20.0, -12.4, 0.0], goal: sterile_store}
```

Ready-made:

| file | agents | |
|---|---|---|
| `config/arrangement_1.yaml` | 1 | the paper's single-robot setting: g1 alone, corridor west -> PACU |
| `config/arrangement_2.yaml` | 2 | + k1: g1 and k1 cross in the surgical corridor |
| `config/arrangement_3.yaml` | 3 | + k2: scrub room -> theatre door (the default, also `config/arrangement.yaml`) |
| `config/arrangement_4.yaml` | 4 | + k3: (14.1, -18.5) -> (30.0, -12.9) |
| `config/arrangement_5.yaml` | 5 | + k4: (15.5, -12.7) -> (29.7, -19.6), the maximum |

Named places: `mg1 goto list` (WardZero's `g1_mission/config/locations.yaml`, its spawns, and the pick/place places
of this project).

## Rules

- Every start and goal must be in free space with room for the G1: at least about 1 m from walls.
- Keep starts and goals at least about 2.5 m from each other.
- The field (the model's workspace) is fitted to all starts and goals: a square of at least 28 m, larger if needed
  (`scripts/arrangement.py FILE workspace`).
- At most **5 agents**: one simulated G1 (first in the file) + up to 4 kinematic G1 teammates. More is refused.
- Each active agent is called every N x 0.3 s. An agent that reaches its goal stops, makes no more calls and stays
  as an obstacle in the others' input, so the cycle shrinks to N - 1 calls.

Helper: `python3 scripts/arrangement.py FILE robots|teammates|spawn|goals|workspace|offline` prints what the
launch scripts read from an arrangement.
