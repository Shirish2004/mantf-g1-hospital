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
| `config/arrangement_1.yaml` | 1 | the paper's single-robot setting |
| `config/arrangement.yaml` | 3 | g1 and k1 cross in the surgical corridor; k2 goes from the scrub room to the theatre |
| `config/arrangement_10.yaml` | 10 | the three above + seven random unique pairs in the surgical wing |

Named places: `mg1 goto list` (WardZero's `g1_mission/config/locations.yaml`, its spawns, and the pick/place places
of this project).

## Rules

- Every start and goal must be in free space with room for the G1: at least about 1 m from walls.
- Keep starts and goals at least about 2.5 m from each other.
- The field (the model's workspace) is fitted to all starts and goals: a square of at least 28 m, larger if needed
  (`scripts/arrangement.py FILE workspace`).
- Any number of agents works. Each is served every N x 0.3 s, and the model's memory holds at least 3 frames per agent.

Helper: `python3 scripts/arrangement.py FILE robots|teammates|spawn|goals|workspace|offline` prints what the
launch scripts read from an arrangement.
