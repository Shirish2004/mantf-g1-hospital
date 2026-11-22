import argparse
import math
import os
import time

import numpy as np
import yaml

from mantf_g1_bridge.astar import path_length, plan_path
from mantf_g1_bridge.local_planner import LocalPlanner
from mantf_g1_bridge.occupancy import GridMap
from mantf_g1_bridge.pure_pursuit import PurePursuit, safety_filter

LOC = os.path.join(os.environ.get("HG1_ROOT", ""), "ros2_ws/src/hospital-sim-ros2/g1_mission/config/locations.yaml")


def walker(t, a, b, speed):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    s = (t * speed) % (2 * L)
    s = s if s < L else 2 * L - s
    return a[0] + (b[0] - a[0]) * s / L, a[1] + (b[1] - a[1]) * s / L


def disc(static, centers, radius):
    data = np.zeros_like(static.data)
    if centers:
        rr, cc = np.mgrid[0:static.height, 0:static.width]
        for x, y in centers:
            r, c = static.world_to_grid(x, y)
            data |= (rr - r) ** 2 + (cc - c) ** 2 <= (radius / static.resolution) ** 2
    return GridMap(data, static.resolution, static.origin, static.yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="theatre_instrument_table")
    ap.add_argument("--goal", default="pre_op")
    ap.add_argument("--no-mantf", action="store_true")
    ap.add_argument("--no-walkers", action="store_true")
    ap.add_argument("--no-safety", action="store_true")
    ap.add_argument("--max-time", type=float, default=120.0)
    ap.add_argument("--out", default="demo_planner.png")
    a = ap.parse_args()

    locs = yaml.safe_load(open(LOC))["locations"]
    s, g = locs[a.start], locs[a.goal]
    static = GridMap.from_yaml(os.environ["MANTF_G1_MAP"])
    crop = static.subgrid((s["x"] + g["x"]) / 2, (s["y"] + g["y"]) / 2,
                          max(abs(s["x"] - g["x"]), abs(s["y"] - g["y"])) / 2 + 8.0)
    pose = [s["x"], s["y"], s["yaw"]]
    goal = (g["x"], g["y"], g["yaw"])
    walkers = [] if a.no_walkers else [((12.0, -12.45), (27.0, -12.45), 0.6), ((25.9, -14.0), (25.9, -11.0), 0.3), ((13.4, -10.6), (17.6, -18.0), 0.45)]

    model = None
    if not a.no_mantf:
        import torch
        from mantf_g1_bridge.mantf_core import make_model
        model = make_model("/tmp/mantf_g1_demo", device="cuda:0" if torch.cuda.is_available() else "cpu", rows=1200)
        t0 = time.time()
        model.pretrain(20)
        print(f"pretrain {time.time() - t0:.1f}s")
    lp = LocalPlanner(model)
    pp = PurePursuit(max_v=0.5, max_vy=0.3, max_w=0.8)
    clearance = 0.38
    gp = plan_path(crop, pose[:2], goal[:2], clearance=clearance, downsample=4)
    print(f"global route {path_length(gp):.1f} m, {len(gp)} pts")

    dt, t, last_plan, last_global = 0.05, 0.0, -1.0, 0.0
    trail, min_gap, hits, path = [], 1e9, 0, gp
    while t < a.max_time:
        people = [walker(t, *w) for w in walkers]
        dyn = disc(crop, people, 0.25)
        if t - last_global > 4.0 or lp.blocked:
            ng = plan_path(GridMap(crop.data | dyn.data, crop.resolution, crop.origin), pose[:2], goal[:2],
                           clearance=clearance, downsample=4)
            gp = ng or gp
            last_global, lp.blocked = t, False
        if t - last_plan >= 0.3:
            path, _ = lp.step(pose, gp, crop, dyn, now=t)
            last_plan = t
        dist = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        if dist < 0.3:
            break
        vx, vy, wz = pp.command(pose, path, dt=dt, distance_to_goal=dist)
        if not a.no_safety and people:
            px, py = min(people, key=lambda q: math.hypot(q[0] - pose[0], q[1] - pose[1]))
            d = math.hypot(px - pose[0], py - pose[1])
            edge = (pose[0] + (px - pose[0]) * (d - 0.25) / d, pose[1] + (py - pose[1]) * (d - 0.25) / d)
            vx, vy, wz = safety_filter((vx, vy, wz), pose, edge)
        c, sn = math.cos(pose[2]), math.sin(pose[2])
        pose = [pose[0] + (c * vx - sn * vy) * dt, pose[1] + (sn * vx + c * vy) * dt, pose[2] + wz * dt]
        trail.append(tuple(pose[:2]))
        if crop.occupied(pose[0], pose[1]) or crop.clearance_at(pose[0], pose[1]) < 0.15:
            hits += 1
        for px, py in people:
            min_gap = min(min_gap, math.hypot(px - pose[0], py - pose[1]))
        t += dt

    reached = math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < 0.3
    print(f"reached={reached} t={t:.1f}s travelled={path_length(trail):.1f}m wall_contacts={hits} "
          f"min_gap_to_person={min_gap:.2f}m planner={lp.stats}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x0, y0 = crop.origin
        ext = [x0, x0 + crop.width * crop.resolution, y0, y0 + crop.height * crop.resolution]
        plt.figure(figsize=(8, 8))
        plt.imshow(crop.data, origin="lower", extent=ext, cmap="gray_r")
        plt.plot(*zip(*gp), "b--", lw=1, label="A* route")
        plt.plot(*zip(*trail), "r-", lw=2, label="robot")
        for w in walkers:
            plt.plot([w[0][0], w[1][0]], [w[0][1], w[1][1]], "g:", lw=3, label="walker lane")
        plt.legend()
        plt.savefig(a.out, dpi=120)
        print(f"saved {a.out}")
    except Exception as exc:
        print(f"plot skipped: {exc}")
    return 0 if reached and hits == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
