"""Multi-agent Active NTFields without a simulator: kinematic G1s with 2-D lidar on the hospital map, people on lanes,
one model, synchronous rounds. Writes the same run folder as the ROS fleet (maps, 3-D map, trajectories, logs, video)."""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws/src/mantf_g1_bridge"))
from mantf_g1_bridge.fleet_core import Fleet, summarize  # noqa: E402
from mantf_g1_bridge.goto import locations  # noqa: E402
from mantf_g1_bridge.occupancy import GridMap  # noqa: E402
from mantf_g1_bridge.walkers_node import People, make_lanes  # noqa: E402


def place(s, locs):
    if s in locs:
        v = locs[s]
        return (v["x"], v["y"], v.get("yaw", 0.0))
    return tuple(float(v) for v in s.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrangement", default=os.path.join(ROOT, "config", "arrangement.yaml"),
                    help="starts and goals of every agent")
    ap.add_argument("--robots", nargs="+", default=None, help="id@start[>goal] (overrides --arrangement); no goal = explore")
    ap.add_argument("--mode", choices=["cooperative", "greedy"], default="cooperative")
    ap.add_argument("--people", type=int, default=5)
    ap.add_argument("--workspace", default=None, help="cx,cy,side (default: fitted to the arrangement)")
    ap.add_argument("--fourier-scale", type=float, default=1.0)
    ap.add_argument("--max-time", type=float, default=300.0)
    ap.add_argument("--round", type=float, default=0.2, help="s of simulated time per model call (one agent per call)")
    ap.add_argument("--max-v", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default=None, help="default runs/offline/<mode>")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    locs = locations()
    gt = GridMap.from_yaml(os.environ["MANTF_G1_MAP"])
    if a.robots is None:
        import subprocess
        a.robots = subprocess.check_output([sys.executable, os.path.join(ROOT, "scripts", "arrangement.py"),
                                            a.arrangement, "offline"], text=True).split()
    specs, goals, names = [], {}, {}
    for s in a.robots:
        rid, rest = s.split("@")
        rest, _, name = rest.partition("=")
        start, _, goal = rest.partition(">")
        specs.append((rid, "kin", place(start, locs)))
        if goal:
            goals[rid] = place(goal, locs)
            names[rid] = name or goal
    folder = os.path.join(a.out or os.path.join(ROOT, "runs", "offline", a.mode), time.strftime("%m%d_%H%M%S"))
    if a.workspace is None:
        import subprocess
        a.workspace = subprocess.check_output([sys.executable, os.path.join(ROOT, "scripts", "arrangement.py"),
                                               a.arrangement, "workspace"], text=True).strip()
    ws = [float(v) for v in a.workspace.split(",")]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    fleet = Fleet(folder, ws, specs, gt, dev, max_v=a.max_v, epochs=a.epochs, mode=a.mode, explore=False,
                  fourier_scale=a.fourier_scale)
    for r in fleet.robots.values():
        r.explore = r.id not in goals
    json.dump({"mode": a.mode, "robots": a.robots}, open(os.path.join(folder, "arrangement.json"), "w"), indent=2)
    people = People(make_lanes(gt, [ws[0] - ws[2] / 2 + 2, ws[1] - ws[2] / 2 + 2, ws[0] + ws[2] / 2 - 2,
                                    ws[1] + ws[2] / 2 - 2], a.people, seed=3), seed=3)
    dt, t, next_round, next_scan, next_view = 0.05, 0.0, 0.0, 0.0, 0.0
    video = None
    for rid, g in goals.items():
        fleet.set_goal(rid, g, names[rid])
    print(f"run {folder}: {[s[0] for s in specs]}, goals {goals}, {len(people.lanes)} people", flush=True)
    t0 = time.time()
    while t < a.max_time:
        pos, _ = people.step(dt, [r.pose[:2] for r in fleet.robots.values()])
        fleet.set_people(pos)
        if t >= next_scan:
            fleet.scan_kin()
            next_scan += 0.1
        if t >= next_round:
            rec = fleet.round()
            next_round += a.round
            img = fleet.view(f"t {t:5.1f} s  round {fleet.field.rounds}  loss {rec['loss'] if rec else 0:.3g}")
            if not a.no_video:
                if video is None:
                    video = cv2.VideoWriter(os.path.join(folder, "fleet_bev_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            10, (img.shape[1], img.shape[0]))
                video.write(img)
            if fleet.field.rounds % 20 == 0:
                st = fleet.status()
                print(f"t {t:6.1f}  round {st['round']}  loss {rec and rec['loss']:.4g}  "
                      + "  ".join(f"{k}:{v['state'][:9]} {v['pose'][0]:.1f},{v['pose'][1]:.1f} clr {v['clearance']:.2f}"
                                  for k, v in st["robots"].items()), flush=True)
        fleet.step_kin(dt)
        if t >= next_view:
            fleet.record_views()
            next_view += 0.2
        fleet.t = t = t + dt
        if goals and all(fleet.robots[k].goal is None for k in goals):
            if all(r.state != "helping" for r in fleet.robots.values()) or t > a.max_time:
                break
    for rid in goals:
        if fleet.robots[rid].goal is not None:
            fleet.finish(rid, False, "timeout")
    fleet.save("final")
    fleet.close_views()
    if video is not None:
        video.release()
        os.system(f"ffmpeg -loglevel error -y -i {folder}/fleet_bev_raw.mp4 -c:v libx264 -pix_fmt yuv420p "
                  f"{folder}/fleet_bev.mp4 && rm {folder}/fleet_bev_raw.mp4")
    summarize(folder)
    contacts = {k: len(r.contacts) for k, r in fleet.robots.items()}
    json.dump({"wall_contacts": contacts, "wall_s": round(time.time() - t0, 1), "sim_s": round(t, 1)},
              open(os.path.join(folder, "offline.json"), "w"), indent=2)
    print(open(os.path.join(folder, "results.md")).read())
    print("wall contacts per agent (G1 footprint touching the true map):", contacts, f"; {time.time() - t0:.0f} s wall for {t:.0f} s sim")
    print("run folder:", folder)


if __name__ == "__main__":
    main()
