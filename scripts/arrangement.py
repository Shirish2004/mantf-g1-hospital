"""Reads config/arrangement.yaml: arrangement.py FILE robots|spawn|goals|offline"""
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HG1 = os.environ.get("HG1_ROOT") or next((p for p in (os.path.join(ROOT, "WardZero"), os.path.join(ROOT, "..", "WardZero"))
                                          if os.path.isfile(os.path.join(p, "env.sh"))), os.path.join(ROOT, "..", "WardZero"))


def places():
    out = {}
    sp = json.load(open(os.path.join(HG1, "g1-mujoco-sim/worlds/spawns.json")))
    for k, v in sp.get("hospital", sp).items():
        if isinstance(v, dict):
            out[k] = (v["x"], v["y"], v.get("yaw", 0.0))
    loc = yaml.safe_load(open(os.path.join(HG1, "ros2_ws/src/hospital-sim-ros2/g1_mission/config/locations.yaml")))
    for k, v in loc["locations"].items():
        out[k] = (v["x"], v["y"], v.get("yaw", 0.0))
    pp = yaml.safe_load(open(os.path.join(ROOT, "ros2_ws/src/mantf_g1_bridge/config/pick_place.yaml"))).get("places", {})
    for k, v in pp.items():
        out[k] = (v["x"], v["y"], v.get("yaw", 0.0))
    return out


def resolve(v, P):
    if isinstance(v, str):
        return tuple(float(a) for a in P[v])
    return tuple(float(a) for a in (list(v) + [0.0])[:3])


def main():
    path, what = sys.argv[1], sys.argv[2]
    P = places()
    robots = yaml.safe_load(open(path))["robots"]
    items = list(robots.items())
    fmt = lambda p: ",".join(f"{a:.4f}" for a in p)  # noqa: E731
    if what == "robots":
        print(", ".join(['"%s:sim"' % items[0][0]] + ['"%s:kin:%s"' % (k, fmt(resolve(v["start"], P))) for k, v in items[1:]]))
    elif what == "teammates":
        print(",".join(k for k, _ in items[1:]))
    elif what == "spawn":
        print(fmt(resolve(items[0][1]["start"], P)))
    elif what == "goals":
        for k, v in items:
            if v.get("goal") is not None:
                g = v["goal"]
                print(k, g if isinstance(g, str) else " ".join(str(a) for a in resolve(g, P)))
    elif what == "workspace":
        pts = []
        for k, v in items:
            pts.append(resolve(v["start"], P)[:2])
            if v.get("goal") is not None:
                pts.append(resolve(v["goal"], P)[:2])
        pts = [list(p) for p in pts]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        side = max(28.0, max(max(xs) - min(xs), max(ys) - min(ys)) + 10.0)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        if side == 28.0 and all(4.5 < x < 32.5 and -28.5 < y < -0.5 for x, y in zip(xs, ys)):
            cx, cy = 18.5, -14.5
        print(f"{cx:.2f},{cy:.2f},{side:.1f}")
    elif what == "offline":
        print(" ".join(f"{k}@{fmt(resolve(v['start'], P))}" + (f">{fmt(resolve(v['goal'], P))}" if v.get("goal") is not None else "")
                       + (f"={v['goal']}" if isinstance(v.get("goal"), str) else "") for k, v in items))


if __name__ == "__main__":
    main()
