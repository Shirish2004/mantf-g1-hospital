"""Arrangements for the evaluation.

  make_arrangement.py nested --upto 12 [--seed 7]
      config/arrangement_6..12.yaml: arrangement_5's agents plus generated teammates k5, k6, ..., so arrangement_n
      holds the first n agents of arrangement_12.
  make_arrangement.py spread --n 4 --spread 3.0 [--seed 0]
      config/spread/arrangement_<n>_r<spread>_s<seed>.yaml: the goals of arrangement_n, every teammate's start
      redrawn within `spread` m of the G1's start (spawning the agents nearer each other).
"""
import argparse
import os
import sys

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, csgraph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws/src/mantf_g1_bridge"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from arrangement import places, resolve  # noqa: E402
from mantf_g1_bridge.occupancy import GridMap  # noqa: E402

BOX = (6.5, -24.0, 31.0, -5.0)


class Geo:
    """G1-traversable cells (0.2 m, >= 0.35 m from walls) and shortest-path lengths between them."""

    def __init__(self, gt, res=0.2, clear=0.35):
        k = int(round(res / gt.resolution))
        h, w = gt.height // k, gt.width // k
        occ = gt.data[:h * k, :w * k].reshape(h, k, w, k).any((1, 3))
        self.edt = distance_transform_edt(~occ) * res
        ok = self.edt >= clear
        self.idx = -np.ones(occ.shape, int)
        self.idx[ok] = np.arange(ok.sum())
        rows, cols, wts = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            c0, c1 = (slice(0, w - dc), slice(dc, w)) if dc >= 0 else (slice(-dc, w), slice(0, w + dc))
            a, b = self.idx[0:h - dr, c0], self.idx[dr:h, c1]
            m = (a >= 0) & (b >= 0)
            rows += list(a[m]); cols += list(b[m]); wts += [res * np.hypot(dr, dc)] * int(m.sum())  # noqa: E702
        G = coo_matrix((wts, (rows, cols)), shape=(ok.sum(), ok.sum())).tocsr()
        self.G = G + G.T
        self.res, self.origin = res, gt.origin

    def cell(self, xy):
        r = int((xy[1] - self.origin[1]) / self.res)
        c = int((xy[0] - self.origin[0]) / self.res)
        return r, c

    def clearance(self, xy):
        r, c = self.cell(xy)
        return float(self.edt[r, c]) if 0 <= r < self.edt.shape[0] and 0 <= c < self.edt.shape[1] else 0.0

    def dist(self, a, b):
        i, j = self.idx[self.cell(a)], self.idx[self.cell(b)]
        if i < 0 or j < 0:
            return np.inf
        return float(csgraph.dijkstra(self.G, indices=i, limit=60.0)[j])


def sample(rng, geo, taken, centre=None, radius=None, clear=1.0, sep=2.5):
    for _ in range(20000):
        if centre is None:
            p = np.array([rng.uniform(BOX[0], BOX[2]), rng.uniform(BOX[1], BOX[3])])
        else:
            a, r = rng.uniform(0, 2 * np.pi), radius * np.sqrt(rng.uniform(0.05, 1.0))
            p = np.asarray(centre) + r * np.array([np.cos(a), np.sin(a)])
        if geo.clearance(p) < clear or any(np.linalg.norm(p - q) < sep for q in taken):
            continue
        return p
    raise RuntimeError("no free cell left with that clearance and separation")


def agents_of(path, P):
    return {k: {"start": list(resolve(v["start"], P)), "goal": list(resolve(v["goal"], P))}
            for k, v in yaml.safe_load(open(path))["robots"].items()}


def dump(path, agents, note):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {note}\nrobots:\n")
        for k, v in agents.items():
            s = ", ".join(f"{a:.2f}" for a in v["start"])
            g = ", ".join(f"{a:.2f}" for a in v["goal"])
            f.write(f"  {k}: {{start: [{s}], goal: [{g}]}}\n")
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["nested", "spread"])
    ap.add_argument("--upto", type=int, default=12)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--spread", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    gt = GridMap.from_yaml(os.environ.get("MANTF_G1_MAP") or os.path.join(
        os.environ["HG1_ROOT"], "ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/hospital_gt.yaml"))
    geo, P, rng = Geo(gt), places(), np.random.default_rng(a.seed)
    if a.what == "nested":
        agents = agents_of(os.path.join(ROOT, "config", "arrangement_5.yaml"), P)
        for i in range(len(agents), a.upto):
            pts = [np.asarray(v["start"][:2]) for v in agents.values()] + [np.asarray(v["goal"][:2]) for v in agents.values()]
            for _ in range(200):
                s = sample(rng, geo, pts)
                g = sample(rng, geo, pts + [s])
                if 8.0 <= geo.dist(s, g) <= 25.0:
                    break
            else:
                raise RuntimeError("no start/goal pair 8-25 m apart")
            yaw = float(np.arctan2(*(g - s)[::-1]))
            agents[f"k{i}"] = {"start": [float(s[0]), float(s[1]), yaw], "goal": [float(g[0]), float(g[1]), 0.0]}
            dump(os.path.join(ROOT, "config", f"arrangement_{i + 1}.yaml"), dict(list(agents.items())[:i + 1]),
                 f"{i + 1} agents: arrangement_{i}'s agents + k{i} (make_arrangement.py nested, seed {a.seed}).")
    else:
        src = os.path.join(ROOT, "config", f"arrangement_{a.n}.yaml")
        agents = agents_of(src, P)
        ids = list(agents)
        g1 = np.asarray(agents[ids[0]]["start"][:2])
        taken = [g1] + [np.asarray(v["goal"][:2]) for v in agents.values()]
        for k in ids[1:]:
            s = sample(rng, geo, taken, centre=g1, radius=a.spread, clear=0.8, sep=1.3)
            taken.append(s)
            g = np.asarray(agents[k]["goal"][:2])
            agents[k]["start"] = [float(s[0]), float(s[1]), float(np.arctan2(*(g - s)[::-1]))]
        d = [np.linalg.norm(np.asarray(agents[k]["start"][:2]) - g1) for k in ids[1:]]
        dump(os.path.join(ROOT, "config", "spread", f"arrangement_{a.n}_r{a.spread:g}_s{a.seed}.yaml"), agents,
             f"arrangement_{a.n} with every teammate spawned within {a.spread} m of the G1 (seed {a.seed}); "
             f"mean distance to the G1 {np.mean(d):.2f} m.")


if __name__ == "__main__":
    main()
