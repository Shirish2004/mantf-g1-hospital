"""The fleet's shared maps: a 2-D log-odds grid (who saw what) and a 3-D voxel map, plus their pictures."""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

COLORS = [(40, 40, 230), (230, 120, 30), (30, 160, 250), (200, 60, 200), (60, 200, 60), (200, 200, 40)]


def color(i):
    return COLORS[i % len(COLORS)]


def densify(path, step=0.1):
    p = np.asarray(path, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.vstack(out)


class SharedGrid:
    def __init__(self, center, size, res, robots):
        self.res = float(res)
        self.n = int(math.ceil(size / res))
        self.origin = np.array([center[0] - size / 2.0, center[1] - size / 2.0])
        self.robots = list(robots)
        self.logodds = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros((len(self.robots), self.n, self.n), bool)
        self.version = 0
        self._edt = (None, -1)

    def idx(self, xy):
        xy = np.asarray(xy, float)
        c = np.floor((xy[..., 0] - self.origin[0]) / self.res).astype(int)
        r = np.floor((xy[..., 1] - self.origin[1]) / self.res).astype(int)
        return r, c

    def world(self, r, c):
        return np.column_stack((self.origin[0] + (c + 0.5) * self.res, self.origin[1] + (r + 0.5) * self.res))

    def _ok(self, r, c):
        return (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)

    def update(self, rid, origin, ends, hit, ray_max=8.0, extra_hits=None):
        o = np.asarray(origin, float)
        ends = np.asarray(ends, float)
        hit = np.asarray(hit, bool)
        step = max(1, len(ends) // 360)
        ends, hit = ends[::step], hit[::step]
        vec = ends - o
        L = np.minimum(np.linalg.norm(vec, axis=1), ray_max)
        u = vec / np.maximum(np.linalg.norm(vec, axis=1), 1e-9)[:, None]
        s = np.arange(0.0, ray_max, self.res * 0.7)
        pts = o + u[:, None] * s[None, :, None]
        mask = s[None] < (L[:, None] - self.res)
        r, c = self.idx(pts[mask])
        ok = self._ok(r, c)
        r, c = r[ok], c[ok]
        lin = np.unique(r * self.n + c)
        self.logodds.ravel()[lin] -= 0.7
        if rid in self.robots:
            self.seen[self.robots.index(rid)].ravel()[lin] = True
        hits = ends[hit & (np.linalg.norm(vec, axis=1) <= ray_max)]
        for pts_h, w in ((hits, 2.0), (extra_hits, 0.9)):
            if pts_h is None or len(pts_h) == 0:
                continue
            r, c = self.idx(np.asarray(pts_h)[:, :2])
            ok = self._ok(r, c)
            lin = np.unique(r[ok] * self.n + c[ok])
            self.logodds.ravel()[lin] += w
            if rid in self.robots:
                self.seen[self.robots.index(rid)].ravel()[lin] = True
        np.clip(self.logodds, -4.0, 6.0, out=self.logodds)
        self.version += 1

    @property
    def occ(self):
        return self.logodds > 0.6

    @property
    def free(self):
        return self.logodds < -0.6

    def is_free(self, xy):
        r, c = self.idx(np.asarray(xy[:2]))
        return bool(self._ok(r, c) and self.free[r, c])

    def frontier_cells(self, pos, dmin, dmax):
        f, u = self.free, ~(self.free | self.occ)
        nb = np.zeros_like(u)
        nb[1:] |= u[:-1]
        nb[:-1] |= u[1:]
        nb[:, 1:] |= u[:, :-1]
        nb[:, :-1] |= u[:, 1:]
        ff = f & nb & (self.clearance() > 0.55) & self.reachable(pos)
        rr, cc = np.nonzero(ff)
        w = self.world(rr, cc)
        d = np.linalg.norm(w - np.asarray(pos[:2]), axis=1)
        return w[(d > dmin) & (d < dmax)]

    def reachable(self, pos, clear=0.36):
        """Mapped free cells connected to pos through mapped free space wide enough for the G1 (8-connected)."""
        from scipy.ndimage import label
        key = (self.version, self.idx(np.asarray(pos[:2])))
        if getattr(self, "_reach_key", None) == key:
            return self._reach
        ok = self.free & (self.clearance() > clear)
        lab, _ = label(ok, structure=np.ones((3, 3), int))
        r, c = self.idx(np.asarray(pos[:2]))
        k = int(np.ceil(0.8 / self.res))
        win = lab[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1]
        ids = win[win > 0]
        out = np.isin(lab, np.unique(ids)) if len(ids) else np.zeros_like(ok)
        self._reach_key, self._reach = key, out
        return out

    def reaches(self, pos, goal):
        """The goal cell (or one within 0.4 m) is mapped free and connected to pos for the G1."""
        r, c = self.idx(np.asarray(goal[:2]))
        k = int(np.ceil(0.4 / self.res))
        win = self.reachable(pos)[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1]
        return bool(win.any())

    def free_cells(self, pos, dmin, dmax, n=150, rng=None, clear=0.55):
        """Mapped free cells with room for the G1 footprint, between dmin and dmax of pos (a random subset)."""
        r0, c0 = self.idx(np.asarray(pos[:2]))
        k = int(np.ceil(dmax / self.res))
        rs, cs = slice(max(0, r0 - k), min(self.n, r0 + k + 1)), slice(max(0, c0 - k), min(self.n, c0 + k + 1))
        m = self.free[rs, cs] & (self.clearance()[rs, cs] > clear) & self.reachable(pos)[rs, cs]
        rr, cc = np.nonzero(m)
        w = self.world(rr + rs.start, cc + cs.start)
        d = np.linalg.norm(w - np.asarray(pos[:2]), axis=1)
        w = w[(d > dmin) & (d < dmax)]
        if len(w) > n:
            w = w[(rng or np.random.default_rng()).choice(len(w), n, replace=False)]
        return w

    def is_frontier(self, xy):
        r, c = self.idx(np.asarray(xy[:2]))
        if not self._ok(r, c):
            return False
        u = ~(self.free | self.occ)
        return bool(u[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any())

    def clearance(self):
        if self._edt[1] != self.version:
            self._edt = (distance_transform_edt(~self.occ) * self.res, self.version)
        return self._edt[0]

    def clearance_at(self, pts):
        r, c = self.idx(np.asarray(pts, float).reshape(-1, 2))
        out = np.full(len(r), 10.0)
        ok = self._ok(r, c)
        out[ok] = self.clearance()[r[ok], c[ok]]
        return out

    def path_clear(self, path, start, radius, skip=0.4):
        p = np.asarray(path, float)
        far = np.linalg.norm(p - np.asarray(start[:2]), axis=1) > skip
        return bool(np.all(self.clearance_at(p[far]) >= radius)) if far.any() else True

    def safe_prefix(self, path, start, radius, skip=0.4, min_len=0.35):
        """The part of a path before it first comes closer than radius to a mapped obstacle (None if too short)."""
        p = densify(path, self.res)
        bad = (self.clearance_at(p) < radius) & (np.linalg.norm(p - np.asarray(start[:2]), axis=1) > skip)
        if bad.any():
            p = p[:int(np.argmax(bad))]
        if len(p) < 2 or np.linalg.norm(np.diff(p, axis=0), axis=1).sum() < min_len:
            return None, True
        return p, bool(bad.any())

    def known_fraction(self, path):
        r, c = self.idx(np.asarray(path, float).reshape(-1, 2))
        ok = self._ok(r, c)
        return float(self.free[r[ok], c[ok]].mean()) if ok.any() else 0.0

    def coverage(self):
        known = self.free | self.occ
        return {"known_m2": float(known.sum() * self.res ** 2),
                "by_robot_m2": {r: float(self.seen[i].sum() * self.res ** 2) for i, r in enumerate(self.robots)}}

    def ros_data(self):
        d = np.full(self.logodds.shape, -1, np.int8)
        d[self.free], d[self.occ] = 0, 100
        return d


class VoxelMap:
    """3-D occupancy as hit counts per voxel; source bit 1 = 3-D lidar, 2 = 2-D lidar extruded."""

    def __init__(self, res=0.1, zmin=-0.2, zmax=3.0):
        self.res, self.zmin, self.zmax = float(res), zmin, zmax
        self.keys = np.zeros(0, np.int64)
        self.hits = np.zeros(0, np.int32)
        self.src = np.zeros(0, np.uint8)
        self.lock = __import__("threading").Lock()

    def _key(self, xyz):
        q = np.floor(np.asarray(xyz, float) / self.res).astype(np.int64) + (1 << 20)
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    def insert(self, xyz, source=1):
        with self.lock:
            self._insert(xyz, source)

    def _insert(self, xyz, source=1):
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        xyz = xyz[(xyz[:, 2] > self.zmin) & (xyz[:, 2] < self.zmax) & np.isfinite(xyz).all(1)]
        if not len(xyz):
            return
        k, cnt = np.unique(self._key(xyz), return_counts=True)
        i = np.searchsorted(self.keys, k)
        old = (i < len(self.keys)) & (self.keys[np.minimum(i, len(self.keys) - 1)] == k) if len(self.keys) else np.zeros(len(k), bool)
        self.hits[i[old]] += cnt[old].astype(np.int32)
        self.src[i[old]] |= source
        if (~old).any():
            keys = np.concatenate((self.keys, k[~old]))
            order = np.argsort(keys, kind="stable")
            self.keys = keys[order]
            self.hits = np.concatenate((self.hits, cnt[~old].astype(np.int32)))[order]
            self.src = np.concatenate((self.src, np.full((~old).sum(), source, np.uint8)))[order]

    def extrude(self, xy, height=2.2):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if not len(xy):
            return
        z = np.arange(0.05, height, self.res * 2)
        self.insert(np.column_stack((np.repeat(xy, len(z), 0), np.tile(z, len(xy)))), source=2)

    def points(self, min_hits=2):
        with self.lock:
            keep = self.hits >= min_hits
            k, src = self.keys[keep], self.src[keep]
        m = (1 << 21) - 1
        q = np.column_stack(((k >> 42) & m, (k >> 21) & m, k & m)) - (1 << 20)
        return (q + 0.5) * self.res, src

    def __len__(self):
        return len(self.keys)

    def save(self, folder, trails=None, people=None):
        os.makedirs(folder, exist_ok=True)
        xyz, src = self.points()
        np.savez_compressed(os.path.join(folder, "map3d.npz"), xyz=xyz.astype(np.float32), source=src, res=self.res)
        z = xyz[:, 2]
        t = np.clip((z - 0.0) / 2.5, 0, 1)
        rgb = (cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1])
        with open(os.path.join(folder, "map3d.ply"), "wb") as f:
            f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
            rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(rec.tobytes())
        self.render(os.path.join(folder, "map3d.png"), xyz, trails, people)
        return len(xyz)

    @staticmethod
    def render(path, xyz, trails=None, people=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 8))
        ax = fig.add_subplot(111, projection="3d")
        if len(xyz):
            s = xyz if len(xyz) < 120000 else xyz[np.random.default_rng(0).choice(len(xyz), 120000, replace=False)]
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s[:, 2], cmap="turbo", s=0.4, vmin=0, vmax=2.5, linewidths=0)
        for i, (name, tr) in enumerate((trails or {}).items()):
            tr = np.asarray(tr)
            if len(tr):
                b, g, r = color(i)
                ax.plot(tr[:, 0], tr[:, 1], np.full(len(tr), 0.05), color=(r / 255, g / 255, b / 255), lw=2.5, label=name)
        for tr in (people or []):
            tr = np.asarray(tr)
            if len(tr):
                ax.plot(tr[:, 0], tr[:, 1], np.full(len(tr), 0.05), color="green", lw=0.8, alpha=0.6)
        if len(xyz):
            lo, hi = np.percentile(xyz[:, :2], 0.5, 0), np.percentile(xyz[:, :2], 99.5, 0)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(0, 3.0)
            ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], 3.0 * 1.5))
        ax.view_init(elev=38, azim=-65)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if trails:
            ax.legend(loc="upper left")
        ax.set_title("shared 3-D map (voxels 0.1 m, colour = height) and robot trajectories")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


def render_fleet(grid, agents, people=(), gt=None, px=900, title="", requests=()):
    """Shared explored map; each robot's cells tinted with its colour, its trail, path, viewpoint and goal."""
    n = grid.n
    img = np.full((n, n, 3), 110, np.uint8)
    if gt is not None:
        img[gt] = (150, 150, 150)
    free, occ = grid.free, grid.occ
    img[free] = (245, 245, 245)
    for i in range(len(grid.robots)):
        tint = np.array(color(i), float)
        m = grid.seen[i] & free
        img[m] = (0.85 * img[m] + 0.15 * tint).astype(np.uint8)
    img[occ] = (20, 20, 20)
    img = cv2.resize(np.flipud(img), (px, px), interpolation=cv2.INTER_NEAREST)
    s = px / (n * grid.res)

    def P(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.column_stack(((xy[:, 0] - grid.origin[0]) * s, px - (xy[:, 1] - grid.origin[1]) * s)).astype(np.int32)

    for p in people:
        cv2.circle(img, tuple(P(p)[0]), max(3, int(0.25 * s)), (0, 170, 0), -1)
    for i, (rid, a) in enumerate(agents.items()):
        col = color(grid.robots.index(rid) if rid in grid.robots else i)
        if a.get("trail") is not None and len(a["trail"]) > 1:
            cv2.polylines(img, [P(np.asarray(a["trail"])[:, :2])], False, col, 2)
        if a.get("path") is not None and len(a["path"]) > 1:
            cv2.polylines(img, [P(a["path"])], False, (255, 0, 255), 1)
        if a.get("viewpoint") is not None:
            cv2.drawMarker(img, tuple(P(a["viewpoint"])[0]), col, cv2.MARKER_TILTED_CROSS, 14, 2)
        if a.get("goal") is not None:
            cv2.drawMarker(img, tuple(P(a["goal"][:2])[0]), col, cv2.MARKER_STAR, 18, 2)
        if a.get("pose") is not None:
            x, y, th = a["pose"]
            c = tuple(P((x, y))[0])
            cv2.circle(img, c, max(4, int(0.32 * s)), col, -1)
            cv2.line(img, c, (int(c[0] + 14 * math.cos(th)), int(c[1] - 14 * math.sin(th))), (255, 255, 255), 2)
            cv2.putText(img, rid, (c[0] + 8, c[1] - 8), 0, 0.5, col, 2)
            if a.get("g") is not None and any(a["g"]):
                gx, gy = a["g"]
                cv2.arrowedLine(img, c, (int(c[0] + 1.5 * s * gx), int(c[1] - 1.5 * s * gy)), (0, 0, 0), 2, tipLength=0.35)
    for q in requests:
        cv2.circle(img, tuple(P(q)[0]), int(2.0 * s), (0, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "request", tuple(P(q)[0] + [6, 18]), 0, 0.4, (0, 140, 255), 1)
    if title:
        cv2.rectangle(img, (0, 0), (px, 24), (40, 40, 40), -1)
        cv2.putText(img, title, (6, 17), 0, 0.5, (255, 255, 255), 1)
    labels = [(i, a["label"]) for i, a in enumerate(agents.values()) if a.get("label")]
    if labels:
        y0 = px - 8 - 18 * len(labels)
        cv2.rectangle(img, (0, y0 - 6), (px, px), (40, 40, 40), -1)
        for j, (i, text) in enumerate(labels):
            cv2.putText(img, text, (8, y0 + 12 + 18 * j), 0, 0.5, color(i), 1)
    return img


def field_panel(S, T, px=450):
    sp = cv2.applyColorMap((np.clip(np.flipud(S), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    tn = np.flipud(T / max(np.nanpercentile(T, 98), 1e-6))
    tm = cv2.applyColorMap((np.clip(tn, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    sp, tm = (cv2.resize(i, (px, px), interpolation=cv2.INTER_NEAREST) for i in (sp, tm))
    lv = cv2.resize((tn * 14).astype(np.float32), (px, px), interpolation=cv2.INTER_LINEAR)
    sp[np.abs(lv - np.round(lv)) < 0.05] = (255, 255, 255)
    cv2.putText(sp, "learned speed S(q) + T contours", (6, 18), 0, 0.5, (0, 255, 255), 1)
    cv2.putText(tm, "arrival time T(q_r, q | o)", (6, 18), 0, 0.5, (0, 255, 255), 1)
    return np.hstack((sp, tm))
