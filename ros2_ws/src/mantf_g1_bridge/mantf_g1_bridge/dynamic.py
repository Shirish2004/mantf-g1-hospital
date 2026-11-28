"""Moving obstacles for the one field (mantf aniANTF, models/se2_dynamic.py).

Every agent has its own SE(2) tracker. At each of its calls the people are ray-cast into that agent's wall scan at
three past instants t - 2g, t - g, t from the agent's current origin (the tracker's three-scans-from-one-pose protocol),
so the tracker sees only what moved. Its bodies and twists give the anisotropic metric A = I - lam u u^T and the
swept-clearance speed labels of the model.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


def cast_discs(origin, ang, base, discs, radius):
    """min(base, first ray-disc hit) per bearing. Ray o + t u meets |p - c| = r at t = t_ca - sqrt(r^2 - d^2),
    t_ca = u.(c - o), d^2 = |c - o|^2 - t_ca^2."""
    out = np.asarray(base, float).copy()
    discs = np.asarray(discs, float).reshape(-1, 2)
    if not len(discs):
        return out
    u = np.column_stack((np.cos(ang), np.sin(ang)))
    f = discs - np.asarray(origin, float)[None, :2]
    tca = u @ f.T
    d2 = (f ** 2).sum(1)[None] - tca ** 2
    r2 = float(radius) ** 2
    hit = (d2 < r2) & (tca > 0)
    t = np.where(hit, tca - np.sqrt(np.clip(r2 - d2, 0, None)), np.inf)
    return np.minimum(out, t.min(1))


class History:
    """People positions over the last `keep` s of sim time, interpolated per person."""

    def __init__(self, keep=4.0):
        self.keep = keep
        self.buf = deque()

    def add(self, t, pos):
        pos = np.asarray(pos, float).reshape(-1, 2)
        if self.buf and len(self.buf[-1][1]) != len(pos):
            self.buf.clear()
        if self.buf and t <= self.buf[-1][0]:
            self.buf[-1] = (self.buf[-1][0], pos)
            return
        self.buf.append((float(t), pos))
        while len(self.buf) > 2 and t - self.buf[1][0] > self.keep:
            self.buf.popleft()

    def at(self, t):
        if not self.buf:
            return np.zeros((0, 2))
        ts = np.array([b[0] for b in self.buf])
        if t <= ts[0]:
            return self.buf[0][1]
        if t >= ts[-1]:
            return self.buf[-1][1]
        k = int(np.searchsorted(ts, t))
        (t0, p0), (t1, p1) = self.buf[k - 1], self.buf[k]
        w = (t - t0) / max(t1 - t0, 1e-9)
        return p0 * (1 - w) + p1 * w

    def velocity(self, t, dt=0.4):
        return (self.at(t) - self.at(t - dt)) / dt

    def span(self):
        return self.buf[-1][0] - self.buf[0][0] if len(self.buf) > 1 else 0.0


class Movers:
    """Per-agent SE(2) trackers in the field's local metric frame (world - center), so that the tracker's
    model-to-metres map s * x matches the field's to_model."""

    def __init__(self, center, scale, gap=0.3, person_radius=0.25, v_max=2.0, horizon=1.0):
        from models.se2_dynamic import SE2Tracker
        self._cls = SE2Tracker
        self.center = np.asarray(center[:2], float)
        self.scale = float(scale)
        self.gap, self.person_radius, self.v_max, self.horizon = float(gap), float(person_radius), float(v_max), float(horizon)
        self.trackers = {}
        self.errors = []

    def tracker(self, rid, beams):
        if rid not in self.trackers:
            self.trackers[rid] = self._cls(self.scale, v_max=self.v_max, n_rays=int(beams))
        return self.trackers[rid]

    def observe(self, rid, origin, ang, wall, history, t):
        """Three virtual scans at t - 2g, t - g, t from `origin` into rid's tracker. Returns the fresh bodies
        as [(centre world, radius, twist)]."""
        tr = self.tracker(rid, len(ang))
        if history.span() < 2 * self.gap:
            return []
        o = np.asarray(origin, float)[:2]
        u = np.column_stack((np.cos(ang), np.sin(ang)))
        for k, dt in enumerate((0.0, self.gap, 2 * self.gap)):
            rng = cast_discs(o, ang, wall, history.at(t - 2 * self.gap + dt), self.person_radius)
            ok = np.isfinite(rng)
            pts = np.zeros((int(ok.sum()), 3))
            pts[:, :2] = o + u[ok] * rng[ok, None] - self.center
            tr.update(pts, np.r_[o - self.center, 0.0], dt)
        return [(t.center[:2] + self.center, float(t.body), np.asarray(t.xi, float), bool(t.have_xi)) for t in tr.tracks if t.hits]

    def score(self, rid, bodies, history, t, round_):
        """Each fresh body against its nearest true person: centroid error (m); twist and heading error only once the
        track has a twist (two detections under one baseline)."""
        gp, gv = history.at(t), history.velocity(t)
        out = []
        if not len(gp):
            return out
        nan = float("nan")
        for c, r, xi, has_v in bodies:
            d = np.linalg.norm(gp - c[None], axis=1)
            k = int(np.argmin(d))
            v = xi[:2]
            sg, se = float(np.linalg.norm(gv[k])), float(np.linalg.norm(v))
            head = (abs(math.degrees(math.atan2(gv[k][0] * v[1] - gv[k][1] * v[0], float(gv[k] @ v))))
                    if has_v and sg > 0.05 and se > 0.05 else nan)
            out.append({"t": round(t, 2), "round": round_, "agent": rid, "person": k, "pos_err_m": round(float(d[k]), 4),
                        "vel_err_mps": round(float(np.linalg.norm(v - gv[k])), 4) if has_v else nan,
                        "speed_gt": round(sg, 3), "speed_est": round(se, 3) if has_v else nan,
                        "heading_err_deg": round(head, 2), "radius_m": round(r, 3)})
        self.errors += out
        return out

    def lam_at(self, xy, model):
        """lam of A = I - lam u u^T at world points, max over every agent's tracker (model._aniso_np)."""
        q = np.zeros((len(xy), 3))
        q[:, :2] = (np.asarray(xy, float) - self.center) / self.scale
        return model._aniso_np(q)

    def swept(self, xy):
        """Distance (m) to the swept bodies over the next `horizon` s, min over every tracker."""
        q = (np.asarray(xy, float) - self.center) / self.scale
        out = np.full(len(q), np.inf)
        for tr in self.trackers.values():
            if tr.has_track():
                out = np.minimum(out, tr.swept_clearance_at(q, horizon=self.horizon) * self.scale)
        return out

    def bodies(self):
        return {k: [(np.asarray(c)[:2] + self.center, float(b), np.asarray(xi, float)) for c, b, xi in tr.bodies_at(0.0, False)]
                for k, tr in self.trackers.items()}


def render_costmap(grid, movers, model, robots, people, gt_mask=None, res=0.2, size=720, title=""):
    """The moving-obstacle cost map: lam of the anisotropic metric (colour) over the shared map, the swept tubes
    (contour at 0), each tracker's bodies with their twist (green), the true people (magenta) and the agents."""
    n = int(grid.n * grid.res / res)
    xs = grid.origin[0] + (np.arange(n) + 0.5) * res
    ys = grid.origin[1] + (np.arange(n) + 0.5) * res
    P = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    lam, _ = movers.lam_at(P, model)
    lam = lam.reshape(n, n)
    base = np.full((grid.n, grid.n, 3), 120, np.uint8)
    base[grid.free] = (235, 235, 235)
    base[grid.occ] = (25, 25, 25)
    if gt_mask is not None:
        base[gt_mask & ~grid.occ & ~grid.free] = (150, 150, 150)
    img = cv2.resize(np.flipud(base), (size, size), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap((np.clip(np.flipud(lam) / 0.8, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(np.flipud(lam), (size, size)) > 0.02
    img[m] = (0.45 * img[m] + 0.55 * heat[m]).astype(np.uint8)
    sc = size / (grid.n * grid.res)
    px = lambda q: (int((q[0] - grid.origin[0]) * sc), int(size - (q[1] - grid.origin[1]) * sc))  # noqa: E731
    for p in np.asarray(people).reshape(-1, 2):
        cv2.circle(img, px(p), max(2, int(0.25 * sc)), (200, 0, 200), 1)
        cv2.drawMarker(img, px(p), (200, 0, 200), cv2.MARKER_CROSS, 6, 1)
    for bl in movers.bodies().values():
        for c, b, xi in bl:
            cv2.circle(img, px(c), max(2, int(b * sc)), (40, 220, 40), 2)
            cv2.arrowedLine(img, px(c), px(c + xi[:2] * 1.5), (40, 220, 40), 2, tipLength=0.3)
    from .maps import color
    for i, (k, xy) in enumerate(robots.items()):
        cv2.circle(img, px(xy), max(3, int(0.32 * sc)), color(i), -1)
        cv2.putText(img, k, (px(xy)[0] + 5, px(xy)[1] - 5), 0, 0.4, color(i), 1)
    cv2.rectangle(img, (0, 0), (size, 22), (30, 30, 30), -1)
    cv2.putText(img, f"{title}  lam max {lam.max():.2f}  tracks {sum(len(v) for v in movers.bodies().values())}",
                (6, 15), 0, 0.45, (255, 255, 255), 1)
    return img
