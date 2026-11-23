"""Holonomic pure pursuit for the G1 base."""
from __future__ import annotations

import math

import numpy as np


def path_length(points):
    if not points or len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(points, float), axis=0), axis=1).sum())


def closest_index(points, x, y):
    a = np.asarray(points, dtype=float)
    return int(np.argmin((a[:, 0] - x) ** 2 + (a[:, 1] - y) ** 2)) if len(a) else 0


def lookahead_point(points, x, y, distance):
    if not points:
        return (x, y)
    i = closest_index(points, x, y)
    remaining = float(distance)
    for j in range(i, len(points) - 1):
        ax, ay = points[j]
        bx, by = points[j + 1]
        seg = math.hypot(bx - ax, by - ay)
        if seg <= 1e-9:
            continue
        if remaining <= seg:
            t = remaining / seg
            return (ax + (bx - ax) * t, ay + (by - ay) * t)
        remaining -= seg
    return tuple(points[-1])


class PurePursuit:
    def __init__(self, lookahead=0.8, max_v=0.5, max_vy=0.30, max_w=0.8,
                 k_v=1.1, k_vy=1.0, k_w=1.8, min_vx=0.0,
                 max_accel=1.2, max_alpha=3.0):
        self.lookahead = float(lookahead)
        self.max_v = float(max_v)
        self.max_vy = float(max_vy)
        self.max_w = float(max_w)
        self.k_v = float(k_v)
        self.k_vy = float(k_vy)
        self.k_w = float(k_w)
        self.min_vx = float(min_vx)
        self.max_accel = float(max_accel)
        self.max_alpha = float(max_alpha)
        self._last = (0.0, 0.0, 0.0)

    def reset(self):
        self._last = (0.0, 0.0, 0.0)

    def command(self, pose, path, dt=0.05, distance_to_goal=None):
        if not path:
            return (0.0, 0.0, 0.0)
        x, y, yaw = pose
        lx, ly = lookahead_point(path, x, y, self.lookahead)
        wx, wy = lx - x, ly - y
        c, s = math.cos(yaw), math.sin(yaw)
        dx = c * wx + s * wy
        dy = -s * wx + c * wy
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) if dist > 1e-6 else 0.0

        wz = max(-self.max_w, min(self.max_w, self.k_w * angle))
        align = max(0.0, math.cos(angle))
        if dx < 0.05 and abs(angle) > 0.4:
            vx, vy = 0.0, 0.0
        else:
            vx = max(-self.max_v, min(self.max_v, self.k_v * dx)) * (0.25 + 0.75 * align)
            vy = max(-self.max_vy, min(self.max_vy, self.k_vy * dy)) * align
            if vx > 0.0 and self.min_vx > 0.0:
                near = distance_to_goal is not None and distance_to_goal < 0.55
                if not near and vx < self.min_vx:
                    vx = self.min_vx

        lvx, lvy, lwz = self._last
        m = self.max_accel * dt
        vx = max(lvx - m, min(lvx + m, vx))
        vy = max(lvy - m, min(lvy + m, vy))
        mw = self.max_alpha * dt
        wz = max(lwz - mw, min(lwz + mw, wz))
        self._last = (vx, vy, wz)
        return (vx, vy, wz)


FRONT, SIDE, BACK = 0.46, 0.34, 0.30   # G1 footprint from the pelvis: hands held in front reach ~0.45 m


def footprint_points(res=0.04):
    """Points filling the G1 footprint (pelvis frame, x forward)."""
    xs = np.arange(-BACK, FRONT + 1e-9, res)
    ys = np.arange(-SIDE, SIDE + 1e-9, res)
    p = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    a = np.where(p[:, 0] > 0, FRONT, BACK)
    return p[(p[:, 0] / a) ** 2 + (p[:, 1] / SIDE) ** 2 <= 1.0]


def footprint_hits(grid, pose, pts=None):
    """Occupied cells of a GridMap under the footprint at pose (x, y, yaw): the number of footprint points on them."""
    pts = footprint_points() if pts is None else pts
    c, s = math.cos(pose[2]), math.sin(pose[2])
    wx = pose[0] + c * pts[:, 0] - s * pts[:, 1]
    wy = pose[1] + s * pts[:, 0] + c * pts[:, 1]
    r, cc = grid.world_to_grid(wx, wy)
    r, cc = np.floor(r).astype(int), np.floor(cc).astype(int)
    ok = (r >= 0) & (r < grid.height) & (cc >= 0) & (cc < grid.width)
    return int((~ok).sum() + grid.data[r[ok], cc[ok]].sum())


def _gap(loc, path, front=FRONT):
    """Smallest distance from the points (body frame) to the footprint placed at each path offset, beyond its edge.
    path: (T, 2) one motion, or (K, T, 2) K motions -> (K,) gaps."""
    many = path.ndim == 3
    P = path if many else path[None]
    rel = loc[None, None] - P[:, :, None]
    a = np.where(rel[..., 0] > 0, front, BACK)
    d = np.sqrt((rel ** 2).sum(-1))
    c, s = rel[..., 0] / np.maximum(d, 1e-9), rel[..., 1] / np.maximum(d, 1e-9)
    edge = 1.0 / np.sqrt((c / a) ** 2 + (s / SIDE) ** 2)
    g = (d - edge).reshape(len(P), -1).min(1)
    return g if many else float(g[0])


def physical_response(U):
    """What the simulated G1 (WardZero placeholder walking policy) does for commanded (vx, vy), measured in MuJoCo:
    forward ~0.5 x vx (+0.1 m/s carry-on while it walks), does not walk backwards, sideways ~0.4 x vy; standing
    still it stays put."""
    U = np.asarray(U, float).reshape(-1, 2)
    fwd = np.where(U[:, 0] > 1e-3, 0.5 * U[:, 0] + 0.1, 0.0)
    return np.column_stack((fwd, 0.4 * U[:, 1]))


def _turn_gap(loc, dth, front):
    """Gap of the footprint after turning in place by dth (points rotated the other way)."""
    c, s = math.cos(-dth), math.sin(-dth)
    return _gap(np.column_stack((c * loc[:, 0] - s * loc[:, 1], s * loc[:, 0] + c * loc[:, 1])), np.zeros((1, 2)), front)


def safe_velocity(cmd, pose, points, discs=(), robot_r=0.32, margin=0.10, disc_r=0.25, horizon=1.2,
                  max_vy=0.3, min_move=0.3, physical=False):
    """Execution guard on the G1 footprint (front 0.46, sides 0.34, back 0.30 m). Among velocities around the commanded
    one keep the closest whose motion over `horizon` s keeps `margin` beyond the footprint from every lidar point and
    agent; when already closer, the one that opens the gap; rotation only when the hands clear everything.
    physical: the simulated G1, whose motion is predicted with physical_response, with 0.20 m of stopping distance in
    front and no backward commands. Returns (cmd, gap) with gap the current distance beyond the footprint."""
    vx, vy, wz = cmd
    front = FRONT + (0.20 if physical else 0.0)
    resp = physical_response if physical else (lambda U: np.asarray(U, float).reshape(-1, 2))
    c, s = math.cos(pose[2]), math.sin(pose[2])
    obs = [np.asarray(points, float).reshape(-1, 2)]
    if len(discs):
        d = np.asarray(discs, float).reshape(-1, 2)
        ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        obs.append((d[:, None] + disc_r * np.stack((np.cos(ang), np.sin(ang)), -1)[None]).reshape(-1, 2))
    rel = np.vstack(obs) - np.asarray(pose[:2])
    loc = np.column_stack((c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]))
    loc = loc[np.einsum("ij,ij->i", loc, loc) < 2.2 ** 2]   # reach of 1.2 s at 0.75 m/s + footprint
    if len(loc) == 0:
        return cmd, 10.0
    loc = np.unique(np.round(loc / 0.08).astype(np.int32), axis=0) * 0.08    # one point per 8 cm cell
    g0 = _gap(loc, np.zeros((1, 2)))                    # the true footprint now
    ts = np.linspace(0.15, horizon, 8)
    near = float(np.sqrt(np.einsum("ij,ij->i", loc, loc).min()))
    if near < FRONT + 0.15 and abs(wz) > 1e-3:
        # turning sweeps the hands: allow it only if every angle of the next 0.8 s keeps 0.04 m (or gains room)
        sweep = [_turn_gap(loc, math.copysign(a, wz), FRONT) for a in np.linspace(0.05, abs(wz) * 0.8, 6)]
        if min(sweep) < min(g0, 0.04):
            wz = 0.0
    speed = math.hypot(vx, vy)
    if speed < 1e-3:
        stand = _gap(loc, ts[:, None] * resp([[0.0, 0.0]])[0][None], front)   # standing still, as it really moves
        if stand >= margin:
            return (0.0, 0.0, wz), g0
    base = math.atan2(vy, vx) if speed > 1e-3 else 0.0
    speed = max(speed, min_move)
    ks, das, us = [], [], []
    for k in (1.0, 0.75, 0.5):
        for da in (0.0, 0.3, -0.3, 0.6, -0.6, 0.95, -0.95, 1.35, -1.35, 1.8, -1.8, 2.4, -2.4, math.pi):
            a = base + da
            ux, uy = speed * k * math.cos(a), max(-max_vy, min(max_vy, speed * k * math.sin(a)))
            if math.hypot(ux, uy) >= min_move * 0.8 and not (physical and ux < 0):
                ks.append(k); das.append(da); us.append((ux, uy))  # noqa: E702
    best = None
    if us:
        U = np.asarray(us)
        gaps = _gap(loc, ts[None, :, None] * resp(U)[:, None, :], front)
        g_ref = _gap(loc, np.zeros((1, 2)), front)
        ok = (gaps >= margin) | ((g_ref < margin) & (gaps > g_ref + 0.03))
        if ok.any():
            cost = (1.0 - np.asarray(ks)) + 0.9 * np.abs(np.asarray(das)) - 0.4 * np.minimum(gaps, 1.0)
            cost[~ok] = np.inf
            best = tuple(U[int(np.argmin(cost))])
    if best is None:
        # nothing safe around the command: escape only along a direction that opens the gap to every point and
        # agent (as the robot really moves); otherwise stand still
        sp = max(min_move, 0.2)
        A = np.linspace(-math.pi, math.pi, 16, endpoint=False)
        U = np.column_stack((sp * np.cos(A), np.clip(sp * np.sin(A), -max_vy, max_vy)))
        if physical:
            U = U[U[:, 0] >= 0]
        gaps = _gap(loc, np.linspace(0.1, 0.6, 6)[None, :, None] * resp(U)[:, None, :], front)
        stand = _gap(loc, np.linspace(0.1, 0.6, 6)[:, None] * resp([[0.0, 0.0]])[0][None], front)
        k = int(np.argmax(gaps))
        if gaps[k] > stand + 0.02:
            return (U[k][0], U[k][1], 0.0), g0
        # no translation opens the gap: turn in place towards the side that opens it (hands swing away), if any
        g_now = _gap(loc, np.zeros((1, 2)))
        best_turn = None
        for sgn in (1.0, -1.0):
            sweep = [_turn_gap(loc, sgn * a, FRONT) for a in np.linspace(0.1, 0.5, 5)]
            # every angle of the turn must keep 0.03 m (or not lose room), and the end must have more room
            if min(sweep) >= min(g_now, 0.03) - 1e-6 and sweep[-1] > g_now + 0.005:
                if best_turn is None or sweep[-1] > best_turn[0]:
                    best_turn = (sweep[-1], sgn)
        if best_turn is not None:
            return (0.0, 0.0, 0.6 * best_turn[1]), g0
        return (0.0, 0.0, 0.0), g0
    return (best[0], best[1], wz), g0
