"""Holonomic pure pursuit for the G1 base."""
from __future__ import annotations

import math

from .astar import lookahead_point


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
