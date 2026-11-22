from __future__ import annotations

import math
import time

import numpy as np

from .astar import closest_index, lookahead_point
from .occupancy import GridMap


class LocalPlanner:
    def __init__(self, model, window_radius=5.0, robot_radius=0.32, static_margin=0.06,
                 dynamic_margin=0.18, lookahead=3.0, warmup_epochs=20, train_epochs=8,
                 retrain_period=1.0):
        self.model = model
        self.window_radius = float(window_radius)
        self.robot_radius = float(robot_radius)
        self.static_margin = float(static_margin)
        self.dynamic_margin = float(dynamic_margin)
        self.lookahead = float(lookahead)
        self.warmup_epochs = int(warmup_epochs)
        self.train_epochs = int(train_epochs)
        self.retrain_period = float(retrain_period)
        self.frames = 0
        self.last_train = -1e9
        self.stats = {"mantf": 0, "fallback": 0}
        self.blocked = False

    def local_goal(self, pose, global_path):
        lx, ly = lookahead_point(global_path, pose[0], pose[1], self.lookahead)
        d = math.hypot(lx - pose[0], ly - pose[1])
        maxd = self.window_radius * 0.8
        if d > maxd:
            lx = pose[0] + (lx - pose[0]) * maxd / d
            ly = pose[1] + (ly - pose[1]) * maxd / d
        return lx, ly

    def window(self, static, dynamic, pose):
        r = self.window_radius
        ssub = static.subgrid(pose[0], pose[1], r)
        occ = ssub.inflated(self.robot_radius + self.static_margin)
        dsub = None
        if dynamic is not None:
            dsub = dynamic.subgrid(pose[0], pose[1], r)
            occ |= dsub.inflated(self.robot_radius + self.dynamic_margin)
        return GridMap(occ, ssub.resolution, ssub.origin, ssub.yaw), dsub

    def path_blocked(self, path, local, pose, horizon=3.0, step=0.1):
        travelled = 0.0
        for (ax, ay), (bx, by) in zip(path[:-1], path[1:]):
            seg = math.hypot(bx - ax, by - ay)
            for t in np.arange(0.0, seg, step):
                x, y = ax + (bx - ax) * t / max(seg, 1e-9), ay + (by - ay) * t / max(seg, 1e-9)
                if math.hypot(x - pose[0], y - pose[1]) > self.robot_radius + 0.15 and local.occupied(x, y):
                    return True
            travelled += seg
            if travelled > horizon:
                break
        return False

    @staticmethod
    def nearest(center, sub, max_dist=2.5):
        if sub is None or not sub.data.any():
            return None
        rr, cc = np.nonzero(sub.data)
        wx, wy = sub.grid_to_world(rr, cc)
        d = (wx - center[0]) ** 2 + (wy - center[1]) ** 2
        k = int(np.argmin(d))
        if d[k] > max_dist * max_dist:
            return None
        return float(wx[k]), float(wy[k])

    def step(self, pose, global_path, static, dynamic=None, other=None, now=None):
        now = time.monotonic() if now is None else now
        local, dsub = self.window(static, dynamic, pose)
        if other is None:
            other = self.nearest(pose[:2], dsub)
        goal = self.local_goal(pose, global_path)
        path = None
        if self.model is not None:
            self.model.set_local_grid(local.data, pose[:2], local.origin)
            if self.frames == 0 or now - self.last_train > self.retrain_period:
                self.model.train_online(self.warmup_epochs if self.frames == 0 else self.train_epochs,
                                        other_world=other)
                self.last_train = now
                self.frames += 1
            path = self.model.plan_local(goal, other_world=other)
        self.blocked = False
        if path is None or len(path) < 2:
            self.stats["fallback"] += 1
            rest = global_path[closest_index(global_path, pose[0], pose[1]):]
            self.blocked = self.path_blocked(rest, local, pose)
            return rest, False
        self.stats["mantf"] += 1
        return path, True
