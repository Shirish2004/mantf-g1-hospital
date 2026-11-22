"""mantf travel-time field on a live occupancy grid.

mantf's own runner samples scans from iGibson.  This module keeps mantf's
network, loss and planner and replaces only the sampler: it draws point pairs
from a 2-D distance transform of the local inflated occupancy grid.  The model
is egocentric, so every local grid is centred on the robot and the robot is at
the model origin.
"""
from __future__ import annotations

import math
import os
import sys
import types

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .occupancy import GridMap

EXPLORATION = 1
SCALE_FACTOR = 10.0
MESH_SCALE = 1.0
MIN_SPEED = 0.007          # model units (0.07 m)
MAX_SPEED = 0.07           # model units (0.70 m)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _install_shims():
    if "igl" not in sys.modules:
        try:
            __import__("igl")
        except Exception:
            _stub("igl")
    if "torchvision" not in sys.modules:
        try:
            __import__("torchvision")
        except Exception:
            tv = _stub("torchvision")
            tr = _stub("torchvision.transforms")
            tr.Compose = lambda fs: (lambda x: _compose(fs, x))
            tv.transforms = tr
    if "igib_runner" not in sys.modules:
        def unavailable(*_a, **_k):
            raise RuntimeError("iGibson sampler is disabled; use GridModel.scan_rows")
        _stub("igib_runner", sample_points_and_speeds_from_pos_new=unavailable,
              sample_points_from_pos=unavailable)


def _compose(fs, x):
    for f in fs:
        x = f(x)
    return x


_install_shims()

MANTF_ROOT = os.environ.get("MANTF_ROOT", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "mantf")))
if MANTF_ROOT not in sys.path:
    sys.path.insert(0, MANTF_ROOT)
try:
    from models import model_igibson_n as md
except Exception as exc:  # pragma: no cover - environment error path
    raise RuntimeError(f"cannot import mantf from MANTF_ROOT={MANTF_ROOT}: {exc}") from exc


def _synthetic_map(kind, size, res, robot_radius, margin):
    n = max(8, int(round(size / res)))
    data = np.zeros((n, n), dtype=bool)
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - n / 2.0) * res
    y = (yy - n / 2.0) * res
    if kind == "corridor":
        data[np.abs(y) > 1.0] = True
    elif kind == "corner":
        data[(x > 1.4) & (np.abs(y) < 1.4)] = True
        data[(y > 1.4) & (np.abs(x) < 1.4)] = True
    elif kind == "pillar":
        data[(x - 1.3) ** 2 + (y - 0.6) ** 2 < 0.45 ** 2] = True
    elif kind == "door":
        data[np.abs(x) > 2.2] = True
        data[(np.abs(x) < 2.2) & (np.abs(y) > 1.6)] = True
        data[(np.abs(x) < 0.5) & (np.abs(y) > 0.4)] = True
    grid = GridMap(data, res, (-n * res / 2.0, -n * res / 2.0))
    return GridMap(grid.inflated(robot_radius + margin), res, grid.origin)


class GridModel(md.Model):
    def __init__(self, model_path, device="cpu", scale_factor=SCALE_FACTOR,
                 mesh_scale=MESH_SCALE, rows=1500, window_radius=5.0,
                 resolution=0.05, robot_radius=0.32, static_margin=0.06,
                 dynamic_margin=0.18, seed=0):
        super().__init__(model_path, 3, scale_factor, EXPLORATION, renderer=None,
                         device=device, mesh_scale=mesh_scale, radius=md.AGENT_RADIUS)
        self.alpha = 1.025
        self.frame_buffer_size = 8
        self.frame_epoch = 10
        self.move_step = 0.05
        self.goal_tol = 0.04
        self.max_rounds = 0
        os.makedirs(self.folder, exist_ok=True)

        self.rows = int(rows)
        self.window_radius = float(window_radius)
        self.resolution = float(resolution)
        self.robot_radius = float(robot_radius)
        self.static_margin = float(static_margin)
        self.dynamic_margin = float(dynamic_margin)
        self.rng = np.random.default_rng(seed)
        self.device_obj = torch.device(device)
        self._occ = None
        self._edt = None
        self._origin = (0.0, 0.0)
        self._center = (0.0, 0.0)
        self._res = self.resolution
        self._signature = None
        self._train_calls = 0
        self.pretrained = False

    # ------------------------------------------------------------------ local grid
    def set_local_grid(self, occ, center, origin=None):
        occ = np.asarray(occ, dtype=bool)
        self._occ = occ.copy()
        self._res = self.resolution
        self._center = (float(center[0]), float(center[1]))
        self._edt = distance_transform_edt(~occ) * self._res
        if origin is None:
            self._origin = (self._center[0] - occ.shape[1] * self._res / 2.0,
                            self._center[1] - occ.shape[0] * self._res / 2.0)
        else:
            self._origin = (float(origin[0]), float(origin[1]))
        self._signature = (int(occ.sum()), occ.shape, self._center)

    def clearance_at_world(self, x, y):
        if self._edt is None:
            return 0.0
        c = int(math.floor((x - self._origin[0]) / self._res))
        r = int(math.floor((y - self._origin[1]) / self._res))
        if not (0 <= r < self._edt.shape[0] and 0 <= c < self._edt.shape[1]):
            return 0.0
        return float(self._edt[r, c])

    def clearance_at_model(self, x, y):
        return self.clearance_at_world(self._center[0] + x * self.scale_factor,
                                       self._center[1] + y * self.scale_factor)

    def _clearance_xy(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        cx = self._center[0] + xy[:, 0] * self.scale_factor
        cy = self._center[1] + xy[:, 1] * self.scale_factor
        c = np.floor((cx - self._origin[0]) / self._res).astype(np.int64)
        r = np.floor((cy - self._origin[1]) / self._res).astype(np.int64)
        valid = ((r >= 0) & (r < self._edt.shape[0]) &
                 (c >= 0) & (c < self._edt.shape[1]))
        out = np.zeros(len(xy), dtype=np.float64)
        out[valid] = self._edt[r[valid], c[valid]]
        return out

    def model_half(self):
        if self._occ is None:
            return 0.5 * self.window_radius / self.scale_factor
        return min(self._occ.shape) * self._res / (2.0 * self.scale_factor)

    def grid_signature(self):
        return self._signature

    # ------------------------------------------------------------------ scan sampler
    def _sample_free(self, count, half, min_clear):
        out = np.zeros((0, 2), dtype=np.float32)
        tries = 0
        while len(out) < count and tries < 20:
            tries += 1
            cand = self.rng.uniform(-half, half, size=(max(64, count * 2), 2))
            keep = cand[self._clearance_xy(cand) >= min_clear]
            out = np.concatenate((out, keep.astype(np.float32)), axis=0)[:count]
        if len(out) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(out) < count:
            extra = self.rng.integers(0, len(out), size=count - len(out))
            out = np.concatenate((out, out[extra]), axis=0)
        return out.astype(np.float32)

    def scan_rows(self, position=None, other=None, rows=None):
        rows = int(rows or self.rows)
        if self._occ is None:
            raise RuntimeError("set_local_grid() before scan_rows()")
        half = self.model_half() * 0.96
        min_clear = MIN_SPEED * self.scale_factor
        max_clear = MAX_SPEED * self.scale_factor
        xy = self._sample_free(rows, half, min_clear)
        if xy.shape[0] == 0:
            xy = np.zeros((rows, 2), dtype=np.float32)
        theta = self.rng.uniform(-math.pi, math.pi, size=(xy.shape[0],))
        clear0 = self._clearance_xy(xy)
        reach = np.clip(clear0, min_clear, max_clear)
        r = self.rng.uniform(min_clear, np.maximum(reach, min_clear + 1e-6))
        x1 = xy + np.column_stack((np.cos(theta), np.sin(theta))) * r[:, None]
        x1 = np.clip(x1, -half, half)
        clear1 = self._clearance_xy(x1)
        speed0 = np.clip(clear0, min_clear, max_clear) / max_clear
        speed1 = np.clip(clear1, min_clear, max_clear) / max_clear

        points = np.column_stack((xy, np.zeros(len(xy)), x1, np.zeros(len(x1))))
        speeds = np.column_stack((speed0, speed1))
        if other is None:
            other_xy = np.array([100.0, 100.0, 0.0], dtype=np.float32)
        elif torch.is_tensor(other):
            other_xy = other.detach().cpu().numpy().reshape(-1)[:3].astype(np.float32)
        else:
            other_xy = np.asarray(other, dtype=np.float32).reshape(-1)[:3]
        others = np.repeat(other_xy[None], len(xy), axis=0)
        row = np.column_stack((points, speeds, others))
        return torch.as_tensor(row, dtype=torch.float32, device=self.Params["Device"])

    # ------------------------------------------------------------------ training
    def train_core(self, epoch, frame_data, is_one_frame=True):
        """mantf's training loop with a repeat cap and no per-call scan dump."""
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1.0
        step = 0
        if is_one_frame:
            if self.all_framedata is None:
                self.all_framedata = frame_data.unsqueeze(0)
            else:
                self.all_framedata = torch.cat((self.all_framedata, frame_data.unsqueeze(0)), dim=0)
            cur_data = self.randomize_data(frame_data)
        else:
            idx = torch.randperm(frame_data.shape[0])[:100000]
            chosen = frame_data[idx]
            chosen_points, chosen_speeds, chosen_others = chosen[:, :6], chosen[:, 6:8], chosen[:, 8:]
            si = torch.randperm(chosen_points.shape[0])
            ei = torch.randperm(chosen_points.shape[0])
            cur_data = torch.cat((chosen_points[si][:, :3], chosen_points[ei][:, 3:6],
                                  chosen_speeds[si][:, 0:1], chosen_speeds[ei][:, 1:2],
                                  chosen_others[si]), dim=1)

        loader = md.FastTensorDataLoader(
            cur_data, batch_size=int(self.Params["Training"]["Batch Size"]), shuffle=True)
        total_diff = torch.tensor(0.0)
        for e in range(int(epoch)):
            self.epoch += 1
            self.optimizer.param_groups[0]["lr"] = 5e-4
            step += 1.0 / 4000.0 / ((int)(self.epoch / 4000) + 1.0) * 2.0
            gamma = 0.001
            state = __import__("pickle").loads(__import__("pickle").dumps(self.network.state_dict()))
            opt = __import__("pickle").loads(__import__("pickle").dumps(self.optimizer.state_dict()))
            self.prev_state_queue.append(state)
            self.prev_optimizer_queue.append(opt)
            if len(self.prev_state_queue) > 5:
                self.prev_state_queue.pop(0)
                self.prev_optimizer_queue.pop(0)

            prev_diff = current_diff
            repeats = 0
            while True:
                total_train_loss = 0.0
                total_diff = 0.0
                batches = 0
                for i, data in enumerate(loader):
                    if i > 5:
                        break
                    data = data[0].to(self.Params["Device"])
                    points = data[:, :2 * self.dim]
                    speed = data[:, 2 * self.dim:2 * self.dim + 2]
                    others = self.augment_others(points, data[:, 2 * self.dim + 2:])
                    speed = self.agent_speed(points, speed, others)
                    speed = speed * speed * (2 - speed) * (2 - speed)
                    speed = self.alpha * speed + 1 - self.alpha
                    self.network.set_others(others if self.agent_aware else None)
                    loss_value, loss_n, _ = self.Loss(points, speed, beta, gamma)
                    self.network.set_others(None)
                    loss_value.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_train_loss += float(loss_value.detach())
                    total_diff += float(loss_n.detach())
                    batches += 1
                if batches == 0:
                    break
                total_train_loss /= batches
                total_diff /= batches
                if not math.isfinite(total_diff):
                    break
                current_diff = total_diff
                ratio = current_diff / max(prev_diff, 1e-12)
                if (ratio < 1.2 and ratio > 0) or e < 10 or repeats >= 3:
                    break
                repeats += 1
                k = int(self.rng.integers(0, len(self.prev_state_queue)))
                with torch.no_grad():
                    self.network.load_state_dict(self.prev_state_queue[k], strict=True)
                    self.optimizer.load_state_dict(self.prev_optimizer_queue[k])
                prev_diff = current_diff
            self.total_train_loss.append(total_train_loss)
            beta = 1.0 / max(total_diff, 1e-9)
        self._train_calls += 1
        return total_diff, cur_data

    def train_online(self, epochs, other_world=None):
        other = None
        if other_world is not None:
            other = np.array([(other_world[0] - self._center[0]) / self.scale_factor,
                              (other_world[1] - self._center[1]) / self.scale_factor, 0.0])
        frame = self.scan_rows(None, other, self.rows)
        self.train_core(int(epochs), frame, True)
        self.frame_idx += 1

    def pretrain(self, epochs=30, size=10.0, kinds=("open", "corridor", "corner", "pillar", "door")):
        if self.pretrained or epochs <= 0:
            return
        for kind in kinds:
            if kind == "open":
                n = int(round(size / self.resolution))
                grid = GridMap(np.zeros((n, n), dtype=bool), self.resolution,
                               (-n * self.resolution / 2.0, -n * self.resolution / 2.0))
            else:
                grid = _synthetic_map(kind, size, self.resolution, self.robot_radius, self.static_margin)
            self.set_local_grid(grid.data, (0.0, 0.0))
            self.train_online(epochs)
        self.pretrained = True

    # ------------------------------------------------------------------ planning
    def plan_local(self, goal_world, other_world=None):
        if self._occ is None:
            return None
        half = self.model_half() * 0.94
        gx = (goal_world[0] - self._center[0]) / self.scale_factor
        gy = (goal_world[1] - self._center[1]) / self.scale_factor
        gx, gy = float(np.clip(gx, -half, half)), float(np.clip(gy, -half, half))
        if other_world is None:
            ox, oy = 100.0, 100.0
        else:
            ox = (other_world[0] - self._center[0]) / self.scale_factor
            oy = (other_world[1] - self._center[1]) / self.scale_factor
        start = torch.zeros(3, dtype=torch.float32)
        goal = torch.tensor([gx, gy, 0.0], dtype=torch.float32)
        other = torch.tensor([ox, oy, 0.0], dtype=torch.float32)
        traj = self.plan(start, goal, other)
        if traj is None or len(traj) < 2:
            return None
        traj = np.asarray(traj, dtype=float)
        world = np.column_stack((self._center[0] + traj[:, 0] * self.scale_factor,
                                 self._center[1] + traj[:, 1] * self.scale_factor))
        for x, y in world:
            if self.clearance_at_world(x, y) <= 0.0:
                return None
        if math.hypot(world[-1, 0] - goal_world[0], world[-1, 1] - goal_world[1]) > 1.0:
            return None
        return [(float(x), float(y)) for x, y in world]

    def save_checkpoint(self, path):
        torch.save({"model_state_dict": self.network.state_dict(),
                    "B_state_dict": self.B, "epoch": self.epoch}, path)
        return path

    def load_checkpoint(self, path):
        self.load(path)
        self.network.train()


def make_model(model_path, device=None, **kwargs):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return GridModel(model_path, device=device, **kwargs)
