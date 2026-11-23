"""One Active NTFields model (arXiv 2410.09883, Alg. 1) shared by a fleet: mantf's network with the other agent as input.

Round-robin calls, agent 1, 2, ..., N, 1, ...: the calling agent shares its x, y, lidar and plan -> its N x m samples
(each row with the agent nearest to it) -> j epochs on the single model (memory M holds every agent's frames) -> that
agent's next viewpoint and Eq. 11 path, given the other agents' shared positions and plans.
"""
from __future__ import annotations

import json
import math
import threading
import os
import pickle
import sys
import time
import types

import numpy as np
import torch
from scipy.spatial import cKDTree

from .maps import SharedGrid

FAR = 100.0


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


for _name in ("igl", "torchvision"):
    try:
        __import__(_name)
    except Exception:
        _tv = _stub(_name)
        if _name == "torchvision":
            _tv.transforms = _stub("torchvision.transforms", Compose=lambda fs: fs)
if "igib_runner" not in sys.modules:
    _stub("igib_runner")

_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
MANTF_ROOT = os.environ.get("MANTF_ROOT") or next(
    (p for p in (os.path.join(_PROJECT, "mantf"), os.path.join(_PROJECT, "..", "mantf"))
     if os.path.isfile(os.path.join(p, "models", "model_igibson_n.py"))), os.path.join(_PROJECT, "..", "mantf"))
if MANTF_ROOT not in sys.path:
    sys.path.insert(0, MANTF_ROOT)
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True
try:
    from models import model_igibson_n as md
finally:
    sys.dont_write_bytecode = _dwb


class FieldModel(md.Model):
    def __init__(self, folder, device, scale, robot_radius, other_radius, frame_buffer=20, fourier_scale=1.0):
        super().__init__(folder, 3, scale, md.EXPLORATION, renderer=None, device=device,
                         radius=0.5 * (robot_radius + other_radius) / scale)
        if fourier_scale != 1.0:
            # mantf's Fourier features B ~ 0.5 N(0, 1) per model unit; a larger workspace per unit needs higher
            # frequencies to resolve the same wall thickness. Same network, same init, only B scaled.
            self.B = self.B * float(fourier_scale)
            self.network = md.NN(self.Params["Device"], self.dim, self.B)
            self.network.apply(self.network.init_weights)
            self.network.zero_agent_branch()
            self.network.to(self.Params["Device"])
            self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                               lr=self.Params["Training"]["Learning Rate"], weight_decay=0.1)
        self.folder = folder
        self.alpha = 1.025
        self.frame_buffer_size = int(frame_buffer)
        self.memory_frames = 600   # M keeps the most recent 600 frames (GPU memory); mantf/paper keep all
        self.rng = np.random.default_rng(0)

    def train_core(self, epoch, frame_data, is_one_frame=True):
        """mantf train_core (randomize_data, augment_others, agent_speed, Loss, rollback) with a rollback cap."""
        self.all_framedata = frame_data[None] if self.all_framedata is None else torch.cat(
            (self.all_framedata[-(self.memory_frames - 1):], frame_data[None]), 0)
        count, self.frame_idx = self.frame_idx, len(self.all_framedata) - 1   # randomize_data samples M[:frame_idx]
        cur = self.randomize_data(frame_data)
        self.frame_idx = count
        loader = md.FastTensorDataLoader(cur, batch_size=int(self.Params["Training"]["Batch Size"]), shuffle=True)
        beta, prev, current = 1.0, 1.0, 1.0
        diff = float("nan")
        for e in range(int(epoch)):
            self.epoch += 1
            self.optimizer.param_groups[0]["lr"] = 5e-4
            self.prev_state_queue.append(pickle.loads(pickle.dumps(self.network.state_dict())))
            self.prev_optimizer_queue.append(pickle.loads(pickle.dumps(self.optimizer.state_dict())))
            if len(self.prev_state_queue) > 5:
                self.prev_state_queue.pop(0)
                self.prev_optimizer_queue.pop(0)
            prev = current
            for repeat in range(4):
                tot, diff, n = 0.0, 0.0, 0
                for i, data in enumerate(loader):
                    if i > 5:
                        break
                    data = data[0].to(self.Params["Device"])
                    points = data[:, :6]
                    others = self.augment_others(points, data[:, 8:])
                    speed = self.agent_speed(points, data[:, 6:8], others)
                    speed = speed * speed * (2 - speed) * (2 - speed)
                    speed = self.alpha * speed + 1 - self.alpha
                    self.network.set_others(others)
                    loss, loss_n, _ = self.Loss(points, speed, beta, 0.001)
                    self.network.set_others(None)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tot += float(loss.detach())
                    diff += float(loss_n.detach())
                    n += 1
                tot, diff = tot / max(n, 1), diff / max(n, 1)
                current = diff
                r = current / max(prev, 1e-12)
                if (0 < r < 1.2) or e < 10 or repeat == 3 or not math.isfinite(diff):
                    break
                k = int(self.rng.integers(0, len(self.prev_state_queue)))
                with torch.no_grad():
                    self.network.load_state_dict(self.prev_state_queue[k])
                    self.optimizer.load_state_dict(self.prev_optimizer_queue[k])
            self.total_train_loss.append(tot)
            beta = 1.0 / max(diff, 1e-9)
        return diff


class Logger:
    TRAIN = ["wall_time", "round", "agent", "epoch", "epochs", "loss", "diff", "robots", "rows", "people", "min_robot_gap",
             "min_people_gap", "train_s"]
    DIAG = ["wall_time", "round", "eikonal_residual", "speed_mae", "speed_corr", "symmetry_err", "agent_ratio",
            "agent_target_ratio", "time_dist_ratio"]

    def __init__(self, folder, tensorboard=True):
        self.files = {}
        for name, fields in (("train_log", self.TRAIN), ("math_log", self.DIAG),
                             ("longrange_log", ["wall_time", "round", "agent", "geo_spearman", "g_into_wall", "pairs"])):
            f = open(os.path.join(folder, f"{name}.csv"), "a", buffering=1)
            if f.tell() == 0:
                f.write(",".join(fields) + "\n")
            self.files[name] = (f, fields)
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(folder, "tb"))
            except Exception:
                pass

    def log(self, name, rec, group):
        f, fields = self.files[name]
        rec = dict(rec, wall_time=round(time.time(), 3))
        f.write(",".join(str(rec.get(k, "")) for k in fields) + "\n")
        if self.tb is not None:
            for k in fields[2:]:
                if k == "agent":
                    continue
                v = rec.get(k)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self.tb.add_scalar(f"{group}/{k}", v, int(rec["round"]))


class CentralField:
    """The single model of the fleet and its shared map."""

    def __init__(self, folder, center, size, robots, device="cuda:0", robot_radius=0.32, other_radius=0.25,
                 n_rays=256, m=12, ray_max=8.0, rows=3000, epochs=5, warmup_epochs=30, frame_buffer=20,
                 grid_res=0.1, tensorboard=True, diag_every=5, plan_step=0.035, fourier_scale=1.0):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.center = np.array(center[:2], float)
        self.size = self.scale = float(size)
        self.robots = list(robots)
        self.robot_radius, self.other_radius = float(robot_radius), float(other_radius)
        self.n_rays, self.m, self.ray_max = int(n_rays), int(m), float(ray_max)
        self.rows, self.epochs, self.warmup = int(rows), int(epochs), int(warmup_epochs)
        self.diag_every = int(diag_every)
        self.plan_step = float(plan_step)   # mantf predict_trajectory2 step (model units); mantf's own plan() uses 0.02
        self.model = FieldModel(folder, device, self.scale, robot_radius, other_radius, frame_buffer, fourier_scale)
        self.fourier_scale = float(fourier_scale)
        self.device = self.model.Params["Device"]
        self.grid = SharedGrid(self.center, self.size, grid_res, self.robots)
        self.log = Logger(folder, tensorboard)
        self.rng = np.random.default_rng(0)
        self.rounds = 0
        self.mlock = threading.RLock()
        self.last = {}
        self.diag = {}
        self.dmin = self.model.minimum * self.scale
        self.dmax = self.model.maximum * self.scale
        json.dump({"algorithm": "Active NTFields Alg. 1, one model for all agents (mantf model_igibson_n)",
                   "robots": self.robots, "center": self.center.tolist(), "size_m": self.size,
                   "scale_factor": self.scale, "dmin_m": self.dmin, "dmax_m": self.dmax,
                   "robot_radius": self.robot_radius, "other_radius": self.other_radius,
                   "agent_radius_model": self.model.radius, "N_rays": self.n_rays, "m": self.m, "ray_max": self.ray_max,
                   "rows_per_robot": self.rows, "epochs_j": self.epochs, "warmup_epochs": self.warmup,
                   "batch": self.model.Params["Training"]["Batch Size"], "lr": 5e-4, "weight_decay": 0.1,
                   "memory_frames": self.model.frame_buffer_size, "alpha": self.model.alpha,
                   "fourier_scale": float(fourier_scale), "plan_step": float(plan_step)},
                  open(os.path.join(folder, "config.json"), "w"), indent=2)

    # ---------------------------------------------------------------- frames
    def to_model(self, xy):
        return (np.asarray(xy, np.float64)[..., :2] - self.center) / self.scale

    def to_world(self, q):
        return np.asarray(q, np.float64)[..., :2] * self.scale + self.center

    def inside(self, xy, margin=0.5):
        return bool(np.all(np.abs(np.asarray(xy[:2]) - self.center) < self.size / 2 - margin))

    def _other(self, xy):
        if xy is None:
            return torch.tensor([FAR, FAR, 0.0])
        q = self.to_model(xy)
        return torch.tensor([q[0], q[1], 0.0], dtype=torch.float32)

    def _drop_agents(self, ends, hit, extra, others):
        """Returns that land on a known agent are not map: agents enter the model as its other-agent input (mantf)."""
        if not len(others):
            return hit, extra
        tree = cKDTree(others)
        r = self.other_radius + 0.2
        hit = np.asarray(hit, bool) & (tree.query(np.asarray(ends, float))[0] > r)
        if extra is not None and len(extra):
            extra = extra[tree.query(np.asarray(extra)[:, :2])[0] > r]
        return hit, extra

    def _rows(self, origin, ends, hit, extra_hits, others):
        o = np.asarray(origin, np.float64)
        vec = np.asarray(ends, np.float64) - o
        length = np.linalg.norm(vec, axis=1)
        cap = np.minimum(length, self.ray_max)
        surf = np.asarray(ends, np.float64)[np.asarray(hit, bool)]
        if extra_hits is not None and len(extra_hits):
            surf = np.vstack((surf, extra_hits))
        if len(surf) < 5:
            return None
        pick = self.rng.choice(len(vec), size=min(self.n_rays, len(vec)), replace=False)
        u = (np.arange(self.m)[None] + self.rng.random((len(pick), self.m))) / self.m
        pts = (o + vec[pick, None] / np.maximum(length[pick], 1e-9)[:, None, None]
               * (u * cap[pick, None])[..., None]).reshape(-1, 2)
        pts = pts[np.all(np.abs(pts - self.center) < self.size / 2 * 0.98, axis=1)]
        if len(pts) < 16:
            return None
        d = cKDTree(surf).query(pts)[0] - self.robot_radius
        pts, d = pts[d > 0], d[d > 0]
        if len(pts) < 16:
            return None
        speed = np.clip(d, self.dmin, self.dmax) / self.dmax
        sel = self.rng.choice(len(pts), size=2 * self.rows, replace=len(pts) < 2 * self.rows)
        pts, speed = pts[sel], speed[sel]
        a, b = pts[0::2], pts[1::2]
        oth = np.zeros((self.rows, 3))
        oth[:, :2] = FAR
        if len(others):
            near = others[np.argmin(((a[:, None] - others[None]) ** 2).sum(-1), axis=1)]
            oth[:, :2] = self.to_model(near)
        z = np.zeros((self.rows, 1))
        return np.hstack((self.to_model(a), z, self.to_model(b), z, speed[0::2, None], speed[1::2, None], oth))

    def observe(self, *a, **k):
        with self.mlock:
            return self._observe(*a, **k)

    def _observe(self, scans, robots, people):
        """scans: id -> (origin, ends, hit, extra_hits); robots: id -> (x, y); people: (K, 2). One frame, one training."""
        people = np.asarray(people, float).reshape(-1, 2)
        blocks = []
        for rid, (origin, ends, hit, extra) in scans.items():
            others = np.asarray([robots[k] for k in robots if k != rid] + list(people), float).reshape(-1, 2)
            hit, extra = self._drop_agents(ends, hit, extra, others)
            self.grid.update(rid, origin, ends, hit, self.ray_max, extra)
            r = self._rows(origin, ends, hit, extra, others)
            if r is not None:
                blocks.append(r)
        if not blocks:
            return None
        rows = np.vstack(blocks)
        need = self.rows * len(scans)
        if len(rows) != need:
            rows = rows[self.rng.choice(len(rows), need, replace=len(rows) < need)]
        frame = torch.as_tensor(rows, dtype=torch.float32, device=self.device)
        self.rounds += 1
        if self.rounds > 1 and self.diag_every > 0 and self.rounds % self.diag_every == 0:
            self.diag = self._diagnostics(frame, robots)
            self.log.log("math_log", dict(self.diag, round=self.rounds), "math")
        t0 = time.monotonic()
        epochs = self.warmup if self.rounds == 1 else self.epochs
        diff = self.model.train_core(epochs, frame)
        self.model.frame_idx += 1
        pos = np.asarray(list(robots.values()), float).reshape(-1, 2)
        rg = [np.linalg.norm(pos[i] - pos[j]) for i in range(len(pos)) for j in range(i + 1, len(pos))]
        pg = np.linalg.norm(pos[:, None] - people[None], axis=-1).min() if len(people) and len(pos) else float("nan")
        self.last = {"round": self.rounds, "agent": ",".join(scans), "epoch": self.model.epoch, "epochs": epochs,
                     "loss": self.model.total_train_loss[-1], "diff": diff, "robots": len(blocks), "rows": len(rows),
                     "people": len(people), "min_robot_gap": min(rg) if rg else float("nan"),
                     "min_people_gap": float(pg), "train_s": round(time.monotonic() - t0, 3)}
        self.log.log("train_log", self.last, "train")
        return self.last

    # ---------------------------------------------------------------- the model
    def _with_other(self, other, fn):
        net = self.model.network
        net.set_others(other if torch.is_tensor(other) else self._other(other).to(self.device))
        try:
            return fn()
        finally:
            net.set_others(None)

    def _pairs(self, start, targets):
        qt = self.to_model(np.asarray(targets, float).reshape(-1, 2))
        X = np.zeros((len(qt), 6), np.float32)
        X[:, 0:2] = self.to_model(start)
        X[:, 3:5] = qt
        return torch.as_tensor(X)

    def travel_times(self, *a, **k):
        with self.mlock:
            return self._travel_times(*a, **k)

    def _travel_times(self, start, targets, other=None):
        X = self._pairs(start, targets)
        return self._with_other(other, lambda: self.model.TravelTimes(X).detach().cpu().numpy()) * self.scale

    def speeds(self, *a, **k):
        with self.mlock:
            return self._speeds(*a, **k)

    def _speeds(self, start, targets, other=None):
        X = self._pairs(start, targets)
        out = []
        for chunk in torch.split(X, 6000):
            out.append(self._with_other(other, lambda c=chunk: self.model.Speed(c).detach().cpu().numpy()))
        return np.concatenate(out)

    def plan(self, *a, **k):
        with self.mlock:
            return self._plan(*a, **k)

    def _plan(self, start, goal, other=None):
        """Eq. 11 (mantf predict_trajectory2). If the two halves never meet, the robot's half (mantf step_along)."""
        qs, qg = self.to_model(start), self.to_model(goal)
        net = self.model.network
        net.set_others(self._other(other).to(self.device))
        try:
            traj = self.model.predict_trajectory2(np.array([qs[0], qs[1], 0.0]), np.array([qg[0], qg[1], 0.0]),
                                                  step_size=self.plan_step, tol=max(0.03, self.plan_step))
        finally:
            net.set_others(None)
        w = self.to_world(np.asarray(traj))
        if len(w) < 2 or not np.all(np.isfinite(w)):
            return None, "nan"
        gaps = np.linalg.norm(np.diff(w, axis=0), axis=1)
        half = (len(w) - 2) // 2
        if gaps.max() > 1.0:
            k = int(np.argmax(gaps[half:] > 1.0)) + half if (gaps[half:] > 1.0).any() else int(np.argmax(gaps))
            w = w[:k + 1]
            if np.linalg.norm(w[-1] - w[0]) < 0.4:
                return None, "stuck"
            return w, "partial"
        return w, "full"

    def next_viewpoint(self, *a, **k):
        with self.mlock:
            return self._next_viewpoint(*a, **k)

    def _next_viewpoint(self, pose, goal, other=None, claimed=(), horizon=5.0, max_cands=300, mode="cooperative",
                        requests=(), tried=(), free_ok=None, goal_paused=False):
        """Next viewpoint (paper Alg. 1 l.17 / Sec. IV-E3, goal-directed, multi-agent).
        The model's T is trusted only through mapped space; across unmapped space the straight-line distance is the
        (optimistic) estimate. Candidates within 1.5 m of a viewpoint this agent already tried are excluded.
          goal reachable through mapped free space: the goal itself (Eq. 11 through the learned field); if the goal
              stalled (goal_paused), mapped free cells within the horizon by T(q, c | o) + T(c, goal | o)
          otherwise, reachable frontier cells (paper):
              cooperative: argmin T(q, c | o) + |c - goal| + claim(c)   (claim: near another agent's shared viewpoint)
              greedy:      argmin |c - goal|
              no goal:     argmin T(q, c | o) + claim(c) - request(c)   (an agent helping others)"""
        g = self.grid
        q = np.asarray(pose[:2], float)
        tried = [np.asarray(t, float) for t in tried]

        def fresh(c):
            if not len(c) or not tried:
                return c
            d = np.min(np.linalg.norm(c[:, None] - np.asarray(tried)[None], axis=-1), axis=1)
            return c[d > 1.5]

        if goal is not None and g.reaches(q, goal):
            gt = np.asarray(goal[:2], float)
            if not goal_paused:
                return tuple(gt), "goal"
            cand = fresh(g.free_cells(q, 1.0, horizon, n=200, rng=self.rng))
            if len(cand):
                cost = self._travel_times(q, cand, other) + self._travel_times(gt, cand, other)
                return tuple(cand[int(np.argmin(cost))]), "waypoint"
            return tuple(gt), "goal"
        cand = fresh(g.frontier_cells(q, 0.8, self.size))
        kind = "frontier"
        if len(cand) == 0:
            cand, kind = fresh(g.free_cells(q, 1.0, horizon, n=200, rng=self.rng)), "waypoint"
        if len(cand) == 0:
            return (tuple(goal[:2]), "goal") if goal is not None else (None, "done")
        if len(cand) > max_cands:
            cand = cand[self.rng.choice(len(cand), max_cands, replace=False)]
        if goal is not None and mode == "greedy":
            cost = np.linalg.norm(cand - np.asarray(goal[:2]), axis=1)
        else:
            cost = self._travel_times(q, cand, other)
            if goal is not None:
                cost = cost + np.linalg.norm(cand - np.asarray(goal[:2]), axis=1)
            for c in claimed:
                cost = cost + 6.0 * np.exp(-((cand - np.asarray(c)) ** 2).sum(1) / (2 * 2.0 ** 2))
            for r in requests:
                cost = cost - 15.0 * np.exp(-((cand - np.asarray(r[:2])) ** 2).sum(1) / (2 * 4.0 ** 2))
                kind = "request"
        return tuple(cand[int(np.argmin(cost))]), kind

    def heading(self, *a, **k):
        with self.mlock:
            return self._heading(*a, **k)

    def _heading(self, start, target, other=None):
        """(gx, gy): the model's descent direction of T at the agent towards target (first step of Eq. 11, mantf
        Gradient), unit length, and the learned speed S at the agent."""
        X = self._pairs(target, [start])
        g = self._with_other(other, lambda: self.model.Gradient(self._pairs(start, [target])).detach().cpu().numpy()[0])
        s = float(self._with_other(other, lambda: self.model.Speed(X).detach().cpu().numpy()[0]))
        v = g[:2]
        n = float(np.linalg.norm(v))
        return (v / n if n > 1e-9 and np.isfinite(n) else np.zeros(2)), s

    def field_maps(self, start, other=None, res=0.25):
        n = int(self.size / res)
        xs = self.center[0] - self.size / 2 + (np.arange(n) + 0.5) * res
        ys = self.center[1] - self.size / 2 + (np.arange(n) + 0.5) * res
        tgt = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        T = self.travel_times(start, tgt, other).reshape(n, n)
        S = self.speeds(start, tgt, other).reshape(n, n)
        return S, T, (xs[0] - res / 2, ys[0] - res / 2, res)

    # ---------------------------------------------------------------- is the math doing what it should
    def diagnostics(self, *a, **k):
        with self.mlock:
            return self._diagnostics(*a, **k)

    def _diagnostics(self, frame, robots, gt_clearance=None):
        """On a frame the model has not trained on yet:
        eikonal_residual  mantf Loss (S* |grad T| = 1), lower is better
        speed_mae/corr    learned S(q) vs the expert target S*(q) of Eq. 4 (+ agent term, mantf transform)
        symmetry_err      |T(a,b) - T(b,a)| / T, should be ~0
        agent_ratio       S near an agent / S with no agent (an agent at r_robot+r_other+0.1 m), should be < 1
        agent_target_ratio the same ratio of the targets (what the loss asks for)
        time_dist_ratio   median T(a,b) / |a-b| on free pairs, >= 1 (1 = straight line at full speed)."""
        m = self.model
        n = min(2000, len(frame))
        f = frame[torch.randperm(len(frame))[:n]]
        pts, sp, oth = f[:, :6], f[:, 6:8], f[:, 8:]
        target = m.agent_speed(pts, sp, oth)
        tt = target * target * (2 - target) * (2 - target)
        tt = m.alpha * tt + 1 - m.alpha
        m.network.set_others(oth)
        _, loss_n, _ = m.Loss(pts, tt, 1.0, 0.001)
        S = m.Speed(pts).detach()
        T_ab = m.TravelTimes(pts).detach()
        T_ba = m.TravelTimes(torch.cat((pts[:, 3:], pts[:, :3]), 1)).detach()
        m.network.set_others(None)
        s, t = S.cpu().numpy(), tt[:, 1].detach().cpu().numpy()
        corr = float(np.corrcoef(s, t)[0, 1]) if s.std() > 1e-9 and t.std() > 1e-9 else float("nan")
        th = torch.rand(n, device=pts.device) * 2 * math.pi
        r = (self.robot_radius + self.other_radius + 0.1) / self.scale
        near = pts[:, 3:].clone()
        near[:, 0] += r * torch.cos(th)
        near[:, 1] += r * torch.sin(th)
        far = torch.full_like(near, FAR)
        far[:, 2] = 0
        free = sp[:, 1] > 0.6
        m.network.set_others(near)
        s_near = m.Speed(pts).detach()
        m.network.set_others(far)
        s_far = m.Speed(pts).detach()
        m.network.set_others(None)
        tgt_near = m.agent_speed(pts, sp, near)[:, 1]
        d = torch.norm(pts[:, 3:5] - pts[:, :2], dim=1)
        ok = free & (d > 0.02)
        return {"eikonal_residual": float(loss_n.detach()), "speed_mae": float(np.abs(s - t).mean()), "speed_corr": corr,
                "symmetry_err": float((torch.abs(T_ab - T_ba) / torch.clamp(T_ab, min=1e-6)).median()),
                "agent_ratio": float((s_near[free] / torch.clamp(s_far[free], min=1e-6)).median()) if free.any() else float("nan"),
                "agent_target_ratio": float((tgt_near[free] / torch.clamp(sp[free, 1], min=1e-6)).median()) if free.any() else float("nan"),
                "time_dist_ratio": float((T_ab[ok] / d[ok]).median()) if ok.any() else float("nan")}
