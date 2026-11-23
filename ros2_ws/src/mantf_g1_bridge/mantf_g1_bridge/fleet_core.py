"""Centralised multi-agent mapping and motion planning: one CentralField, one shared map, round-robin calls.

Call k serves agent k mod N: it shares x, y, lidar and plan; the model trains on its frame and returns its viewpoint,
its (gx, gy) = -grad T towards it and its Eq. 11 plan, which it follows until its next call.

Every robot shares its pose, its lidar (into the shared map and the one model's training frame) and its plan
(viewpoint + Eq. 11 path, so the others' next viewpoints avoid it). People are other agents for the model.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .fields import CentralField
from .maps import VoxelMap, color, field_panel, render_fleet
from .occupancy import raycast
from .pure_pursuit import PurePursuit, closest_index, footprint_hits, footprint_points, path_length, safe_velocity


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class KinBody:
    """A kinematic G1 for the extra robots: integrates its command, collides with the true map using the G1 footprint
    (front 0.46, sides 0.34, back 0.30 m), 2-D lidar on the true map."""

    def __init__(self, pose, gt, accel=1.0):
        self.pose = np.array(pose, float)
        self.v = np.zeros(3)
        self.gt, self.accel = gt, accel
        self.fp = footprint_points()
        self.blocked = False

    def _agent_gap(self, pose, a):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        w = np.column_stack((pose[0] + c * self.fp[:, 0] - s * self.fp[:, 1], pose[1] + s * self.fp[:, 0] + c * self.fp[:, 1]))
        return float((np.linalg.norm(w[:, None] - a[None, :, :2], axis=-1) - a[None, :, 2]).min())

    def free(self, pose, agents=()):
        """Footprint off the true map, and either clear of every agent or moving apart from the nearest one."""
        if footprint_hits(self.gt, pose, self.fp) > 0:
            return False
        a = np.asarray(agents, float).reshape(-1, 3)
        if not len(a):
            return True
        gap = self._agent_gap(pose, a)
        return gap > 0.0 or gap > self._agent_gap(self.pose, a) + 1e-4

    def step(self, cmd, dt, agents=()):
        """agents: (x, y, radius) of the others. Returns True when the motion was stopped (wall or agent)."""
        self.v += np.clip(np.asarray(cmd, float) - self.v, -self.accel * dt * 2, self.accel * dt * 2)
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        nth = wrap(th + self.v[2] * dt)
        nx, ny = x + (c * self.v[0] - s * self.v[1]) * dt, y + (s * self.v[0] + c * self.v[1]) * dt
        for cand in ((nx, ny, nth), (nx, ny, th), (x, y, nth)):
            if self.free(cand, agents):
                hit = cand != (nx, ny, nth)
                self.pose = np.array(cand)
                break
        else:
            hit = True
        if hit:
            self.v[:] = 0
        self.blocked = hit
        return hit

    def scan(self, discs=(), beams=360, rmax=10.0):
        x, y, th = self.pose
        ang, rng = raycast(self.gt, x, y, th, beams, rmax, discs=np.asarray(discs).reshape(-1, 2), disc_r=0.28)
        hit = np.isfinite(rng)
        r = np.where(hit, rng, rmax)
        ends = np.column_stack((x + r * np.cos(ang + th), y + r * np.sin(ang + th)))
        return (x, y), ends, hit


class Robot:
    def __init__(self, rid, idx, kind, cfg):
        self.id, self.idx, self.kind = rid, idx, kind
        self.pose = None
        self.scan = None
        self.goal = None
        self.explore = kind == "kin" and cfg.get("explore", True)
        self.helper = False
        self.goal_name = ""
        self.label = ""
        self.calls = 0
        self.gxy = None
        self.s_here = 0.0
        self.plan_s = 0.0
        self.contacts = []
        self.in_contact = False
        self.tried = []
        self.progress = []
        self.yields = 0
        self.agent_contacts = []
        self.goal_pause = -1.0
        self.ep_video = None
        self.call_log = []
        self.scans = []
        self.path = None
        self.path_kind = ""
        self.viewpoint = None
        self.view_kind = ""
        self.trail = []
        self.state = "idle"
        self.stats = {"goal": 0, "frontier": 0, "waypoint": 0, "request": 0, "opportunity": 0, "full": 0, "partial": 0, "stuck": 0, "blocked": 0, "nan": 0}
        self.ctrl = PurePursuit(lookahead=cfg["lookahead"], max_v=cfg["max_v"], max_vy=cfg["max_vy"],
                                max_w=cfg["max_w"], min_vx=cfg["min_vx"] if kind == "sim" else 0.0,
                                max_accel=2.0)
        self.body = None
        self.episode = None
        self.cmd = (0.0, 0.0, 0.0)
        self.clear = 10.0


class Fleet:
    DEFAULTS = {"robot_radius": 0.32, "other_radius": 0.25, "n_rays": 256, "m": 12, "ray_max": 8.0, "rows": 3000,
                "epochs": 5, "warmup_epochs": 30, "frame_buffer": 20, "grid_res": 0.1, "horizon": 5.0,
                "lookahead": 0.7, "max_v": 0.75, "max_vy": 0.35, "max_w": 0.9, "min_vx": 0.36, "goal_tolerance": 0.35,
                "yaw_tolerance": 0.45, "guard_margin": 0.10, "guard_horizon": 1.2, "plan_clearance": 0.42,
                "explore": True, "mode": "cooperative", "help": True, "opportunity_range": 18.0, "tensorboard": True, "diag_every": 5, "voxel_res": 0.1, "fourier_scale": 1.0}

    def __init__(self, folder, workspace, robots, gt=None, device="cuda:0", **cfg):
        self.cfg = dict(self.DEFAULTS, **{k: v for k, v in cfg.items() if v is not None})
        c = self.cfg
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.robots = {}
        for i, (rid, kind, pose) in enumerate(robots):
            r = Robot(rid, i, kind, c)
            if kind == "kin":
                r.body = KinBody(pose, gt)
                r.pose = tuple(pose)
            self.robots[rid] = r
        self.gt = gt
        self.gt_edt = distance_transform_edt(~gt.data) * gt.resolution if gt is not None else None
        self.field = CentralField(folder, workspace[:2], workspace[2], list(self.robots), device=device,
                                  robot_radius=c["robot_radius"], other_radius=c["other_radius"], n_rays=c["n_rays"],
                                  m=c["m"], ray_max=c["ray_max"], rows=c["rows"], epochs=c["epochs"],
                                  warmup_epochs=c["warmup_epochs"], frame_buffer=max(c["frame_buffer"], 3 * len(self.robots)),
                                  grid_res=c["grid_res"], tensorboard=c["tensorboard"], diag_every=c["diag_every"],
                                  fourier_scale=c["fourier_scale"])
        self.voxels = VoxelMap(c["voxel_res"])
        self.people = np.zeros((0, 2))
        self.people_trails = {}
        self.requests = {}
        self.events = []
        self.turn = 0
        self.shared = {}
        self.stuck_requests = {}
        self.vlock = threading.Lock()
        self._geo = None
        self._fp = footprint_points()
        self.lock = threading.RLock()
        self.episodes = []
        self.n_episodes = 0
        self.rec = {}
        self.t = 0.0
        self._gt_mask = None
        if gt is not None:
            g = self.field.grid
            w = g.world(*np.nonzero(np.ones((g.n, g.n), bool)))
            r, cc = gt.world_to_grid(w[:, 0], w[:, 1])
            r, cc = np.floor(r).astype(int), np.floor(cc).astype(int)
            ok = (r >= 0) & (r < gt.height) & (cc >= 0) & (cc < gt.width)
            m = np.zeros(len(w), bool)
            m[ok] = gt.data[r[ok], cc[ok]]
            self._gt_mask = m.reshape(g.n, g.n)

    # ------------------------------------------------------------- shared state
    def positions(self, exclude=None):
        return {k: r.pose[:2] for k, r in self.robots.items() if r.pose is not None and k != exclude}

    def others_of(self, rid):
        pts = [np.asarray(p) for p in self.positions(rid).values()] + list(self.people)
        return np.asarray(pts, float).reshape(-1, 2)

    def nearest_other(self, rid, within=6.0):
        o = self.others_of(rid)
        r = self.robots[rid]
        if not len(o) or r.pose is None:
            return None
        d = np.linalg.norm(o - np.asarray(r.pose[:2]), axis=1)
        k = int(np.argmin(d))
        return tuple(o[k]) if d[k] < within else None

    def event(self, text):
        with self.lock:
            self.events.append((round(self.t, 2), text))
            self.events = self.events[-50:]
        with open(os.path.join(self.folder, "events.log"), "a") as f:
            f.write(f"{self.t:9.2f}  {text}\n")

    def gt_clearance(self, xy):
        if self.gt is None:
            return float("nan")
        r, c = self.gt.world_to_index(*xy[:2])
        return float(self.gt_edt[r, c]) if self.gt.inside(r, c) else 0.0

    def set_goal(self, rid, goal, name=""):
        r = self.robots[rid]
        r.goal_name = name
        with self.lock:
            r.goal = None if goal is None else tuple(goal)
            r.path, r.viewpoint = None, None
            r.ctrl.reset()
            if goal is not None:
                r.state = "navigating"
                self.n_episodes += 1
                r.episode = {"id": self.n_episodes, "robot": rid, "goal": list(goal), "t0": self.t,
                             "start": list(r.pose) if r.pose is not None else None, "min_wall": 10.0,
                             "min_people": 10.0, "min_robot": 10.0, "dist": 0.0, "last": None,
                             "stats0": dict(r.stats)}

    # ------------------------------------------------------------- one call of the model (round-robin)
    def forecast(self, rid, ahead=1.0):
        """Where robot rid will be in `ahead` s along its shared plan (its position when it has none)."""
        r = self.robots[rid]
        if r.pose is None:
            return None
        if not r.path or len(r.path) < 2:
            return tuple(r.pose[:2])
        from .pure_pursuit import lookahead_point
        return tuple(lookahead_point(r.path, r.pose[0], r.pose[1], self.cfg["max_v"] * ahead))

    def other_for_plan(self, rid, within=6.0):
        """The model's other-agent input for rid's plan: nearest person (now) or robot (where its plan takes it)."""
        r = self.robots[rid]
        cands = [np.asarray(self.forecast(k)) for k in self.robots if k != rid and self.robots[k].pose is not None]
        cands += list(self.people)
        if not cands or r.pose is None:
            return None
        c = np.asarray(cands, float).reshape(-1, 2)
        d = np.linalg.norm(c - np.asarray(r.pose[:2]), axis=1)
        k = int(np.argmin(d))
        return tuple(c[k]) if d[k] < within else None

    def round(self):
        """One call: the next agent in turn (1, 2, ..., N, 1, ...) shares x, y, lidar and plan; the one model trains on
        it and plans for it. N calls make one cycle."""
        ids = list(self.robots)
        rid = None
        for _ in range(len(ids)):
            k = ids[self.turn % len(ids)]
            self.turn += 1
            if self.robots[k].scan is not None and self.robots[k].pose is not None:
                rid = k
                break
        if rid is None:
            return None
        r = self.robots[rid]
        with self.lock:
            scan = r.scan
            robots = self.positions()
            people = self.people.copy()
            self.shared[rid] = {"x": round(r.pose[0], 3), "y": round(r.pose[1], 3), "goal": r.goal_name or r.goal,
                                "viewpoint": r.viewpoint, "path": None if r.path is None else [list(map(float, q)) for q in r.path[::4]],
                                "t": round(self.t, 2)}
        rec = self.field.observe({rid: scan}, robots, people)
        if rec:
            self.rec = rec
        r.scans.append((round(self.t, 2), np.asarray(scan[0], np.float32), np.asarray(scan[1][scan[2]], np.float16)))
        g = self.field.grid
        with self.lock:
            self.requests = {k: q.goal[:2] for k, q in self.robots.items() if q.goal is not None and not g.is_free(q.goal)}
            self.requests.update({f"{k}_stuck": v for k, (t, v) in self.stuck_requests.items()
                                  if self.t - t < 60.0 and self.robots[k].goal is not None})
        self.plan_for(r)
        self._view_cache = self.view()   # the shared map at this call, reused by the agents' own-view videos
        if self.gt is not None and self.field.rounds % 16 == 0:
            self.long_range_check(r)
        return rec

    def long_range_check(self, r, n=200, reach=15.0):
        """Does the model's T know the building? Rank correlation of T(q_r, c | o) with the true shortest-path length
        (G1-traversable cells of the ground-truth map) over mapped free cells c; and how often the agent's last (gx, gy)
        pointed into a wall within 0.5 m."""
        from scipy.sparse import csgraph
        from scipy.stats import spearmanr
        if self._geo is None:
            res = 0.2
            k = int(round(res / self.gt.resolution))
            h, w = self.gt.height // k, self.gt.width // k
            occ = self.gt.data[:h * k, :w * k].reshape(h, k, w, k).any((1, 3))
            ok = (distance_transform_edt(~occ) * res) >= 0.3
            idx = -np.ones(occ.shape, int)
            idx[ok] = np.arange(ok.sum())
            rows, cols, wts = [], [], []
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                c0, c1 = (slice(0, w - dc), slice(dc, w)) if dc >= 0 else (slice(-dc, w), slice(0, w + dc))
                a, b = idx[0:h - dr, c0], idx[dr:h, c1]
                m = (a >= 0) & (b >= 0)
                rows += list(a[m]); cols += list(b[m]); wts += [res * math.hypot(dr, dc)] * int(m.sum())  # noqa: E702
            from scipy.sparse import coo_matrix
            G = coo_matrix((wts, (rows, cols)), shape=(ok.sum(), ok.sum())).tocsr()
            G = G + G.T
            self._geo = (G, idx, k, res)
        G, idx, k, res = self._geo
        cell = lambda xy: idx[min(idx.shape[0] - 1, int((xy[1] - self.gt.origin[1]) / res)),  # noqa: E731
                              min(idx.shape[1] - 1, int((xy[0] - self.gt.origin[0]) / res))]
        src = cell(r.pose)
        if src < 0:
            return
        dist = csgraph.dijkstra(G, indices=src, limit=reach * 1.5)
        cand = self.field.grid.free_cells(r.pose, 1.0, reach, n=n, rng=np.random.default_rng(self.field.rounds), clear=0.35)
        if len(cand) < 20:
            return
        geo = np.array([dist[cell(c)] if cell(c) >= 0 else np.inf for c in cand])
        keep = np.isfinite(geo)
        if keep.sum() < 20:
            return
        T = self.field.travel_times(r.pose[:2], cand[keep], self.other_for_plan(r.id))
        rho = float(spearmanr(T, geo[keep])[0])
        recent = r.call_log[-15:]
        into = [self.gt_clearance((c["pose"][0] + 0.5 * c["g"][0], c["pose"][1] + 0.5 * c["g"][1])) < 0.3 for c in recent]
        rec = {"round": self.field.rounds, "agent": r.id, "geo_spearman": round(rho, 4),
               "g_into_wall": round(float(np.mean(into)), 3) if into else float("nan"), "pairs": int(keep.sum())}
        self.field.log.log("longrange_log", rec, "longrange")
        self.field.diag.update({"geo_spearman": rec["geo_spearman"], "g_into_wall": rec["g_into_wall"]})

    def _progress(self, r):
        """No progress towards the goal over the last 8 s (never in an episode's first 10 s): a frontier/waypoint
        viewpoint is marked tried (excluded for 90 s); if the agent was heading to its goal itself, it takes local
        waypoints for 15 s and then tries the goal again. Either way it asks the others to map around it."""
        if r.goal is None or r.pose is None or r.episode is None or self.t - r.episode["t0"] < 10.0:
            return
        d = math.hypot(r.goal[0] - r.pose[0], r.goal[1] - r.pose[1])
        r.progress.append((self.t, d, r.pose[0], r.pose[1]))
        old = [p for p in r.progress if self.t - p[0] >= 8.0]
        if not old or r.viewpoint is None:
            return
        old = old[-1]
        r.progress = [p for p in r.progress if self.t - p[0] < 8.0]
        if old[1] - d < 0.3 and math.hypot(r.pose[0] - old[2], r.pose[1] - old[3]) < 1.0:
            if r.view_kind in ("goal", "opportunity"):
                r.goal_pause = self.t + 15.0
                self.event(f"{r.id}: no progress towards its goal, local waypoints for 15 s")
            else:
                r.tried.append((self.t, tuple(r.viewpoint)))
                self.event(f"{r.id}: no progress, viewpoint ({r.viewpoint[0]:.1f}, {r.viewpoint[1]:.1f}) marked tried")
            self.stuck_requests[r.id] = (self.t, tuple(r.pose[:2]))

    def plan_for(self, r):
        g = self.field.grid
        coop = self.cfg["mode"] != "greedy"
        helping = r.goal is None and r.helper and bool(self.requests)
        if r.goal is None and r.helper and not helping and not r.explore:
            if r.state == "helping":
                r.state, r.path, r.viewpoint = "done", None, None
                r.label = f"{r.id} done (no open requests)"
            return
        if r.pose is None or (r.goal is None and not r.explore and not helping) or r.state == "explored":
            return
        other = self.other_for_plan(r.id)
        claimed = [q.viewpoint for k, q in self.robots.items() if k != r.id and q.viewpoint is not None] if coop else []
        t0 = time.monotonic()
        reqs = [q for k, q in self.requests.items() if k != r.id] if r.goal is None else []
        self._progress(r)
        tried = [v for t, v in r.tried if self.t - t < 90.0]
        vp, kind = self.field.next_viewpoint(r.pose, r.goal, other, claimed, self.cfg["horizon"],
                                             mode=self.cfg["mode"], requests=reqs, tried=tried,
                                             goal_paused=self.t < r.goal_pause)
        if vp is None:
            r.state, r.path, r.viewpoint = ("done" if r.helper else "explored"), None, None
            return
        path, pk = self.field.plan(r.pose[:2], vp, other)
        if path is not None:
            full = pk == "full"
            path, cut = g.safe_prefix(path, r.pose, self.cfg["plan_clearance"])
            pk = "blocked" if path is None else ("partial" if cut else pk)
            if kind == "goal" and full and path is not None and not cut and g.known_fraction(path) > 0.9:
                kind = "opportunity" if r.view_kind not in ("goal", "opportunity") else "goal"
                if kind == "opportunity":
                    self.event(f"{r.id}: opportunity, the model's path to its goal runs through mapped free space")
        plan_s = time.monotonic() - t0
        gxy, s_here = self.field.heading(r.pose[:2], vp, other)
        r.stats[kind] += 1
        r.stats[pk] += 1
        with self.lock:
            r.gxy, r.s_here = (float(gxy[0]), float(gxy[1])), s_here
            r.viewpoint, r.view_kind = vp, kind
            if path is not None:
                r.path, r.path_kind = [tuple(p) for p in path], pk
            elif r.path is not None:
                i = closest_index(r.path, r.pose[0], r.pose[1])
                rest = r.path[i:]
                rest = g.safe_prefix(rest, r.pose, self.cfg["plan_clearance"])[0] if len(rest) > 1 else None
                r.path = None if rest is None else [tuple(q) for q in rest]
            if r.goal is None:
                r.state = "helping" if helping else "exploring"
            togo = math.hypot(r.goal[0] - r.pose[0], r.goal[1] - r.pose[1]) if r.goal is not None else 0.0
            g_txt = f"g=({r.gxy[0]:+.2f},{r.gxy[1]:+.2f}) S={s_here:.2f}"
            r.label = (f"{r.id} -> {r.goal_name or 'goal'} {togo:.1f} m | {kind} | {g_txt} | path {pk}" if r.goal is not None
                       else f"{r.id} {r.state} | {kind} | {g_txt} | path {pk}")
            r.calls += 1
            r.plan_s = plan_s
        rec_call = {"t": round(self.t, 2), "call": self.turn, "agent": r.id, "pose": [round(v, 3) for v in r.pose],
                    "shared": self.shared.get(r.id), "received": {
                        "other_agent_input": None if other is None else [round(float(v), 3) for v in other],
                        "claimed_viewpoints": [[round(float(v), 2) for v in q] for q in claimed],
                        "requests": {k: [round(float(v), 2) for v in q] for k, q in self.requests.items()},
                        "positions": {k: [round(float(v), 3) for v in q] for k, q in self.positions(r.id).items()},
                        "people": self.people.round(2).tolist()},
                    "goal": r.goal_name or (None if r.goal is None else list(r.goal)), "viewpoint": [round(float(v), 3) for v in vp],
                    "kind": kind, "g": [round(r.gxy[0], 4), round(r.gxy[1], 4)], "S": round(s_here, 4),
                    "path_kind": pk, "path": None if path is None else [[round(float(a), 3), round(float(b), 3)] for a, b in path],
                    "plan_s": round(plan_s, 3), "train": {k: self.rec.get(k) for k in ("loss", "diff", "train_s", "epoch")}}
        r.call_log.append(rec_call)
        d = os.path.join(self.folder, "agents", r.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "calls.jsonl"), "a") as f:
            f.write(json.dumps(rec_call) + "\n")
        new = not os.path.exists(os.path.join(self.folder, "calls.csv"))
        with open(os.path.join(self.folder, "calls.csv"), "a") as f:
            if new:
                f.write("t,call,agent,x,y,gx,gy,S,viewpoint_x,viewpoint_y,kind,path,path_points,other_x,other_y,goal\n")
            o = other if other is not None else (float("nan"), float("nan"))
            f.write(f"{self.t:.2f},{self.turn},{r.id},{r.pose[0]:.3f},{r.pose[1]:.3f},{r.gxy[0]:.4f},{r.gxy[1]:.4f},"
                    f"{s_here:.3f},{vp[0]:.3f},{vp[1]:.3f},{kind},{pk},{0 if path is None else len(path)},"
                    f"{o[0]:.3f},{o[1]:.3f},{r.goal_name}\n")

    # ------------------------------------------------------------- control of one robot
    def control(self, rid, dt):
        """Returns (cmd, event) with event in (None, 'succeeded')."""
        c = self.cfg
        r = self.robots[rid]
        with self.lock:
            pose, path, goal, scan = r.pose, r.path, r.goal, r.scan
        if pose is None or (goal is None and r.state not in ("exploring", "helping")):
            return (0.0, 0.0, 0.0), None
        event = None
        if goal is not None and math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < c["goal_tolerance"]:
            dyaw = wrap(goal[2] - pose[2]) if len(goal) > 2 else 0.0
            if abs(dyaw) < c["yaw_tolerance"]:
                cmd, event = (0.0, 0.0, 0.0), "succeeded"
            else:
                cmd = (0.0, 0.0, max(-c["max_w"], min(c["max_w"], 1.5 * dyaw)))
        elif path and path_length(path[closest_index(path, pose[0], pose[1]):]) > 0.25:
            dist = math.hypot(goal[0] - pose[0], goal[1] - pose[1]) if goal is not None else None
            cmd = r.ctrl.command(pose, path, dt=dt, distance_to_goal=dist)
        elif r.gxy is not None and any(r.gxy):
            # plan used up before the agent's next call: hold position, only turn towards (gx, gy)
            err = wrap(math.atan2(r.gxy[1], r.gxy[0]) - pose[2])
            cmd = (0.0, 0.0, max(-c["max_w"], min(c["max_w"], 1.5 * err)) if abs(err) > 0.15 else 0.0)
        else:
            cmd = (0.0, 0.0, 0.4)
        pts = np.zeros((0, 2))
        if scan is not None:
            pts = scan[1][scan[2]]
            if scan[3] is not None and len(scan[3]):
                pts = np.vstack((pts, scan[3][:, :2]))
        cmd, r.clear = safe_velocity(cmd, pose, pts, self.others_of(rid), robot_r=c["robot_radius"],
                                     margin=c["guard_margin"] + (0.05 if r.kind == "sim" else 0.0),
                                     horizon=c["guard_horizon"], max_vy=c["max_vy"], physical=r.kind == "sim",
                                     min_move=c["min_vx"] if r.kind == "sim" else 0.05, disc_r=0.40)
        r.cmd = cmd
        self._track(r)
        if event:
            self.finish(rid, True)
        return cmd, event

    def _contact(self, r, touching):
        """Contact bookkeeping: one event when an agent starts touching a wall, its duration when it stops."""
        if touching and not r.in_contact:
            r.contacts.append({"t": round(self.t, 2), "x": round(r.pose[0], 3), "y": round(r.pose[1], 3),
                               "yaw": round(r.pose[2], 3), "duration_s": 0.0, "cmd": [round(v, 2) for v in r.cmd]})
            self.event(f"{r.id}: wall contact at ({r.pose[0]:.2f}, {r.pose[1]:.2f})")
        elif touching and r.contacts:
            r.contacts[-1]["duration_s"] = round(self.t - r.contacts[-1]["t"], 2)
        r.in_contact = touching

    def _track(self, r):
        ep = r.episode
        if r.pose is None:
            return
        if not r.trail or self.t - r.trail[-1][0] >= 0.2:
            r.trail.append((self.t, r.pose[0], r.pose[1], r.pose[2]))
            if r.body is None and self.gt is not None:
                self._contact(r, footprint_hits(self.gt, r.pose, self._fp) > 0)
        p0 = np.asarray(r.pose[:2])
        close = [(k, float(np.linalg.norm(np.asarray(q) - p0))) for k, q in self.positions(r.id).items()]
        close += [(f"person{i}", float(np.linalg.norm(q - p0))) for i, q in enumerate(self.people)]
        for k, d in close:
            lim = 0.60 if not k.startswith("person") else 0.55
            if d < lim and (not r.agent_contacts or r.agent_contacts[-1]["with"] != k or self.t - r.agent_contacts[-1]["t"] > 2.0):
                r.agent_contacts.append({"t": round(self.t, 2), "with": k, "distance": round(d, 3),
                                         "x": round(r.pose[0], 3), "y": round(r.pose[1], 3)})
                self.event(f"{r.id}: too close to {k} ({d:.2f} m between centres)")
        if ep is None:
            return
        p = np.asarray(r.pose[:2])
        if ep["last"] is not None:
            ep["dist"] += float(np.linalg.norm(p - ep["last"]))
        ep["last"] = p
        ep["min_wall"] = min(ep["min_wall"], self.gt_clearance(p))
        if len(self.people):
            ep["min_people"] = min(ep["min_people"], float(np.linalg.norm(self.people - p, axis=1).min()))
        o = self.positions(r.id)
        if o:
            ep["min_robot"] = min(ep["min_robot"], float(min(np.linalg.norm(np.asarray(q) - p) for q in o.values())))

    def finish(self, rid, ok, why=""):
        r = self.robots[rid]
        ep = r.episode
        with self.lock:
            r.goal, r.path, r.viewpoint = None, None, None
            r.state = "succeeded" if ok else ("failed " + why).strip()
            r.episode = None
            r.helper = ok and self.cfg["help"]
        if ep is None:
            return None
        self._close_view(r, ok)
        self.event(f"{rid}: {'reached' if ok else 'failed'} {r.goal_name or ep['goal']} {why}".strip())
        dur = self.t - ep["t0"]
        res = {"episode": ep["id"], "robot": rid, "success": ok, "reason": why, "goal": ep["goal"], "start": ep["start"],
               "time_s": round(dur, 2), "path_m": round(ep["dist"], 2), "mean_speed": round(ep["dist"] / max(dur, 1e-6), 3),
               "min_wall_clearance_m": round(ep["min_wall"], 3), "min_people_m": round(ep["min_people"], 3),
               "min_robot_m": round(ep["min_robot"], 3),
               "contacts": sum(1 for c in r.contacts if c["t"] >= ep["t0"]),
               "viewpoints": {k: r.stats[k] - ep["stats0"].get(k, 0) for k in r.stats},
               "rounds": self.field.rounds, "math": self.field.diag, "mode": self.cfg["mode"],
               "goal_name": r.goal_name}
        self.episodes.append(res)
        with open(os.path.join(self.folder, "episodes.json"), "w") as f:
            json.dump(self.episodes, f, indent=2)
        threading.Thread(target=self.save, args=(f"ep{ep['id']:02d}_{rid}_{'success' if ok else 'fail'}",), daemon=True).start()
        return res

    # ------------------------------------------------------------- kinematic robots
    def step_kin(self, dt, beams=360):
        with self.lock:
            people = self.people.copy()
        for r in self.robots.values():
            if r.body is None:
                continue
            cmd, _ = self.control(r.id, dt)
            others = [(p[0], p[1], 0.30) for k, p in self.positions(r.id).items()] + [(p[0], p[1], 0.25) for p in people]
            moving = abs(cmd[0]) + abs(cmd[1]) + abs(cmd[2]) > 1e-3
            hit = r.body.step(cmd, dt, others) and moving
            if hit and footprint_hits(self.gt, r.body.pose, r.body.fp) == 0:
                hit = False  # held back by an agent: a stop, not a contact
                r.yields += 1
            with self.lock:
                r.pose = tuple(r.body.pose)
            self._contact(r, hit)

    def scan_kin(self):
        for r in self.robots.values():
            if r.body is None:
                continue
            discs = self.others_of(r.id)
            org, ends, hit = r.body.scan(discs, rmax=self.cfg["ray_max"] + 2)
            with self.lock:
                r.scan = (org, ends, hit, None)
            self.voxels.extrude(ends[hit][::2])

    def set_people(self, pos):
        with self.lock:
            self.people = np.asarray(pos, float).reshape(-1, 2)
            for i, p in enumerate(self.people):
                tr = self.people_trails.setdefault(i, [])
                if not tr or np.linalg.norm(np.asarray(tr[-1]) - p) > 0.2:
                    tr.append(tuple(p))

    # ------------------------------------------------------------- pictures and files
    def view(self, title=""):
        with self.lock:
            agents = {k: {"pose": r.pose, "trail": [t[1:3] for t in r.trail[-3000:]] or None, "path": r.path,
                          "viewpoint": r.viewpoint, "goal": r.goal, "label": r.label, "g": r.gxy} for k, r in self.robots.items()}
            people = self.people.copy()
            reqs = list(self.requests.values())
        return render_fleet(self.field.grid, agents, people, self._gt_mask, title=f"[{self.cfg['mode']}] {title}",
                            requests=reqs)

    def status(self):
        with self.lock:
            return {"t": round(self.t, 2), "round": self.field.rounds, "train": self.rec, "math": self.field.diag,
                    "coverage": self.field.grid.coverage(), "people": self.people.round(2).tolist(),
                    "robots": {k: {"pose": None if r.pose is None else [round(v, 3) for v in r.pose],
                                   "state": r.state, "goal": r.goal, "viewpoint": r.viewpoint, "view": r.view_kind,
                                   "path_kind": r.path_kind, "path": None if r.path is None else
                                   [[round(x, 2), round(y, 2)] for x, y in r.path[::3]],
                                   "cmd": [round(v, 2) for v in r.cmd], "clearance": round(r.clear, 2),
                                   "stats": r.stats, "plan": r.label, "kind": r.kind, "calls": r.calls,
                                   "g": r.gxy, "S": round(r.s_here, 3), "goal_name": r.goal_name,
                                   "helper": r.helper} for k, r in self.robots.items()},
                    "mode": self.cfg["mode"], "requests": {k: list(v) for k, v in self.requests.items()},
                    "events": self.events[-8:], "episodes": len(self.episodes),
                    "turn": list(self.robots)[(self.turn - 1) % len(self.robots)] if self.turn else None,
                    "shared": self.shared}

    def ego_views(self, size=400, span=16.0):
        """Each agent's own view: the shared map around it, rotated so it faces up, with its lidar, plan and (gx, gy)."""
        base = self._view_cache if getattr(self, "_view_cache", None) is not None else self.view()
        g = self.field.grid
        px = base.shape[0]
        sc = px / (g.n * g.res)
        out = {}
        for i, (k, r) in enumerate(self.robots.items()):
            if r.pose is None:
                continue
            P = lambda xy: np.column_stack(((np.asarray(xy)[:, 0] - g.origin[0]) * sc,  # noqa: E731
                                            px - (np.asarray(xy)[:, 1] - g.origin[1]) * sc))
            c = P(np.asarray([r.pose[:2]]))[0]
            M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), 90.0 - math.degrees(r.pose[2]), size / (span * sc))
            M[:, 2] += np.array([size / 2, size / 2]) - c
            img = cv2.warpAffine(base, M, (size, size), borderValue=(90, 90, 90))
            T = lambda xy: (np.hstack((P(xy), np.ones((len(xy), 1)))) @ M.T).astype(np.int32)  # noqa: E731
            if r.scan is not None and r.scan[2].any():
                for q in T(r.scan[1][r.scan[2]]):
                    cv2.circle(img, tuple(q), 2, (255, 0, 255), -1)
            if r.path:
                cv2.polylines(img, [T(np.asarray(r.path))], False, (0, 200, 255), 2)
            if r.gxy is not None:
                tip = T(np.asarray([[r.pose[0] + 1.5 * r.gxy[0], r.pose[1] + 1.5 * r.gxy[1]]]))[0]
                cv2.arrowedLine(img, (size // 2, size // 2), tuple(tip), (0, 0, 0), 2, tipLength=0.3)
            cv2.rectangle(img, (0, 0), (size, 40), (35, 35, 35), -1)
            cv2.putText(img, f"{k}  t {self.t:6.1f}  calls {r.calls}  contacts {len(r.contacts)}", (6, 15), 0, 0.45,
                        (255, 255, 255), 1)
            cv2.putText(img, (r.label or r.state)[:62], (6, 33), 0, 0.4, (200, 230, 255), 1)
            out[k] = img
        return out

    def record_views(self):
        """Each agent's own view, written to agents/<id>/view_epNN while it has a goal; kept as _success / _fail."""
        with self.vlock:
            for k, img in self.ego_views().items():
                r = self.robots[k]
                if r.episode is None:
                    continue
                v = r.ep_video
                if v is None or v["ep"] != r.episode["id"]:
                    d = os.path.join(self.folder, "agents", k)
                    os.makedirs(d, exist_ok=True)
                    path = os.path.join(d, f"view_ep{r.episode['id']:02d}_raw.mp4")
                    v = r.ep_video = {"ep": r.episode["id"], "path": path, "w": cv2.VideoWriter(
                        path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (img.shape[1], img.shape[0]))}
                v["w"].write(img)

    def _close_view(self, r, ok):
        with self.vlock:
            v, r.ep_video = r.ep_video, None
        if v is None:
            return
        v["w"].release()
        dst = v["path"].replace("_raw.mp4", f"_{'success' if ok else 'fail'}.mp4")
        threading.Thread(target=self._h264, args=(v["path"], dst), daemon=True).start()

    @staticmethod
    def _h264(src, dst):
        import shutil
        import subprocess
        if shutil.which("ffmpeg") and subprocess.call(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:v", "libx264",
                                                       "-pix_fmt", "yuv420p", dst]) == 0:
            os.remove(src)
        else:
            os.replace(src, dst)

    def close_views(self):
        for r in self.robots.values():
            self._close_view(r, False)

    def save_agents(self):
        g = self.field.grid
        out = {}
        for i, (k, r) in enumerate(self.robots.items()):
            d = os.path.join(self.folder, "agents", k)
            os.makedirs(d, exist_ok=True)
            tr = np.asarray(r.trail).reshape(-1, 4)
            np.savetxt(os.path.join(d, "trajectory.csv"), tr, delimiter=",", header="t,x,y,yaw", comments="", fmt="%.3f")
            with open(os.path.join(d, "contacts.csv"), "w") as f:
                f.write("t,x,y,yaw,duration_s,cmd_vx,cmd_vy,cmd_wz\n")
                for c in r.contacts:
                    f.write(f"{c['t']},{c['x']},{c['y']},{c['yaw']},{c['duration_s']},{c['cmd'][0]},{c['cmd'][1]},{c['cmd'][2]}\n")
            json.dump(r.agent_contacts, open(os.path.join(d, "agent_proximity.json"), "w"), indent=2)
            if r.scans:
                np.savez_compressed(os.path.join(d, "scans.npz"), t=np.array([q[0] for q in r.scans]),
                                    origin=np.array([q[1] for q in r.scans]),
                                    hits=np.array([q[2] for q in r.scans], dtype=object))
            cov = np.full((g.n, g.n, 3), 110, np.uint8)
            cov[g.free] = (235, 235, 235)
            cov[g.seen[i] & g.free] = color(i)
            cov[g.occ] = (20, 20, 20)
            cov = cv2.resize(np.flipud(cov), (700, 700), interpolation=cv2.INTER_NEAREST)
            cv2.putText(cov, f"cells {k} added to the shared map: {g.seen[i].sum() * g.res ** 2:.0f} m2", (6, 18), 0, 0.5,
                        (0, 0, 0), 1)
            cv2.imwrite(os.path.join(d, "coverage.png"), cov)
            eps = [e for e in self.episodes if e["robot"] == k]
            json.dump(eps, open(os.path.join(d, "episodes.json"), "w"), indent=2)
            ts = [c["t"] for c in r.call_log]
            walls = [self.gt_clearance(q[1:3]) for q in r.trail] if self.gt is not None and r.trail else []
            summ = {"agent": k, "kind": r.kind, "goal": r.goal_name, "calls": r.calls,
                    "mean_s_between_own_calls": round(float(np.diff(ts).mean()), 2) if len(ts) > 1 else None,
                    "mean_plan_s": round(float(np.mean([c["plan_s"] for c in r.call_log])), 3) if r.call_log else None,
                    "contacts": len(r.contacts), "contact_s": round(sum(c["duration_s"] for c in r.contacts), 2),
                    "agent_contacts": len(r.agent_contacts), "yields": r.yields,
                    "min_wall_clearance_center_m": round(float(min(walls)), 3) if walls else None,
                    "path_m": round(float(np.linalg.norm(np.diff(tr[:, 1:3], axis=0), axis=1).sum()), 2) if len(tr) > 1 else 0.0,
                    "map_cells_m2": round(float(g.seen[i].sum() * g.res ** 2), 1),
                    "viewpoints": {q: r.stats[q] for q in ("goal", "opportunity", "frontier", "waypoint", "request")},
                    "paths": {q: r.stats[q] for q in ("full", "partial", "stuck", "blocked", "nan")},
                    "episodes": [{"goal": e.get("goal_name") or e["goal"], "success": e["success"], "time_s": e["time_s"],
                                  "path_m": e["path_m"]} for e in eps]}
            json.dump(summ, open(os.path.join(d, "summary.json"), "w"), indent=2)
            out[k] = summ
        return out

    def save(self, tag="latest"):
        out = os.path.join(self.folder, "maps")
        os.makedirs(out, exist_ok=True)
        with self.lock:
            g = self.field.grid
            trails = {k: np.asarray(r.trail) for k, r in self.robots.items()}
            robots = {k: r for k, r in self.robots.items()}
            people_tr = {i: np.asarray(t) for i, t in self.people_trails.items()}
        fleet = self.view(f"{tag}  round {self.field.rounds}  known {g.coverage()['known_m2']:.0f} m2")
        panels = []
        fields = {}
        for k, r in robots.items():
            if r.pose is None:
                continue
            S, T, f0 = self.field.field_maps(r.pose[:2], self.nearest_other(k))
            fields[k] = (S, T)
            p = field_panel(S, T, 450)
            cv2.putText(p, f"field of {k} (nearest agent as input)", (6, 440), 0, 0.5, (255, 255, 255), 1)
            panels.append(p)
        cv2.imwrite(os.path.join(out, f"{tag}_fleet.png"), fleet)
        if panels:
            cv2.imwrite(os.path.join(out, f"{tag}_fields.png"), np.vstack(panels))
        np.savez_compressed(os.path.join(out, f"{tag}.npz"), logodds=g.logodds, seen=g.seen, robots=np.array(g.robots),
                            grid_origin=g.origin, grid_res=g.res,
                            **{f"speed_{k}": v[0] for k, v in fields.items()},
                            **{f"time_{k}": v[1] for k, v in fields.items()},
                            **{f"trail_{k}": v for k, v in trails.items()},
                            **{f"person_{i}": v for i, v in people_tr.items()})
        traj = os.path.join(self.folder, "trajectories")
        os.makedirs(traj, exist_ok=True)
        np.savez_compressed(os.path.join(traj, "trajectories.npz"), **{k: v for k, v in trails.items()},
                            **{f"person_{i}": v for i, v in people_tr.items()})
        for k, v in trails.items():
            np.savetxt(os.path.join(traj, f"{k}.csv"), v, delimiter=",", header="t,x,y,yaw", comments="", fmt="%.3f")
        cv2.imwrite(os.path.join(traj, "trajectories.png"), fleet)
        n3 = self.voxels.save(os.path.join(self.folder, "map3d"), {k: v[:, 1:3] for k, v in trails.items() if len(v)},
                              [v for v in people_tr.values() if len(v)])
        cv2.imwrite(os.path.join(out, "latest_fleet.png"), fleet)
        self.save_agents()
        return {"fleet": os.path.join(out, f"{tag}_fleet.png"), "voxels": n3}


def summarize(folder):
    """results.md from config, logs and episodes of one run."""
    import csv
    cfg = json.load(open(os.path.join(folder, "config.json")))
    eps = json.load(open(os.path.join(folder, "episodes.json"))) if os.path.exists(os.path.join(folder, "episodes.json")) else []
    rows = list(csv.DictReader(open(os.path.join(folder, "train_log.csv"))))
    diag = list(csv.DictReader(open(os.path.join(folder, "math_log.csv"))))
    f = lambda v: f"{float(v):.4g}" if v not in ("", None) else "-"  # noqa: E731
    lines = [f"# Run {os.path.basename(folder)}", "", f"robots: {', '.join(cfg['robots'])}; workspace {cfg['size_m']} m "
             f"at {cfg['center']}; dmin/dmax {cfg['dmin_m']:.2f}/{cfg['dmax_m']:.2f} m; rounds {len(rows)}", ""]
    if rows:
        k = max(1, len(rows) // 8)
        lines += ["## Training (one model, all robots per round)", "", "| round | loss | eikonal diff | rows | robots | "
                  "people | min robot gap | min people gap | train s |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows[::k] + ([rows[-1]] if (len(rows) - 1) % k else []):
            lines.append(f"| {r['round']} | {f(r['loss'])} | {f(r['diff'])} | {r['rows']} | {r['robots']} | {r['people']} | "
                         f"{f(r['min_robot_gap'])} | {f(r['min_people_gap'])} | {f(r['train_s'])} |")
    if diag:
        k = max(1, len(diag) // 8)
        lines += ["", "## Math checks (on each new frame before training on it)", "",
                  "| round | eikonal residual | S vs S* MAE | S vs S* corr | T symmetry err | S near agent / S alone | "
                  "target ratio | T / distance |", "|---|---|---|---|---|---|---|---|"]
        for r in diag[::k] + ([diag[-1]] if (len(diag) - 1) % k else []):
            lines.append(f"| {r['round']} | {f(r['eikonal_residual'])} | {f(r['speed_mae'])} | {f(r['speed_corr'])} | "
                         f"{f(r['symmetry_err'])} | {f(r['agent_ratio'])} | {f(r['agent_target_ratio'])} | "
                         f"{f(r['time_dist_ratio'])} |")
    if eps:
        lines += ["", "## Episodes", "", "| # | robot | success | time s | path m | mean m/s | min wall m | min person m | "
                  "min robot m | wall contacts | viewpoints (goal/frontier) | paths (full/partial/stuck/blocked) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in eps:
            v = e["viewpoints"]
            lines.append(f"| {e['episode']} | {e['robot']} | {e['success']} | {e['time_s']} | {e['path_m']} | {e['mean_speed']} | "
                         f"{e['min_wall_clearance_m']} | {e['min_people_m']} | {e['min_robot_m']} | {e.get('contacts', '-')} | "
                         f"{v['goal']}/{v['frontier']} | "
                         f"{v['full']}/{v['partial']}/{v['stuck']}/{v['blocked']} |")
    ag = os.path.join(folder, "agents")
    if os.path.isdir(ag):
        lines += ["", "## Every agent", "", "| agent | goal | reached | calls | s between own calls | plan s/call | wall contacts | agent contacts | "
                  "contact s | min wall (centre) m | path m | map cells m2 | viewpoints goal/opportunity/frontier/waypoint/request |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(os.listdir(ag)):
            p = os.path.join(ag, k, "summary.json")
            if not os.path.exists(p):
                continue
            a = json.load(open(p))
            v = a["viewpoints"]
            ok = ", ".join(str(e["success"]) for e in a["episodes"]) or "-"
            lines.append(f"| {k} | {a['goal'] or '-'} | {ok} | {a['calls']} | {a['mean_s_between_own_calls']} | {a['mean_plan_s']} | "
                         f"{a['contacts']} | {a.get('agent_contacts', '-')} | {a['contact_s']} | {a['min_wall_clearance_center_m']} | {a['path_m']} | "
                         f"{a['map_cells_m2']} | {v['goal']}/{v['opportunity']}/{v['frontier']}/{v.get('waypoint', 0)}/{v['request']} |")
    open(os.path.join(folder, "results.md"), "w").write("\n".join(lines) + "\n")
    return os.path.join(folder, "results.md")
