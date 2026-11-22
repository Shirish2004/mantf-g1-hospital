"""Global A* route on the static/dynamic occupancy grid."""
from __future__ import annotations

import heapq
import math

import numpy as np

from .occupancy import GridMap

_SQRT2 = math.sqrt(2.0)
_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]


def astar(free, start, goal, max_expand=2_000_000):
    h, w = free.shape
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if not free[sr, sc] or not free[gr, gc]:
        return None

    def heuristic(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (_SQRT2 - 2.0) * min(dr, dc)

    g_score = np.full((h, w), np.inf, dtype=np.float64)
    came = np.full((h, w), -1, dtype=np.int32)
    g_score[sr, sc] = 0.0
    open_heap = [(heuristic(sr, sc), 0.0, sr, sc)]
    closed = np.zeros((h, w), dtype=bool)
    expanded = 0
    while open_heap:
        _, g, r, c = heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        if r == gr and c == gc:
            path = []
            cur = r * w + c
            while cur >= 0:
                pr, pc = divmod(cur, w)
                path.append((pr, pc))
                cur = int(came[pr, pc])
            path.reverse()
            return path
        closed[r, c] = True
        expanded += 1
        if expanded > max_expand:
            return None
        for dr, dc, cost in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc] or closed[nr, nc]:
                continue
            if dr and dc and not (free[r, nc] and free[nr, c]):
                continue
            ng = g + cost
            if ng + 1e-12 < g_score[nr, nc]:
                g_score[nr, nc] = ng
                came[nr, nc] = r * w + c
                heapq.heappush(open_heap, (ng + heuristic(nr, nc), ng, nr, nc))
    return None


def line_of_sight(free, a, b):
    ar, ac = int(a[0]), int(a[1])
    br, bc = int(b[0]), int(b[1])
    n = int(max(abs(br - ar), abs(bc - ac)) * 2) + 1
    for t in np.linspace(0.0, 1.0, n):
        r = int(round(ar + (br - ar) * t))
        c = int(round(ac + (bc - ac) * t))
        if not (0 <= r < free.shape[0] and 0 <= c < free.shape[1]) or not free[r, c]:
            return False
    return True


def smooth_path(free, path):
    if not path or len(path) < 3:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(free, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan_path(grid: GridMap, start_xy, goal_xy, clearance=0.35, downsample=4,
              max_expand=2_000_000):
    """Return a smoothed list of world (x, y) waypoints, or None."""
    coarse = grid.coarsen(downsample)
    blocked = coarse.inflated(clearance)
    free = ~blocked
    start = coarse.world_to_index(*start_xy)
    goal = coarse.world_to_index(*goal_xy)
    start_world = (float(start_xy[0]), float(start_xy[1]))
    goal_world = (float(goal_xy[0]), float(goal_xy[1]))
    if not free[start]:
        alt = coarse.nearest_free(*start_xy, max_radius=3.0, min_clearance=clearance)
        if alt is None:
            return None
        start_world = alt
        start = coarse.world_to_index(*alt)
    if not free[goal]:
        alt = coarse.nearest_free(*goal_xy, max_radius=4.0, min_clearance=clearance)
        if alt is None:
            return None
        goal_world = alt
        goal = coarse.world_to_index(*alt)
    route = astar(free, start, goal, max_expand=max_expand)
    if route is None:
        return None

    fine_free = ~grid.inflated(clearance)
    fine = []
    for r, c in route:
        wx, wy = coarse.grid_to_world(r, c)
        fr, fc = grid.world_to_index(wx, wy)
        if not (0 <= fr < grid.height and 0 <= fc < grid.width):
            continue
        if not fine_free[fr, fc]:
            alt = grid.nearest_free(wx, wy, max_radius=1.0, min_clearance=clearance)
            if alt is None:
                continue
            fr, fc = grid.world_to_index(*alt)
        fine.append((fr, fc))
    if not fine:
        return None
    fine = smooth_path(fine_free, fine)
    points = [grid.grid_to_world(r, c) for r, c in fine]
    points[0] = start_world
    points[-1] = goal_world
    return points


def path_length(points):
    if not points or len(points) < 2:
        return 0.0
    a = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())


def closest_index(points, x, y):
    if not points:
        return 0
    a = np.asarray(points, dtype=float)
    d = (a[:, 0] - x) ** 2 + (a[:, 1] - y) ** 2
    return int(np.argmin(d))


def lookahead_point(points, x, y, distance):
    """Point `distance` along the polyline from its closest point to (x, y)."""
    if not points:
        return (x, y)
    i = closest_index(points, x, y)
    px, py = points[i]
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
