"""2-D occupancy grid in the map frame, independent of ROS message lifetimes."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt


@dataclass
class GridMap:
    data: np.ndarray            # bool, shape (height, width), row 0 = minimum y
    resolution: float
    origin: tuple[float, float]  # world position of the lower-left cell corner
    yaw: float = 0.0

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=bool)
        self.resolution = float(self.resolution)
        self.origin = (float(self.origin[0]), float(self.origin[1]))
        self.yaw = float(self.yaw)

    @property
    def height(self):
        return int(self.data.shape[0])

    @property
    def width(self):
        return int(self.data.shape[1])

    def world_to_grid(self, x, y):
        dx, dy = x - self.origin[0], y - self.origin[1]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        gx = c * dx + s * dy
        gy = -s * dx + c * dy
        return gy / self.resolution, gx / self.resolution

    def grid_to_world(self, row, col):
        gx = (col + 0.5) * self.resolution
        gy = (row + 0.5) * self.resolution
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.origin[0] + c * gx - s * gy,
                self.origin[1] + s * gx + c * gy)

    def corner_to_world(self, row, col):
        gx = col * self.resolution
        gy = row * self.resolution
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.origin[0] + c * gx - s * gy,
                self.origin[1] + s * gx + c * gy)

    def world_to_index(self, x, y):
        r, c = self.world_to_grid(x, y)
        return int(math.floor(r)), int(math.floor(c))

    def inside(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

    def occupied(self, x, y):
        r, c = self.world_to_index(x, y)
        if not self.inside(r, c):
            return True
        return bool(self.data[r, c])

    def clearance(self):
        return distance_transform_edt(~self.data) * self.resolution

    def inflated(self, clearance):
        if clearance <= 0.0:
            return self.data.copy()
        return self.clearance() < float(clearance)

    def subgrid(self, cx, cy, radius):
        radius = float(radius)
        gr, gc = self.world_to_grid(cx, cy)
        n = max(2, int(math.ceil(2.0 * radius / self.resolution)))
        r0 = int(math.floor(gr - n / 2.0))
        c0 = int(math.floor(gc - n / 2.0))
        rows = np.arange(r0, r0 + n)
        cols = np.arange(c0, c0 + n)
        out = np.ones((n, n), dtype=bool)
        rv = (rows >= 0) & (rows < self.height)
        cv = (cols >= 0) & (cols < self.width)
        if rv.any() and cv.any():
            out[np.ix_(rv, cv)] = self.data[np.ix_(rows[rv], cols[cv])]
        ox, oy = self.corner_to_world(r0, c0)
        return GridMap(out, self.resolution, (ox, oy), self.yaw)

    def nearest_free(self, x, y, max_radius=3.0, min_clearance=None):
        if not self.occupied(x, y):
            if min_clearance is None or self.clearance_at(x, y) >= min_clearance:
                return (x, y)
        clear = self.clearance()
        r0, c0 = self.world_to_index(x, y)
        for radius in (1.0, 2.0, max(3.0, max_radius)):
            n = max(2, int(radius / self.resolution))
            rr = np.arange(max(0, r0 - n), min(self.height, r0 + n + 1))
            cc = np.arange(max(0, c0 - n), min(self.width, c0 + n + 1))
            if rr.size == 0 or cc.size == 0:
                continue
            sub = clear[np.ix_(rr, cc)]
            if min_clearance is not None:
                ok = sub >= min_clearance
            else:
                ok = ~self.data[np.ix_(rr, cc)]
            if ok.any():
                ii, jj = np.nonzero(ok)
                px, py = self.grid_to_world(rr[ii], cc[jj])
                d = (px - x) ** 2 + (py - y) ** 2
                k = int(np.argmin(d))
                return (float(px[k]), float(py[k]))
        return None

    def clearance_at(self, x, y):
        r, c = self.world_to_index(x, y)
        if not self.inside(r, c):
            return 0.0
        return float(distance_transform_edt(~self.data)[r, c] * self.resolution)

    def coarsen(self, factor):
        factor = max(1, int(factor))
        if factor == 1:
            return self
        h, w = self.height, self.width
        ph = (-h) % factor
        pw = (-w) % factor
        data = np.pad(self.data, ((0, ph), (0, pw)), constant_values=True)
        data = data.reshape(data.shape[0] // factor, factor,
                            data.shape[1] // factor, factor).max(axis=(1, 3))
        return GridMap(data, self.resolution * factor, self.origin, self.yaw)

    @classmethod
    def from_yaml(cls, path, occupied_thresh=0.65, free_thresh=0.25, unknown_is_occupied=True):
        with open(path) as f:
            meta = yaml.safe_load(f)
        image = meta["image"]
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(os.path.abspath(path)), image)
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"map image not found: {image}")
        img = np.flipud(img).astype(np.float32)
        prob = img / 255.0 if int(meta.get("negate", 0)) else (255.0 - img) / 255.0
        occupied = prob > float(meta.get("occupied_thresh", occupied_thresh))
        free = prob < float(meta.get("free_thresh", free_thresh))
        data = occupied | (~free & ~occupied) if unknown_is_occupied else occupied
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        return cls(data, float(meta["resolution"]), (origin[0], origin[1]), float(origin[2]))

    @classmethod
    def from_ros(cls, msg, unknown_is_occupied=True):
        h, w = int(msg.info.height), int(msg.info.width)
        raw = np.asarray(msg.data, dtype=np.int8).reshape(h, w)
        occupied = raw >= 50
        data = occupied | (raw < 0) if unknown_is_occupied else occupied
        q = msg.info.origin.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        p = msg.info.origin.position
        return cls(data, float(msg.info.resolution), (p.x, p.y), yaw)


def merge(*maps: GridMap) -> GridMap:
    """Overlay same-frame maps (e.g. static + dynamic) by logical OR."""
    base = maps[0]
    data = base.data.copy()
    for m in maps[1:]:
        if m.data.shape != data.shape:
            raise ValueError("cannot merge maps with different shapes")
        data |= m.data
    return GridMap(data, base.resolution, base.origin, base.yaw)
