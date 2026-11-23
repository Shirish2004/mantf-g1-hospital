import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .ros_util import cloud_xy

PANEL_W, PANEL_H, BEV = 520, 293, 586


def _rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _image(msg):
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR)
    if enc in ("32fc1",):
        return buf.view(np.float32).reshape(h, msg.step // 4)[:, :w].copy()
    if enc in ("16uc1", "mono16"):
        return buf.view(np.uint16).reshape(h, msg.step // 2)[:, :w].astype(np.float32) / 1000.0
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(h, msg.step)[:, :w], cv2.COLOR_GRAY2BGR)
    return None


class SensorViewer(Node):
    def __init__(self):
        super().__init__("mantf_sensor_viewer")
        p = self.declare_parameter
        p("rgb_topic", "/g1/head_camera/image_raw")
        p("depth_topic", "/g1/head_camera/depth")
        p("scan_topic", "/scan")
        p("cloud_topic", "/livox/lidar")
        p("base_frame", "pelvis")
        p("bev_range", 8.0)
        p("depth_max", 6.0)
        p("title", "mantf G1 sensors")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.lock = threading.Lock()
        self.rgb = self.depth = None
        self.scan = self.cloud = None
        self.path = self.walkers = None
        self.pose = None
        self.status = {}
        self.rate = {k: [0, time.monotonic(), 0.0] for k in ("rgb", "depth", "scan", "cloud")}
        self.tf = Buffer()
        TransformListener(self.tf, self)
        q = qos_profile_sensor_data
        self.create_subscription(Image, g("rgb_topic"), lambda m: self._set("rgb", _image(m)), q)
        self.create_subscription(Image, g("depth_topic"), lambda m: self._set("depth", _image(m)), q)
        self.create_subscription(LaserScan, g("scan_topic"), self._on_scan, q)
        self.create_subscription(PointCloud2, g("cloud_topic"), self._on_cloud, q)
        self.create_subscription(Odometry, "/sim/ground_truth", self._on_gt, 10)
        self.create_subscription(Path, "/mantf/path", lambda m: setattr(self, "path", [
            (ps.pose.position.x, ps.pose.position.y) for ps in m.poses]), 10)
        self.create_subscription(PointCloud2, "/mantf/walkers", lambda m: setattr(self, "walkers", cloud_xy(m)), q)
        self.create_subscription(String, "/mantf/status", lambda m: setattr(self, "status", json.loads(m.data)), 10)
        self.fleet = {}
        self.create_subscription(String, "/mantf/fleet", lambda m: setattr(self, "fleet", json.loads(m.data)), 10)

    def _tick(self, key):
        r = self.rate[key]
        r[0] += 1
        now = time.monotonic()
        if now - r[1] > 1.0:
            r[2], r[0], r[1] = r[0] / (now - r[1]), 0, now

    def _set(self, key, val):
        if val is None:
            return
        with self.lock:
            setattr(self, key, val)
        self._tick(key)

    def _to_base(self, pts, frame):
        try:
            t = self.tf.lookup_transform(self.g("base_frame"), frame, Time())
        except Exception:
            return pts
        tr = t.transform.translation
        return pts @ _rot(t.transform.rotation).T + np.array([tr.x, tr.y, tr.z])

    def _on_gt(self, m):
        p = m.pose.pose
        yaw = math.atan2(2 * (p.orientation.w * p.orientation.z + p.orientation.x * p.orientation.y),
                         1 - 2 * (p.orientation.y ** 2 + p.orientation.z ** 2))
        self.pose = (p.position.x, p.position.y, yaw)

    def _on_scan(self, m):
        r = np.asarray(m.ranges, dtype=np.float32)
        a = m.angle_min + np.arange(r.size, dtype=np.float32) * m.angle_increment
        ok = np.isfinite(r) & (r > m.range_min) & (r < m.range_max)
        pts = np.column_stack((r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum())))
        self._set("scan", self._to_base(pts, m.header.frame_id))

    def _on_cloud(self, m):
        n = m.width * m.height
        if n == 0:
            return
        off = {f.name: f.offset for f in m.fields}
        raw = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(-1, m.point_step)
        xyz = np.column_stack([raw[:, off[k]:off[k] + 4].copy().view(np.float32).ravel() for k in "xyz"])
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) > 30000:
            xyz = xyz[np.random.default_rng(0).choice(len(xyz), 30000, replace=False)]
        self._set("cloud", self._to_base(xyz, m.header.frame_id))

    def _panel(self, img, label, fps):
        out = np.full((PANEL_H, PANEL_W, 3), 30, np.uint8)
        if img is not None:
            h, w = img.shape[:2]
            s = min(PANEL_W / w, PANEL_H / h)
            r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            y0, x0 = (PANEL_H - r.shape[0]) // 2, (PANEL_W - r.shape[1]) // 2
            out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        else:
            cv2.putText(out, "waiting...", (PANEL_W // 2 - 60, PANEL_H // 2), 0, 0.7, (150, 150, 150), 2)
        cv2.putText(out, f"{label}  {fps:4.1f} Hz", (8, 22), 0, 0.6, (0, 255, 255), 2)
        return out

    def _depth_vis(self, d):
        if d is None:
            return None
        dm = float(self.g("depth_max"))
        v = np.nan_to_num(d, nan=0.0, posinf=0.0)
        u8 = (np.clip(v / dm, 0, 1) * 255).astype(np.uint8)
        col = cv2.applyColorMap(255 - u8, cv2.COLORMAP_TURBO)
        col[v <= 0.01] = 0
        return col

    def _bev(self):
        img = np.full((BEV, BEV, 3), 18, np.uint8)
        rng = float(self.g("bev_range"))
        s = BEV / (2 * rng)
        c = BEV // 2

        def px(x, y):
            return (c - y * s).astype(int), (c - x * s).astype(int)

        for r in range(2, int(rng) + 1, 2):
            cv2.circle(img, (c, c), int(r * s), (50, 50, 50), 1)
        with self.lock:
            cloud, scan = self.cloud, self.scan
        if cloud is not None and len(cloud):
            z = cloud[:, 2]
            keep = (np.abs(cloud[:, 0]) < rng) & (np.abs(cloud[:, 1]) < rng) & (z > -0.75) & (z < 1.2)
            u, v = px(cloud[keep, 0], cloud[keep, 1])
            hue = np.clip((z[keep] + 0.75) / 1.95 * 255, 0, 255).astype(np.uint8)
            col = cv2.applyColorMap(hue.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
            img[np.clip(v, 0, BEV - 1), np.clip(u, 0, BEV - 1)] = col
        if scan is not None and len(scan):
            u, v = px(scan[:, 0], scan[:, 1])
            ok = (u >= 0) & (u < BEV) & (v >= 0) & (v < BEV)
            for a, b in zip(u[ok], v[ok]):
                cv2.circle(img, (int(a), int(b)), 2, (255, 0, 255), -1)
        pose = self.pose
        if pose is not None:
            cy, sy = math.cos(pose[2]), math.sin(pose[2])

            def local(pts):
                pts = np.asarray(pts, float)
                dx, dy = pts[:, 0] - pose[0], pts[:, 1] - pose[1]
                return cy * dx + sy * dy, -sy * dx + cy * dy

            if self.path:
                lx, ly = local(self.path)
                u, v = px(lx, ly)
                cv2.polylines(img, [np.column_stack((u, v)).astype(np.int32)], False, (60, 60, 255), 2)
            for i, (k, r) in enumerate((self.fleet.get("robots") or {}).items()):
                if k == "g1" or not r.get("pose"):
                    continue
                lx, ly = local([r["pose"][:2]])
                u, v = px(lx, ly)
                col = [(230, 120, 30), (30, 160, 250), (200, 60, 200)][(i - 1) % 3]
                cv2.circle(img, (int(u[0]), int(v[0])), int(0.32 * s), col, 2)
                cv2.putText(img, k, (int(u[0]) + 6, int(v[0]) - 6), 0, 0.45, col, 1)
            if self.walkers is not None and len(self.walkers):
                lx, ly = local(self.walkers)
                u, v = px(lx, ly)
                for a, b in zip(u, v):
                    cv2.circle(img, (int(a), int(b)), 2, (0, 255, 0), -1)
        cv2.circle(img, (c, c), int(0.32 * s), (255, 255, 255), 2)
        cv2.line(img, (c, c), (c, c - int(0.7 * s)), (255, 255, 255), 2)
        cv2.putText(img, f"lidar BEV  scan {self.rate['scan'][2]:.1f} Hz  cloud {self.rate['cloud'][2]:.1f} Hz",
                    (8, 22), 0, 0.55, (0, 255, 255), 2)
        cv2.putText(img, "magenta /scan  jet /livox/lidar (height)  red path  green people  rings teammates",
                    (8, BEV - 10), 0, 0.42, (200, 200, 200), 1)
        return img

    def frame(self):
        with self.lock:
            rgb, depth = self.rgb, self.depth
        left = np.vstack((self._panel(rgb, "RGB", self.rate["rgb"][2]),
                          self._panel(self._depth_vis(depth), "depth", self.rate["depth"][2])))
        right = self._bev()
        canvas = np.hstack((left, right))
        st = self.status
        pose = st.get("pose")
        fl = self.fleet or {}
        line = (f"g1 {st.get('state', '-')}  rounds {st.get('frames', 0)}  eikonal {st.get('loss') or 0:.3g}  "
                f"robots {len(fl.get('robots') or {})}  people {len(fl.get('people') or [])}  "
                f"known {(fl.get('coverage') or {}).get('known_m2', 0):.0f} m2  planner {st.get('planner', {})}")
        if pose:
            line += f"  pose {pose[0]:.2f} {pose[1]:.2f} {math.degrees(pose[2]):.0f}deg"
        bar = np.full((28, canvas.shape[1], 3), 45, np.uint8)
        cv2.putText(bar, line, (8, 19), 0, 0.5, (255, 255, 255), 1)
        return np.vstack((canvas, bar))


def main():
    rclpy.init()
    node = SensorViewer()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    title = node.g("title")
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        while True:
            cv2.imshow(title, node.frame())
            k = cv2.waitKey(33) & 0xFF
            if k in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
