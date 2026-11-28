"""People (dynamic obstacles) walking back and forth on straight lanes cut from the free space of the hospital map.
They pause when a robot stands right in front of them. Published for the fleet (/mantf/agents), the viewer
(/mantf/walkers), RViz (markers) and Isaac (/mantf/people PoseArray, which moves the Isaac characters)."""
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.ndimage import distance_transform_edt
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from .occupancy import GridMap
from .ros_util import cloud_xy, xy_cloud


def make_lanes(grid, box, n, seed=0, clear=0.55, min_len=5.0, max_len=14.0, speed=0.0):
    edt = distance_transform_edt(~grid.data) * grid.resolution
    rng = np.random.default_rng(seed)
    lanes = []
    for _ in range(4000):
        if len(lanes) >= n:
            break
        x, y = rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])
        r, c = grid.world_to_index(x, y)
        if not grid.inside(r, c) or edt[r, c] < clear + 0.2:
            continue
        th = rng.choice([0.0, math.pi / 2]) if rng.random() < 0.8 else rng.uniform(0, math.pi)
        u = np.array([math.cos(th), math.sin(th)])
        ends = []
        for sgn in (1, -1):
            s = 0.0
            while s < max_len / 2:
                q = np.array([x, y]) + sgn * (s + 0.1) * u
                r, c = grid.world_to_index(*q)
                if (not grid.inside(r, c) or edt[r, c] < clear
                        or not (box[0] <= q[0] <= box[2] and box[1] <= q[1] <= box[3])):
                    break
                s += 0.1
            ends.append(np.array([x, y]) + sgn * s * u)
        a, b = ends
        mid = (a + b) / 2
        if np.linalg.norm(b - a) < min_len or any(np.linalg.norm(mid - (l[0] + l[1]) / 2) < 3.0 for l in lanes):
            continue
        v = float(rng.uniform(0.5, 0.9))
        lanes.append((a, b, speed if speed > 0 else v))
    return lanes


class People:
    def __init__(self, lanes, seed=0, yield_distance=1.1):
        rng = np.random.default_rng(seed)
        self.lanes = lanes
        self.s = [float(rng.uniform(0, np.linalg.norm(b - a))) for a, b, _ in lanes]
        self.dir = [1.0 if rng.random() < 0.5 else -1.0 for _ in lanes]
        self.yield_distance = yield_distance
        self.wait = [0.0] * len(lanes)

    def step(self, dt, robots=()):
        robots = np.asarray(robots, float).reshape(-1, 2)
        pos, yaw = [], []
        for i, (a, b, v) in enumerate(self.lanes):
            L = float(np.linalg.norm(b - a))
            u = (b - a) / L
            x = a + u * self.s[i]
            head = u * self.dir[i]
            rel = robots - x
            nxt = a + u * (self.s[i] + self.dir[i] * v * dt)
            # never step closer than 0.8 m to a robot, and yield to a robot ahead; after 4 s of waiting, turn back
            now_d = np.linalg.norm(rel, axis=1) if len(robots) else np.zeros(0)
            nxt_d = np.linalg.norm(robots - nxt, axis=1) if len(robots) else np.zeros(0)
            # a step is blocked only if it brings the person closer than 0.8 m to a robot (stepping apart is free)
            block = bool(np.any((now_d < self.yield_distance) & (rel @ head > 0.0) & (nxt_d < now_d))) or bool(
                np.any((nxt_d < 0.8) & (nxt_d < now_d)))
            self.wait[i] = self.wait[i] + dt if block else 0.0
            if self.wait[i] > 4.0:
                self.dir[i] *= -1
                self.wait[i] = 0.0
            elif not block:
                self.s[i] += self.dir[i] * v * dt
                if not 0.0 <= self.s[i] <= L:
                    self.dir[i] *= -1
                    self.s[i] = min(max(self.s[i], 0.0), L)
            x = a + u * self.s[i]
            pos.append((float(x[0]), float(x[1])))
            yaw.append(math.atan2(head[1], head[0]))
        return np.asarray(pos).reshape(-1, 2), yaw


class Walkers(Node):
    def __init__(self):
        super().__init__("mantf_walkers")
        p = self.declare_parameter
        p("map_yaml", os.environ.get("MANTF_G1_MAP", ""))
        p("box", [6.5, -24.0, 31.0, -5.0])
        p("count", 5)
        p("seed", 3)
        p("speed", 0.0)
        p("radius", 0.25)
        p("rate", 20.0)
        p("frame", "map")
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.r = float(g("radius"))
        self.frame = g("frame")
        lanes = make_lanes(GridMap.from_yaml(g("map_yaml")), g("box"), int(g("count")), int(g("seed")), speed=float(g("speed")))
        self.people = People(lanes, int(g("seed")))
        self.robots = np.zeros((0, 2))
        self.create_subscription(PointCloud2, "/mantf/fleet_poses", lambda m: setattr(self, "robots", cloud_xy(m)),
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(PointCloud2, "/mantf/walkers", 5)
        self.pub_a = self.create_publisher(PointCloud2, "/mantf/agents", 5)
        self.pub_p = self.create_publisher(PoseArray, "/mantf/people", 5)
        self.pub_m = self.create_publisher(MarkerArray, "/mantf/walkers_markers", 5)
        self.last = None
        self.create_timer(1.0 / float(g("rate")), self.tick)
        for a, b, v in lanes:
            self.get_logger().info(f"person lane ({a[0]:.1f},{a[1]:.1f}) -> ({b[0]:.1f},{b[1]:.1f}) at {v:.2f} m/s")

    def tick(self):
        now = self.get_clock().now()
        dt = 0.0 if self.last is None else min(0.2, (now - self.last).nanoseconds * 1e-9)
        self.last = now
        pos, yaw = self.people.step(dt, self.robots)
        stamp = now.to_msg()
        ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        ring = [(x + self.r * math.cos(t), y + self.r * math.sin(t)) for x, y in pos for t in ang] + [tuple(q) for q in pos]
        self.pub.publish(xy_cloud(ring, self.frame, stamp))
        self.pub_a.publish(xy_cloud(pos, self.frame, stamp))
        pa = PoseArray()
        pa.header.frame_id, pa.header.stamp = self.frame, stamp
        ma = MarkerArray()
        for i, ((x, y), th) in enumerate(zip(pos, yaw)):
            ps = Pose()
            ps.position.x, ps.position.y = float(x), float(y)
            ps.orientation.z, ps.orientation.w = math.sin(th / 2), math.cos(th / 2)
            pa.poses.append(ps)
            m = Marker()
            m.header = pa.header
            m.ns, m.id, m.type, m.action = "people", i, Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.85
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2 * self.r
            m.scale.z = 1.7
            m.color.g, m.color.b, m.color.a = 0.75, 0.3, 0.8
            ma.markers.append(m)
        self.pub_p.publish(pa)
        self.pub_m.publish(ma)


def main():
    rclpy.init()
    node = Walkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
