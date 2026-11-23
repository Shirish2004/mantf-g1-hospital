import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_ros import TransformBroadcaster

from .occupancy import GridMap
from .ros_util import cloud_xy


class KinematicSim(Node):
    def __init__(self):
        super().__init__("g1_kinematic_sim")
        p = self.declare_parameter
        p("map_yaml", os.environ.get("MANTF_G1_MAP", ""))
        p("start", [8.93, -12.45, 0.0])
        p("rate", 50.0)
        p("scan_rate", 10.0)
        p("scan_range", 8.0)
        p("beams", 360)
        p("cmd_topics", ["/g1/cmd_vel_nav", "/cmd_vel"])
        p("walkers_topic", "/mantf/walkers")
        p("cmd_timeout", 0.5)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.map = GridMap.from_yaml(g("map_yaml"), unknown_is_occupied=True)
        self.pose = [float(v) for v in g("start")]
        self.cmd, self.cmd_t = (0.0, 0.0, 0.0), 0.0
        self.walkers = np.zeros((0, 2), dtype=np.float32)
        for t in g("cmd_topics"):
            self.create_subscription(Twist, t, self.on_cmd, 10)
        self.create_subscription(PointCloud2, g("walkers_topic"), lambda m: setattr(self, "walkers", cloud_xy(m)),
                                 qos_profile_sensor_data)
        self.pub_gt = self.create_publisher(Odometry, "/sim/ground_truth", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.tf = TransformBroadcaster(self)
        self.dt = 1.0 / float(g("rate"))
        self.create_timer(self.dt, self.step)
        self.create_timer(1.0 / float(g("scan_rate")), self.scan)
        self.get_logger().info(f"kinematic G1 at {self.pose}")

    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.linear.y, m.angular.z)
        self.cmd_t = self.get_clock().now().nanoseconds * 1e-9

    def step(self):
        now = self.get_clock().now()
        vx, vy, wz = self.cmd if now.nanoseconds * 1e-9 - self.cmd_t < self.get_parameter("cmd_timeout").value \
            else (0.0, 0.0, 0.0)
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        nx, ny = x + (c * vx - s * vy) * self.dt, y + (s * vx + c * vy) * self.dt
        if not self.map.occupied(nx, ny):
            x, y = nx, ny
        th = (th + wz * self.dt + math.pi) % (2 * math.pi) - math.pi
        self.pose = [x, y, th]
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = now.to_msg(), "map", "pelvis"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = x, y, 0.78
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = vx, vy, wz
        self.pub_gt.publish(o)
        self.pub_odom.publish(o)
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = o.header.stamp, "odom", "pelvis"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, 0.78
        t.transform.rotation = o.pose.pose.orientation
        self.tf.sendTransform(t)

    def scan(self):
        x, y, th = self.pose
        n, rmax = int(self.get_parameter("beams").value), float(self.get_parameter("scan_range").value)
        ang = np.linspace(-math.pi, math.pi, n, endpoint=False)
        steps = np.arange(0.1, rmax, self.map.resolution * 0.8)
        wx = x + np.cos(ang + th)[:, None] * steps[None]
        wy = y + np.sin(ang + th)[:, None] * steps[None]
        r, c = self.map.world_to_grid(wx, wy)
        r, c = np.floor(r).astype(int), np.floor(c).astype(int)
        inside = (r >= 0) & (r < self.map.height) & (c >= 0) & (c < self.map.width)
        hit = ~inside
        hit[inside] = self.map.data[r[inside], c[inside]]
        if len(self.walkers):
            d = np.min((wx[..., None] - self.walkers[:, 0]) ** 2 + (wy[..., None] - self.walkers[:, 1]) ** 2, axis=-1)
            hit |= d < 0.06 ** 2 * 4
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        ranges = np.where(first >= 0, steps[np.clip(first, 0, None)], np.inf).astype(np.float32)
        m = LaserScan()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), "pelvis"
        m.angle_min, m.angle_max, m.angle_increment = -math.pi, math.pi - 2 * math.pi / n, 2 * math.pi / n
        m.range_min, m.range_max = 0.1, rmax
        m.ranges = ranges.tolist()
        self.pub_scan.publish(m)


def main():
    rclpy.init()
    node = KinematicSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
