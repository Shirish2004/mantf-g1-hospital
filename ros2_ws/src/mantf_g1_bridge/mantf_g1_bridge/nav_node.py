"""Nav2-compatible NavigateToPose server driven by mantf + pure pursuit.

The mission/skills in WardZero keep calling `/navigate_to_pose`; this node
replaces Nav2's controller with mantf's local travel-time field on a live
occupancy grid.  Static walls come from the saved map, dynamic cells from
`/scan` (or a point cloud), and the base follows the field with holonomic pure
pursuit on `/g1/cmd_vel_nav` (forwarded by g1_teleop_manager in NAV or AUTO).
"""
from __future__ import annotations

import json
import math
import os
import signal
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .astar import path_length, plan_path
from .occupancy import GridMap
from .local_planner import LocalPlanner
from .pure_pursuit import PurePursuit, safety_filter

try:
    from sensor_msgs_py import point_cloud2
except Exception:  # pragma: no cover
    point_cloud2 = None


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _pose_of(msg):
    p = msg.pose.pose.position
    return (float(p.x), float(p.y), _yaw(msg.pose.pose.orientation))


def _pose_stamped(msg):
    p = msg.pose.position
    return (float(p.x), float(p.y), _yaw(msg.pose.orientation))


class MantfNav(Node):
    def __init__(self):
        super().__init__("mantf_nav")
        p = self.declare_parameter
        p("map_source", "auto")                 # auto | yaml | topic
        p("map_yaml", "")
        p("map_topic", "/map")
        p("map_frame", "map")
        p("base_frame", "pelvis")
        p("pose_source", "ground_truth")        # ground_truth | odom | tf
        p("ground_truth_topic", "/sim/ground_truth")
        p("odom_topic", "/odom")
        p("scan_topic", "/scan")
        p("cloud_topic", "")
        p("use_scan", True)
        p("unknown_is_occupied", True)
        p("robot_radius", 0.32)
        p("safety_margin", 0.06)
        p("dynamic_margin", 0.18)
        p("dynamic_ttl", 2.0)
        p("dynamic_mark_radius", 0.10)
        p("window_radius", 5.0)
        p("local_resolution", 0.05)
        p("global_downsample", 4)
        p("local_goal_lookahead", 3.0)
        p("pursuit_lookahead", 0.8)
        p("goal_tolerance", 0.30)
        p("yaw_tolerance", 0.40)
        p("control_rate", 20.0)
        p("replan_period", 0.30)
        p("global_replan_period", 4.0)
        p("retrain_period", 1.0)
        p("path_timeout", 1.0)
        p("max_v", 0.50)
        p("max_vy", 0.30)
        p("max_w", 0.80)
        p("min_vx", 0.0)
        p("rows", 1200)
        p("train_epochs", 8)
        p("warmup_epochs", 20)
        p("pretrain_epochs", 20)
        p("model_path", "")
        p("device", "auto")
        p("action_name", "/navigate_to_pose")
        p("cmd_topic", "/g1/cmd_vel_nav")
        p("path_topic", "/mantf/path")
        p("status_topic", "/mantf/status")
        p("publish_path", True)
        p("save_checkpoint", "")
        p("use_safety_filter", True)
        p("safety_stop", 0.45)
        p("safety_slow", 1.6)
        p("use_mantf", True)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g

        self.cb_group = ReentrantCallbackGroup()
        self.map_lock = threading.RLock()
        self.pose_lock = threading.RLock()
        self.path_lock = threading.RLock()
        self.model_lock = threading.RLock()
        self.active_lock = threading.Lock()
        self.static_map = None
        self.dynamic_last = None
        self.pose = None
        self.pose_time = 0.0
        self.map_ready = threading.Event()
        self.model_ready = threading.Event()
        self.model_error = ""
        self.model = None
        self._active = False
        self._cancel_event = threading.Event()
        self._planner_stop = threading.Event()
        self._planner = None
        self._global_path = None
        self._local_path = None
        self._local_path_time = 0.0
        self._dynamic_cells = 0
        self._state = "idle"
        self.local_planner = None
        self._fallback_planner = LocalPlanner(None, window_radius=g("window_radius"),
                                              robot_radius=g("robot_radius"),
                                              static_margin=g("safety_margin"),
                                              dynamic_margin=g("dynamic_margin"))
        self._planner_stats = {}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if g("model_path"):
            self.model_path = os.path.expanduser(g("model_path"))
        else:
            root = os.environ.get("MANTF_ROOT", os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "..", "mantf")))
            self.model_path = os.path.join(root, "Experiments")

        self.controller = PurePursuit(
            lookahead=g("pursuit_lookahead"), max_v=g("max_v"), max_vy=g("max_vy"),
            max_w=g("max_w"), min_vx=g("min_vx"))

        self.pub_cmd = self.create_publisher(Twist, g("cmd_topic"), 10)
        self.pub_path = self.create_publisher(Path, g("path_topic"), 10)
        self.pub_status = self.create_publisher(String, g("status_topic"), 10)

        self._setup_map()
        if g("use_scan"):
            self.create_subscription(LaserScan, g("scan_topic"), self._scan_cb,
                                     qos_profile_sensor_data, callback_group=self.cb_group)
        if g("cloud_topic"):
            self.create_subscription(PointCloud2, g("cloud_topic"), self._cloud_cb,
                                     qos_profile_sensor_data, callback_group=self.cb_group)
        src = g("pose_source")
        if src == "ground_truth":
            self.create_subscription(Odometry, g("ground_truth_topic"), self._odom_cb, 10,
                                     callback_group=self.cb_group)
        elif src == "odom":
            self.create_subscription(Odometry, g("odom_topic"), self._odom_cb, 10,
                                     callback_group=self.cb_group)

        self._server = ActionServer(
            self, NavigateToPose, g("action_name"), execute_callback=self._execute,
            goal_callback=self._goal_cb, cancel_callback=self._cancel_cb,
            callback_group=self.cb_group)
        self.create_timer(0.5, self._status_timer)
        self.create_timer(0.5, self._check_pose_timer)

        if g("use_mantf"):
            threading.Thread(target=self._init_model, daemon=True).start()
        signal.signal(signal.SIGINT, self._sigint)
        signal.signal(signal.SIGTERM, self._sigint)
        self.get_logger().info(f"mantf_nav up: map_source={g('map_source')} pose={src} "
                               f"action={g('action_name')} cmd={g('cmd_topic')}")

    def _sigint(self, *_):
        self.get_logger().info("shutting down")
        os._exit(0)

    # ------------------------------------------------------------------ map
    def _setup_map(self):
        source = str(self.g("map_source")).lower()
        path = os.path.expanduser(str(self.g("map_yaml") or ""))
        if not path:
            path = os.environ.get("MANTF_G1_MAP", "")
        if source in ("auto", "yaml") and path and os.path.isfile(path):
            try:
                grid = GridMap.from_yaml(path, unknown_is_occupied=bool(self.g("unknown_is_occupied")))
                self._set_map(grid)
                self.get_logger().info(f"map loaded: {path} ({grid.width}x{grid.height} @ {grid.resolution} m)")
                return
            except Exception as exc:
                self.get_logger().error(f"cannot load map {path}: {exc}")
                if source == "yaml":
                    raise
        if source in ("auto", "topic"):
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(OccupancyGrid, self.g("map_topic"), self._map_cb, qos,
                                     callback_group=self.cb_group)
            self.get_logger().info(f"waiting for {self.g('map_topic')}")
        elif not self.map_ready.is_set():
            raise RuntimeError(f"no map: map_source={source} map_yaml={path}")

    def _set_map(self, grid):
        with self.map_lock:
            self.static_map = grid
            self.dynamic_last = np.zeros(grid.data.shape, dtype=np.int64)
        self.map_ready.set()

    def _map_cb(self, msg):
        if self.map_ready.is_set():
            return
        self._set_map(GridMap.from_ros(msg, unknown_is_occupied=bool(self.g("unknown_is_occupied"))))
        self.get_logger().info(f"map received: {msg.info.width}x{msg.info.height} @ {msg.info.resolution} m")

    def _dynamic_mask(self):
        if self.static_map is None:
            return None
        now = time.monotonic_ns()
        ttl = int(float(self.g("dynamic_ttl")) * 1e9)
        with self.map_lock:
            age = now - self.dynamic_last
            data = (age >= 0) & (age < ttl)
        return GridMap(data, self.static_map.resolution, self.static_map.origin, self.static_map.yaw)

    def _merged_grid(self):
        with self.map_lock:
            static = self.static_map
        if static is None:
            return None
        dyn = self._dynamic_mask()
        if dyn is None:
            return static
        self._dynamic_cells = int(dyn.data.sum())
        return GridMap(static.data | dyn.data, static.resolution, static.origin, static.yaw)

    # ------------------------------------------------------------------ pose
    def _odom_cb(self, msg):
        with self.pose_lock:
            self.pose = _pose_of(msg)
            self.pose_time = time.monotonic()

    def _pose_now(self):
        if self.g("pose_source") == "tf":
            try:
                t = self.tf_buffer.lookup_transform(self.g("map_frame"), self.g("base_frame"), Time())
                tr, r = t.transform.translation, t.transform.rotation
                return (float(tr.x), float(tr.y), _yaw(r))
            except Exception:
                return None
        with self.pose_lock:
            if self.pose is None or time.monotonic() - self.pose_time > 1.0:
                return None
            return self.pose

    def _wait_pose(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            pose = self._pose_now()
            if pose is not None:
                return pose
            time.sleep(0.05)
        return None

    def _check_pose_timer(self):
        if self._state in ("navigating", "planning") and self._pose_now() is None:
            self.get_logger().warn("no base pose", throttle_duration_sec=2.0)

    # ------------------------------------------------------------------ scan / dynamic obstacles
    def _sensor_transform(self, frame_id):
        if not frame_id or frame_id in (self.g("map_frame"), "world"):
            return (0.0, 0.0, 0.0)
        for parent in (self.g("map_frame"), self.g("base_frame")):
            try:
                t = self.tf_buffer.lookup_transform(parent, frame_id, Time())
                tr, r = t.transform.translation, t.transform.rotation
                yaw = _yaw(r)
                if parent == self.g("map_frame"):
                    return (float(tr.x), float(tr.y), yaw)
                pose = self._pose_now()
                if pose is None:
                    continue
                c, s = math.cos(pose[2]), math.sin(pose[2])
                return (pose[0] + c * tr.x - s * tr.y,
                        pose[1] + s * tr.x + c * tr.y,
                        _wrap(pose[2] + yaw))
            except Exception:
                continue
        pose = self._pose_now()
        return pose if pose is not None else (0.0, 0.0, 0.0)

    def _mark_points(self, points, transform):
        if self.static_map is None or points.size == 0:
            return
        tx, ty, tyaw = transform
        c, s = math.cos(tyaw), math.sin(tyaw)
        wx = tx + c * points[:, 0] - s * points[:, 1]
        wy = ty + s * points[:, 0] + c * points[:, 1]
        r, col = self.static_map.world_to_grid(wx, wy)
        r = np.floor(r).astype(np.int64)
        col = np.floor(col).astype(np.int64)
        rad = max(1, int(round(float(self.g("dynamic_mark_radius")) / self.static_map.resolution)))
        now = time.monotonic_ns()
        offsets = [(dr, dc) for dr in range(-rad, rad + 1) for dc in range(-rad, rad + 1)
                   if dr * dr + dc * dc <= rad * rad + 1]
        with self.map_lock:
            for dr, dc in offsets:
                rr, cc = r + dr, col + dc
                ok = (rr >= 0) & (rr < self.dynamic_last.shape[0]) & (cc >= 0) & (cc < self.dynamic_last.shape[1])
                self.dynamic_last[rr[ok], cc[ok]] = now

    def _scan_cb(self, msg):
        if self.static_map is None:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
        ok = np.isfinite(ranges) & (ranges >= max(0.05, msg.range_min)) & (ranges <= msg.range_max)
        if not ok.any():
            return
        rr, aa = ranges[ok], angles[ok]
        local = np.column_stack((rr * np.cos(aa), rr * np.sin(aa)))
        self._mark_points(local, self._sensor_transform(msg.header.frame_id))

    def _cloud_cb(self, msg):
        if point_cloud2 is None or self.static_map is None:
            return
        try:
            pts = np.array([[p[0], p[1]] for p in point_cloud2.read_points(
                msg, field_names=("x", "y"), skip_nans=True)])
        except Exception:
            return
        if pts.size:
            self._mark_points(pts.astype(np.float32), self._sensor_transform(msg.header.frame_id))

    def _nearest_dynamic(self, pose, radius=2.0):
        if self.static_map is None:
            return None
        m = self.static_map
        r0, c0 = m.world_to_index(pose[0], pose[1])
        n = int(radius / m.resolution)
        ra, rb = max(0, r0 - n), min(m.height, r0 + n + 1)
        ca, cb = max(0, c0 - n), min(m.width, c0 + n + 1)
        if ra >= rb or ca >= cb:
            return None
        ttl = int(float(self.g("dynamic_ttl")) * 1e9)
        with self.map_lock:
            age = time.monotonic_ns() - self.dynamic_last[ra:rb, ca:cb]
        rr, cc = np.nonzero((age >= 0) & (age < ttl))
        if rr.size == 0:
            return None
        wx, wy = m.grid_to_world(rr + ra, cc + ca)
        d = (wx - pose[0]) ** 2 + (wy - pose[1]) ** 2
        near = d > (0.5 * float(self.g("robot_radius"))) ** 2
        if not near.any():
            return None
        k = int(np.argmin(np.where(near, d, np.inf)))
        return (float(wx[k]), float(wy[k]))

    # ------------------------------------------------------------------ planner model
    def _init_model(self):
        try:
            from .mantf_core import make_model
            device = self.g("device")
            if device == "auto":
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = make_model(
                self.model_path, device=device, rows=int(self.g("rows")),
                window_radius=float(self.g("window_radius")),
                resolution=float(self.g("local_resolution")),
                robot_radius=float(self.g("robot_radius")),
                static_margin=float(self.g("safety_margin")),
                dynamic_margin=float(self.g("dynamic_margin")))
            model.pretrain(int(self.g("pretrain_epochs")))
            with self.model_lock:
                self.model = model
            self.model_ready.set()
            self.get_logger().info(f"mantf field ready on {device}, folder {model.folder}")
        except Exception as exc:
            self.model_error = str(exc)
            self.get_logger().error(f"mantf model failed: {exc}")

    # ------------------------------------------------------------------ action server
    def _goal_cb(self, _goal):
        if not self.map_ready.is_set():
            self.get_logger().warn("rejecting goal: map not ready")
            return GoalResponse.REJECT
        with self.active_lock:
            if self._active:
                self.get_logger().warn("rejecting goal: another goal is active")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal_handle):
        self._cancel_event.set()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        with self.active_lock:
            self._active = True
        self._cancel_event.clear()
        self._planner_stop.clear()
        self._state = "planning"
        start = time.monotonic()
        goal = _pose_stamped(goal_handle.request.pose)
        self.get_logger().info(f"goal {goal[0]:.2f},{goal[1]:.2f} yaw {goal[2]:.2f}")
        try:
            pose = self._wait_pose(10.0)
            if pose is None:
                self.get_logger().error("no base pose; aborting")
                goal_handle.abort()
                return NavigateToPose.Result()
            grid = self._merged_grid()
            if grid is None:
                self.get_logger().error("no map; aborting")
                goal_handle.abort()
                return NavigateToPose.Result()
            clearance = float(self.g("robot_radius")) + float(self.g("safety_margin"))
            path = plan_path(grid, pose[:2], goal[:2], clearance=clearance,
                             downsample=int(self.g("global_downsample")))
            if path is None:
                self.get_logger().error("no global route to goal")
                goal_handle.abort()
                return NavigateToPose.Result()
            with self.path_lock:
                self._global_path = path
                self._local_path = path
                self._local_path_time = time.monotonic()
            self.get_logger().info(f"global route: {len(path)} points, {path_length(path):.1f} m")
            self._state = "navigating"
            self._planner = threading.Thread(target=self._planner_loop, args=(goal,), daemon=True)
            self._planner.start()

            dt = 1.0 / max(1.0, float(self.g("control_rate")))
            while rclpy.ok():
                if self._cancel_event.is_set():
                    self._publish_zero()
                    goal_handle.canceled()
                    return NavigateToPose.Result()
                pose = self._pose_now()
                if pose is None:
                    self._publish_zero()
                    time.sleep(dt)
                    continue
                dist = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
                dyaw = abs(_wrap(goal[2] - pose[2]))
                if dist < float(self.g("goal_tolerance")) and dyaw < float(self.g("yaw_tolerance")):
                    self._publish_zero()
                    self._state = "succeeded"
                    goal_handle.succeed()
                    self.get_logger().info(f"goal reached in {time.monotonic() - start:.1f} s")
                    return NavigateToPose.Result()
                if dist < float(self.g("goal_tolerance")):
                    w = max(-float(self.g("max_w")), min(float(self.g("max_w")), 1.5 * _wrap(goal[2] - pose[2])))
                    self._publish_cmd((0.0, 0.0, w))
                    self._publish_feedback(goal_handle, pose, dist, start)
                    time.sleep(dt)
                    continue
                path = self._current_path()
                cmd = self.controller.command(pose, path, dt=dt, distance_to_goal=dist)
                if bool(self.g("use_safety_filter")):
                    cmd = safety_filter(cmd, pose, self._nearest_dynamic(pose), stop=float(self.g("safety_stop")),
                                        slow=float(self.g("safety_slow")), max_vy=float(self.g("max_vy")))
                self._publish_cmd(cmd)
                if bool(self.g("publish_path")) and path:
                    self._publish_path(path)
                self._publish_feedback(goal_handle, pose, dist, start)
                time.sleep(dt)
            goal_handle.abort()
            return NavigateToPose.Result()
        except Exception as exc:
            self.get_logger().error(f"navigation failed: {exc}")
            self._publish_zero()
            goal_handle.abort()
            return NavigateToPose.Result()
        finally:
            self._planner_stop.set()
            if self._planner is not None:
                self._planner.join(timeout=1.0)
            self._publish_zero()
            with self.active_lock:
                self._active = False

    def _current_path(self):
        with self.path_lock:
            p, t = self._local_path, self._local_path_time
        if p and time.monotonic() - t < float(self.g("path_timeout")):
            return p
        with self.path_lock:
            return self._global_path

    # ------------------------------------------------------------------ local planner thread
    def _planner_loop(self, goal):
        last_global = 0.0
        while not self._planner_stop.is_set():
            pose = self._pose_now()
            if pose is None:
                time.sleep(0.1)
                continue
            now = time.monotonic()
            if now - last_global > float(self.g("global_replan_period")):
                grid = self._merged_grid()
                if grid is not None:
                    clearance = float(self.g("robot_radius")) + float(self.g("safety_margin"))
                    gp = plan_path(grid, pose[:2], goal[:2], clearance=clearance,
                                   downsample=int(self.g("global_downsample")))
                    if gp:
                        with self.path_lock:
                            self._global_path = gp
                        last_global = now
            with self.path_lock:
                gp = self._global_path
            if not gp:
                time.sleep(0.1)
                continue

            with self.map_lock:
                static = self.static_map
            dyn = self._dynamic_mask()
            if static is None:
                time.sleep(0.1)
                continue
            with self.model_lock:
                if self.local_planner is None and self.model is not None:
                    self.local_planner = LocalPlanner(
                        self.model, window_radius=float(self.g("window_radius")),
                        robot_radius=float(self.g("robot_radius")),
                        static_margin=float(self.g("safety_margin")),
                        dynamic_margin=float(self.g("dynamic_margin")),
                        lookahead=float(self.g("local_goal_lookahead")),
                        warmup_epochs=int(self.g("warmup_epochs")),
                        train_epochs=int(self.g("train_epochs")),
                        retrain_period=float(self.g("retrain_period")))
                lp = self.local_planner or self._fallback_planner
                local_path, _ = lp.step(pose, gp, static, dyn)
                if lp.blocked:
                    last_global = 0.0
                self._planner_stats = dict(lp.stats)
            with self.path_lock:
                self._local_path = local_path
                self._local_path_time = time.monotonic()
            time.sleep(float(self.g("replan_period")))

    # ------------------------------------------------------------------ outputs
    def _publish_cmd(self, cmd):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = (float(v) for v in cmd)
        self.pub_cmd.publish(msg)

    def _publish_zero(self):
        self.controller.reset()
        self._publish_cmd((0.0, 0.0, 0.0))

    def _publish_path(self, points):
        msg = Path()
        msg.header.frame_id = self.g("map_frame")
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def _publish_feedback(self, goal_handle, pose, dist, start):
        fb = NavigateToPose.Feedback()
        fb.current_pose.header.frame_id = self.g("map_frame")
        fb.current_pose.header.stamp = self.get_clock().now().to_msg()
        fb.current_pose.pose.position.x, fb.current_pose.pose.position.y = pose[0], pose[1]
        fb.current_pose.pose.orientation.z = math.sin(pose[2] / 2.0)
        fb.current_pose.pose.orientation.w = math.cos(pose[2] / 2.0)
        elapsed = time.monotonic() - start
        fb.navigation_time.sec = int(elapsed)
        fb.navigation_time.nanosec = int((elapsed % 1.0) * 1e9)
        fb.estimated_time_remaining.sec = 0
        fb.number_of_recoveries = 0
        fb.distance_remaining = float(dist)
        goal_handle.publish_feedback(fb)

    def _status_timer(self):
        pose = self._pose_now()
        with self.path_lock:
            lp, gp = self._local_path, self._global_path
        status = {
            "state": self._state,
            "pose": list(pose) if pose else None,
            "model_ready": self.model_ready.is_set(),
            "model_error": self.model_error,
            "dynamic_cells": self._dynamic_cells,
            "local_points": len(lp) if lp else 0,
            "global_points": len(gp) if gp else 0,
            "planner": self._planner_stats,
        }
        self.pub_status.publish(String(data=json.dumps(status)))


def main(args=None):
    rclpy.init(args=args)
    node = MantfNav()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
