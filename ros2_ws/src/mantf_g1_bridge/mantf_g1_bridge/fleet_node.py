"""The centralised fleet: one Active NTFields model and one shared map for the simulated G1 (MuJoCo / Isaac) and
kinematic G1 teammates; people from /mantf/agents are other agents. Synchronous rounds, NavigateToPose per robot,
robot-camera video per episode, 2-D/3-D maps and trajectories in the run folder."""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState, LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .fleet_core import Fleet, summarize
from .maps import color
from .occupancy import GridMap
from .ros_util import cloud_xy, xy_cloud
from .sensor_viewer import _image, _rot

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def xyz_cloud(xyz, frame, stamp):
    from sensor_msgs.msg import PointField
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    m = PointCloud2()
    m.header.frame_id, m.header.stamp = frame, stamp
    m.height, m.width = 1, len(xyz)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    m.is_bigendian, m.point_step, m.row_step, m.is_dense = False, 12, 12 * len(xyz), True
    m.data = xyz.tobytes()
    return m


def g1_joints():
    try:
        from ament_index_python.packages import get_package_share_directory
        root = ET.parse(os.path.join(get_package_share_directory("g1_description"), "urdf",
                                     "g1_29dof_brainco_ros.urdf")).getroot()
        return [j.get("name") for j in root.iter("joint") if j.get("type") != "fixed" and j.find("mimic") is None]
    except Exception:
        return []


class FleetNode(Node):
    def __init__(self):
        super().__init__("mantf_fleet")
        p = self.declare_parameter
        for k, v in {
            "robots": ["g1:sim", "k1:kin:25.93,-12.45,3.1416", "k2:kin:15.93,-16.45,1.5708"],
            "explore": True, "workspace": [18.5, -14.5, 28.0], "map_yaml": "", "map_frame": "map",
            "odom_frame": "odom", "base_frame": "pelvis", "ground_truth_topic": "/sim/ground_truth",
            "scan_topic": "/scan", "cloud_topic": "/livox/lidar", "camera_topic": "/g1/head_camera/image_raw",
            "depth_topic": "/g1/head_camera/depth", "agents_topic": "/mantf/agents", "cmd_topic": "/g1/cmd_vel_nav",
            "robot_radius": 0.32, "other_radius": 0.25, "n_rays": 256, "m": 12, "ray_max": 8.0, "rows": 3000,
            "epochs": 5, "warmup_epochs": 30, "frame_buffer": 20, "horizon": 5.0, "replan_period": 0.3, "fourier_scale": 1.0,
            "control_rate": 20.0, "goal_tolerance": 0.35, "yaw_tolerance": 0.45, "lookahead": 0.7, "max_v": 0.75,
            "max_vy": 0.35, "max_w": 0.9, "min_vx": 0.36, "guard_margin": 0.10, "guard_horizon": 1.2,
            "obstacle_min_z": 0.12, "obstacle_max_z": 1.8, "episode_timeout": 600.0, "map_period": 20.0,
            "video": True, "video_fps": 10.0, "device": "auto", "runs_dir": "", "tensorboard": True,
            "publish_map_odom": True, "diag_every": 5, "mode": "cooperative", "help": True, "arrangement": "",
        }.items():
            p(k, v)
        g = self.g = lambda n: self.get_parameter(n).value
        self.cb = ReentrantCallbackGroup()
        dev = ("cuda:0" if torch.cuda.is_available() else "cpu") if g("device") == "auto" else g("device")
        map_yaml = os.path.expanduser(g("map_yaml") or os.environ.get("MANTF_G1_MAP", ""))
        self.gt = GridMap.from_yaml(map_yaml) if os.path.isfile(map_yaml) else None
        specs = []
        for s in g("robots"):
            parts = s.split(":")
            pose = tuple(float(v) for v in parts[2].split(",")) if len(parts) > 2 else None
            specs.append((parts[0], parts[1], pose))
        self.sim_id = next((s[0] for s in specs if s[1] == "sim"), None)
        runs = os.path.expanduser(g("runs_dir") or os.path.join(os.environ.get("MANTF_G1_ROOT", "."), "runs", "fleet"))
        self.folder = os.path.join(runs, time.strftime("%m%d_%H%M%S"))
        os.makedirs(self.folder, exist_ok=True)
        if os.path.isfile(os.path.expanduser(g("arrangement"))):
            shutil.copy(os.path.expanduser(g("arrangement")), os.path.join(self.folder, "arrangement.yaml"))
        from .goto import locations
        self.places = {k: (v["x"], v["y"]) for k, v in locations().items()}
        cfg = {k: g(k) for k in ("robot_radius", "other_radius", "n_rays", "m", "ray_max", "rows", "epochs",
                                 "warmup_epochs", "frame_buffer", "horizon", "lookahead", "max_v", "max_vy", "max_w",
                                 "min_vx", "goal_tolerance", "yaw_tolerance", "guard_margin", "guard_horizon",
                                 "tensorboard", "diag_every", "explore", "mode", "help", "fourier_scale")}
        self.fleet = Fleet(self.folder, list(g("workspace")), specs, self.gt, dev, **cfg)
        F = self.fleet
        self.pose_R, self.pose_z, self.pose_t = np.eye(3), 0.78, 0.0
        self.fallen = False
        self.map_odom = None
        self.joint_names = g1_joints()
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.tfb = TransformBroadcaster(self)
        self.pub_cmd = self.create_publisher(Twist, g("cmd_topic"), 10)
        self.pub_status = self.create_publisher(String, "/mantf/status", 10)
        self.pub_fleet = self.create_publisher(String, "/mantf/fleet", 10)
        self.pub_poses = self.create_publisher(PointCloud2, "/mantf/fleet_poses", 5)
        self.pub_map = self.create_publisher(OccupancyGrid, "/mantf/map", LATCHED)
        self.pub_explored = self.create_publisher(OccupancyGrid, "/mantf/explored", 2)
        self.pub_map3d = self.create_publisher(PointCloud2, "/mantf/map3d", 2)
        self.pub_markers = self.create_publisher(MarkerArray, "/mantf/fleet_markers", 5)
        self.pub_path = {k: self.create_publisher(Path, "/mantf/path" if k == self.sim_id else f"/mantf/{k}/path", 5)
                         for k in F.robots}
        self.pub_trail = {k: self.create_publisher(Path, f"/mantf/{k}/trail", 5) for k in F.robots}
        self.pub_js = {k: self.create_publisher(JointState, f"/{k}/joint_states", 5)
                       for k, r in F.robots.items() if r.kind == "kin"}
        if self.sim_id:
            self.create_subscription(Odometry, g("ground_truth_topic"), self._on_gt, 10, callback_group=self.cb)
            self.create_subscription(LaserScan, g("scan_topic"), self._on_scan, qos_profile_sensor_data, callback_group=self.cb)
            self.create_subscription(PointCloud2, g("cloud_topic"), self._on_cloud, qos_profile_sensor_data, callback_group=self.cb)
            self.create_subscription(Image, g("camera_topic"), self._on_rgb, qos_profile_sensor_data, callback_group=self.cb)
            self.create_subscription(Image, g("depth_topic"), self._on_depth, qos_profile_sensor_data, callback_group=self.cb)
        self.create_subscription(PointCloud2, g("agents_topic"), lambda m: F.set_people(cloud_xy(m)),
                                 qos_profile_sensor_data, callback_group=self.cb)
        self.servers = []
        for k in F.robots:
            name = "/navigate_to_pose" if k == self.sim_id else f"/{k}/navigate_to_pose"
            self.servers.append(ActionServer(self, NavigateToPose, name, execute_callback=lambda gh, k=k: self._execute(k, gh),
                                             goal_callback=lambda _g: GoalResponse.ACCEPT,
                                             cancel_callback=lambda _g: CancelResponse.ACCEPT, callback_group=self.cb))
            topic = "/goal_pose" if k == self.sim_id else f"/{k}/goal_pose"
            self.create_subscription(PoseStamped, topic, lambda m, k=k: self._topic_goal(k, m), 10, callback_group=self.cb)
        self.cloud_t = 0.0
        self.extra = None
        self.rgb = self.depth = None
        self.video = None
        self.vlock = threading.Lock()
        self.bev = None
        self.last_save = time.monotonic()
        self.running = True
        self.create_timer(1.0 / float(g("control_rate")), self._control, callback_group=self.cb)
        self.create_timer(0.1, self._kin_scans, callback_group=self.cb)
        self.views_lock = threading.Lock()
        self.create_timer(0.2, self._views, callback_group=self.cb)
        self.create_timer(0.5, self._publish, callback_group=self.cb)
        self.create_timer(2.0, self._publish_slow, callback_group=self.cb)
        if g("publish_map_odom") and self.sim_id:
            self.create_timer(0.05, self._map_odom, callback_group=self.cb)
        self._publish_display_map()
        threading.Thread(target=self._rounds, daemon=True).start()
        signal.signal(signal.SIGINT, self._exit)
        signal.signal(signal.SIGTERM, self._exit)
        self.get_logger().info(f"fleet {list(F.robots)} on {dev}, one model; run -> {self.folder}")

    # ---------------------------------------------------------------- inputs of the simulated G1
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_gt(self, m):
        p = m.pose.pose
        r = self.fleet.robots[self.sim_id]
        with self.fleet.lock:
            r.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        self.pose_R, self.pose_z, self.pose_t = _rot(p.orientation), p.position.z, time.monotonic()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, self.pose_R[2, 2]))))
        if (p.position.z < 0.45 or tilt > 45.0) and not self.fallen:
            self.fallen = True
            self.fleet.event(f"{self.sim_id}: FELL (pelvis z {p.position.z:.2f} m, tilt {tilt:.0f} deg); no more commands")
            self.get_logger().error(f"{self.sim_id} fell (pelvis z {p.position.z:.2f} m, tilt {tilt:.0f} deg)")
            if r.episode is not None:
                self.fleet.finish(self.sim_id, False, "fell")
                self._end_video(False)

    def _sensor_tf(self, frame):
        try:
            t = self.tf.lookup_transform(self.g("base_frame"), frame, Time())
            tr = t.transform.translation
            return _rot(t.transform.rotation), np.array([tr.x, tr.y, tr.z])
        except Exception:
            return np.eye(3), np.zeros(3)

    def _to_world(self, pts, frame):
        r = self.fleet.robots[self.sim_id]
        if r.pose is None:
            return None, None
        Rs, ts = self._sensor_tf(frame)
        w0 = np.array([r.pose[0], r.pose[1], self.pose_z])
        return (pts @ Rs.T + ts) @ self.pose_R.T + w0, ts @ self.pose_R.T + w0

    def _on_scan(self, m):
        r = np.asarray(m.ranges, np.float32)
        a = m.angle_min + np.arange(r.size, dtype=np.float32) * m.angle_increment
        hit = np.isfinite(r) & (r > m.range_min) & (r < m.range_max)
        rr = np.where(hit, r, min(m.range_max, 30.0))
        world, org = self._to_world(np.column_stack((rr * np.cos(a), rr * np.sin(a), np.zeros_like(rr))), m.header.frame_id)
        if world is None:
            return
        band = (world[:, 2] > self.g("obstacle_min_z")) & (world[:, 2] < self.g("obstacle_max_z"))
        keep = ~(hit & ~band)
        F = self.fleet
        with F.lock:
            F.robots[self.sim_id].scan = (org[:2], world[keep, :2], (hit & band)[keep], self.extra)

    def _on_cloud(self, m):
        now = time.monotonic()
        if now - self.cloud_t < 0.3 or m.width * m.height == 0:
            return
        self.cloud_t = now
        off = {f.name: f.offset for f in m.fields}
        raw = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(-1, m.point_step)
        xyz = np.column_stack([raw[:, off[k]:off[k] + 4].copy().view(np.float32).ravel() for k in "xyz"])
        xyz = xyz[np.isfinite(xyz).all(1) & (np.einsum("ij,ij->i", xyz, xyz) < 15.0 ** 2)]
        world, _ = self._to_world(xyz.astype(float), m.header.frame_id)
        if world is None:
            return
        world = np.unique(np.floor(world / 0.05).astype(np.int64), axis=0) * 0.05 + 0.025
        self.fleet.voxels.insert(world, 1)
        band = world[(world[:, 2] > self.g("obstacle_min_z")) & (world[:, 2] < self.g("obstacle_max_z"))]
        r = self.fleet.robots[self.sim_id]
        if r.pose is not None and len(band):
            band = band[np.linalg.norm(band[:, :2] - np.asarray(r.pose[:2]), axis=1) < self.g("ray_max")]
            self.extra = np.unique(np.floor(band[:, :2] / 0.1).astype(np.int64), axis=0) * 0.1 + 0.05

    def _on_rgb(self, m):
        self.rgb = _image(m)
        self._video_frame()

    def _on_depth(self, m):
        self.depth = _image(m)

    # ---------------------------------------------------------------- the synchronous rounds
    def _rounds(self):
        F = self.fleet
        while self.running:
            t0 = time.monotonic()
            try:
                rec = F.round()
                if rec is not None:
                    img = F.view(f"t {F.t:6.1f}  round {F.field.rounds}  eikonal {rec['diff']:.3f}  "
                                 f"known {F.field.grid.coverage()['known_m2']:.0f} m2")
                    if self.bev is None:
                        self.bev = cv2.VideoWriter(os.path.join(self.folder, "fleet_bev_raw.mp4"),
                                                   cv2.VideoWriter_fourcc(*"mp4v"), 2.0 / max(self.g("replan_period"), 0.1) / 2,
                                                   (img.shape[1], img.shape[0]))
                    self.bev.write(img)
                    self._publish_paths()
            except Exception as exc:
                self.get_logger().error(f"round failed: {exc}", throttle_duration_sec=2.0)
            if time.monotonic() - self.last_save > float(self.g("map_period")):
                self.last_save = time.monotonic()
                try:
                    F.save("latest")
                except Exception as exc:
                    self.get_logger().warn(f"save failed: {exc}")
            time.sleep(max(0.0, float(self.g("replan_period")) - (time.monotonic() - t0)))

    # ---------------------------------------------------------------- control (all robots, 20 Hz)
    def _control(self):
        F = self.fleet
        now = self._now()
        dt = 1.0 / float(self.g("control_rate"))
        F.t = now
        F.step_kin(dt)
        if self.sim_id:
            r = F.robots[self.sim_id]
            fresh = time.monotonic() - self.pose_t < 1.0 and not self.fallen
            if r.goal is not None or r.state == "exploring":
                cmd, event = F.control(self.sim_id, dt) if fresh else ((0.0, 0.0, 0.0), None)
                self._cmd(cmd)
                if event == "succeeded":
                    self._cmd((0.0, 0.0, 0.0))
                    self._end_video(True)
                elif r.episode is not None and F.t - r.episode["t0"] > float(self.g("episode_timeout")):
                    F.finish(self.sim_id, False, "timeout")
                    self._end_video(False)
        stamp = self.get_clock().now().to_msg()
        for k, r in F.robots.items():
            if r.kind != "kin" or r.pose is None:
                continue
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.g("map_frame"), f"{k}/pelvis"
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = r.pose[0], r.pose[1], 0.78
            t.transform.rotation.z, t.transform.rotation.w = math.sin(r.pose[2] / 2), math.cos(r.pose[2] / 2)
            self.tfb.sendTransform(t)
            if self.joint_names:
                js = JointState()
                js.header.stamp = stamp
                js.name = self.joint_names
                ph = (F.t * 6.0) if math.hypot(*r.body.v[:2]) > 0.05 else 0.0
                js.position = [0.35 * math.sin(ph) * (1 if "left" in n else -1) if "hip_pitch" in n else
                               0.3 * (1 - math.cos(ph)) if "knee" in n else 0.0 for n in self.joint_names]
                self.pub_js[k].publish(js)
        self.pub_poses.publish(xy_cloud([r.pose[:2] for r in F.robots.values() if r.pose is not None],
                                        self.g("map_frame"), stamp))

    def _views(self):
        if self.running and self.views_lock.acquire(blocking=False):
            try:
                self.fleet.record_views()
            except Exception as exc:
                self.get_logger().warn(f"own-view video: {exc}", throttle_duration_sec=10.0)
            finally:
                self.views_lock.release()

    def _kin_scans(self):
        self.fleet.scan_kin()

    def _cmd(self, v):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = (float(x) for x in v)
        self.pub_cmd.publish(t)

    # ---------------------------------------------------------------- goals
    def _topic_goal(self, rid, msg):
        p = msg.pose
        self._start(rid, (p.position.x, p.position.y, yaw_of(p.orientation)))

    def _start(self, rid, goal):
        F = self.fleet
        if F.robots[rid].episode is not None:
            F.finish(rid, False, "preempted")
            if rid == self.sim_id:
                self._end_video(False)
        name = next((k for k, (x, y) in self.places.items() if math.hypot(x - goal[0], y - goal[1]) < 0.3), "")
        F.set_goal(rid, goal, name)
        if rid == self.sim_id:
            self._begin_video(F.robots[rid].episode["id"], goal)
        self.get_logger().info(f"{rid}: goal {goal[0]:.2f},{goal[1]:.2f} (episode {F.robots[rid].episode['id']})")

    def _execute(self, rid, gh):
        p = gh.request.pose.pose
        F = self.fleet
        r = F.robots[rid]
        self._start(rid, (p.position.x, p.position.y, yaw_of(p.orientation)))
        ep = r.episode["id"]
        res = NavigateToPose.Result()
        while rclpy.ok():
            time.sleep(0.1)
            if gh.is_cancel_requested:
                F.finish(rid, False, "canceled")
                if rid == self.sim_id:
                    self._end_video(False)
                gh.canceled()
                return res
            done = next((e for e in reversed(F.episodes) if e["episode"] == ep), None)
            if done is not None:
                gh.succeed() if done["success"] else gh.abort()
                return res
            if r.pose is not None:
                fb = NavigateToPose.Feedback()
                fb.current_pose.header.frame_id = self.g("map_frame")
                fb.current_pose.pose.position.x, fb.current_pose.pose.position.y = r.pose[0], r.pose[1]
                fb.distance_remaining = float(math.hypot(p.position.x - r.pose[0], p.position.y - r.pose[1]))
                gh.publish_feedback(fb)
        gh.abort()
        return res

    # ---------------------------------------------------------------- robot-camera video per episode
    def _begin_video(self, ep, goal):
        if not self.g("video"):
            return
        os.makedirs(os.path.join(self.folder, "agents", self.sim_id), exist_ok=True)
        path = os.path.join(self.folder, "agents", self.sim_id, f"camera_ep{ep:02d}_raw.mp4")
        self.video = {"path": path, "writer": None, "last": 0.0, "goal": goal, "ep": ep, "t0": self.fleet.t}

    def _video_frame(self):
        if not self.vlock.acquire(blocking=False):
            return
        try:
            self._video_frame_locked()
        finally:
            self.vlock.release()

    def _video_frame_locked(self):
        v = self.video
        if v is None or self.rgb is None:
            return
        now = time.monotonic()
        if now - v["last"] < 1.0 / float(self.g("video_fps")):
            return
        v["last"] = now
        rgb = cv2.resize(self.rgb, (640, 360))
        if self.depth is not None:
            d = np.nan_to_num(self.depth, nan=0.0, posinf=0.0)
            dm = cv2.applyColorMap(255 - (np.clip(d / 6.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            dm[d <= 0.01] = 0
            frame = np.hstack((rgb, cv2.resize(dm, (640, 360))))
        else:
            frame = rgb
        r = self.fleet.robots[self.sim_id]
        txt = (f"ep {v['ep']}  t {self.fleet.t - v['t0']:5.1f} s  goal {v['goal'][0]:.1f},{v['goal'][1]:.1f}  "
               f"cmd {r.cmd[0]:+.2f} {r.cmd[1]:+.2f} {r.cmd[2]:+.2f}  clearance {r.clear:.2f} m  {r.view_kind}/{r.path_kind}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (30, 30, 30), -1)
        cv2.putText(frame, txt, (6, 17), 0, 0.5, (255, 255, 255), 1)
        if v["writer"] is None:
            v["writer"] = cv2.VideoWriter(v["path"], cv2.VideoWriter_fourcc(*"mp4v"), float(self.g("video_fps")),
                                          (frame.shape[1], frame.shape[0]))
        v["writer"].write(frame)

    def _end_video(self, ok):
        with self.vlock:
            v, self.video = self.video, None
        if v is None or v["writer"] is None:
            return
        v["writer"].release()
        out = v["path"].replace("_raw.mp4", f"_{'success' if ok else 'fail'}.mp4")
        threading.Thread(target=self._h264, args=(v["path"], out), daemon=True).start()
        self.get_logger().info(f"robot-camera video -> {out}")

    @staticmethod
    def _h264(src, dst):
        if shutil.which("ffmpeg") and subprocess.call(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:v", "libx264",
                                                       "-pix_fmt", "yuv420p", dst]) == 0:
            os.remove(src)
        else:
            os.replace(src, dst)

    # ---------------------------------------------------------------- outputs
    def _grid(self, data, res, origin):
        m = OccupancyGrid()
        m.header.frame_id = self.g("map_frame")
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = float(res)
        m.info.height, m.info.width = data.shape
        m.info.origin.position.x, m.info.origin.position.y = float(origin[0]), float(origin[1])
        m.info.origin.orientation.w = 1.0
        m.data = data.astype(np.int8).ravel().tolist()
        return m

    def _publish_display_map(self):
        if self.gt is not None:
            self.pub_map.publish(self._grid(self.gt.data.astype(np.int8) * 100, self.gt.resolution, self.gt.origin))

    def _path_msg(self, pts):
        msg = Path()
        msg.header.frame_id = self.g("map_frame")
        msg.header.stamp = self.get_clock().now().to_msg()
        for q in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(q[0]), float(q[1])
            ps.pose.position.z = 0.05
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _publish_paths(self):
        for k, r in self.fleet.robots.items():
            if r.path:
                self.pub_path[k].publish(self._path_msg(r.path))

    def _publish(self):
        F = self.fleet
        st = F.status()
        self.pub_fleet.publish(String(data=json.dumps(st)))
        if self.sim_id:
            r = st["robots"][self.sim_id]
            self.pub_status.publish(String(data=json.dumps({
                "state": r["state"], "pose": r["pose"], "model_ready": st["round"] > 0, "frames": st["round"],
                "loss": (st["train"] or {}).get("diff"), "planner": r["stats"], "viewpoint": r["viewpoint"],
                "agents": len(st["people"]), "robots": {k: v["pose"] for k, v in st["robots"].items()},
                "run": self.folder})))
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (k, r) in enumerate(F.robots.items()):
            b, gg, rr = color(i)
            for j, (kind, xy) in enumerate((("viewpoint", r.viewpoint), ("goal", r.goal))):
                m = Marker()
                m.header.frame_id, m.header.stamp = self.g("map_frame"), stamp
                m.ns, m.id = f"{k}_{kind}", 0
                if xy is None:
                    m.action = Marker.DELETE
                else:
                    m.type = Marker.SPHERE if kind == "viewpoint" else Marker.CYLINDER
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(xy[0]), float(xy[1]), 0.15
                    m.pose.orientation.w = 1.0
                    m.scale.x = m.scale.y = 0.35 if kind == "viewpoint" else 0.5
                    m.scale.z = 0.35 if kind == "viewpoint" else 0.05
                    m.color.r, m.color.g, m.color.b, m.color.a = rr / 255, gg / 255, b / 255, 0.9
                ma.markers.append(m)
            if r.pose is not None:
                m = Marker()
                m.header.frame_id, m.header.stamp = self.g("map_frame"), stamp
                m.ns, m.id, m.type = f"{k}_label", 0, Marker.TEXT_VIEW_FACING
                m.pose.position.x, m.pose.position.y, m.pose.position.z = r.pose[0], r.pose[1], 2.0
                m.pose.orientation.w = 1.0
                m.scale.z = 0.4
                m.color.r, m.color.g, m.color.b, m.color.a = rr / 255, gg / 255, b / 255, 1.0
                m.text = r.label or f"{k}: {r.state}"
                ma.markers.append(m)
        for j, q in enumerate(list(F.requests.values()) + [None] * 4):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.g("map_frame"), stamp
            m.ns, m.id = "requests", j
            if q is None:
                m.action = Marker.DELETE
            else:
                m.type = Marker.CYLINDER
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(q[0]), float(q[1]), 0.02
                m.pose.orientation.w = 1.0
                m.scale.x, m.scale.y, m.scale.z = 4.0, 4.0, 0.02
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.55, 0.0, 0.25
            ma.markers.append(m)
            if len(r.trail) > 1:
                self.pub_trail[k].publish(self._path_msg([t[1:3] for t in r.trail[-4000:]]))
        self.pub_markers.publish(ma)

    def _publish_slow(self):
        g = self.fleet.field.grid
        self.pub_explored.publish(self._grid(g.ros_data(), g.res, g.origin))
        xyz, _ = self.fleet.voxels.points()
        if len(xyz):
            self.pub_map3d.publish(xyz_cloud(xyz, self.g("map_frame"), self.get_clock().now().to_msg()))

    def _map_odom(self):
        r = self.fleet.robots[self.sim_id]
        if r.pose is not None:
            try:
                t = self.tf.lookup_transform(self.g("odom_frame"), self.g("base_frame"), Time())
                tr = t.transform.translation
                Tmb, Tob = np.eye(4), np.eye(4)
                Tmb[:3, :3], Tmb[:3, 3] = self.pose_R, (r.pose[0], r.pose[1], self.pose_z)
                Tob[:3, :3], Tob[:3, 3] = _rot(t.transform.rotation), (tr.x, tr.y, tr.z)
                T = Tmb @ np.linalg.inv(Tob)
                self.map_odom = (T[0, 3], T[1, 3], T[2, 3], math.atan2(T[1, 0], T[0, 0]))
            except Exception:
                pass
        if self.map_odom is None:
            return
        x, y, z, yaw = self.map_odom
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = self.g("map_frame"), self.g("odom_frame")
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = x, y, z
        m.transform.rotation.z, m.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.tfb.sendTransform(m)

    def _exit(self, *_):
        self.running = False
        try:
            self._cmd((0.0, 0.0, 0.0))
            F = self.fleet
            for k, r in F.robots.items():
                if r.episode is not None:
                    F.finish(k, False, "shutdown")
            self._end_video(False)
            F.save("final")
            with self.views_lock:
                F.close_views()
            if self.bev is not None:
                self.bev.release()
                self._h264(os.path.join(self.folder, "fleet_bev_raw.mp4"), os.path.join(self.folder, "fleet_bev.mp4"))
            summarize(self.folder)
            self.get_logger().info(f"saved {self.folder}")
        except Exception as exc:
            print(f"save on exit failed: {exc}", flush=True)
        os._exit(0)


def main():
    rclpy.init()
    node = FleetNode()
    # every goal holds one thread while its episode runs: room for all of them plus the timers and subscriptions
    ex = MultiThreadedExecutor(num_threads=max(8, len(node.fleet.robots) + 6))
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        node._exit()


if __name__ == "__main__":
    main()
