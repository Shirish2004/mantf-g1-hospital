import math
import os
import sys
import time

import rclpy
import yaml
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def locations():
    out = {}
    hg1 = os.environ.get("HG1_ROOT", "")
    p = os.path.join(hg1, "ros2_ws/src/hospital-sim-ros2/g1_mission/config/locations.yaml")
    if os.path.isfile(p):
        out.update(yaml.safe_load(open(p)).get("locations", {}))
    p = os.path.join(os.environ.get("MANTF_G1_ROOT", ""), "ros2_ws/src/mantf_g1_bridge/config/pick_place.yaml")
    if os.path.isfile(p):
        out.update(yaml.safe_load(open(p)).get("places", {}))
    return out


def ensure_nav_mode(node, timeout=3.0):
    try:
        from g1_interfaces.msg import Mode
        from g1_interfaces.srv import SetMode
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError:
        return
    mode = [None]
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(Mode, "/g1/mode", lambda m: mode.__setitem__(0, m.mode), qos)
    cli = node.create_client(SetMode, "/g1/set_mode")
    end = time.monotonic() + timeout
    while mode[0] is None and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if mode[0] is None or not cli.wait_for_service(timeout_sec=1.0):
        print("no mode manager: sending the goal directly")
        return
    for want in ([1, 4] if mode[0] == 0 else [4] if mode[0] == 1 else []):
        fut = cli.call_async(SetMode.Request(mode=want, source="mantf"))
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        r = fut.result()
        print(f"set_mode {want}: {'ok' if r and r.accepted else (r.message if r else 'timeout')}")
        if want == 1:
            time.sleep(2.0)
    if mode[0] not in (0, 1, 4, 5):
        print(f"robot in mode {mode[0]}: the manager forwards /g1/cmd_vel_nav only in NAV or AUTO")


def main():
    args = sys.argv[1:]
    no_mode = "--no-mode" in args
    args = [a for a in args if a != "--no-mode"]
    robot = "g1"
    if "--robot" in args:
        i = args.index("--robot")
        robot = args[i + 1]
        args = args[:i] + args[i + 2:]
    locs = locations()
    if not args or args[0] in ("-h", "--help", "list"):
        print("usage: goto [--robot ID] <location> | goto [--robot ID] <x> <y> [yaw]\n" + "\n".join(
            f"  {k:24s} {v['x']:7.2f} {v['y']:7.2f} {v.get('yaw', 0):6.2f}" for k, v in locs.items()))
        return 0
    if args[0] in locs:
        loc = locs[args[0]]
        x, y, yaw = loc["x"], loc["y"], loc.get("yaw", 0.0)
    else:
        x, y = float(args[0]), float(args[1])
        yaw = float(args[2]) if len(args) > 2 else 0.0
    rclpy.init()
    node = Node("mantf_goto")
    if not no_mode and robot == "g1":
        ensure_nav_mode(node)
    action = "/navigate_to_pose" if robot == "g1" else f"/{robot}/navigate_to_pose"
    cli = ActionClient(node, NavigateToPose, action)
    if not cli.wait_for_server(timeout_sec=20.0):
        print(f"no {action} server")
        os._exit(2)
    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = "map"
    goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
    goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    last = [0.0]

    def fb(m):
        d = m.feedback.distance_remaining
        if abs(d - last[0]) > 0.5:
            last[0] = d
            p = m.feedback.current_pose.pose.position
            print(f"  at {p.x:6.2f} {p.y:6.2f}, {d:5.1f} m to go", flush=True)

    fut = cli.send_goal_async(goal, feedback_callback=fb)
    rclpy.spin_until_future_complete(node, fut)
    gh = fut.result()
    if not gh.accepted:
        print("goal rejected")
        os._exit(3)
    print(f"goal accepted: {x:.2f} {y:.2f} {yaw:.2f}")
    res = gh.get_result_async()
    try:
        rclpy.spin_until_future_complete(node, res)
    except KeyboardInterrupt:
        gh.cancel_goal_async()
        os._exit(130)
    status = res.result().status
    print("succeeded" if status == 4 else f"finished with status {status}")
    os._exit(0 if status == 4 else 1)


if __name__ == "__main__":
    main()
