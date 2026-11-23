import json
import os
import sys
import time

import rclpy
from std_msgs.msg import String


def _node():
    rclpy.init()
    return rclpy.create_node(f"mantf_cli_{os.getpid()}")


def ensure_stand(n, timeout=3.0):
    try:
        from g1_interfaces.msg import Mode
        from g1_interfaces.srv import SetMode
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError:
        return
    mode = [None]
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    n.create_subscription(Mode, "/g1/mode", lambda m: mode.__setitem__(0, m.mode), qos)
    cli = n.create_client(SetMode, "/g1/set_mode")
    end = time.monotonic() + timeout
    while mode[0] is None and time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    if mode[0] in (None, 1, 5) or not cli.wait_for_service(timeout_sec=1.0):
        return
    for want in ([1] if mode[0] == 0 else [1]):
        fut = cli.call_async(SetMode.Request(mode=want, source="mantf"))
        rclpy.spin_until_future_complete(n, fut, timeout_sec=5.0)
        r = fut.result()
        print(f"set_mode STAND: {'ok' if r and r.accepted else (r.message if r else 'timeout')}")
    time.sleep(2.0)


def mission(cmd):
    n = _node()
    if cmd.startswith(("pickplace", "start")):
        ensure_stand(n)
    pub = n.create_publisher(String, "/g1/mission/command", 10)
    end = time.monotonic() + 3.0
    while pub.get_subscription_count() == 0 and time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print("no mission node on /g1/mission/command (start the stack with --mission)")
        os._exit(2)
    pub.publish(String(data=cmd))
    rclpy.spin_once(n, timeout_sec=0.3)
    print(f"sent '{cmd}'")
    watch(n, mission_only=True)


def watch(n=None, mission_only=False, once=False):
    n = n or _node()
    st = {}
    n.create_subscription(String, "/mantf/fleet", lambda m: st.__setitem__("fleet", json.loads(m.data)), 10)
    n.create_subscription(String, "/g1/mission/status", lambda m: st.__setitem__("mission", json.loads(m.data)), 10)
    last = 0.0
    try:
        while True:
            rclpy.spin_once(n, timeout_sec=0.2)
            fl, mis = st.get("fleet"), st.get("mission", {})
            if fl and time.monotonic() - last > 1.0:
                last = time.monotonic()
                tr, dg = fl.get("train") or {}, fl.get("math") or {}
                print(f"t {fl['t']:.1f}  round {fl['round']}  eikonal {tr.get('diff', 0):.3f}  train {tr.get('train_s', 0)} s  "
                      f"known {fl['coverage']['known_m2']:.0f} m2  people {len(fl['people'])}  "
                      f"S~S* corr {dg.get('speed_corr', float('nan')):.2f}  S near agent/alone {dg.get('agent_ratio', float('nan')):.2f}",
                      flush=True)
                for k, r in fl["robots"].items():
                    p = r["pose"]
                    print(f"   {k:3s} {r['state']:11s} pose {('%.2f %.2f %.2f' % tuple(p)) if p else '-':20s} "
                          f"view {r['view'] or '-':8s} path {r['path_kind'] or '-':7s} cmd {r['cmd']} clear {r['clearance']}",
                          flush=True)
                if mis:
                    print(f"   mission {mis.get('state', '-')}: {mis.get('message', '')}", flush=True)
                if once:
                    break
            if mission_only and mis.get("state") in ("succeeded", "failed", "aborted"):
                break
    except KeyboardInterrupt:
        pass
    os._exit(0)


def main():
    a = sys.argv[1:]
    if not a or a[0] == "status":
        watch(once="--once" in a)
    elif a[0] == "pickplace":
        mission(":".join(["pickplace"] + a[1:]))
    elif a[0] == "mission":
        mission(" ".join(a[1:]))
    else:
        print("usage: cli status | pickplace [instrument [place [pick [home]]]] | mission <start|abort|pause|resume|goto:loc>")


if __name__ == "__main__":
    main()
