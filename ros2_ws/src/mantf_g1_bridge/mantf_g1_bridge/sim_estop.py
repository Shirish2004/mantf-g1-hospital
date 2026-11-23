import os

import rclpy
from std_msgs.msg import Bool


def main():
    rclpy.init()
    n = rclpy.create_node("sim_estop_heartbeat")
    pub = n.create_publisher(Bool, "/g1/estop/heartbeat", 10)
    n.create_timer(0.1, lambda: pub.publish(Bool(data=True)))
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
