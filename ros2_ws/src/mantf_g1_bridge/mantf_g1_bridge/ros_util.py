import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def xy_cloud(points, frame, stamp):
    pts = np.zeros((len(points), 3), dtype=np.float32)
    if len(points):
        pts[:, :2] = np.asarray(points, dtype=np.float32)
    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    msg.height, msg.width = 1, len(pts)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 12, 12 * len(pts), True
    msg.data = pts.tobytes()
    return msg


def cloud_xy(msg):
    if msg.width * msg.height == 0:
        return np.zeros((0, 2), dtype=np.float32)
    off = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    x = raw[:, off["x"]:off["x"] + 4].copy().view(np.float32).ravel()
    y = raw[:, off["y"]:off["y"] + 4].copy().view(np.float32).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return np.column_stack((x[ok], y[ok]))
