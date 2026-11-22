import math
import os

import numpy as np
import pytest

from mantf_g1_bridge.astar import lookahead_point, path_length, plan_path
from mantf_g1_bridge.occupancy import GridMap
from mantf_g1_bridge.pure_pursuit import PurePursuit

MAP = os.environ.get("MANTF_G1_MAP", "")


def box_world(n=200, res=0.05):
    data = np.zeros((n, n), dtype=bool)
    data[0, :] = data[-1, :] = data[:, 0] = data[:, -1] = True
    data[:150, 95:105] = True
    return GridMap(data, res, (0.0, 0.0))


def test_grid_roundtrip():
    g = box_world()
    r, c = g.world_to_index(2.52, 7.13)
    x, y = g.grid_to_world(r, c)
    assert abs(x - 2.52) < g.resolution and abs(y - 7.13) < g.resolution


def test_astar_goes_around_wall():
    g = box_world()
    p = plan_path(g, (2.0, 2.0), (8.0, 2.0), clearance=0.3, downsample=2)
    assert p is not None
    assert max(y for _, y in p) > 7.5
    for x, y in p:
        assert not g.occupied(x, y)


def test_pure_pursuit_heads_to_path():
    pp = PurePursuit(max_accel=100, max_alpha=100)
    vx, vy, wz = pp.command((0, 0, 0), [(0, 0), (2, 0)], dt=0.05)
    assert vx > 0 and abs(vy) < 1e-6 and abs(wz) < 1e-6
    vx, vy, wz = pp.command((0, 0, 0), [(0, 0), (0, 2)], dt=0.05)
    assert wz > 0


def test_lookahead():
    assert lookahead_point([(0, 0), (1, 0), (1, 1)], 0, 0, 1.5) == pytest.approx((1.0, 0.5))


@pytest.mark.skipif(not os.path.isfile(MAP), reason="hospital map not found")
def test_hospital_route():
    g = GridMap.from_yaml(MAP)
    p = plan_path(g, (25.58, -20.12), (15.93, -16.45), clearance=0.38, downsample=4)
    assert p is not None and path_length(p) > 9.0


def test_mantf_local_plan_avoids_pillar():
    torch = pytest.importorskip("torch")
    from mantf_g1_bridge.mantf_core import _synthetic_map, make_model
    m = make_model("/tmp/mantf_g1_test", device="cuda:0" if torch.cuda.is_available() else "cpu",
                   rows=800, window_radius=5.0)
    m.pretrain(15)
    g = _synthetic_map("pillar", 10.0, 0.05, 0.32, 0.06)
    m.set_local_grid(g.data, (0.0, 0.0), g.origin)
    for _ in range(6):
        m.train_online(20)
    path = m.plan_local((3.0, 0.6))
    if path is not None:
        assert all(m.clearance_at_world(x, y) > 0 for x, y in path)
        assert math.hypot(path[-1][0] - 3.0, path[-1][1] - 0.6) < 1.0
