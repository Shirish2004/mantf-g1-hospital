import math
import os

import numpy as np
import pytest

from mantf_g1_bridge.occupancy import GridMap, raycast
from mantf_g1_bridge.pure_pursuit import PurePursuit, lookahead_point, safe_velocity

GT = os.path.join(os.environ.get("HG1_ROOT", ""), "ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/hospital_gt.yaml")


def corridor(n=240, res=0.05):
    d = np.zeros((n, n), bool)
    d[0, :] = d[-1, :] = d[:, 0] = d[:, -1] = True
    d[:150, 115:125] = True
    return GridMap(d, res, (-n * res / 2, -n * res / 2))


def test_grid_roundtrip():
    g = corridor()
    r, c = g.world_to_index(1.52, -2.13)
    x, y = g.grid_to_world(r, c)
    assert abs(x - 1.52) < g.resolution and abs(y + 2.13) < g.resolution


def test_raycast_hits_wall():
    g = corridor()
    ang, rng = raycast(g, -3.0, 0.0, 0.0, beams=8, rmax=10.0)
    assert abs(rng[4] - 2.75) < 0.15


def test_pure_pursuit_and_guard():
    pp = PurePursuit(max_accel=100, max_alpha=100)
    vx, vy, wz = pp.command((0, 0, 0), [(0, 0), (2, 0)], dt=0.05)
    assert vx > 0 and abs(wz) < 1e-6
    wall = np.column_stack((np.full(40, 0.6), np.linspace(-2, 2, 40)))
    (gx, gy, _), clear = safe_velocity((0.6, 0.0, 0.0), (0, 0, 0), wall)
    assert clear == pytest.approx(0.6 - 0.46, abs=0.06)
    assert gx < 0.2
    (gx, gy, _), _ = safe_velocity((0.6, 0.0, 0.0), (0, 0, 0), wall + [3.0, 0.0])
    assert gx == pytest.approx(0.6)
    assert lookahead_point([(0, 0), (1, 0), (1, 1)], 0, 0, 1.5) == pytest.approx((1.0, 0.5))


def test_shared_grid_and_voxels(tmp_path):
    from mantf_g1_bridge.maps import SharedGrid, VoxelMap
    sg = SharedGrid((0, 0), 10.0, 0.1, ["a", "b"])
    ends = [(4.0 * math.cos(a), 4.0 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 90, endpoint=False)]
    sg.update("a", (0, 0), ends, [True] * len(ends))
    assert sg.free.sum() > 500 and sg.occ.sum() > 30 and sg.seen[0].any() and not sg.seen[1].any()
    vm = VoxelMap(0.1)
    vm.insert(np.random.default_rng(0).uniform(0, 2, (2000, 3)))
    vm.insert(np.random.default_rng(0).uniform(0, 2, (2000, 3)))
    assert len(vm) > 1000 and vm.save(str(tmp_path)) > 0 and (tmp_path / "map3d.ply").exists()


def test_fleet_one_model_rounds(tmp_path):
    torch = pytest.importorskip("torch")
    from mantf_g1_bridge.fleet_core import Fleet, summarize
    g = corridor()
    f = Fleet(str(tmp_path), (0.0, 0.0, 12.0), [("a", "kin", (-3.0, -3.0, 0.0)), ("b", "kin", (3.0, -3.0, 3.14))], g,
              "cuda:0" if torch.cuda.is_available() else "cpu", rows=800, warmup_epochs=10, tensorboard=False,
              diag_every=2)
    f.set_people([(0.0, -4.0)])
    f.set_goal("a", (0.0, -3.0, 0.0))
    agents = []
    for _ in range(4):
        f.scan_kin()
        rec = f.round()
        agents.append(rec["agent"])
        for _ in range(10):
            f.step_kin(0.05)
            f.t += 0.05
    assert agents == ["a", "b", "a", "b"] and rec["rows"] == 800 and math.isfinite(rec["diff"])
    assert f.robots["a"].gxy is not None and abs(math.hypot(*f.robots["a"].gxy) - 1.0) < 1e-3
    assert os.path.exists(os.path.join(str(tmp_path), "calls.csv"))
    assert f.field.diag and math.isfinite(f.field.diag["speed_mae"])
    assert f.robots["a"].viewpoint is not None
    out = f.save("t")
    assert os.path.exists(out["fleet"])
    f.finish("a", False, "test")
    assert os.path.exists(summarize(str(tmp_path)))


@pytest.mark.skipif(not os.path.isfile(GT), reason="hospital_gt map not found")
def test_hospital_gt_map_loads_and_lanes():
    from mantf_g1_bridge.walkers_node import make_lanes
    g = GridMap.from_yaml(GT)
    assert not g.occupied(25.93, -12.45)
    lanes = make_lanes(g, [6.5, -24.0, 31.0, -5.0], 5, seed=3)
    assert len(lanes) == 5
    for a, b, _ in lanes:
        for t in np.linspace(0, 1, 50):
            q = a + (b - a) * t
            assert not g.occupied(*q)


def test_close_person_and_robot_can_separate():
    from mantf_g1_bridge.fleet_core import KinBody
    from mantf_g1_bridge.walkers_node import People
    g = corridor()
    robot = (-3.0, -3.0)
    people = People([(np.array([-5.0, -3.5]), np.array([0.0, -3.5]), 0.8)], seed=0)
    people.s[0] = 2.0 + 0.0                       # the person starts 0.5 m from the robot
    start = np.linalg.norm(people.step(0.0, [robot])[0][0] - robot)
    for _ in range(100):
        pos, _ = people.step(0.1, [robot])
    assert np.linalg.norm(pos[0] - robot) > start  # the person gets away instead of freezing
    body = KinBody((-3.0, -3.0, 0.0), g)
    person = [(-2.6, -3.0, 0.25)]                  # in front, inside the footprint's safety radius
    body.step((-0.3, 0.0, 0.0), 0.5, person)       # backing away is allowed
    assert body.pose[0] < -3.0
    x = body.pose[0]
    body.step((0.3, 0.0, 0.0), 0.5, person)        # moving into the person is not
    assert body.pose[0] <= x + 1e-9


def test_guard_escape_never_backs_into_something_else():
    # a wall 0.4 m ahead and another 0.35 m behind: backing straight away from the nearest point would hit the
    # one behind; the guard must pick a sideways escape or stand still
    front = np.column_stack((np.full(30, 0.40), np.linspace(-1.5, 1.5, 30)))
    back = np.column_stack((np.full(30, -0.35), np.linspace(-1.5, 1.5, 30)))
    (vx, vy, _), gap = safe_velocity((0.6, 0.0, 0.0), (0, 0, 0), np.vstack((front, back)), min_move=0.36)
    assert vx <= 0.05 and vx >= -0.05      # neither into the front wall nor into the back one


def test_physical_guard_stops_early_and_never_reverses():
    from mantf_g1_bridge.pure_pursuit import physical_response
    wall = np.column_stack((np.full(40, 0.95), np.linspace(-2, 2, 40)))
    # a kinematic G1 may keep walking at 0.95 m; the physical one (keeps moving ~0.1 m/s and needs stopping distance)
    # must not head straight on
    (kx, _, _), _ = safe_velocity((0.6, 0.0, 0.0), (0, 0, 0), wall, min_move=0.05, margin=0.10)
    (px, py, _), _ = safe_velocity((0.6, 0.0, 0.0), (0, 0, 0), wall, min_move=0.36, margin=0.15, physical=True)
    assert kx >= 0.29                                 # the kinematic G1 still walks on (slower)
    assert not (px > 0.3 and abs(py) < 0.05)
    close = np.column_stack((np.full(40, 0.55), np.linspace(-2, 2, 40)))
    (vx, _, _), _ = safe_velocity((0.0, 0.0, 0.0), (0, 0, 0), close, min_move=0.36, margin=0.15, physical=True)
    assert vx >= 0.0                                  # never a backward command for the physical G1
    assert physical_response([[0.0, 0.0]])[0][0] == 0 and physical_response([[0.5, 0.0]])[0][0] > 0.25


def test_physical_guard_turns_out_of_a_corner_it_cannot_back_out_of():
    # wall right in front (inside the margin) and a wall on the right: the physical G1 cannot reverse, so it must
    # turn towards the open left side instead of freezing
    front = np.column_stack((np.full(30, 0.55), np.linspace(-1.5, 1.5, 30)))
    right = np.column_stack((np.linspace(-1.0, 0.55, 20), np.full(20, -0.40)))
    (vx, vy, wz), _ = safe_velocity((0.0, 0.0, 0.0), (0, 0, 0), np.vstack((front, right)), min_move=0.36,
                                    margin=0.15, physical=True)
    assert vx >= 0.0 and (wz > 0 or vy > 0)
