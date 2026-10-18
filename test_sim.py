"""Faithfulness checks. Run: python test_sim.py (or pytest)."""
import filecmp
import json
import tempfile

import numpy as np

import sim

BBOX = [31.40, 49.68, 31.56, 49.80]


def flat(height=100.0):
    return sim.Terrain(np.full((200, 200), height), BBOX)


def test_power_falls_with_distance_on_flat_ground():
    t = flat()
    rx = np.array([0.0, -5000.0, 110.0])
    tx = np.stack([np.zeros(50), np.linspace(-4900, 5000, 50), np.full(50, 160.0)], 1)
    _, fspl, diff, _ = sim.path_loss(t, rx, tx, 2437.5)
    assert np.all(np.diff(fspl + diff) > 0)


def test_ridge_blocks_signal():
    t = flat()
    open_ = sim.path_loss(t, np.array([0.0, -4000.0, 110.0]), np.array([[0.0, 3000.0, 160.0]]), 2437.5)
    z = t.z.copy()
    z[95:105, :] = 250.0  # 150 m ridge across the middle of the box
    ridge = sim.Terrain(z, BBOX)
    blocked = sim.path_loss(ridge, np.array([0.0, -4000.0, 110.0]), np.array([[0.0, 3000.0, 160.0]]), 2437.5)
    loss_open, loss_ridge = open_[1] + open_[2], blocked[1] + blocked[2]
    assert loss_ridge[0] - loss_open[0] > 15, (loss_open, loss_ridge)
    assert blocked[3][0] > 0 > open_[3][0]  # Fresnel v: ridge above the ray vs clear path


def test_bearing_noise_grows_as_snr_drops():
    s = sim.bearing_sigma(np.array([30.0, 20.0, 10.0, 0.0]), 2.0)
    assert np.all(np.diff(s) > 0) and s[1] == 2.0


def test_crossed_bearings_find_the_target():
    target = np.array([1000.0, 2000.0])
    pts = np.array([[-3000.0, -2000.0], [3000.0, -2000.0]])
    b = np.degrees(np.arctan2(*(target - pts).T)) % 360
    est, ce = sim.cross_bearings(pts, b, np.array([2.0, 2.0]))
    assert np.hypot(*(est - target)) < 1 and ce > 0


def test_same_seed_same_bytes():
    sc = json.loads(open("scenario.json").read())
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sim.run(sc, a)
        sim.run(sc, b)
        files = ["truth.csv", "rf_raw.csv", "rf_budget.csv", "fused.csv", "fused.cot.xml", "bundle.json"]
        match, mismatch, err = filecmp.cmpfiles(a, b, files, shallow=False)
        assert not mismatch and not err, (mismatch, err)


def test_adding_a_drone_does_not_move_the_others():
    area = {"name": "t", "region": "t", "bbox": BBOX}
    a, b = sim.make_scenario(area, 3, 2, 7), sim.make_scenario(area, 4, 3, 7)
    assert a["sensors"] == b["sensors"][:3] and a["objects"] == b["objects"][:2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
