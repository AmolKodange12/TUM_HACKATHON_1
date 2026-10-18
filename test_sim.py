"""Faithfulness checks. Run: python test_sim.py (or pytest)."""
import filecmp
import json
import math
import tempfile
from pathlib import Path

import numpy as np

import flight
import sim

BBOX = [31.40, 49.68, 31.56, 49.80]
DRONES = json.loads((Path(__file__).parent / "drones.json").read_text())


def flat(height=100.0):
    return sim.Terrain(np.full((200, 200), height), BBOX)


def ll(x, y):
    lat, lon = flat().latlon(x, y)
    return {"lat": float(lat), "lon": float(lon)}


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


def test_radar_power_falls_with_distance_on_flat_ground():
    t = flat()
    rx = np.array([0.0, -5000.0, 110.0])
    tx = np.stack([np.zeros(50), np.linspace(-4900, 5000, 50), np.full(50, 160.0)], 1)
    _, fspl, diff, _ = sim.path_loss(t, rx, tx, 9500.0)
    power = sim.radar_echo_power(45.0, 30.0, fspl, diff, -10.0, 9500.0, 25.0)
    assert np.all(np.diff(power) < 0)


def test_radial_velocity_sign_matches_closing_or_receding():
    rx = np.array([0.0, 0.0, 0.0])
    tx = np.array([[0.0, 1000.0, 0.0]])  # due north of the radar
    closing = sim.radial_velocity(rx, tx, np.array([0.0]), np.array([-20.0]))  # flying south: closing
    receding = sim.radial_velocity(rx, tx, np.array([0.0]), np.array([20.0]))  # flying north: receding
    assert closing[0] < 0 < receding[0]


def test_radar_fix_with_no_noise_finds_the_target():
    rx = np.array([0.0, 0.0])
    target = np.array([1000.0, 2000.0])
    rng_m = float(np.hypot(*(target - rx)))
    bearing = math.degrees(math.atan2(target[0] - rx[0], target[1] - rx[1])) % 360
    xy, ce = sim.radar_fix(rx, rng_m, bearing, 0.001, 0.001)
    assert np.hypot(*(xy - target)) < 1 and ce > 0


def test_crossed_bearings_find_the_target():
    target = np.array([1000.0, 2000.0])
    pts = np.array([[-3000.0, -2000.0], [3000.0, -2000.0]])
    b = np.degrees(np.arctan2(*(target - pts).T)) % 360
    est, ce = sim.cross_bearings(pts, b, np.array([2.0, 2.0]))
    assert np.hypot(*(est - target)) < 1 and ce > 0


def test_quadcopter_respects_its_limits_and_lands_on_target():
    q = DRONES["quadcopter"]
    obj = {"id": "q", "start": {**ll(0, 0), "alt_m": 0}, "plan": [
        {"takeoff": {"alt_m": 100}}, {"goto": {**ll(0, 2000), "speed_mps": 15}}, {"hover": {"duration_s": 30}},
        {"goto": ll(0, 0)}, {"land": {}}]}
    tr = flight.fly(q, obj, flat().xy)
    vh = np.hypot(tr[:, 5], tr[:, 6])
    acc = np.linalg.norm(np.diff(tr[:, 5:7], axis=0), axis=1) / flight.DT
    tilt = np.degrees(np.arccos(np.cos(np.radians(tr[:, 10])) * np.cos(np.radians(tr[:, 11]))))
    assert vh.max() <= 15.01 and acc.max() <= q["flight"]["max_accel_mps2"] + 1e-6
    assert tilt.max() <= q["flight"]["max_tilt_deg"] + 0.5
    hover = tr[tr[:, 1] == 2]
    assert np.hypot(hover[:, 2], hover[:, 3] - 2000).max() < 1.0
    assert tr[-1, 4] == 0 and np.hypot(tr[-1, 2], tr[-1, 3]) < 1.0


def test_fixed_wing_banks_and_holds_an_orbit_in_wind():
    r = DRONES["fixed_wing_recon"]
    obj = {"id": "r", "start": {**ll(-6000, 0), "alt_m": 1500, "heading_deg": 90}, "plan": [
        {"orbit": {**ll(0, 0), "radius_m": 1500, "duration_s": 3600, "speed_mps": 30}}]}
    tr = flight.fly(r, obj, flat().xy, flight.wind_xy({"from_deg": 270, "speed_mps": 15}))
    late = tr[tr[:, 0] > 900]
    gs = np.hypot(late[:, 5], late[:, 6])
    assert np.abs(np.hypot(late[:, 2], late[:, 3]) - 1500).max() < 100
    assert np.abs(tr[:, 11]).max() <= r["flight"]["max_bank_deg"] + 1e-6
    assert gs.max() - gs.min() > 25  # ~30 +/- 15 m/s over the ground around the circle
    assert np.abs(late[:, 8] - 30).max() < 0.01  # airspeed held


def test_same_seed_same_bytes():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sim.run("scenario.json", a)
        sim.run("scenario.json", b)
        files = ["truth.csv", "rf_raw.csv", "rf_budget.csv", "radar_raw.csv", "radar_budget.csv", "fused.csv", "fused.cot.xml", "bundle.json"]
        match, mismatch, err = filecmp.cmpfiles(a, b, files, shallow=False)
        assert not mismatch and not err, (mismatch, err)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
