"""Faithfulness checks. Run: python test_sim.py (or pytest)."""
import filecmp
import json
import math
import tempfile
from datetime import datetime, timezone
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


def test_ridge_hides_a_drone_from_a_camera():
    rx, tx = np.array([0.0, -4000.0, 110.0]), np.array([[0.0, 3000.0, 160.0]])
    z = flat().z.copy()
    z[95:105, :] = 250.0
    clear = -sim.profile(flat(), rx, tx)[3].max(axis=1)
    hidden = -sim.profile(sim.Terrain(z, BBOX), rx, tx)[3].max(axis=1)
    assert clear[0] > 0 > hidden[0]


def test_cameras_get_fewer_pixels_further_away_and_recognise_with_more():
    px = sim.pixels_on_target(0.35, np.array([500.0, 1000.0, 2000.0]), 6, 1920)
    assert np.all(np.diff(px) < 0) and abs(px[0] - 2 * px[1]) < 1e-9
    p = sim.johnson(np.array([4.0, 8.0, 16.0, 1e4]), 8)
    assert np.all(np.diff(p) > 0) and abs(p[1] - 0.5) < 1e-12 and p[-1] <= 1


def test_sun_is_up_at_midday_and_down_at_midnight():
    noon = sim.sun_elevation(49.74, 31.48, datetime(2026, 6, 21, 9, 56, tzinfo=timezone.utc))
    night = sim.sun_elevation(49.74, 31.48, datetime(2026, 6, 21, 21, 56, tzinfo=timezone.utc))
    assert 62 < noon < 65 and night < -15  # 90 - lat +/- 23.4 deg at the solstice


def test_sound_points_to_where_the_drone_was():
    ts = np.arange(0, 120.0)
    x, y, z = -3000 + 50 * ts, np.full_like(ts, 3000.0), np.full_like(ts, 400.0)  # eastbound, 50 m/s
    s = {"freq_hz": 500, "absorption_db_per_km": 5, "ambient_db": 40, "mics": 6, "processing_gain_db": 10,
         "threshold_db": 6, "bearing_sigma_deg": 3.0, "bearing_bias_deg": 0.0}
    L = sim.acoustic_link(flat(), s, np.array([0.0, 0.0, 102.0]), (x, y, z), DRONES["delta_wing"], ts, 1.0, 0.0,
                          np.random.default_rng(0))
    lag = math.hypot(1000, 3000, 298) / sim.SOUND  # at t = 80 s the drone is at x = 1000 m
    now = math.degrees(math.atan2(1000, 3000))
    assert abs(L["lag"][80] - lag) < 1e-6 and L["true_bearing"][80] < now - 5


def test_crossed_bearings_find_the_target():
    target = np.array([1000.0, 2000.0])
    pts = np.array([[-3000.0, -2000.0], [3000.0, -2000.0]])
    b = np.degrees(np.arctan2(*(target - pts).T)) % 360
    est, ce = sim.cross_bearings(pts, b, np.array([2.0, 2.0]))
    assert np.hypot(*(est - target)) < 1 and ce > 0


def test_coverage_falls_with_distance_and_needs_two_receivers():
    t = flat()
    rx = lambda x, y: {"x": x, "y": y, "z": 110.0, "gain_dbi": 6, "noise_figure_db": 6, "threshold_db": 6, "bearing_sigma_deg": 2.0}
    em = {"freq_mhz": 2437.5, "eirp_dbm": 10, "bandwidth_mhz": 10}
    grid, heard, located, err = sim.coverage(t, [rx(0.0, 0.0)], em, [100], 4.0, n=20)
    row = heard[0, grid["ny"] // 2]  # west -> east through the receiver
    mid = grid["nx"] // 2
    assert row[mid] > 0.99 and np.all(np.diff(row[mid:]) <= 1e-9) and row[-1] < row[mid]
    assert located.max() < 1e-9 and np.isnan(err).all()  # one receiver hears, but cannot locate
    strong = {**em, "eirp_dbm": 27}
    _, _, located2, err2 = sim.coverage(t, [rx(-3000.0, 0.0), rx(3000.0, 0.0)], strong, [100], 4.0, n=20)
    assert located2.max() > 0.9 and np.nanmin(err2) < 400  # best off the baseline, where the bearings cross square


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
    sc = json.loads(open("scenario.json").read())
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sim.run(sc, a)
        sim.run(sc, b)
        files = ["truth.csv", "rf_raw.csv", "rf_budget.csv", "radar_raw.csv", "radar_budget.csv", "camera_raw.csv",
                 "camera_budget.csv", "acoustic_raw.csv", "acoustic_budget.csv", "remote_id_raw.csv", "remote_id_budget.csv",
                 "fused.csv", "fused.cot.xml", "bundle.json"]
        match, mismatch, err = filecmp.cmpfiles(a, b, files, shallow=False)
        assert not mismatch and not err, (mismatch, err)


def test_adding_a_drone_does_not_move_the_others():
    area = {"name": "t", "region": "t", "bbox": BBOX}
    a, b = sim.make_scenario(area, 3, 2, 7), sim.make_scenario(area, 4, 3, 7)
    assert a["sensors"] == b["sensors"][:3] and a["objects"] == b["objects"][:2]
    assert all(a[k] == b[k] and len(a[k]) > 0 for k in ("radars", "cameras", "acoustic_arrays", "remote_id_receivers"))


def test_no_sensor_detects_a_drone_before_it_appears():
    sc = sim.make_scenario({"name": "t", "region": "t", "bbox": BBOX}, 3, 2, 1, {"mix": "quadcopters", "duration_s": 600})
    sc["objects"][0]["start_s"] = 300
    with tempfile.TemporaryDirectory() as d:
        b = sim.run(sc, d)
    oid, early = sc["objects"][0]["id"], np.array(b["t"]) < 300
    kinds = {s["kind"] for s in b["sensors"] if f"{s['id']}|{oid}" in b["links"]}
    assert kinds == {"rf", "radar", "eo", "ir", "acoustic", "remote_id"}
    assert not any(np.array(b["links"][f"{s['id']}|{oid}"]["det"])[early].any()
                   for s in b["sensors"] if f"{s['id']}|{oid}" in b["links"])
    assert all(f[0] >= 300 for f in b["fused"][oid])


def test_only_quadcopters_broadcast_remote_id():
    sc = sim.make_scenario({"name": "t", "region": "t", "bbox": BBOX}, 3, 6, 7)
    assert all(o["remote_id"] == (o["type"] == "quadcopter") for o in sc["objects"])


def test_layout_settings_reach_the_scenario_and_are_clamped():
    area = {"name": "t", "region": "t", "bbox": BBOX}
    sc = sim.make_scenario(area, 3, 6, 7, {"mix": "attack", "mast_m": 99, "duration_s": 600, "wind_mps": 8, "mix_typo": 1})
    assert {o["type"] for o in sc["objects"]} <= {"fixed_wing_recon", "delta_wing", "decoy"}
    assert all(s["mast_m"] == 40 for s in sc["sensors"]) and sc["duration_s"] == 600 and sc["wind"]["speed_mps"] == 8
    assert sim.make_scenario(area, 3, 2, 7)["objects"] == sim.make_scenario(area, 3, 2, 7, {})["objects"]  # defaults unchanged


def test_game_birds_are_only_heard_while_they_are_there():
    area = {"name": "t", "region": "t", "bbox": BBOX}
    huts = [[49.71, 31.44], [49.77, 31.47], [49.72, 31.53]]
    sc = sim.make_game(area, huts, 4, 5)
    assert sim.make_game(area, huts + [[49.75, 31.50]], 6, 5)["objects"][:4] == sc["objects"]  # more birds or huts move nobody
    with tempfile.TemporaryDirectory() as d:
        b = sim.run(sc, d)
    for o in sc["objects"]:
        first, last = b["spawn"][o["id"]]
        assert first == o["start_s"] and last > first
        for s in b["sensors"]:
            det = np.array(b["links"][f"{s['id']}|{o['id']}"]["det"])
            assert not det[(np.array(b["t"]) < first) | (np.array(b["t"]) > last)].any()
        assert all(first <= f[0] <= last for f in b["fused"][o["id"]])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
