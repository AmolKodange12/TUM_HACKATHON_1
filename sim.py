"""Mirror World MVP: scenario.json -> truth, RF sensor data, fused TAK events.

Usage: python sim.py [scenario.json] [out_dir]
"""
import csv
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import flight

C = 299_792_458.0
K_EARTH_R = 4 / 3 * 6_371_000  # effective earth radius, standard refraction
CACHE = Path(__file__).parent / ".cache"


def fetch_dem(bbox):
    """Copernicus GLO-30 for bbox [west, south, east, north], rows north -> south."""
    west, south, east, north = bbox
    # ponytail: one 1x1 degree tile only; mosaic tiles if a box ever crosses a degree line
    assert math.floor(west) == math.floor(east) and math.floor(south) == math.floor(north), "bbox crosses a tile edge"
    cached = CACHE / f"dem_{'_'.join(map(str, bbox))}.npy"
    if cached.exists():
        return np.load(cached)
    import rasterio
    from rasterio.windows import from_bounds

    name = f"Copernicus_DSM_COG_10_N{math.floor(south):02d}_00_E{math.floor(west):03d}_00_DEM"
    with rasterio.open(f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif") as src:
        z = src.read(1, window=from_bounds(west, south, east, north, src.transform)).astype(np.float64)
    CACHE.mkdir(exist_ok=True)
    np.save(cached, z)
    return z


class Terrain:
    """DEM grid in a local metre frame: x east, y north, origin at the bbox centre."""

    def __init__(self, z, bbox):
        west, south, east, north = bbox
        self.z, self.bbox = z, bbox
        self.lat0, self.lon0 = (south + north) / 2, (west + east) / 2
        # ponytail: flat-earth projection, < 0.1% error at 15 km; use pyproj for bigger areas
        self.mx, self.my = 111_320 * math.cos(math.radians(self.lat0)), 110_540
        self.xw, self.yn = self.xy(north, west)
        self.dx = (east - west) * self.mx / z.shape[1]
        self.dy = (north - south) * self.my / z.shape[0]

    def xy(self, lat, lon):
        return (lon - self.lon0) * self.mx, (lat - self.lat0) * self.my

    def latlon(self, x, y):
        return self.lat0 + np.asarray(y) / self.my, self.lon0 + np.asarray(x) / self.mx

    def h(self, x, y):
        """Bilinear ground height (m MSL) at x, y; clamps to the grid edge."""
        ny, nx = self.z.shape
        c = np.clip((np.asarray(x) - self.xw) / self.dx - 0.5, 0, nx - 1.001)
        r = np.clip((self.yn - np.asarray(y)) / self.dy - 0.5, 0, ny - 1.001)
        c0, r0 = c.astype(int), r.astype(int)
        fc, fr = c - c0, r - r0
        z = self.z
        top = z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc
        bot = z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bot * fr


def track(terrain, obj, ts, drone, wind=np.zeros(2)):
    """Truth path from the flight model (flight.py), at the object's height above the ground.

    The drone starts at waypoints[0] at alt_agl_m and speed_ms, then flies `plan` if given,
    else a goto through each further waypoint (repeated if `loop`). When the plan ends early,
    a multirotor hovers and a fixed-wing circles its last point.
    Returns x, y, z (m MSL), heading, pitch, roll, height above ground, ground speed and vx, vy (m/s).
    """
    (lat0, lon0), rest, speed = obj["waypoints"][0], obj["waypoints"][1:], obj["speed_ms"]
    plan = obj.get("plan")
    if plan is None:
        pts = rest + [[lat0, lon0]] if obj.get("loop") else rest
        if obj.get("loop"):
            xy = np.array([terrain.xy(*p) for p in [[lat0, lon0], *pts]])
            lap = np.hypot(*np.diff(xy, axis=0).T).sum()
            pts = pts * (math.ceil(ts[-1] * speed / lap) + 1)
        plan = [{"goto": {"lat": la, "lon": lo}} for la, lo in pts]
    plan = [{k: {"speed_mps": speed, **(a or {})}} for leg in plan for k, a in leg.items()]
    end = plan[-1][next(iter(plan[-1]))] if plan else {"lat": lat0, "lon": lon0}
    if drone["type"] == "multirotor":
        plan.append({"hover": {"duration_s": ts[-1] + 1}})
    else:
        plan.append({"orbit": {"lat": end.get("lat", lat0), "lon": end.get("lon", lon0), "duration_s": ts[-1] + 1,
                               "radius_m": round(2 * flight.min_orbit_radius(drone, wind)), "speed_mps": speed}})
    start = {"lat": lat0, "lon": lon0, "alt_m": obj["alt_agl_m"], "speed_mps": speed}
    rows = flight.fly(drone, {"id": obj["id"], "start": start, "plan": plan}, terrain.xy, wind, t_end=ts[-1])
    r = rows[np.minimum(np.rint(ts / flight.DT).astype(int), len(rows) - 1)]
    x, y, agl = r[:, 2], r[:, 3], r[:, 4]
    # ponytail: terrain following at a fixed height above ground; no climbs over ridges or minimum MSL
    return x, y, terrain.h(x, y) + agl, r[:, 9], r[:, 10], r[:, 11], agl, np.hypot(r[:, 5], r[:, 6]), r[:, 5], r[:, 6]


def path_loss(terrain, rx, tx, freq_mhz, n=64):
    """Free-space + single knife-edge diffraction (ITU-R P.526) between rx (3,) and tx (T, 3).

    Returns 3D distance, FSPL dB, diffraction dB, and the worst Fresnel-Kirchhoff v
    (v < 0: the direct path clears all terrain).
    """
    f = np.linspace(0, 1, n + 2)[1:-1]
    dxy = tx[:, :2] - rx[:2]
    d = np.maximum(np.hypot(dxy[:, 0], dxy[:, 1]), 1.0)[:, None]
    ground = terrain.h(rx[0] + f * dxy[:, :1], rx[1] + f * dxy[:, 1:2])
    d1, d2 = f * d, (1 - f) * d
    ray = rx[2] + f * (tx[:, 2:3] - rx[2])
    excess = ground + d1 * d2 / (2 * K_EARTH_R) - ray
    lam = C / (freq_mhz * 1e6)
    v = (excess * np.sqrt(2 * d / (lam * d1 * d2))).max(axis=1)
    # ponytail: worst single edge only; Deygout (3 edges) when ridges stack up
    diff = np.where(v > -0.78, 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1), 0.0)
    d3 = np.maximum(np.hypot(d[:, 0], tx[:, 2] - rx[2]), 1.0)
    fspl = 20 * np.log10(d3) + 20 * np.log10(freq_mhz * 1e6) - 147.55
    return d3, fspl, diff, v


def shadowing(rng, n, sigma_db, tau_s, dt):
    """Log-normal shadowing that drifts slowly (AR(1)) instead of jumping every tick."""
    rho = math.exp(-dt / tau_s)
    e = rng.standard_normal(n) * sigma_db
    s = np.empty(n)
    s[0] = e[0]
    for i in range(1, n):
        s[i] = rho * s[i - 1] + math.sqrt(1 - rho**2) * e[i]
    return s


def meas_sigma(snr_db, sigma_at_20db, cap):
    """Measurement noise that grows as 1/sqrt(SNR) below 20 dB, capped."""
    return np.minimum(sigma_at_20db * 10 ** ((20 - snr_db) / 20), cap)


def bearing_sigma(snr_db, sigma_at_20db):
    """Bearing noise grows as 1/sqrt(SNR); capped at 30 deg."""
    return meas_sigma(snr_db, sigma_at_20db, 30.0)


def radar_echo_power(tx_dbm, gain_dbi, fspl_db, diff_db, rcs_dbsm, freq_mhz, integration_gain_db=0.0):
    """Monostatic radar equation: the two-way path loss, the target's reflected aperture (RCS), and the
    SNR gain of integrating many pulses over a dwell (a single pulse at this power would rarely detect anything).
    """
    lam = C / (freq_mhz * 1e6)
    gain_rcs = 10 * math.log10(4 * math.pi) - 20 * math.log10(lam) + rcs_dbsm
    return tx_dbm + 2 * gain_dbi - 2 * (fspl_db + diff_db) + gain_rcs + integration_gain_db


def radial_velocity(rx, tx, vx, vy):
    """Signed closing speed of tx (T, 3) with ground velocity (vx, vy) as seen from rx (3,).

    Horizontal only (matches the bearing calculation below). Positive: moving away from rx.
    """
    dx, dy = tx[:, 0] - rx[0], tx[:, 1] - rx[1]
    d = np.maximum(np.hypot(dx, dy), 1.0)
    return (vx * dx + vy * dy) / d


def radar_fix(rx, rng_m, bearing_deg, rng_sigma_m, bearing_sigma_deg):
    """Direct range+bearing fix from a single radar (no crossing needed). Returns (xy, ce_m)."""
    b = math.radians(bearing_deg)
    xy = rx + rng_m * np.array([math.sin(b), math.cos(b)])
    ce = math.hypot(rng_sigma_m, rng_m * math.radians(bearing_sigma_deg))
    return xy, ce


def cross_bearings(points, bearings_deg, sigmas_deg):
    """Weighted least-squares crossing of bearing lines. Returns (xy, ce_m) or None."""
    b = np.radians(bearings_deg)
    u = np.stack([np.sin(b), np.cos(b)], 1)
    nrm = np.stack([np.cos(b), -np.sin(b)], 1)
    sin_pairs = [abs(math.sin(bi - bj)) for i, bi in enumerate(b) for bj in b[i + 1 :]]
    if max(sin_pairs) < math.sin(math.radians(10)):
        return None  # lines nearly parallel: no usable fix
    w = 1 / np.radians(sigmas_deg)
    for _ in range(2):  # 2nd pass weights by cross-range error in metres, which needs a range
        a = nrm * w[:, None]
        est = np.linalg.lstsq(a, (nrm * points).sum(1) * w, rcond=None)[0]
        r = np.maximum(np.hypot(*(est - points).T), 1.0)
        w = 1 / (r * np.radians(sigmas_deg))
    if np.any(((est - points) * u).sum(1) <= 0):
        return None  # fix lies behind a receiver
    cov = np.linalg.inv(a.T @ a)
    return est, float(math.sqrt(np.trace(cov)))


def make_scenario(area, n_rx, n_drones, seed):
    """Random but presentable layout for an area from areas.json.

    Receivers go on high ground (top 25% of the inner box), spread far apart. Drones fly
    random loops inside the inner 80% of the box. Each receiver and drone has its own RNG
    stream, so adding one never moves the others.
    """
    terrain = Terrain(fetch_dem(area["bbox"]), area["bbox"])
    z, (ny, nx) = terrain.z, terrain.z.shape
    r0, c0 = ny // 10, nx // 10
    inner = z[r0:ny - r0, c0:nx - c0]
    rows, cols = np.nonzero(inner >= np.percentile(inner, 75))
    cand = np.stack([terrain.xw + (cols + c0 + 0.5) * terrain.dx, terrain.yn - (rows + r0 + 0.5) * terrain.dy], 1)

    sensors, placed = [], []
    for i in range(n_rx):
        rng = np.random.default_rng([seed, 0, i])
        pick = cand[rng.choice(len(cand), 200)]
        if placed:  # farthest of 200 samples from the receivers already placed
            pick = pick[[np.argmax(np.linalg.norm(pick[:, None] - np.array(placed)[None], axis=2).min(1))]]
        placed.append(pick[0])
        lat, lon = terrain.latlon(*pick[0])
        sensors.append({"id": f"RF{i + 1}", "lat": round(float(lat), 5), "lon": round(float(lon), 5), "mast_m": 10,
                        "gain_dbi": 6, "noise_figure_db": 6, "threshold_db": 6, "bearing_sigma_deg": 2.0,
                        "bearing_bias_deg": round(float(rng.normal(0, 1)), 1)})

    drones = json.loads((Path(__file__).parent / "drones.json").read_text())
    types, weights = ["quadcopter", "fixed_wing_recon", "delta_wing", "decoy"], [0.5, 0.25, 0.15, 0.1]
    objects, xmax, ymax = [], -terrain.xw * 0.8, terrain.yn * 0.8
    for i in range(n_drones):
        rng = np.random.default_rng([seed, 1, i])
        kind = types[rng.choice(len(types), p=weights)]
        quad = drones[kind]["type"] == "multirotor"
        n = int(rng.integers(3, 6))
        lat, lon = terrain.latlon(rng.uniform(-xmax, xmax, n), rng.uniform(-ymax, ymax, n))
        objects.append({
            "id": f"drone_{i + 1:02d}", "type": kind,
            "freq_mhz": [2412.0, 2437.0, 2462.0][i % 3] if quad else [5745.0, 5785.0, 5825.0][i % 3],
            "eirp_dbm": 27 if quad else 30, "bandwidth_mhz": 10 if quad else 20,
            "alt_agl_m": int(rng.integers(60, 121)) if quad else int(rng.integers(200, 401)),
            "speed_ms": drones[kind]["flight"]["cruise_speed_mps"], "loop": True,
            "waypoints": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)],
        })

    return {"name": area["name"], "region": area["region"], "bbox": area["bbox"], "start_utc": "2026-11-03T14:00:00Z",
            "duration_s": 1200, "dt_s": 1.0, "seed": seed, "shadowing_db": 4.0, "shadowing_tau_s": 15.0,
            "sensors": sensors, "objects": objects}


def terrain_json(terrain, max_cells):
    """Downsampled DEM grid for the viewer, plus the frame to turn metres back into lat/lon."""
    step = max(1, math.ceil(max(terrain.z.shape) / max_cells))
    zs = terrain.z[::step, ::step]
    return ({"nx": zs.shape[1], "ny": zs.shape[0], "x0": terrain.xw + terrain.dx * step / 2,
             "y0": terrain.yn - terrain.dy * step / 2, "dx": terrain.dx * step, "dy": terrain.dy * step,
             "z": [round(float(v)) for v in zs.ravel()]},
            {"lat0": terrain.lat0, "lon0": terrain.lon0, "mx": terrain.mx, "my": terrain.my})


def run(sc, out_dir):
    """Simulate scenario dict sc into out_dir. Returns the viewer bundle."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    terrain = Terrain(fetch_dem(sc["bbox"]), sc["bbox"])
    dt = sc.get("dt_s", 1.0)
    ts = np.arange(0, sc["duration_s"], dt)
    start = datetime.fromisoformat(sc["start_utc"].replace("Z", "+00:00"))
    iso = lambda t: (start + timedelta(seconds=float(t))).strftime("%Y-%m-%dT%H:%M:%SZ")
    drones = json.loads((Path(__file__).parent / "drones.json").read_text())
    wind = flight.wind_xy(sc.get("wind"))
    for o in sc["objects"]:
        if o["type"] not in drones or o["type"].startswith("_"):
            raise ValueError(f"{o['id']}: unknown type {o['type']!r}, use one of "
                             f"{', '.join(k for k in drones if not k.startswith('_'))} (drones.json)")

    truth = {}
    with open(out / "truth.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_utc", "t_s", "object_id", "lat", "lon", "alt_msl_m", "alt_agl_m", "heading_deg",
                     "pitch_deg", "roll_deg", "speed_ms"])
        for o in sc["objects"]:
            x, y, z, hd, pitch, roll, agl, speed, vx, vy = truth[o["id"]] = track(terrain, o, ts, drones[o["type"]], wind)
            lat, lon = terrain.latlon(x, y)
            for i, t in enumerate(ts):
                wr.writerow([iso(t), t, o["id"], f"{lat[i]:.6f}", f"{lon[i]:.6f}", f"{z[i]:.1f}", f"{agl[i]:.1f}",
                             f"{hd[i]:.1f}", f"{pitch[i]:.1f}", f"{roll[i]:.1f}", f"{speed[i]:.1f}"])

    sensors, links = [], {}
    for si, s in enumerate(sc["sensors"]):
        sx, sy = terrain.xy(s["lat"], s["lon"])
        rx = np.array([sx, sy, float(terrain.h(sx, sy)) + s["mast_m"]])
        sensors.append({**s, "x": sx, "y": sy, "z": rx[2]})
        noise_dbm = -174 + 60 + s["noise_figure_db"]  # +60 = 10*log10(1 MHz), scaled per emitter below
        for oi, o in enumerate(sc["objects"]):
            rng = np.random.default_rng([sc["seed"], si, oi])
            x, y, z = truth[o["id"]][:3]
            tx = np.stack([x, y, z], 1)
            d3, fspl, diff, v = path_loss(terrain, rx, tx, o["freq_mhz"])
            shadow = shadowing(rng, len(ts), sc["shadowing_db"], sc["shadowing_tau_s"], dt)
            power = o["eirp_dbm"] + s["gain_dbi"] - fspl - diff - shadow
            noise = noise_dbm + 10 * math.log10(o["bandwidth_mhz"])
            snr = power - noise
            sigma = bearing_sigma(snr, s["bearing_sigma_deg"])
            true_b = np.degrees(np.arctan2(x - sx, y - sy)) % 360
            bearing = (true_b + s["bearing_bias_deg"] + rng.normal(0, 1, len(ts)) * sigma) % 360
            links[(s["id"], o["id"])] = dict(
                power=power, snr=snr, det=snr >= s["threshold_db"], bearing=bearing, sigma=sigma,
                true_bearing=true_b, dist=d3, fspl=fspl, diff=diff, shadow=shadow, v=v, noise=noise,
            )

    # Radar: active (own transmitter, own frequency), so it hears a silent drone too. One site already
    # gives a position fix (range + bearing), no crossing with another sensor needed.
    radars, radar_links = [], {}
    for ri, s in enumerate(sc.get("radars", [])):
        sx, sy = terrain.xy(s["lat"], s["lon"])
        rx = np.array([sx, sy, float(terrain.h(sx, sy)) + s["mast_m"]])
        radars.append({**s, "x": sx, "y": sy, "z": rx[2]})
        noise_dbm = -174 + 60 + s["noise_figure_db"] + 10 * math.log10(s["bandwidth_mhz"])
        for oi, o in enumerate(sc["objects"]):
            rng = np.random.default_rng([sc["seed"], 100 + ri, oi])
            x, y, z = truth[o["id"]][:3]
            vx, vy = truth[o["id"]][8], truth[o["id"]][9]
            tx = np.stack([x, y, z], 1)
            rcs_dbsm = drones[o["type"]]["rcs_dbsm"]
            d3, fspl, diff, v = path_loss(terrain, rx, tx, s["freq_mhz"])
            shadow = shadowing(rng, len(ts), sc["shadowing_db"], sc["shadowing_tau_s"], dt)
            power = radar_echo_power(s["tx_power_dbm"], s["gain_dbi"], fspl, diff, rcs_dbsm, s["freq_mhz"],
                                      s.get("integration_gain_db", 0.0)) - shadow
            snr = power - noise_dbm
            det = snr >= s["threshold_db"]
            sigma = bearing_sigma(snr, s["bearing_sigma_deg"])
            true_b = np.degrees(np.arctan2(x - sx, y - sy)) % 360
            bearing = (true_b + s["bearing_bias_deg"] + rng.normal(0, 1, len(ts)) * sigma) % 360
            range_sigma = meas_sigma(snr, s["range_sigma_m"], 2000.0)
            rng_meas = d3 + rng.normal(0, 1, len(ts)) * range_sigma
            radial_true = radial_velocity(rx, tx, vx, vy)
            radial_sigma = meas_sigma(snr, s.get("radial_sigma_mps", 1.0), 50.0)
            radial = radial_true + rng.normal(0, 1, len(ts)) * radial_sigma
            radar_links[(s["id"], o["id"])] = dict(
                power=power, snr=snr, det=det, bearing=bearing, sigma=sigma, true_bearing=true_b,
                dist=d3, fspl=2 * fspl, diff=2 * diff, shadow=shadow, v=v, noise=noise_dbm,
                range=rng_meas, range_sigma=range_sigma, radial=radial, radial_true=radial_true, doppler=-2 * radial / (C / (s["freq_mhz"] * 1e6)),
            )

    # What each receiver actually reports: detections only.
    with open(out / "rf_raw.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_utc", "t_s", "sensor_id", "freq_mhz", "power_dbm", "snr_db", "bearing_deg", "source_object_id"])
        for i, t in enumerate(ts):
            for (sid, oid), L in links.items():
                if L["det"][i]:
                    freq = next(o["freq_mhz"] for o in sc["objects"] if o["id"] == oid)
                    wr.writerow([iso(t), t, sid, freq, f"{L['power'][i]:.1f}", f"{L['snr'][i]:.1f}", f"{L['bearing'][i]:.1f}", oid])

    # Why each value is what it is: full link budget for every tick, detected or not.
    with open(out / "rf_budget.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "sensor_id", "object_id", "dist_m", "fspl_db", "diffraction_db", "shadow_db", "fresnel_v",
                     "power_dbm", "noise_dbm", "snr_db", "detected", "true_bearing_deg", "bearing_sigma_deg"])
        for (sid, oid), L in links.items():
            for i, t in enumerate(ts):
                wr.writerow([t, sid, oid, f"{L['dist'][i]:.0f}", f"{L['fspl'][i]:.1f}", f"{L['diff'][i]:.1f}", f"{L['shadow'][i]:.1f}",
                             f"{L['v'][i]:.2f}", f"{L['power'][i]:.1f}", f"{L['noise']:.1f}", f"{L['snr'][i]:.1f}", int(L["det"][i]),
                             f"{L['true_bearing'][i]:.1f}", f"{L['sigma'][i]:.1f}"])

    # What each radar actually reports: detections only. A direct range+bearing fix, no second radar needed.
    with open(out / "radar_raw.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_utc", "t_s", "radar_id", "range_m", "bearing_deg", "radial_velocity_mps", "snr_db", "source_object_id"])
        for i, t in enumerate(ts):
            for (sid, oid), L in radar_links.items():
                if L["det"][i]:
                    wr.writerow([iso(t), t, sid, f"{L['range'][i]:.0f}", f"{L['bearing'][i]:.1f}", f"{L['radial'][i]:.1f}",
                                 f"{L['snr'][i]:.1f}", oid])

    # Why each radar value is what it is: full two-way link budget for every tick, detected or not.
    with open(out / "radar_budget.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "radar_id", "object_id", "dist_m", "two_way_fspl_db", "two_way_diffraction_db", "shadow_db",
                     "power_dbm", "noise_dbm", "snr_db", "detected", "true_bearing_deg", "bearing_sigma_deg",
                     "range_sigma_m", "radial_velocity_true_mps", "doppler_hz"])
        for (sid, oid), L in radar_links.items():
            for i, t in enumerate(ts):
                wr.writerow([t, sid, oid, f"{L['dist'][i]:.0f}", f"{L['fspl'][i]:.1f}", f"{L['diff'][i]:.1f}", f"{L['shadow'][i]:.1f}",
                             f"{L['power'][i]:.1f}", f"{L['noise']:.1f}", f"{L['snr'][i]:.1f}", int(L["det"][i]),
                             f"{L['true_bearing'][i]:.1f}", f"{L['sigma'][i]:.1f}", f"{L['range_sigma'][i]:.1f}",
                             f"{L['radial_true'][i]:.1f}", f"{L['doppler'][i]:.0f}"])

    # ponytail: association uses source_object_id (answer key); a real system must associate by freq/time
    fused = {o["id"]: [] for o in sc["objects"]}
    events, n = [], 0
    with open(out / "fused.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["uid", "time_utc", "t_s", "lat", "lon", "ce_m", "n_sensors", "source_object_id", "error_m", "method"])
        for i, t in enumerate(ts):
            for o in sc["objects"]:
                heard = [(s, links[(s["id"], o["id"])]) for s in sensors if links[(s["id"], o["id"])]["det"][i]]
                fix = None if len(heard) < 2 else cross_bearings(
                    np.array([[s["x"], s["y"]] for s, _ in heard]),
                    np.array([L["bearing"][i] for _, L in heard]),
                    np.array([L["sigma"][i] for _, L in heard]))
                if fix is not None:
                    (ex, ey), ce = fix
                    tx_, ty_ = truth[o["id"]][0][i], truth[o["id"]][1][i]
                    lat, lon = terrain.latlon(ex, ey)
                    uid = f"sig-{n:05d}"
                    n += 1
                    wr.writerow([uid, iso(t), t, f"{lat:.6f}", f"{lon:.6f}", f"{ce:.0f}", len(heard), o["id"],
                                 f"{math.hypot(ex - tx_, ey - ty_):.0f}", "crossed_bearings"])
                    fused[o["id"]].append([float(t), round(float(ex)), round(float(ey)), round(ce)])
                    stale = iso(t + 60)
                    sig = "\n".join(
                        f'      <__signal sensor="{s["id"]}" freq_mhz="{o["freq_mhz"]}" power_dbm="{L["power"][i]:.1f}" bearing_deg="{L["bearing"][i]:.1f}"/>'
                        for s, L in heard)
                    events.append(
                        f'  <event version="2.0" uid="{uid}" type="a-u-A" how="m-r" time="{iso(t)}" start="{iso(t)}" stale="{stale}">\n'
                        f'    <point lat="{lat:.6f}" lon="{lon:.6f}" hae="9999999.0" ce="{ce:.0f}" le="9999999.0"/>\n'
                        f'    <detail>\n      <contact callsign="Unknown drone, {o["freq_mhz"] / 1000:.1f} GHz"/>\n'
                        f'      <remarks>Heard by {", ".join(s["id"] for s, _ in heard)}, position from crossed bearings</remarks>\n'
                        f'{sig}\n    </detail>\n  </event>')

                # Radar: a single site already gives range + bearing, so one detection is a fix.
                for s in radars:
                    L = radar_links[(s["id"], o["id"])]
                    if not L["det"][i]:
                        continue
                    (ex, ey), ce = radar_fix(np.array([s["x"], s["y"]]), L["range"][i], L["bearing"][i], L["range_sigma"][i], L["sigma"][i])
                    tx_, ty_ = truth[o["id"]][0][i], truth[o["id"]][1][i]
                    lat, lon = terrain.latlon(ex, ey)
                    uid = f"sig-{n:05d}"
                    n += 1
                    wr.writerow([uid, iso(t), t, f"{lat:.6f}", f"{lon:.6f}", f"{ce:.0f}", 1, o["id"],
                                 f"{math.hypot(ex - tx_, ey - ty_):.0f}", "radar"])
                    fused[o["id"]].append([float(t), round(float(ex)), round(float(ey)), round(ce)])
                    stale = iso(t + 60)
                    events.append(
                        f'  <event version="2.0" uid="{uid}" type="a-u-A" how="m-r" time="{iso(t)}" start="{iso(t)}" stale="{stale}">\n'
                        f'    <point lat="{lat:.6f}" lon="{lon:.6f}" hae="9999999.0" ce="{ce:.0f}" le="9999999.0"/>\n'
                        f'    <detail>\n      <contact callsign="Unknown drone, radar return"/>\n'
                        f'      <remarks>Heard by {s["id"]}, range {L["range"][i]:.0f} m, radial velocity {L["radial"][i]:.1f} m/s</remarks>\n'
                        f'      <__signal sensor="{s["id"]}" freq_mhz="{s["freq_mhz"]}" power_dbm="{L["power"][i]:.1f}" bearing_deg="{L["bearing"][i]:.1f}"/>\n'
                        f'    </detail>\n  </event>')
    (out / "fused.cot.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<events>\n' + "\n".join(events) + "\n</events>\n")
    (out / "scenario.json").write_text(json.dumps(sc, indent=2))

    # Viewer bundle: downsampled terrain + every series in the local metre frame.
    r1 = lambda a: [round(float(v), 1) for v in a]
    tj, frame = terrain_json(terrain, 256)
    bundle = {
        "scenario": sc,
        "frame": frame,
        "terrain": tj,
        "t": [float(t) for t in ts],
        "sensors": [{"id": s["id"], "x": s["x"], "y": s["y"], "z": s["z"], "threshold_db": s["threshold_db"], "kind": "rf"} for s in sensors]
                  + [{"id": s["id"], "x": s["x"], "y": s["y"], "z": s["z"], "threshold_db": s["threshold_db"], "kind": "radar"} for s in radars],
        "truth": {k: {"x": r1(v[0]), "y": r1(v[1]), "z": r1(v[2]), "h": r1(v[3]), "p": r1(v[4]), "r": r1(v[5])}
                  for k, v in truth.items()},
        "drones": {o["id"]: {"name": drones[o["type"]]["name"], "model": drones[o["type"]]["model"]} for o in sc["objects"]},
        "links": {f"{sid}|{oid}": {k: (r1(L[k]) if k != "det" else [int(b) for b in L[k]])
                                   for k in ("power", "snr", "det", "bearing", "sigma", "dist", "fspl", "diff", "shadow")}
                  | {"noise": round(L["noise"], 1)}
                  | ({"range": r1(L["range"]), "range_sigma": r1(L["range_sigma"]), "radial": r1(L["radial"]), "doppler": r1(L["doppler"])}
                     if "range" in L else {})
                  for (sid, oid), L in {**links, **radar_links}.items()},
        "fused": fused,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))
    return bundle


if __name__ == "__main__":
    sc = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "scenario.json").read_text())
    b = run(sc, sys.argv[2] if len(sys.argv) > 2 else "out")
    for k, L in b["links"].items():
        print(f"{k}: detected {np.mean(L['det']):5.1%}  blocked by terrain {np.mean(np.array(L['diff']) > 10):5.1%}")
    for oid, f in b["fused"].items():
        print(f"{oid}: {len(f)} fixes")
