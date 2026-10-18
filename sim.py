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


def track(terrain, obj, ts):
    """Truth path: constant speed along the waypoints, fixed height above ground."""
    pts = np.array([terrain.xy(lat, lon) for lat, lon in obj["waypoints"]])
    if obj.get("loop"):
        pts = np.vstack([pts, pts[:1]])
    seg = np.diff(pts, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    s = obj["speed_ms"] * ts
    s = s % cum[-1] if obj.get("loop") else np.minimum(s, cum[-1])
    # ponytail: straight segments with sharp turns; spline the waypoints if turns look wrong
    x, y = np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])
    i = np.clip(np.searchsorted(cum, s, "right") - 1, 0, len(seg) - 1)
    heading = np.degrees(np.arctan2(seg[i, 0], seg[i, 1])) % 360
    return x, y, terrain.h(x, y) + obj["alt_agl_m"], heading


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


def bearing_sigma(snr_db, sigma_at_20db):
    """Bearing noise grows as 1/sqrt(SNR); capped at 30 deg."""
    return np.minimum(sigma_at_20db * 10 ** ((20 - snr_db) / 20), 30.0)


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


def run(scenario_path, out_dir):
    sc = json.loads(Path(scenario_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    terrain = Terrain(fetch_dem(sc["bbox"]), sc["bbox"])
    dt = sc.get("dt_s", 1.0)
    ts = np.arange(0, sc["duration_s"], dt)
    start = datetime.fromisoformat(sc["start_utc"].replace("Z", "+00:00"))
    iso = lambda t: (start + timedelta(seconds=float(t))).strftime("%Y-%m-%dT%H:%M:%SZ")

    truth = {}
    with open(out / "truth.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_utc", "t_s", "object_id", "lat", "lon", "alt_msl_m", "alt_agl_m", "heading_deg", "speed_ms"])
        for o in sc["objects"]:
            x, y, z, hd = truth[o["id"]] = track(terrain, o, ts)
            lat, lon = terrain.latlon(x, y)
            for i, t in enumerate(ts):
                wr.writerow([iso(t), t, o["id"], f"{lat[i]:.6f}", f"{lon[i]:.6f}", f"{z[i]:.1f}", o["alt_agl_m"], f"{hd[i]:.1f}", o["speed_ms"]])

    sensors, links = [], {}
    for si, s in enumerate(sc["sensors"]):
        sx, sy = terrain.xy(s["lat"], s["lon"])
        rx = np.array([sx, sy, float(terrain.h(sx, sy)) + s["mast_m"]])
        sensors.append({**s, "x": sx, "y": sy, "z": rx[2]})
        noise_dbm = -174 + 60 + s["noise_figure_db"]  # +60 = 10*log10(1 MHz), scaled per emitter below
        for oi, o in enumerate(sc["objects"]):
            rng = np.random.default_rng([sc["seed"], si, oi])
            x, y, z, _ = truth[o["id"]]
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

    # ponytail: association uses source_object_id (answer key); a real system must associate by freq/time
    fused = {o["id"]: [] for o in sc["objects"]}
    events, n = [], 0
    with open(out / "fused.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["uid", "time_utc", "t_s", "lat", "lon", "ce_m", "n_sensors", "source_object_id", "error_m"])
        for i, t in enumerate(ts):
            for o in sc["objects"]:
                heard = [(s, links[(s["id"], o["id"])]) for s in sensors if links[(s["id"], o["id"])]["det"][i]]
                if len(heard) < 2:
                    continue
                fix = cross_bearings(np.array([[s["x"], s["y"]] for s, _ in heard]),
                                     np.array([L["bearing"][i] for _, L in heard]),
                                     np.array([L["sigma"][i] for _, L in heard]))
                if fix is None:
                    continue
                (ex, ey), ce = fix
                tx_, ty_ = truth[o["id"]][0][i], truth[o["id"]][1][i]
                lat, lon = terrain.latlon(ex, ey)
                uid = f"sig-{n:05d}"
                n += 1
                wr.writerow([uid, iso(t), t, f"{lat:.6f}", f"{lon:.6f}", f"{ce:.0f}", len(heard), o["id"], f"{math.hypot(ex - tx_, ey - ty_):.0f}"])
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
    (out / "fused.cot.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<events>\n' + "\n".join(events) + "\n</events>\n")
    (out / "scenario.json").write_text(json.dumps(sc, indent=2))

    # Viewer bundle: downsampled terrain + every series in the local metre frame.
    step = max(1, math.ceil(max(terrain.z.shape) / 256))
    zs = terrain.z[::step, ::step]
    r1 = lambda a: [round(float(v), 1) for v in a]
    bundle = {
        "scenario": sc,
        "frame": {"lat0": terrain.lat0, "lon0": terrain.lon0, "mx": terrain.mx, "my": terrain.my},
        "terrain": {"nx": zs.shape[1], "ny": zs.shape[0], "x0": terrain.xw + terrain.dx * step / 2,
                    "y0": terrain.yn - terrain.dy * step / 2, "dx": terrain.dx * step, "dy": terrain.dy * step,
                    "z": [round(float(v)) for v in zs.ravel()]},
        "t": [float(t) for t in ts],
        "sensors": [{"id": s["id"], "x": s["x"], "y": s["y"], "z": s["z"], "threshold_db": s["threshold_db"]} for s in sensors],
        "truth": {k: {"x": r1(v[0]), "y": r1(v[1]), "z": r1(v[2])} for k, v in truth.items()},
        "links": {f"{sid}|{oid}": {k: (r1(L[k]) if k != "det" else [int(b) for b in L[k]])
                                   for k in ("power", "snr", "det", "bearing", "sigma", "dist", "fspl", "diff", "shadow")}
                  | {"noise": round(L["noise"], 1)} for (sid, oid), L in links.items()},
        "fused": fused,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))
    return links, fused, ts


if __name__ == "__main__":
    links, fused, ts = run(sys.argv[1] if len(sys.argv) > 1 else "scenario.json", sys.argv[2] if len(sys.argv) > 2 else "out")
    for (sid, oid), L in links.items():
        print(f"{sid} <- {oid}: detected {L['det'].mean():5.1%}  blocked by terrain {(L['diff'] > 10).mean():5.1%}")
    for oid, f in fused.items():
        print(f"{oid}: {len(f)} fixes")
