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
SOUND = 343.0  # m/s at 20 C
RID_EIRP_DBM = 14.0  # Remote ID broadcast (Wi-Fi beacon / Bluetooth 5 long range), typical
RID_GNSS_SIGMA_M = 3.0  # error of the position a drone reports about itself


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
    a multirotor hovers and a fixed-wing circles its last point, unless `leave` is set: then it
    is gone. `start_s` delays its appearance (it waits, invisible, at its first waypoint).
    Returns x, y, z (m MSL), heading, pitch, roll, height above ground, ground speed,
    `present` (True while the drone is in the scenario) and vx, vy (m/s).
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
    t0 = float(obj.get("start_s", 0))
    if not obj.get("leave") and drone["type"] == "multirotor":
        plan.append({"hover": {"duration_s": ts[-1] + 1}})
    elif not obj.get("leave"):
        plan.append({"orbit": {"lat": end.get("lat", lat0), "lon": end.get("lon", lon0), "duration_s": ts[-1] + 1,
                               "radius_m": round(2 * flight.min_orbit_radius(drone, wind)), "speed_mps": speed}})
    start = {"lat": lat0, "lon": lon0, "alt_m": obj["alt_agl_m"], "speed_mps": speed}
    rows = flight.fly(drone, {"id": obj["id"], "start": start, "plan": plan}, terrain.xy, wind, t_end=max(ts[-1] - t0, 0))
    r = rows[np.clip(np.rint((ts - t0) / flight.DT).astype(int), 0, len(rows) - 1)]
    present = (ts >= t0) & ((ts - t0 <= rows[-1, 0]) if obj.get("leave") else True)
    x, y, agl = r[:, 2], r[:, 3], r[:, 4]
    # ponytail: terrain following at a fixed height above ground; no climbs over ridges or minimum MSL
    return x, y, terrain.h(x, y) + agl, r[:, 9], r[:, 10], r[:, 11], agl, np.hypot(r[:, 5], r[:, 6]), present, r[:, 5], r[:, 6]


def profile(terrain, rx, tx, n=64):
    """Terrain under the straight line rx (3,) -> tx (T, 3), with the earth bulge (k = 4/3).

    Returns the horizontal distance d (T, 1), the distances d1, d2 from each end (T, n), and how far
    the terrain rises above the line at each sample (excess > 0: terrain above the line).
    """
    f = np.linspace(0, 1, n + 2)[1:-1]
    dxy = tx[:, :2] - rx[:2]
    d = np.maximum(np.hypot(dxy[:, 0], dxy[:, 1]), 1.0)[:, None]
    ground = terrain.h(rx[0] + f * dxy[:, :1], rx[1] + f * dxy[:, 1:2])
    d1, d2 = f * d, (1 - f) * d
    ray = rx[2] + f * (tx[:, 2:3] - rx[2])
    return d, d1, d2, ground + d1 * d2 / (2 * K_EARTH_R) - ray


def knife_edge(d, d1, d2, excess, lam):
    """Worst Fresnel-Kirchhoff v and its knife-edge loss in dB (ITU-R P.526); any wave, any wavelength."""
    v = (excess * np.sqrt(2 * d / (lam * d1 * d2))).max(axis=1)
    # ponytail: worst single edge only; Deygout (3 edges) when ridges stack up
    return v, np.where(v > -0.78, 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1), 0.0)


def path_loss(terrain, rx, tx, freq_mhz, n=64):
    """Free-space + single knife-edge diffraction (ITU-R P.526) between rx (3,) and tx (T, 3).

    Returns 3D distance, FSPL dB, diffraction dB, and the worst Fresnel-Kirchhoff v
    (v < 0: the direct path clears all terrain).
    """
    d, d1, d2, excess = profile(terrain, rx, tx, n)
    v, diff = knife_edge(d, d1, d2, excess, C / (freq_mhz * 1e6))
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


def sun_elevation(lat, lon, when):
    """Sun elevation in degrees at a UTC datetime (NOAA low-precision formulas, good to about 0.5 deg)."""
    hour = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * math.pi / 365 * (when.timetuple().tm_yday - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))  # minutes
    ha = math.radians((hour * 60 + eqtime + 4 * lon) / 4 - 180)
    phi = math.radians(lat)
    cos_z = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_z))))


def pixels_on_target(size_m, dist_m, hfov_deg, pixels):
    """How many pixels the drone's largest dimension covers on the camera's sensor."""
    return size_m / (dist_m * math.radians(hfov_deg) / pixels)


def johnson(px, px50):
    """Chance of a task (detect, recognise) from pixels across the target: the Johnson / TTPF curve."""
    r = px / px50
    return 1 / (1 + np.exp(-(2.7 + 0.7 * r) * np.log(r)))  # r^E / (1 + r^E) without overflow


def site(terrain, s, kind):
    """A sensor's mast top rx (3,) in the local frame, and its entry for the outputs."""
    sx, sy = terrain.xy(s["lat"], s["lon"])
    rx = np.array([sx, sy, float(terrain.h(sx, sy)) + s["mast_m"]])
    return rx, {**s, "x": sx, "y": sy, "z": rx[2], "kind": kind}


def camera_link(terrain, s, rx, tr, drone, ts, dt, sc, sun_el, rng):
    """Visible or thermal camera on a pan-tilt head, already pointed at the drone (e.g. cued by a radar).

    Detected when the sight line clears the terrain, enough pixels cover the drone, and enough contrast
    (visible) or temperature difference (thermal) is left after the atmosphere.
    """
    x, y, z = tr[:3]
    n, visible = len(ts), s["band"] == "visible"
    # ponytail: always pointed (no search pattern or slew time); heat and contrast are the same from every side
    d, _, _, excess = profile(terrain, rx, np.stack([x, y, z], 1))
    clear = -excess.max(axis=1)  # metres between the sight line and the worst terrain point; < 0: hidden
    dist = np.maximum(np.hypot(d[:, 0], z - rx[2]), 1.0)
    px = pixels_on_target(max(drone["size_m"].values()), dist, s["hfov_deg"], s["pixels"])
    fill = np.minimum(px, 1.0) ** 2  # a drone smaller than one pixel is smeared over that pixel
    if visible:  # Koschmieder: contrast falls to 2% at the meteorological visibility
        signal = s["target_contrast"] * fill * np.exp(-3.912 * dist / (sc.get("visibility_km", 10.0) * 1000))
        need, lit = s["contrast_threshold"], sun_el > -6  # civil twilight
    else:
        signal = drone["thermal_dt_k"] * fill * np.exp(-s["extinction_per_km"] * dist / 1000)
        need, lit = s["netd_k"] * s["snr_required"], np.ones(n, bool)
    scint = shadowing(rng, n, s.get("scintillation_db", 2.0), 10.0, dt)
    margin = np.minimum(20 * np.log10(px / s["px_detect"]), 20 * np.log10(signal / need)) - scint
    sigma = np.full(n, s["bearing_sigma_deg"])
    true_b = np.degrees(np.arctan2(x - rx[0], y - rx[1])) % 360
    return dict(
        snr=margin, det=(margin >= s["threshold_db"]) & (clear >= 0) & lit, sigma=sigma, true_bearing=true_b,
        bearing=(true_b + s["bearing_bias_deg"] + rng.normal(0, 1, n) * sigma) % 360,
        elev=np.degrees(np.arctan2(z - rx[2], d[:, 0])) + rng.normal(0, 1, n) * sigma,
        dist=dist, clear=clear, px=px, signal=signal, need=need, shadow=scint, conf=johnson(px, s["px_recognise"]),
        **({"sun": sun_el} if visible else {}),
    )


def acoustic_link(terrain, s, rx, tr, drone, ts, dt, wind_mps, rng):
    """Microphone array. Sound needs dist / 343 s to arrive, so the array hears (and points at) the drone
    where it was then. Spherical spreading, air absorption, and the radio knife-edge maths with the
    sound's wavelength.
    """
    x, y, z = tr[:3]
    n = len(ts)
    lag = np.hypot(np.hypot(x - rx[0], y - rx[1]), z - rx[2]) / SOUND
    xe, ye, ze = (np.interp(ts - lag, ts, a) for a in (x, y, z))  # where the sound left the drone
    d, d1, d2, excess = profile(terrain, rx, np.stack([xe, ye, ze], 1))
    _, diff = knife_edge(d, d1, d2, excess, SOUND / s["freq_hz"])
    dist = np.maximum(np.hypot(d[:, 0], ze - rx[2]), 1.0)
    spread, absorb = 20 * np.log10(dist), s["absorption_db_per_km"] * dist / 1000
    shadow = shadowing(rng, n, 3.0, 5.0, dt)  # turbulence
    level = drone["spl_1m_db"] - spread - absorb - diff - shadow
    # ponytail: rough wind-noise rule; no refraction of sound by wind or temperature gradients
    noise = s["ambient_db"] + 20 * math.log10(1 + wind_mps / 5)
    gain = 10 * math.log10(s["mics"]) + s["processing_gain_db"]
    snr = level + gain - noise
    sigma = bearing_sigma(snr, s["bearing_sigma_deg"])
    true_b = np.degrees(np.arctan2(xe - rx[0], ye - rx[1])) % 360
    return dict(
        power=level, snr=snr, det=snr >= s["threshold_db"], sigma=sigma, true_bearing=true_b,
        bearing=(true_b + s["bearing_bias_deg"] + rng.normal(0, 1, n) * sigma) % 360,
        dist=dist, lag=lag, spread=spread, absorb=absorb, diff=diff, shadow=shadow, noise=noise, gain=gain,
        src=drone["spl_1m_db"],
    )


def remote_id_link(terrain, s, rx, o, tr, ts, dt, sc, rng):
    """Remote ID receiver. The drone broadcasts its own position, speed and operator location once a
    second, so one receiver is enough for a fix. Cooperative drones only ("remote_id": true).
    """
    x, y, z = tr[:3]
    n = len(ts)
    d3, fspl, diff, _ = path_loss(terrain, rx, np.stack([x, y, z], 1), s["freq_mhz"])
    shadow = shadowing(rng, n, sc["shadowing_db"], sc["shadowing_tau_s"], dt)
    power = RID_EIRP_DBM + s["gain_dbi"] - fspl - diff - shadow
    noise = -174 + 60 + s["noise_figure_db"] + 10 * math.log10(s["bandwidth_mhz"])
    gnss = lambda: shadowing(rng, n, RID_GNSS_SIGMA_M, 30.0, dt)  # GNSS error drifts slowly
    return dict(
        power=power, snr=power - noise, det=power - noise >= s["threshold_db"], dist=d3, fspl=fspl, diff=diff,
        shadow=shadow, noise=noise, eirp=RID_EIRP_DBM, rep_x=x + gnss(), rep_y=y + gnss(), rep_alt=z + gnss(),
        rep_speed=tr[7] + rng.normal(0, 0.3, n), uas_id=f"MW-{o['id'].upper()}", operator=o["waypoints"][0],
    )


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)


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


def p_detect(snr_db, threshold_db, shadowing_db):
    """Chance that SNR minus log-normal shadowing still clears the threshold."""
    z = (np.asarray(snr_db) - threshold_db) / max(shadowing_db, 1e-6)
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


def coverage(terrain, sensors, emitter, alts, shadowing_db, n=40):
    """Where a drone with this emitter can be heard and located, per height above ground.

    For a grid over the box: heard = chance at least one receiver detects it, located = chance
    at least two do (a fix needs two bearings), err = expected fix error in metres from the
    bearing noise and the crossing geometry of the receivers that probably hear it (NaN if fewer
    than two). Same terrain and radio physics as the per-second links. Arrays are [alt, row, col],
    rows north -> south like the terrain grid.
    """
    xe, ys = terrain.xy(terrain.bbox[1], terrain.bbox[2])
    nx = n
    ny = max(2, round(n * (terrain.yn - ys) / (xe - terrain.xw)))
    dx, dy = (xe - terrain.xw) / nx, (terrain.yn - ys) / ny
    gx, gy = np.meshgrid(terrain.xw + dx * (np.arange(nx) + 0.5), terrain.yn - dy * (np.arange(ny) + 0.5))
    gx, gy = gx.ravel(), gy.ravel()
    ground = terrain.h(gx, gy)
    heard, located, err = [], [], []
    for alt in alts:
        tx = np.stack([gx, gy, ground + alt], 1)
        ps, normal, n_heard = [], np.zeros((len(gx), 3)), 0  # normal matrix entries: xx, xy, yy
        for s in sensors:
            rx = np.array([s["x"], s["y"], s["z"]])
            _, fspl, diff, _ = path_loss(terrain, rx, tx, emitter["freq_mhz"])
            noise = -174 + 60 + s["noise_figure_db"] + 10 * math.log10(emitter["bandwidth_mhz"])
            snr = emitter["eirp_dbm"] + s["gain_dbi"] - fspl - diff - noise
            p = p_detect(snr, s["threshold_db"], shadowing_db)
            ps.append(p)
            b = np.arctan2(gx - s["x"], gy - s["y"])
            r = np.maximum(np.hypot(gx - s["x"], gy - s["y"]), 1.0)
            w2 = (p >= 0.5) / (r * np.radians(bearing_sigma(snr, s["bearing_sigma_deg"]))) ** 2
            n_heard = n_heard + (p >= 0.5)
            normal += w2[:, None] * np.stack([np.cos(b) ** 2, -np.cos(b) * np.sin(b), np.sin(b) ** 2], 1)
        ps = np.array(ps)
        miss = np.prod(1 - ps, axis=0)
        exactly_one = sum(ps[k] * np.prod(np.delete(1 - ps, k, axis=0), axis=0) for k in range(len(ps)))
        det = normal[:, 0] * normal[:, 2] - normal[:, 1] ** 2
        # ponytail: ignores receiver bias and the fusion's <10 deg crossing cut; good enough to compare places
        with np.errstate(divide="ignore", invalid="ignore"):
            ce = np.where((n_heard >= 2) & (det > 0), np.sqrt((normal[:, 0] + normal[:, 2]) / det), np.nan)
        heard.append((1 - miss).reshape(ny, nx))
        located.append((1 - miss - exactly_one).clip(0, 1).reshape(ny, nx))
        err.append(ce.reshape(ny, nx))
    grid = {"nx": nx, "ny": ny, "x0": terrain.xw + dx / 2, "y0": terrain.yn - dy / 2, "dx": dx, "dy": dy}
    return grid, np.array(heard), np.array(located), np.array(err)


# Typical settings for each sensor type, used by make_scenario.
RADAR = {"mast_m": 8, "tx_power_dbm": 45, "gain_dbi": 30, "freq_mhz": 9500, "bandwidth_mhz": 5, "integration_gain_db": 25,
         "noise_figure_db": 4, "threshold_db": 13, "bearing_sigma_deg": 1.0, "bearing_bias_deg": 0.0,
         "range_sigma_m": 15, "radial_sigma_mps": 1.0}
EO_CAMERA = {"band": "visible", "mast_m": 6, "hfov_deg": 6, "pixels": 1920, "px_detect": 3, "px_recognise": 8,
             "target_contrast": 0.3, "contrast_threshold": 0.02, "threshold_db": 0, "bearing_sigma_deg": 0.1,
             "bearing_bias_deg": 0.0}
IR_CAMERA = {"band": "thermal", "mast_m": 6, "hfov_deg": 6, "pixels": 640, "px_detect": 3, "px_recognise": 8,
             "netd_k": 0.05, "snr_required": 5, "extinction_per_km": 0.2, "threshold_db": 0, "bearing_sigma_deg": 0.15,
             "bearing_bias_deg": 0.0}
ACOUSTIC = {"mast_m": 2, "mics": 6, "freq_hz": 500, "ambient_db": 40, "processing_gain_db": 10, "absorption_db_per_km": 5,
            "threshold_db": 6, "bearing_sigma_deg": 3.0, "bearing_bias_deg": 0.0}
REMOTE_ID = {"mast_m": 3, "freq_mhz": 2437, "gain_dbi": 2, "noise_figure_db": 6, "bandwidth_mhz": 20, "threshold_db": 4}

TYPES = ["quadcopter", "fixed_wing_recon", "delta_wing", "decoy"]
MIXES = {"balanced": [0.5, 0.25, 0.15, 0.1], "quadcopters": [0.85, 0.1, 0.05, 0.0],
         "fixed_wing": [0.1, 0.7, 0.15, 0.05], "attack": [0.0, 0.1, 0.6, 0.3]}  # weights over TYPES
SETTINGS = {"duration_s": 1200, "shadowing_db": 4.0, "mix": "balanced", "mast_m": 10, "bearing_sigma_deg": 2.0,
            "wind_from_deg": 270, "wind_mps": 0}


def scenario_settings(given=None):
    """Layout settings with defaults, clamped to ranges the engine and the viewer handle well."""
    st = {**SETTINGS, **(given or {})}
    clamp = lambda k, lo, hi: min(max(float(st[k]), lo), hi)
    return {"duration_s": int(clamp("duration_s", 60, 3600)), "shadowing_db": clamp("shadowing_db", 0, 12),
            "mix": st["mix"] if st["mix"] in MIXES else "balanced", "mast_m": clamp("mast_m", 2, 40),
            "bearing_sigma_deg": clamp("bearing_sigma_deg", 0.5, 10), "wind_from_deg": clamp("wind_from_deg", 0, 360) % 360,
            "wind_mps": clamp("wind_mps", 0, 25)}


def make_scenario(area, n_rx, n_drones, seed, settings=None):
    """Random but presentable layout for an area from areas.json.

    Receivers go on high ground (top 25% of the inner box), spread far apart. Drones fly
    random loops inside the inner 80% of the box. Each receiver and drone has its own RNG
    stream, so adding one never moves the others. `settings` (see SETTINGS) sets the length,
    shadowing, drone mix, receiver mast and bearing accuracy, and the wind.
    """
    st = scenario_settings(settings)
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
        sensors.append({"id": f"RF{i + 1}", "lat": round(float(lat), 5), "lon": round(float(lon), 5), "mast_m": st["mast_m"],
                        "gain_dbi": 6, "noise_figure_db": 6, "threshold_db": 6, "bearing_sigma_deg": st["bearing_sigma_deg"],
                        "bearing_bias_deg": round(float(rng.normal(0, 1)), 1)})

    drones = json.loads((Path(__file__).parent / "drones.json").read_text())
    types, weights = TYPES, MIXES[st["mix"]]
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
            "speed_ms": drones[kind]["flight"]["cruise_speed_mps"], "loop": True, "remote_id": quad,
            "waypoints": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)],
        })

    def spot(k, high):  # own RNG stream per sensor, so changing the counts never moves it
        rng = np.random.default_rng([seed, 2, k])
        p = cand[rng.integers(len(cand))] if high else (rng.uniform(-xmax, xmax), rng.uniform(-ymax, ymax))
        lat, lon = terrain.latlon(*p)
        return {"lat": round(float(lat), 5), "lon": round(float(lon), 5)}

    sc = {"name": area["name"], "region": area["region"], "bbox": area["bbox"], "start_utc": "2026-11-03T14:00:00Z",
          "duration_s": st["duration_s"], "dt_s": 1.0, "seed": seed, "shadowing_db": st["shadowing_db"], "shadowing_tau_s": 15.0,
          "visibility_km": 10.0, "sensors": sensors,
          "radars": [{"id": "RADAR1", **spot(0, True), **RADAR}],
          "cameras": [{"id": "EO1", **spot(1, True), **EO_CAMERA}, {"id": "IR1", **spot(2, True), **IR_CAMERA}],
          "acoustic_arrays": [{"id": "AC1", **spot(3, False), **ACOUSTIC}],
          "remote_id_receivers": [{"id": "RID1", **spot(4, False), **REMOTE_ID}],
          "objects": objects, "settings": st}
    if st["wind_mps"]:
        sc["wind"] = {"from_deg": st["wind_from_deg"], "speed_mps": st["wind_mps"]}
    return sc


def make_game(area, huts, n_birds, seed, duration_s=600):
    """One round of the bird game: receivers where the player put them ([lat, lon] or [lat, lon,
    mast_m] each, mast 2-40 m, default 10), and drones that appear at random times on one edge,
    cross the box and leave by another edge.

    Same drone types, radios and heights as make_scenario. Each bird has its own RNG stream,
    so changing the number of birds never moves the others.
    """
    west, south, east, north = area["bbox"]
    terrain = Terrain(fetch_dem(area["bbox"]), area["bbox"])
    sensors = []
    for i, hut in enumerate(huts):
        lat, lon = hut[0], hut[1]
        mast = min(max(float(hut[2]), 2.0), 40.0) if len(hut) > 2 else 10.0
        if not (south <= lat <= north and west <= lon <= east):
            raise ValueError(f"station {i + 1} is outside the area")
        rng = np.random.default_rng([seed, 0, i])
        sensors.append({"id": f"RF{i + 1}", "lat": round(float(lat), 5), "lon": round(float(lon), 5), "mast_m": mast,
                        "gain_dbi": 6, "noise_figure_db": 6, "threshold_db": 6, "bearing_sigma_deg": 2.0,
                        "bearing_bias_deg": round(float(rng.normal(0, 1)), 1)})

    drones = json.loads((Path(__file__).parent / "drones.json").read_text())
    types, weights = ["quadcopter", "fixed_wing_recon", "delta_wing", "decoy"], [0.4, 0.3, 0.2, 0.1]
    xmax, ymax, objects = -terrain.xw * 0.92, terrain.yn * 0.92, []
    for i in range(n_birds):
        rng = np.random.default_rng([seed, 1, i])
        kind = types[rng.choice(len(types), p=weights)]
        quad = drones[kind]["type"] == "multirotor"
        side_in = int(rng.integers(4))
        side_out = (side_in + int(rng.integers(1, 4))) % 4  # leave by any other edge
        edge = lambda side, u: [(-xmax, u * ymax), (xmax, u * ymax), (u * xmax, -ymax), (u * xmax, ymax)][side]
        pts = ([edge(side_in, rng.uniform(-0.8, 0.8))]
               + [(rng.uniform(-0.6, 0.6) * xmax, rng.uniform(-0.6, 0.6) * ymax) for _ in range(int(rng.integers(1, 3)))]
               + [edge(side_out, rng.uniform(-0.8, 0.8))])
        lat, lon = terrain.latlon(*np.array(pts).T)
        objects.append({
            "id": f"drone_{i + 1:02d}", "type": kind, "leave": True,
            "start_s": float(round(rng.uniform(0, duration_s * 0.6))),
            "freq_mhz": [2412.0, 2437.0, 2462.0][i % 3] if quad else [5745.0, 5785.0, 5825.0][i % 3],
            "eirp_dbm": 27 if quad else 30, "bandwidth_mhz": 10 if quad else 20,
            "alt_agl_m": int(rng.integers(60, 121)) if quad else int(rng.integers(200, 401)),
            "speed_ms": drones[kind]["flight"]["cruise_speed_mps"],
            "waypoints": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)],
        })

    return {"name": area["name"], "region": area["region"], "bbox": area["bbox"], "start_utc": "2026-11-03T14:00:00Z",
            "duration_s": duration_s, "dt_s": 1.0, "seed": seed, "shadowing_db": 4.0, "shadowing_tau_s": 15.0,
            "sensors": sensors, "objects": objects, "game": True}


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
    # radio and speed default to the drone type's; anything the scenario sets wins
    sc["objects"] = [{**{k: v for k, v in drones[o["type"]]["emitter"].items() if k != "what"},
                      "speed_ms": drones[o["type"]]["flight"]["cruise_speed_mps"], **o} for o in sc["objects"]]

    truth = {}
    with open(out / "truth.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_utc", "t_s", "object_id", "lat", "lon", "alt_msl_m", "alt_agl_m", "heading_deg",
                     "pitch_deg", "roll_deg", "speed_ms"])
        for o in sc["objects"]:
            x, y, z, hd, pitch, roll, agl, speed, present, vx, vy = truth[o["id"]] = track(terrain, o, ts, drones[o["type"]], wind)
            lat, lon = terrain.latlon(x, y)
            for i, t in enumerate(ts):
                if not present[i]:
                    continue
                wr.writerow([iso(t), t, o["id"], f"{lat[i]:.6f}", f"{lon[i]:.6f}", f"{z[i]:.1f}", f"{agl[i]:.1f}",
                             f"{hd[i]:.1f}", f"{pitch[i]:.1f}", f"{roll[i]:.1f}", f"{speed[i]:.1f}"])

    sensors, links = [], {}
    for si, s in enumerate(sc["sensors"]):
        sx, sy = terrain.xy(s["lat"], s["lon"])
        rx = np.array([sx, sy, float(terrain.h(sx, sy)) + s["mast_m"]])
        sensors.append({**s, "x": sx, "y": sy, "z": rx[2], "kind": "rf"})
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
                power=power, snr=snr, det=(snr >= s["threshold_db"]) & truth[o["id"]][8], bearing=bearing, sigma=sigma,
                true_bearing=true_b, dist=d3, fspl=fspl, diff=diff, shadow=shadow, v=v, noise=noise,
            )

    # Radar: active (own transmitter, own frequency), so it hears a silent drone too. One site already
    # gives a position fix (range + bearing), no crossing with another sensor needed.
    radars, radar_links = [], {}
    for ri, s in enumerate(sc.get("radars", [])):
        sx, sy = terrain.xy(s["lat"], s["lon"])
        rx = np.array([sx, sy, float(terrain.h(sx, sy)) + s["mast_m"]])
        radars.append({**s, "x": sx, "y": sy, "z": rx[2], "kind": "radar"})
        noise_dbm = -174 + 60 + s["noise_figure_db"] + 10 * math.log10(s["bandwidth_mhz"])
        for oi, o in enumerate(sc["objects"]):
            rng = np.random.default_rng([sc["seed"], 100 + ri, oi])
            x, y, z = truth[o["id"]][:3]
            vx, vy = truth[o["id"]][9], truth[o["id"]][10]
            tx = np.stack([x, y, z], 1)
            rcs_dbsm = drones[o["type"]]["rcs_dbsm"]
            d3, fspl, diff, v = path_loss(terrain, rx, tx, s["freq_mhz"])
            shadow = shadowing(rng, len(ts), sc["shadowing_db"], sc["shadowing_tau_s"], dt)
            power = radar_echo_power(s["tx_power_dbm"], s["gain_dbi"], fspl, diff, rcs_dbsm, s["freq_mhz"],
                                      s.get("integration_gain_db", 0.0)) - shadow
            snr = power - noise_dbm
            det = (snr >= s["threshold_db"]) & truth[o["id"]][8]
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

    # Cameras, microphone arrays and Remote ID receivers (models in camera_link, acoustic_link, remote_id_link).
    sun_el = np.array([sun_elevation(terrain.lat0, terrain.lon0, start + timedelta(seconds=float(t))) for t in ts])
    wind_mps = float(np.hypot(*wind))
    cameras, acoustics, rid_rxs, other_links = [], [], [], {}
    for k, s in enumerate(sc.get("cameras", [])):
        rx, entry = site(terrain, s, "eo" if s["band"] == "visible" else "ir")
        cameras.append(entry)
        for oi, o in enumerate(sc["objects"]):
            other_links[(s["id"], o["id"])] = camera_link(terrain, s, rx, truth[o["id"]], drones[o["type"]], ts, dt, sc,
                                                          sun_el, np.random.default_rng([sc["seed"], 200 + k, oi]))
    for k, s in enumerate(sc.get("acoustic_arrays", [])):
        rx, entry = site(terrain, s, "acoustic")
        acoustics.append(entry)
        for oi, o in enumerate(sc["objects"]):
            other_links[(s["id"], o["id"])] = acoustic_link(terrain, s, rx, truth[o["id"]], drones[o["type"]], ts, dt,
                                                            wind_mps, np.random.default_rng([sc["seed"], 300 + k, oi]))
    for k, s in enumerate(sc.get("remote_id_receivers", [])):
        rx, entry = site(terrain, s, "remote_id")
        rid_rxs.append(entry)
        for oi, o in enumerate(sc["objects"]):
            if o.get("remote_id"):  # a drone that does not broadcast has no link at all
                other_links[(s["id"], o["id"])] = remote_id_link(terrain, s, rx, o, truth[o["id"]], ts, dt, sc,
                                                                 np.random.default_rng([sc["seed"], 400 + k, oi]))
    for (_, oid), L in other_links.items():
        L["det"] &= truth[oid][8]  # nothing to see or hear before a drone appears or after it leaves
    by_kind = lambda *kinds: [(s, o, other_links[(s["id"], o["id"])]) for s in cameras + acoustics + rid_rxs
                              if s["kind"] in kinds for o in sc["objects"] if (s["id"], o["id"]) in other_links]

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

    # Cameras: azimuth, elevation and a class with a confidence. Detections only, then the full budget.
    cams, mics, rids = by_kind("eo", "ir"), by_kind("acoustic"), by_kind("remote_id")
    write_csv(out / "camera_raw.csv",
              ["time_utc", "t_s", "camera_id", "band", "azimuth_deg", "elevation_deg", "class", "confidence", "source_object_id"],
              ([iso(t), t, s["id"], s["band"], f"{L['bearing'][i]:.2f}", f"{L['elev'][i]:.2f}",
                drones[o["type"]]["type"] if L["conf"][i] >= 0.5 else "unknown", f"{L['conf'][i]:.2f}", o["id"]]
               for i, t in enumerate(ts) for s, o, L in cams if L["det"][i]))
    write_csv(out / "camera_budget.csv",
              ["t_s", "camera_id", "band", "object_id", "dist_m", "clearance_m", "pixels_on_target", "signal", "signal_needed",
               "scintillation_db", "margin_db", "detected", "sun_elevation_deg", "recognition_conf", "true_bearing_deg"],
              ([t, s["id"], s["band"], o["id"], f"{L['dist'][i]:.0f}", f"{L['clear'][i]:.1f}", f"{L['px'][i]:.2f}",
                f"{L['signal'][i]:.4f}", f"{L['need']:.4f}", f"{L['shadow'][i]:.1f}", f"{L['snr'][i]:.1f}", int(L["det"][i]),
                f"{sun_el[i]:.1f}", f"{L['conf'][i]:.2f}", f"{L['true_bearing'][i]:.1f}"]
               for s, o, L in cams for i, t in enumerate(ts)))

    # Microphone arrays: sound level and a bearing to where the drone was when the sound left it.
    write_csv(out / "acoustic_raw.csv",
              ["time_utc", "t_s", "array_id", "level_db", "snr_db", "bearing_deg", "source_object_id"],
              ([iso(t), t, s["id"], f"{L['power'][i]:.1f}", f"{L['snr'][i]:.1f}", f"{L['bearing'][i]:.1f}", o["id"]]
               for i, t in enumerate(ts) for s, o, L in mics if L["det"][i]))
    write_csv(out / "acoustic_budget.csv",
              ["t_s", "array_id", "object_id", "dist_m", "delay_s", "source_db", "spreading_db", "absorption_db",
               "diffraction_db", "shadow_db", "level_db", "array_gain_db", "noise_db", "snr_db", "detected",
               "true_bearing_deg", "bearing_sigma_deg"],
              ([t, s["id"], o["id"], f"{L['dist'][i]:.0f}", f"{L['lag'][i]:.2f}", f"{L['src']:.1f}", f"{L['spread'][i]:.1f}",
                f"{L['absorb'][i]:.1f}", f"{L['diff'][i]:.1f}", f"{L['shadow'][i]:.1f}", f"{L['power'][i]:.1f}",
                f"{L['gain']:.1f}", f"{L['noise']:.1f}", f"{L['snr'][i]:.1f}", int(L["det"][i]),
                f"{L['true_bearing'][i]:.1f}", f"{L['sigma'][i]:.1f}"]
               for s, o, L in mics for i, t in enumerate(ts)))

    # Remote ID: the message as decoded (the drone's own report of itself), then the link budget.
    write_csv(out / "remote_id_raw.csv",
              ["time_utc", "t_s", "receiver_id", "uas_id", "lat", "lon", "alt_msl_m", "speed_ms", "operator_lat",
               "operator_lon", "rssi_dbm", "source_object_id"],
              ([iso(t), t, s["id"], L["uas_id"], *(f"{v:.6f}" for v in terrain.latlon(L["rep_x"][i], L["rep_y"][i])),
                f"{L['rep_alt'][i]:.1f}", f"{L['rep_speed'][i]:.1f}", *(f"{v:.6f}" for v in L["operator"]),
                f"{L['power'][i]:.1f}", o["id"]]
               for i, t in enumerate(ts) for s, o, L in rids if L["det"][i]))
    write_csv(out / "remote_id_budget.csv",
              ["t_s", "receiver_id", "object_id", "dist_m", "fspl_db", "diffraction_db", "shadow_db", "power_dbm",
               "noise_dbm", "snr_db", "detected"],
              ([t, s["id"], o["id"], f"{L['dist'][i]:.0f}", f"{L['fspl'][i]:.1f}", f"{L['diff'][i]:.1f}",
                f"{L['shadow'][i]:.1f}", f"{L['power'][i]:.1f}", f"{L['noise']:.1f}", f"{L['snr'][i]:.1f}", int(L["det"][i])]
               for s, o, L in rids for i, t in enumerate(ts)))

    # ponytail: association uses source_object_id (answer key); a real system must associate by freq/time
    fused = {o["id"]: [] for o in sc["objects"]}
    events, rows = [], []

    def add_fix(o, i, ex, ey, ce, n_sensors, method, callsign, remarks, signals):
        t, (lat, lon) = ts[i], terrain.latlon(ex, ey)
        uid = f"sig-{len(rows):05d}"
        rows.append([uid, iso(t), t, f"{lat:.6f}", f"{lon:.6f}", f"{ce:.0f}", n_sensors, o["id"],
                     f"{math.hypot(ex - truth[o['id']][0][i], ey - truth[o['id']][1][i]):.0f}", method])
        fused[o["id"]].append([float(t), round(float(ex)), round(float(ey)), round(ce)])
        events.append(
            f'  <event version="2.0" uid="{uid}" type="a-u-A" how="m-r" time="{iso(t)}" start="{iso(t)}" stale="{iso(t + 60)}">\n'
            f'    <point lat="{lat:.6f}" lon="{lon:.6f}" hae="9999999.0" ce="{ce:.0f}" le="9999999.0"/>\n'
            f'    <detail>\n      <contact callsign="{callsign}"/>\n      <remarks>{remarks}</remarks>\n'
            + "".join(f"      <__signal {sig}/>\n" for sig in signals) + "    </detail>\n  </event>")

    # Any sensor that measures a bearing (radio, camera, microphones) can join a crossing.
    bearers = {o["id"]: [(s, L) for s, o2, L in by_kind("eo", "ir", "acoustic") if o2 is o] for o in sc["objects"]}
    for i, t in enumerate(ts):
        for o in sc["objects"]:
            heard = [(s, links[(s["id"], o["id"])]) for s in sensors if links[(s["id"], o["id"])]["det"][i]]
            heard += [(s, L) for s, L in bearers[o["id"]] if L["det"][i]]
            fix = None if len(heard) < 2 else cross_bearings(
                np.array([[s["x"], s["y"]] for s, _ in heard]),
                np.array([L["bearing"][i] for _, L in heard]),
                np.array([L["sigma"][i] for _, L in heard]))
            if fix is not None:
                rf = any(s["kind"] == "rf" for s, _ in heard)
                add_fix(o, i, *fix[0], fix[1], len(heard), "crossed_bearings",
                        f"Unknown drone, {o['freq_mhz'] / 1000:.1f} GHz" if rf else "Unknown drone",
                        f"Detected by {', '.join(s['id'] for s, _ in heard)}, position from crossed bearings",
                        [f'sensor="{s["id"]}" kind="{s["kind"]}"'
                         + (f' freq_mhz="{o["freq_mhz"]}" power_dbm="{L["power"][i]:.1f}"' if s["kind"] == "rf" else "")
                         + f' bearing_deg="{L["bearing"][i]:.1f}"' for s, L in heard])

            # Radar: a single site already gives range + bearing, so one detection is a fix.
            for s in radars:
                L = radar_links[(s["id"], o["id"])]
                if L["det"][i]:
                    (ex, ey), ce = radar_fix(np.array([s["x"], s["y"]]), L["range"][i], L["bearing"][i], L["range_sigma"][i], L["sigma"][i])
                    add_fix(o, i, ex, ey, ce, 1, "radar", "Unknown drone, radar return",
                            f"Heard by {s['id']}, range {L['range'][i]:.0f} m, radial velocity {L['radial'][i]:.1f} m/s",
                            [f'sensor="{s["id"]}" kind="radar" freq_mhz="{s["freq_mhz"]}" power_dbm="{L["power"][i]:.1f}" '
                             f'bearing_deg="{L["bearing"][i]:.1f}"'])

            # Remote ID: the drone says where it is.
            for s, o2, L in rids:
                if o2 is o and L["det"][i]:
                    op_lat, op_lon = L["operator"]
                    add_fix(o, i, L["rep_x"][i], L["rep_y"][i], RID_GNSS_SIGMA_M * math.sqrt(2), 1, "remote_id", L["uas_id"],
                            f"Remote ID from {L['uas_id']} via {s['id']}, operator at {op_lat:.5f}, {op_lon:.5f}",
                            [f'sensor="{s["id"]}" kind="remote_id" uas_id="{L["uas_id"]}" rssi_dbm="{L["power"][i]:.1f}"'])
    write_csv(out / "fused.csv", ["uid", "time_utc", "t_s", "lat", "lon", "ce_m", "n_sensors", "source_object_id", "error_m",
                                  "method"], rows)
    (out / "fused.cot.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<events>\n' + "\n".join(events) + "\n</events>\n")
    (out / "scenario.json").write_text(json.dumps(sc, indent=2))

    # Coverage heatmap, once per distinct emitter (drones of one type usually share one).
    cov_cfg = sc.get("coverage", {})
    alts = cov_cfg.get("alts_agl_m", [50, 100, 250, 500, 1000])
    cov = {"alts": alts, "emitters": {}, "object_emitter": {}}
    for o in sc["objects"]:
        key = f"{o['freq_mhz']}|{o['eirp_dbm']}|{o['bandwidth_mhz']}"
        cov["object_emitter"][o["id"]] = key
        if key in cov["emitters"]:
            continue
        grid, heard, located, err = coverage(terrain, sensors, o, alts, sc["shadowing_db"], cov_cfg.get("grid", 40))
        cov |= grid
        pct = lambda a: [int(v) for v in np.rint(a.ravel() * 100)]
        cov["emitters"][key] = {"label": f"{o['freq_mhz'] / 1000:.2f} GHz · {o['eirp_dbm']} dBm",
                                "heard": pct(heard), "located": pct(located),
                                "err": [-1 if np.isnan(v) else int(round(min(v, 99_999))) for v in err.ravel()]}

    # Viewer bundle: downsampled terrain + every series in the local metre frame.
    r1 = lambda a: [round(float(v), 1) for v in a]
    digits = {"conf": 2, "px": 2, "signal": 4, "need": 4}  # everything else: 0.1
    internal = {"true_bearing", "v", "radial_true", "rep_x", "rep_y", "sun"}  # in the CSVs or as fixes already

    def export(L):
        e = {}
        for k, v in L.items():
            if k in internal:
                continue
            if k == "det":
                e[k] = [int(b) for b in v]
            elif isinstance(v, np.ndarray):
                e[k] = [round(float(x), digits.get(k, 1)) for x in v]
            elif isinstance(v, float):
                e[k] = round(v, digits.get(k, 1))
            else:
                e[k] = v
        return e

    tj, frame = terrain_json(terrain, 256)
    bundle = {
        "scenario": sc,
        "frame": frame,
        "terrain": tj,
        "t": [float(t) for t in ts],
        "sun": r1(sun_el),
        "sensors": [{"id": s["id"], "x": s["x"], "y": s["y"], "z": s["z"], "threshold_db": s["threshold_db"], "kind": s["kind"]}
                    for s in sensors + radars + cameras + acoustics + rid_rxs],
        "truth": {k: {"x": r1(v[0]), "y": r1(v[1]), "z": r1(v[2]), "h": r1(v[3]), "p": r1(v[4]), "r": r1(v[5])}
                  for k, v in truth.items()},
        "drones": {o["id"]: {"name": drones[o["type"]]["name"], "model": drones[o["type"]]["model"], "type": o["type"],
                             "class": drones[o["type"]]["type"], "size_m": max(drones[o["type"]]["size_m"].values())}
                   for o in sc["objects"]},
        # seconds [first, last] each drone is in the scenario; the whole run unless it spawns or leaves
        "spawn": {k: [float(ts[v[8]][0]), float(ts[v[8]][-1])] if v[8].any() else [0.0, 0.0] for k, v in truth.items()},
        "links": {f"{sid}|{oid}": export(L) for (sid, oid), L in {**links, **radar_links, **other_links}.items()},
        "fused": fused,
        "coverage": cov,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))
    return bundle


if __name__ == "__main__":
    sc = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "scenario.json").read_text())
    b = run(sc, sys.argv[2] if len(sys.argv) > 2 else "out")
    for k, L in b["links"].items():
        blocked = f"  blocked by terrain {np.mean(np.array(L['diff']) > 10):5.1%}" if "diff" in L else ""
        print(f"{k}: detected {np.mean(L['det']):5.1%}{blocked}")
    for oid, f in b["fused"].items():
        print(f"{oid}: {len(f)} fixes")
