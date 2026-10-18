"""Kinematic flight model: a drone type (drones.json) plus a flight plan in, a 10 Hz track out.

Frame: x east, y north in metres (sim.Terrain), height above ground. Angles: heading is
the nose direction in degrees clockwise from north, pitch is positive nose up, roll is
positive right side down.

Plan legs, flown in order (alt_m carries over, speed_mps defaults to the cruise speed):
  takeoff {alt_m}                        multirotor climbs straight up, fixed-wing climbs out
  goto    {lat, lon, alt_m?, speed_mps?} a multirotor brakes to a stop unless a goto follows
  orbit   {lat, lon, radius_m, duration_s, direction: cw|ccw, alt_m?, speed_mps?}
  hover   {duration_s}                   multirotor only
  land    {}                             multirotor only
"""
import math

import numpy as np

G = 9.81
DT = 0.1  # s, 10 Hz
K_TURN = 0.4  # 1/s, commanded turn rate per radian of heading error (fixed-wing)
K_ORBIT = 2.0  # how hard the orbit guidance pulls back onto the circle
MAX_LEG_S = 12 * 3600  # guard against a plan that can never finish a leg
COLUMNS = ("t_s", "leg", "x", "y", "agl", "vx", "vy", "vz", "airspeed", "heading", "pitch", "roll")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _approach(err, up_max, down_max, accel):
    """Speed that closes `err` metres and brakes in time: sqrt braking, linear at the end."""
    v = min(math.sqrt(2 * accel * 0.8 * abs(err)), abs(err))
    return min(v, up_max) if err > 0 else -min(v, down_max)


def wind_xy(wind):
    """{'from_deg', 'speed_mps'} -> (east, north) velocity the air moves with."""
    if not wind:
        return np.zeros(2)
    to = math.radians(wind["from_deg"] + 180)
    return wind["speed_mps"] * np.array([math.sin(to), math.cos(to)])


def min_orbit_radius(drone, wind=np.zeros(2)):
    f = drone["flight"]
    return (f["max_speed_mps"] + np.linalg.norm(wind)) ** 2 / (G * math.tan(math.radians(f["max_bank_deg"])))


def fly(drone, obj, xy, wind=np.zeros(2), t_end=math.inf):
    """Fly obj = {id, start: {lat, lon, alt_m, heading_deg?, speed_mps?}, plan: [legs]}.
    xy(lat, lon) -> (x, y) metres. Stops after t_end s. Returns rows of COLUMNS."""
    f, multirotor = drone["flight"], drone["type"] == "multirotor"
    start, plan = obj["start"], obj["plan"]
    p = np.array([*xy(start["lat"], start["lon"]), float(start.get("alt_m", 0))])
    first = next(iter(plan[0].items()))
    if "heading_deg" in start:
        psi = math.radians(start["heading_deg"])
    elif first[0] == "goto":  # default: nose toward the first waypoint
        tx, ty = xy(first[1]["lat"], first[1]["lon"])
        psi = math.atan2(tx - p[0], ty - p[1])
    else:
        psi = 0.0
    speed0 = start.get("speed_mps", 0.0 if multirotor else f["cruise_speed_mps"])
    st = {"p": p, "psi": psi, "v": speed0 * np.array([math.sin(psi), math.cos(psi), 0.0]),
          "va": speed0, "w": 0.0, "phi": 0.0, "pitch": 0.0, "roll": 0.0}
    alt, t = p[2], 0.0
    rows = [(t, 0, *p, *st["v"], speed0, math.degrees(psi) % 360, 0.0, 0.0)]
    drag_k = G * math.tan(math.radians(f.get("max_tilt_deg", 30))) / f["max_speed_mps"] ** 2

    for i, leg in enumerate(plan):
        kind, a = next(iter(leg.items()))
        a = a or {}
        if not multirotor and kind in ("hover", "land"):
            raise ValueError(f"{obj['id']}: a fixed-wing drone cannot {kind}")
        alt = float(a.get("alt_m", alt if kind != "land" else 0.0))
        speed = float(a.get("speed_mps", f["cruise_speed_mps"]))
        hold = st["p"][:2].copy()
        target = np.array(xy(a["lat"], a["lon"])) if kind in ("goto", "orbit") else hold
        nxt = next(iter(plan[i + 1])) if i + 1 < len(plan) else None
        stop = multirotor and kind == "goto" and nxt != "goto"
        if kind == "orbit" and not multirotor and a["radius_m"] < (r_min := 1.2 * min_orbit_radius(drone, wind)):
            raise ValueError(f"{obj['id']}: orbit radius {a['radius_m']} m is too tight, needs at least {r_min:.0f} m in this wind")
        t_leg, course0 = 0.0, st["psi"]
        while True:
            d_vec = target - st["p"][:2]
            d = float(np.linalg.norm(d_vec))
            if kind == "orbit":
                gamma = math.atan2(-d_vec[0], -d_vec[1])  # bearing from the centre to the drone
                lam = -1 if a.get("direction", "cw") == "ccw" else 1
                course = gamma + lam * (math.pi / 2 + math.atan(K_ORBIT * (d - a["radius_m"]) / a["radius_m"]))
                hspeed = speed
            elif kind == "takeoff" and not multirotor:
                course, hspeed = course0, speed
            else:
                course = math.atan2(d_vec[0], d_vec[1]) if d > 1e-6 else st["psi"]
                hspeed = speed if (kind == "goto" and not stop) else min(speed, _approach(d, speed, speed, f["max_accel_mps2"]))

            if multirotor:
                _multirotor_step(st, course, hspeed, alt, f, wind, drag_k)
            else:
                _fixed_wing_step(st, course, hspeed, alt, f, wind)
            if kind == "land" and st["p"][2] <= 0.05:  # touchdown
                st["p"][2], st["v"][:] = 0.0, 0.0
            t += DT
            t_leg += DT
            vh = float(np.linalg.norm(st["v"][:2]))
            rows.append((t, i, *st["p"], *st["v"], st["va"], math.degrees(st["psi"]) % 360, st["pitch"], st["roll"]))
            if t >= t_end:
                return np.array(rows)

            dz = abs(alt - st["p"][2])
            if kind == "takeoff":
                done = dz < (0.5 if multirotor else 2.0) and abs(st["v"][2]) < 0.3
            elif kind == "goto" and stop:
                done = d < 0.5 and vh < 0.3 and dz < 0.5
            elif kind == "goto":
                if multirotor:
                    accept = max(2.0, 1.5 * vh)
                else:  # turn radius: any point inside the turning circle still gets reached
                    accept = max(50.0, st["va"] ** 2 / (G * math.tan(math.radians(f["max_bank_deg"]))))
                done = d < accept
            elif kind == "land":
                done = st["p"][2] == 0.0
            else:  # hover, orbit
                done = t_leg >= a["duration_s"]
            if done:
                break
            if t_leg > MAX_LEG_S:
                raise RuntimeError(f"{obj['id']}: leg {i} ({kind}) did not finish in {MAX_LEG_S} s")

    if t > drone.get("endurance_s", math.inf):
        print(f"warning: {obj['id']} flies {t / 60:.0f} min, longer than its {drone['endurance_s'] / 60:.0f} min endurance")
    return np.array(rows)


def _multirotor_step(st, course, hspeed, alt, f, wind, drag_k):
    v, amax = st["v"], f["max_accel_mps2"]
    want = hspeed * np.array([math.sin(course), math.cos(course)])
    acc = (want - v[:2]) / DT
    if np.linalg.norm(acc) > amax:
        acc *= amax / np.linalg.norm(acc)
    vu = _approach(alt - st["p"][2], f["max_climb_mps"], f["max_descent_mps"], amax)
    au = float(np.clip((vu - v[2]) / DT, -amax, amax))
    # thrust must also beat drag on the airspeed, and the tilt limit caps the total
    v_air = v[:2] - wind
    drag = drag_k * np.linalg.norm(v_air) * v_air
    force = acc + drag
    fmax = G * math.tan(math.radians(f["max_tilt_deg"]))
    if np.linalg.norm(force) > fmax:
        force *= fmax / np.linalg.norm(force)
    v[:2] += (force - drag) * DT
    v[2] += au * DT
    st["p"] += v * DT
    if np.linalg.norm(v[:2]) > 1.0:  # nose follows the direction of travel
        turn = _wrap(math.atan2(v[0], v[1]) - st["psi"])
        lim = math.radians(f["max_yaw_rate_dps"]) * DT
        st["psi"] = _wrap(st["psi"] + max(-lim, min(lim, turn)))
    fwd = np.array([math.sin(st["psi"]), math.cos(st["psi"])])
    right = np.array([math.cos(st["psi"]), -math.sin(st["psi"])])
    tf, tr, tu = force @ fwd, force @ right, G + au
    st["pitch"] = -math.degrees(math.atan2(tf, tu))  # accelerating forward = nose down
    st["roll"] = math.degrees(math.atan2(tr, math.hypot(tf, tu)))
    st["va"] = float(np.linalg.norm(np.append(v_air, v[2])))


def _fixed_wing_step(st, course, airspeed, alt, f, wind):
    amax = f["max_accel_mps2"]
    target = min(max(airspeed, f["min_speed_mps"]), f["max_speed_mps"])
    st["va"] += max(-amax * DT, min(amax * DT, target - st["va"]))
    w = _approach(alt - st["p"][2], f["max_climb_mps"], f["max_descent_mps"], amax)
    st["w"] += max(-amax * DT, min(amax * DT, w - st["w"]))
    vh = math.sqrt(max(st["va"] ** 2 - st["w"] ** 2, 1.0))
    # crab: the heading that makes good the desired course over the ground in this wind
    cross = wind @ np.array([math.cos(course), -math.sin(course)])
    psi_d = course + math.asin(max(-1.0, min(1.0, -cross / vh)))
    bank_max = math.radians(f["max_bank_deg"])
    phi_d = max(-bank_max, min(bank_max, math.atan(K_TURN * _wrap(psi_d - st["psi"]) * vh / G)))
    lim = math.radians(f["roll_rate_dps"]) * DT
    st["phi"] += max(-lim, min(lim, phi_d - st["phi"]))
    st["psi"] = _wrap(st["psi"] + G * math.tan(st["phi"]) / vh * DT)
    st["v"] = np.array([vh * math.sin(st["psi"]) + wind[0], vh * math.cos(st["psi"]) + wind[1], st["w"]])
    st["p"] += st["v"] * DT
    st["pitch"] = math.degrees(math.asin(st["w"] / st["va"]))
    st["roll"] = math.degrees(st["phi"])
