# Mirror World

A synthetic scenario generator for aerial objects over real terrain in Ukraine. You give it a scenario file. It gives back the truth (where each drone was) and the data that each radio receiver recorded, on one shared clock.

A single LLM prompt can produce numbers that look correct. This engine computes every value from the terrain and the radio physics, so you can explain every record. Click a receiver row in the viewer to see the full loss breakdown for that moment.

The drones fly a kinematic flight model with each type's real limits (speed, acceleration, climb, tilt or bank), so they brake, lean and bank through turns. The viewer shows each one as a 3D model with that attitude.

![The four drone models, from above and as a ground camera sees them](models/previews/all.png)

## Run

```bash
uv venv && uv pip install -r requirements.txt    # or: pip install -r requirements.txt
.venv/bin/python serve.py                        # then open http://localhost:8765/viewer/
.venv/bin/python test_sim.py                     # faithfulness checks
.venv/bin/python sim.py                          # optional: scenario.json -> out/, no viewer
```

1. Choose one of the 4 areas.
2. Set the number of receivers and drones. The engine runs again (in a few seconds) and puts them at random positions.
3. Click **New layout** to get a new seed and new positions.

The first visit to an area downloads part of one Copernicus GLO-30 tile from public S3 (no key). The engine keeps it in `.cache/`.

The URL always holds the current layout, so you can share it. Example: `viewer/?area=kaniv&rx=3&dr=2&seed=42&t=450&ground=map&sel=RF2|drone_01`. The parameters are:
- `area`: the area id from `areas.json`
- `rx`, `dr`: the number of receivers and drones
- `seed`: the random seed
- `t`: the time in seconds
- `ground`: `sat`, `map` or `elev`
- `show`: `truth`, `sensed` or `both`
- `sel`: a receiver|object pair

## Files

| File | What it does |
|---|---|
| `areas.json` | The example areas. To add an area, add a line. The box must stay inside one 1° tile |
| `scenario.json` | A hand-written example scenario for `sim.py`, with receivers and a radar |
| `sim.py` | Engine: random layout → DEM → truth → radio/radar links → fixes → outputs |
| `serve.py` | Serves the viewer and runs the engine for each request |
| `flight.py` | Flight model: drone type + flight plan → 10 Hz track with heading, pitch and roll |
| `drones.json` | Drone types: size, mass, endurance, flight limits, 3D model |
| `models/` | One GLB model per drone type, plus the Blender scripts that build and render them |
| `test_sim.py` | Checks that the physics behaves as it should |
| `viewer/index.html` | 3D replay. One file, libraries come from a CDN, no build step |

## Random layout

- Receivers go on high ground (the top 25% of the inner box). Each new receiver takes the position farthest from the receivers that are already placed.
- Each drone gets a random type from `drones.json`:
  - `quadcopter` (50%): 60–120 m above ground, 2.4 GHz
  - `fixed_wing_recon` (25%), `delta_wing` (15%), `decoy` (10%): 200–400 m above ground, 5.8 GHz
- Each drone flies a loop of 3–5 random waypoints at the cruise speed of its type.
- Each receiver and drone has its own random stream from `(seed, kind, index)`. When you add a drone, the others do not move.

## Drones

Each object's `type` is a key in `drones.json`:

| Type | Class | Size | Cruise (max) | Turns by |
|---|---|---|---|---|
| `quadcopter` | DJI Mavic 3 | 0.35 m, 0.9 kg | 12 m/s (21) | tilting up to 35°, can hover and land |
| `fixed_wing_recon` | Orlan-10 | 3.1 m span, 18 kg | 30 m/s (42) | banking up to 30° |
| `delta_wing` | Shahed-136 / Geran-2 | 2.5 m span, 200 kg | 51 m/s (56) | banking up to 30° |
| `decoy` | Gerbera | 2.4 m span, 18 kg | 42 m/s (47) | banking up to 30°; rough figures |

The figures are rounded public numbers for the class, not the spec sheet of one airframe.

A drone starts at `waypoints[0]`, at `alt_agl_m` and `speed_ms`, and flies to each further waypoint (`loop` repeats them). For other trajectories, give a `plan` instead of more waypoints, as a list of legs:

```json
"plan": [{"orbit": {"lat": 49.75, "lon": 31.50, "radius_m": 800, "duration_s": 600, "direction": "ccw"}},
         {"goto": {"lat": 49.70, "lon": 31.45, "alt_m": 120, "speed_mps": 15}},
         {"hover": {"duration_s": 120}}, {"land": {}}]
```

| Leg | Fields | What it does |
|---|---|---|
| `takeoff` | `alt_m` | climbs to that height above ground |
| `goto` | `lat`, `lon`, opt. `alt_m`, `speed_mps` | flies to a point; a multirotor brakes to a stop unless another `goto` follows |
| `orbit` | `lat`, `lon`, `radius_m`, `duration_s`, opt. `direction` (`cw`/`ccw`), `alt_m`, `speed_mps` | flies to the circle and circles it; refuses a radius too tight for the drone |
| `hover` | `duration_s` | multirotor only |
| `land` | none | multirotor only |

When the plan ends before the scenario does, a multirotor hovers and a fixed-wing circles its last point. An optional top-level `"wind": {"from_deg": 270, "speed_mps": 10}` makes fixed-wing drones crab and change speed over the ground, and makes a multirotor lean into the wind.

The models are at real size in metres, nose −Z, right +X, up +Y (glTF axes), with the propellers as separate `prop_*` objects. The viewer draws them 44 px across at any zoom, and paints each one in its path colour. To rebuild them or render one as a camera would see it (attitude from a `truth.csv` row):

```bash
blender -b -P models/build.py
blender -b -P models/render.py -- models/delta_wing.glb frame.png --heading 250 --roll 12 --distance 300 --cam-el 20 --fov 3
```

## Outputs (`out/<area>/`)

| File | Contents | Who sees it |
|---|---|---|
| `truth.csv` | Position, altitude, speed, heading, pitch and roll of each object, every second | Answer key |
| `rf_raw.csv` | What each receiver reports: power, SNR, bearing. Detections only | Sensor data |
| `rf_budget.csv` | Full link budget for every receiver, object and second, detected or not | Explanation |
| `radar_raw.csv` | What the radar reports: range, bearing, radial velocity, SNR. Detections only | Sensor data |
| `radar_budget.csv` | Full two-way link budget for every radar, object and second, detected or not | Explanation |
| `fused.cot.xml` | Positions from crossed bearings or a radar's own range+bearing, as TAK Cursor on Target events | Sensor data |
| `fused.csv` | The same fixes, with `source_object_id`, `method` and the error against the truth | Answer key |
| `bundle.json` | Everything the viewer needs | Viewer |

Every sensor record has a `source_object_id` column. All files use the same UTC clock.

## Radio model

For each receiver, each drone and each second:

1. Sample the terrain on the straight line between the receiver and the drone. Add the earth bulge (k = 4/3).
2. Calculate the free-space path loss.
3. Find the worst obstacle and calculate its knife-edge diffraction loss (ITU-R P.526).
4. Add log-normal shadowing. The shadowing changes slowly over time (AR(1)), so it does not jump at random each second.
5. Received power = EIRP + antenna gain − losses. SNR = power − noise floor. Detected if SNR ≥ threshold.
6. Bearing = true bearing + fixed receiver bias + noise. The noise grows as the SNR drops.

When two or more receivers detect the same drone, a weighted least-squares crossing gives a position. `ce` comes from the covariance of that solution.

## Radar model

A radar is active: it transmits on its own frequency and hears its own echo, so it detects a drone whether or not that
drone is transmitting. For each radar, each drone and each second:

1. Run the same terrain-and-diffraction path loss as the radio model, once, since the radar sits at both ends of the link.
2. Double it (the signal crosses that path twice, out and back) and add the target's reflected aperture, from its
   radar cross-section (`rcs_dbsm` in `drones.json`) and the radar's wavelength.
3. Add the gain from integrating many pulses over a dwell (`integration_gain_db`) — a single pulse at this power would
   rarely detect anything.
4. Received power = transmit power + antenna gain (×2, the same antenna sends and listens) − losses + RCS gain +
   integration gain. SNR = power − noise floor. Detected if SNR ≥ threshold.
5. Bearing = true bearing + fixed bias + noise, same as a receiver. Range and radial velocity (Doppler) get their own
   noise, also growing as the SNR drops.

One radar already gives a full fix (range + bearing), no crossing with a second sensor needed.

## Checks (`test_sim.py`)

- On flat ground, the loss always increases with distance.
- A 150 m ridge between the receiver and the drone adds more than 15 dB of loss.
- Bearing noise grows as the SNR drops.
- Crossed bearings with no noise find the exact target.
- Radar echo power falls with distance; a closing target has negative radial velocity, a receding one positive.
- A radar range+bearing fix with no noise finds the exact target.
- A quadcopter keeps to its speed, acceleration and tilt limits, and lands where it took off.
- A fixed-wing holds an orbit in a 15 m/s wind without passing its bank limit, and its speed over the ground swings by the wind.
- The same seed gives byte-identical output files.
- Adding a receiver or a drone does not move the others.

## Known limits

Each limit has a `ponytail:` comment in `sim.py` that tells where to extend the code.

- Diffraction uses one obstacle only (not Deygout). There is no land-cover clutter.
- Each drone has one emitter that transmits all the time. There is no duty cycle and no interference.
- The fusion step uses `source_object_id` to group detections. A real system must group them by frequency and time.
- The area must stay inside one 1° DEM tile. The local metre frame is flat (good to about 15 km).
- Drones follow the ground at a fixed height above it. They do not climb over ridges.
- Radar RCS is one fixed number per drone class; a real target's RCS swings with aspect angle. Radial velocity uses
  horizontal ground velocity only (no climb/descent component).
- The wind is the same at every height. Propellers do not spin in the viewer.

## Next tasks

These do not change the structure. `scenario.json` is the input and `out/` is the output, so the engine and the viewer can change independently.

- Deygout diffraction and ESA WorldCover clutter
- Duty cycle, interference between drones, a jammer
- Remote ID as a second sensor type
- Error ellipse from the fix covariance
- Terrain profile chart in the "why" card
- LLM prompt → `scenario.json`
- Camera and satellite sensors (`models/render.py` already renders a drone at a given range and attitude)

## Data credits

- Terrain: Copernicus GLO-30 DEM, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.
- Satellite: Esri, Maxar, Earthstar Geographics.
- Map: Esri World Topographic Map, © OpenStreetMap contributors and others.
