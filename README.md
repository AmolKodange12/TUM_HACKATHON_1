# Mirror World

A synthetic scenario generator for aerial objects over real terrain in Ukraine. You give it a scenario file. It gives back the truth (where each drone was) and the data that each radio receiver recorded, on one shared clock.

A single LLM prompt can produce numbers that look correct. This engine computes every value from the terrain and the radio physics, so you can explain every record. Click a receiver row in the viewer to see the full loss breakdown for that moment.

## Run

```bash
uv venv && uv pip install -r requirements.txt    # or: pip install -r requirements.txt
.venv/bin/python serve.py                        # then open http://localhost:8765/viewer/
.venv/bin/python test_sim.py                     # faithfulness checks
.venv/bin/python sim.py                          # optional: scenario.json -> out/, no viewer
```

1. Choose one of the 4 areas.
2. Set the number of receivers and drones. The engine runs again (in less than 1 s) and puts them at random positions.
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
| `scenario.json` | A hand-written example scenario for `sim.py` |
| `sim.py` | Engine: random layout → DEM → truth → radio links → crossed bearings → outputs |
| `serve.py` | Serves the viewer and runs the engine for each request |
| `test_sim.py` | Checks that the physics behaves as it should |
| `viewer/index.html` | 3D replay. One file, libraries come from a CDN, no build step |

## Random layout

- Receivers go on high ground (the top 25% of the inner box). Each new receiver takes the position farthest from the receivers that are already placed.
- Each drone gets a random type:
  - quadcopter: 60–120 m above ground, 2.4 GHz
  - fixed-wing: 200–400 m above ground, 5.8 GHz
- Each drone flies a loop of 3–5 random waypoints.
- Each receiver and drone has its own random stream from `(seed, kind, index)`. When you add a drone, the others do not move.

## Outputs (`out/<area>/`)

| File | Contents | Who sees it |
|---|---|---|
| `truth.csv` | Position, altitude and heading of each object, every second | Answer key |
| `rf_raw.csv` | What each receiver reports: power, SNR, bearing. Detections only | Sensor data |
| `fused.cot.xml` | Positions from crossed bearings, as TAK Cursor on Target events | Sensor data |
| `fused.csv` | The same fixes, with `source_object_id` and the error against the truth | Answer key |
| `rf_budget.csv` | Full link budget for every receiver, object and second, detected or not | Explanation |
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

## Checks (`test_sim.py`)

- On flat ground, the loss always increases with distance.
- A 150 m ridge between the receiver and the drone adds more than 15 dB of loss.
- Bearing noise grows as the SNR drops.
- Crossed bearings with no noise find the exact target.
- The same seed gives byte-identical output files.
- Adding a receiver or a drone does not move the others.

## Known limits

Each limit has a `ponytail:` comment in `sim.py` that tells where to extend the code.

- Diffraction uses one obstacle only (not Deygout). There is no land-cover clutter.
- Each drone has one emitter that transmits all the time. There is no duty cycle and no interference.
- The fusion step uses `source_object_id` to group detections. A real system must group them by frequency and time.
- The area must stay inside one 1° DEM tile. The local metre frame is flat (good to about 15 km).
- Paths are straight lines between waypoints at a fixed height above the ground.

## Next tasks

These do not change the structure. `scenario.json` is the input and `out/` is the output, so the engine and the viewer can change independently.

- Deygout diffraction and ESA WorldCover clutter
- Duty cycle, interference between drones, a jammer
- Remote ID as a second sensor type
- Error ellipse from the fix covariance
- Terrain profile chart in the "why" card
- LLM prompt → `scenario.json`
- Camera and satellite sensors

## Data credits

- Terrain: Copernicus GLO-30 DEM, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.
- Satellite: Esri, Maxar, Earthstar Geographics.
- Map: Esri World Topographic Map, © OpenStreetMap contributors and others.
