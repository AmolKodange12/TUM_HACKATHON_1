"""Viewer + engine API. Run: python serve.py [port], then open http://localhost:8765/viewer/

GET  /api/areas                 areas.json
GET  /api/terrain?area=<id>[&cells=64]   DEM grid for the area previews (16-256 cells across)
POST /api/run {area, receivers, drones, seed, settings?}   generate + simulate, returns the viewer bundle
     settings: duration_s, shadowing_db, mix (balanced|quadcopters|fixed_wing|attack), mast_m, bearing_sigma_deg, wind_from_deg, wind_mps
POST /api/game {area, huts: [[lat, lon, mast_m?]], birds, seed}   a round of the bird game, returns the bundle
POST /api/simulate <scenario>   simulate a scenario from the planner into out/planned/
GET  /api/export?run=<dir>      the run in out/<dir>/ as a ZIP: data files, scenario, manifest, README
"""
import io
import json
import re
import subprocess
import sys
import threading
import traceback
import zipfile
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import sim

ROOT = Path(__file__).parent
AREAS = {a["id"]: a for a in json.loads((ROOT / "areas.json").read_text())}
LOCK = threading.Lock()  # ponytail: one run at a time; per-area locks if several people share a server
MAX_BODY = 2_000_000

# Export: what each output file is. "answer key" files hold the truth; keep them away from a model under test.
FILE_ROLES = {
    "truth.csv": ("answer key", "Where each object really was, every second: position, height, speed, heading, pitch, roll."),
    "fused.csv": ("answer key", "Position fixes with their estimated error (ce_m), the method, source_object_id and the real error_m."),
    "fused.cot.xml": ("sensor data", "The same fixes as TAK Cursor on Target events. ATAK and TAK Server can read them."),
    "rf_raw.csv": ("sensor data", "What each radio receiver reports: power, SNR, bearing. Detections only."),
    "radar_raw.csv": ("sensor data", "What each radar reports: range, bearing, radial speed, SNR. Detections only."),
    "camera_raw.csv": ("sensor data", "What each daylight or thermal camera reports: azimuth, elevation, class, confidence. Detections only."),
    "acoustic_raw.csv": ("sensor data", "What each microphone array reports: bearing and sound level. Detections only."),
    "remote_id_raw.csv": ("sensor data", "Decoded Remote ID messages: the position, height and speed that the drone broadcasts."),
    "rf_budget.csv": ("explanation", "Full radio link budget for each receiver, object and second, detected or not."),
    "radar_budget.csv": ("explanation", "Full two-way radar budget for each radar, object and second."),
    "camera_budget.csv": ("explanation", "Line of sight, pixels on target and contrast for each camera, object and second."),
    "acoustic_budget.csv": ("explanation", "Sound path losses and travel time for each microphone array, object and second."),
    "remote_id_budget.csv": ("explanation", "Remote ID link budget for each receiver, object and second."),
    "scenario.json": ("input", "The exact scenario. Run `python sim.py scenario.json out` to get the same files again."),
}
EXPORT_README = """Mirror World scenario export

All files use the same UTC clock. Each sensor record has a source_object_id column:
the object that caused the record. manifest.json gives the role of each file.

Roles:
- sensor data: what the sensors report. The source_object_id column is the label.
- answer key: the truth. Do not give these files to a model that you test.
- explanation: why each value is what it is (losses, line of sight), for every second.
- input: the scenario. The same scenario and seed always give the same files.

To get the same files again:
  python sim.py scenario.json out
"""


def export_zip(run_dir):
    """One run folder -> (zip bytes, file name). Leaves out bundle.json, which only the viewer reads."""
    sc = json.loads((run_dir / "scenario.json").read_text())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    files = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run_dir.iterdir()):
            if f.name not in FILE_ROLES:
                continue
            data = f.read_bytes()
            z.writestr(f.name, data)
            role, about = FILE_ROLES[f.name]
            rows = data.count(b"<event ") if f.suffix == ".xml" else max(data.count(b"\n") - 1, 0) if f.suffix == ".csv" else None
            files.append({"name": f.name, "role": role, "rows": rows, "bytes": len(data), "description": about})
        lists = {"sensors": "rf_receivers", "radars": "radars", "cameras": "cameras",
                 "acoustic_arrays": "acoustic_arrays", "remote_id_receivers": "remote_id_receivers"}
        manifest = {
            "name": sc.get("name"), "region": sc.get("region"), "bbox": sc.get("bbox"),
            "start_utc": sc.get("start_utc"), "duration_s": sc.get("duration_s"), "seed": sc.get("seed"),
            "sensors": {out: [x["id"] for x in sc[k]] for k, out in lists.items() if sc.get(k)},
            "objects": [{"id": o["id"], "type": o["type"]} for o in sc.get("objects", [])],
            "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "engine_commit": commit or None,
            "files": files,
        }
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("README.txt", EXPORT_README)
    slug = re.sub(r"[^a-z0-9]+", "-", (sc.get("name") or run_dir.name).lower()).strip("-")
    return buf.getvalue(), f"mirror-world_{slug}_seed{sc.get('seed', 0)}.zip"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # a new run must show up on reload
        super().end_headers()

    def send_json(self, obj, code=200):
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def api(self, fn):
        try:
            self.send_json(fn())
        except (KeyError, ValueError, TypeError, AssertionError) as e:
            self.send_json({"error": f"Bad request: {e}"}, 400)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def body(self):
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_BODY:
            raise ValueError(f"the request must be 1 byte to {MAX_BODY // 1000} kB")
        return json.loads(self.rfile.read(size))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/areas":
            return self.send_json(list(AREAS.values()))
        if u.path == "/api/terrain":
            def terrain():
                q = parse_qs(u.query)
                area, cells = AREAS[q["area"][0]], min(max(int(q.get("cells", ["64"])[0]), 16), 256)
                with LOCK:
                    t = sim.Terrain(sim.fetch_dem(area["bbox"]), area["bbox"])
                tj, frame = sim.terrain_json(t, cells)
                return {"terrain": tj, "frame": frame}
            return self.api(terrain)
        if u.path == "/api/export":
            name = parse_qs(u.query).get("run", [""])[0]
            run_dir = ROOT / "out" / name
            if not re.fullmatch(r"[a-z0-9_-]{1,40}", name) or not (run_dir / "scenario.json").is_file():
                return self.send_json({"error": "No run with that name. Run a scenario first."}, 404)
            with LOCK:  # a run in progress must not change the files while they are read
                data, filename = export_zip(run_dir)
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/run":
            def run():
                req = self.body()
                area = AREAS[req["area"]]
                n_rx = min(max(int(req["receivers"]), 1), 6)
                n_dr = min(max(int(req["drones"]), 1), 6)
                seed = int(req["seed"]) % 1_000_000
                with LOCK:
                    return sim.run(sim.make_scenario(area, n_rx, n_dr, seed, req.get("settings")), ROOT / "out" / area["id"])
            return self.api(run)
        if self.path == "/api/game":
            def game():
                req = self.body()
                area = AREAS[req["area"]]
                huts = [[float(v) for v in h[:3]] for h in req["huts"]][:6]  # [lat, lon] or [lat, lon, mast_m]
                if not huts:
                    raise ValueError("place at least one station")
                n = min(max(int(req.get("birds", 6)), 1), 12)
                seed = int(req.get("seed", 1)) % 1_000_000
                with LOCK:
                    return sim.run(sim.make_game(area, huts, n, seed), ROOT / "out" / f"game-{area['id']}")
            return self.api(game)
        if self.path == "/api/simulate":
            def simulate():
                sc = self.body()
                if not isinstance(sc, dict) or not sc.get("sensors") or not sc.get("objects"):
                    raise ValueError("a scenario needs at least one receiver and one drone")
                with LOCK:
                    b = sim.run(sc, ROOT / "out" / "planned")
                return {"ok": True, "fixes": {k: len(v) for k, v in b["fused"].items()}}
            return self.api(simulate)
        self.send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://localhost:{port}/viewer/   planner: http://localhost:{port}/planner/")
    ThreadingHTTPServer(("", port), Handler).serve_forever()
