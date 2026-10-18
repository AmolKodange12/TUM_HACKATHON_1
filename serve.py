"""Viewer + engine API. Run: python serve.py [port], then open http://localhost:8765/viewer/

GET  /api/areas                 areas.json
GET  /api/terrain?area=<id>[&cells=64]   DEM grid for the area previews (16-256 cells across)
POST /api/run {area, receivers, drones, seed, settings?}   generate + simulate, returns the viewer bundle
     settings: duration_s, shadowing_db, mix (balanced|quadcopters|fixed_wing|attack), mast_m, bearing_sigma_deg, wind_from_deg, wind_mps
POST /api/game {area, huts: [[lat, lon, mast_m?]], birds, seed}   a round of the bird game, returns the bundle
POST /api/simulate <scenario>   simulate a scenario from the planner into out/planned/
"""
import json
import sys
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import sim

ROOT = Path(__file__).parent
AREAS = {a["id"]: a for a in json.loads((ROOT / "areas.json").read_text())}
LOCK = threading.Lock()  # ponytail: one run at a time; per-area locks if several people share a server
MAX_BODY = 2_000_000


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
