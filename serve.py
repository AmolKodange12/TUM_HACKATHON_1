"""Viewer + engine API. Run: python serve.py [port], then open http://localhost:8765/viewer/

GET  /api/areas                 areas.json
GET  /api/terrain?area=<id>     small DEM grid for the area previews
POST /api/run {area, receivers, drones, seed}   generate + simulate, returns the viewer bundle
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


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

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
        except (KeyError, ValueError, TypeError) as e:
            self.send_json({"error": f"Bad request: {e}"}, 400)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/areas":
            return self.send_json(list(AREAS.values()))
        if u.path == "/api/terrain":
            def terrain():
                area = AREAS[parse_qs(u.query)["area"][0]]
                with LOCK:
                    t = sim.Terrain(sim.fetch_dem(area["bbox"]), area["bbox"])
                tj, frame = sim.terrain_json(t, 64)
                return {"terrain": tj, "frame": frame}
            return self.api(terrain)
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/run":
            return self.send_json({"error": "Not found"}, 404)

        def run():
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            area = AREAS[req["area"]]
            n_rx = min(max(int(req["receivers"]), 1), 6)
            n_dr = min(max(int(req["drones"]), 1), 6)
            seed = int(req["seed"]) % 1_000_000
            with LOCK:
                return sim.run(sim.make_scenario(area, n_rx, n_dr, seed), ROOT / "out" / area["id"])
        self.api(run)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://localhost:{port}/viewer/")
    ThreadingHTTPServer(("", port), Handler).serve_forever()
