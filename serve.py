"""Local server: the viewer, the planner, and POST /api/simulate (scenario JSON -> sim.run -> out/).

Usage: .venv/bin/python serve.py [port]   then open http://localhost:8765/planner/
"""
import json
import sys
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sim

ROOT = Path(__file__).parent
OUT = ROOT / "out"
MAX_BODY = 2_000_000
RUN_LOCK = threading.Lock()  # one simulation at a time; they all write out/


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # a new run must show up on reload
        super().end_headers()

    def do_POST(self):
        if self.path != "/api/simulate":
            return self.send_error(404)
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_BODY:
            return self._json(413, {"error": f"scenario must be 1 byte to {MAX_BODY // 1000} kB"})
        try:
            sc = json.loads(self.rfile.read(size))
            if not isinstance(sc, dict) or not sc.get("sensors") or not sc.get("objects"):
                raise ValueError("a scenario needs at least one receiver and one drone")
            OUT.mkdir(exist_ok=True)
            path = OUT / "planned.json"
            path.write_text(json.dumps(sc, indent=2))
            with RUN_LOCK:
                _, fused, _ = sim.run(path, OUT)
            self._json(200, {"ok": True, "fixes": {k: len(v) for k, v in fused.items()}})
        except (ValueError, KeyError, TypeError, AssertionError, json.JSONDecodeError) as e:
            self._json(400, {"error": f"{type(e).__name__}: {e}"})
        except Exception as e:  # DEM download, rasterio, ...: show it, keep serving
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"planner: http://localhost:{port}/planner/   replay: http://localhost:{port}/viewer/")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
