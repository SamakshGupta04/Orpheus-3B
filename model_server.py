#!/usr/bin/env python3
"""Minimal stdlib-only HTTP JSON server for hosting one warm inference model.

Used by infer_model{1,2,3}.py's --serve mode. Deliberately dependency-free
(no flask/fastapi) so it drops into each model's isolated venv without any
extra pip install.

Model load and every inference job run on a single dedicated worker thread
(never the HTTP handler thread), so libraries that do main-thread-style
global monkey-patching at import time (unsloth, bitsandbytes) always see
load and inference happen on the same thread, in order, one at a time.

Protocol:
    GET  /health            -> {"model_state": "loading|ready|busy|error", "model_error": str|None}
    GET  /status?job_id=N   -> {"model_state":, "model_error":, "job": {"id","done","error","result","log"} | None}
    POST /infer   {..payload..}        -> 202 {"job_id": N}  |  409 {"model_state": ...} if not ready
         (Content-Type: application/json) — payload's "ref_audio"/"output" are
         filesystem paths already reachable by this process. This is how
         app.py talks to the server (same container, same filesystem).
    POST /infer   multipart/form-data  -> same response as above
         Fields: "ref_audio" (file, required), "target_text" (text,
         required), "ref_transcript" (text, optional), plus any
         model-specific text fields (max_new_tokens / cfg_value / timesteps /
         cfg_scale). The uploaded file is saved server-side and an output
         path is generated automatically — this is the Postman-friendly path,
         since callers don't need filesystem access to the server.
    GET  /audio?job_id=N    -> 200 audio/wav bytes for a completed job, or 404
    POST /shutdown -> 200, then process exits shortly after
"""

import json
import os
import queue
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_TAIL = 500  # rolling log cap, lines
UPLOAD_DIR = os.path.join("outputs", "live", "postman")


def _query_param(path: str, name: str, cast=str):
    if "?" not in path:
        return None
    for part in path.split("?", 1)[1].split("&"):
        if part.startswith(name + "="):
            try:
                return cast(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _parse_multipart(body: bytes, content_type: str) -> dict:
    """Parse a multipart/form-data body without cgi (removed in 3.13) or extra deps.

    Returns {field_name: str} for text fields and
    {field_name: {"filename", "content_type", "data": bytes}} for file fields.
    """
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type)
    if not m:
        raise ValueError("no multipart boundary in Content-Type")
    boundary = (m.group(1) or m.group(2)).strip().encode()
    fields = {}
    for part in body.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        headers_raw, content = part.split(b"\r\n\r\n", 1)
        content = content[:-2] if content.endswith(b"\r\n") else content
        disp = ""
        content_type_h = "application/octet-stream"
        for line in headers_raw.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition:"):
                disp = line.decode(errors="replace")
            elif line.lower().startswith(b"content-type:"):
                content_type_h = line.split(b":", 1)[1].strip().decode(errors="replace")
        name_m = re.search(r'name="([^"]*)"', disp)
        if not name_m:
            continue
        name = name_m.group(1)
        filename_m = re.search(r'filename="([^"]*)"', disp)
        if filename_m and filename_m.group(1):
            fields[name] = {
                "filename": filename_m.group(1),
                "content_type": content_type_h,
                "data": content,
            }
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields


def _payload_from_multipart(body: bytes, content_type: str) -> dict:
    """Save the uploaded ref_audio file server-side and build an /infer payload.

    Mirrors the JSON payload shape (ref_audio/output become real paths) so
    infer_model{1,2,3}.py's run_inference() needs no changes for this path.
    """
    fields = _parse_multipart(body, content_type)
    ref_file = fields.get("ref_audio")
    if not isinstance(ref_file, dict):
        raise ValueError("missing file field 'ref_audio'")
    if "target_text" not in fields:
        raise ValueError("missing required field 'target_text'")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    stamp = int(time.time() * 1000)
    suffix = os.path.splitext(ref_file["filename"] or "")[1] or ".wav"
    ref_path = os.path.join(UPLOAD_DIR, f"upload_{stamp}{suffix}")
    with open(ref_path, "wb") as f:
        f.write(ref_file["data"])

    payload = {k: v for k, v in fields.items() if isinstance(v, str)}
    payload["ref_audio"] = ref_path
    payload["output"] = os.path.join(UPLOAD_DIR, f"result_{stamp}.wav")
    return payload


class ModelServer:
    def __init__(self, name, load_fn, infer_fn):
        """
        load_fn(log) -> objs                     called once, on the worker thread
        infer_fn(objs, payload, log) -> result    called once per job, on the worker thread
        `log` is a callable(str) -> None that both prints and records the line.
        """
        self.name = name
        self._load_fn = load_fn
        self._infer_fn = infer_fn

        self.state = "loading"  # loading | ready | busy | error
        self.model_error = None
        self.model_objs = None

        self.log_lines = []
        self.job = None  # {"id", "log_start", "done", "error", "result"}

        self._lock = threading.Lock()
        self._job_queue = queue.Queue()
        self._job_counter = 0

    def _log(self, line):
        with self._lock:
            self.log_lines.append(line)
            if len(self.log_lines) > LOG_TAIL:
                self.log_lines = self.log_lines[-LOG_TAIL:]
        print(line, flush=True)

    def submit_job(self, payload):
        with self._lock:
            if self.state != "ready":
                return None
            self._job_counter += 1
            job_id = self._job_counter
            self.job = {
                "id": job_id,
                "log_start": len(self.log_lines),
                "done": False,
                "error": None,
                "result": None,
            }
            self.state = "busy"
        self._job_queue.put((job_id, payload))
        return job_id

    def get_job_result_path(self, job_id):
        """Output file path for a completed, error-free job, or None."""
        with self._lock:
            if self.job and self.job["id"] == job_id and self.job["done"] and not self.job["error"]:
                result = self.job.get("result") or {}
                return result.get("output_path")
        return None

    def get_status(self, job_id=None):
        with self._lock:
            job = None
            if self.job is not None and (job_id is None or self.job["id"] == job_id):
                job = dict(self.job)
                job["log"] = self.log_lines[self.job["log_start"]:]
            return {
                "model_state": self.state,
                "model_error": self.model_error,
                "log_tail": self.log_lines[-50:],
                "job": job,
            }

    def _worker_loop(self, dry_run_label=""):
        try:
            self.model_objs = self._load_fn(self._log)
            with self._lock:
                self.state = "ready"
            self._log(f"[{self.name}] Ready.{dry_run_label}")
        except Exception as e:
            with self._lock:
                self.state = "error"
                self.model_error = str(e)
            self._log(f"[{self.name}] ERROR loading model: {e}\n{traceback.format_exc()}")
            return  # worker thread ends; server keeps reporting the error state

        while True:
            job_id, payload = self._job_queue.get()
            try:
                result = self._infer_fn(self.model_objs, payload, self._log)
                with self._lock:
                    self.job["result"] = result
                    self.job["done"] = True
            except Exception as e:
                self._log(f"[{self.name}] ERROR: {e}\n{traceback.format_exc()}")
                with self._lock:
                    self.job["error"] = str(e)
                    self.job["done"] = True
            finally:
                with self._lock:
                    self.state = "ready"


def serve(server: ModelServer, port: int, dry_run_label: str = ""):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                status = server.get_status()
                self._send_json(200, {
                    "model_state": status["model_state"],
                    "model_error": status["model_error"],
                })
                return
            if self.path.startswith("/status"):
                job_id = _query_param(self.path, "job_id", int)
                self._send_json(200, server.get_status(job_id))
                return
            if self.path.startswith("/audio"):
                job_id = _query_param(self.path, "job_id", int)
                path = server.get_job_result_path(job_id) if job_id is not None else None
                if not path or not os.path.exists(path):
                    self._send_json(404, {"error": "no completed job with that job_id, or its file is gone"})
                    return
                with open(path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
                self.end_headers()
                self.wfile.write(data)
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/infer":
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b""
                content_type = self.headers.get("Content-Type", "")
                if content_type.startswith("multipart/form-data"):
                    try:
                        payload = _payload_from_multipart(raw, content_type)
                    except Exception as e:
                        self._send_json(400, {"error": f"invalid multipart body: {e}"})
                        return
                else:
                    try:
                        payload = json.loads(raw or b"{}")
                    except json.JSONDecodeError:
                        self._send_json(400, {"error": "invalid JSON body"})
                        return
                job_id = server.submit_job(payload)
                if job_id is None:
                    self._send_json(409, {"model_state": server.state})
                else:
                    self._send_json(202, {"job_id": job_id})
                return
            if self.path == "/shutdown":
                self._send_json(200, {"status": "shutting down"})

                def _die():
                    time.sleep(0.2)
                    os._exit(0)

                threading.Thread(target=_die, daemon=True).start()
                return
            self._send_json(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            pass  # silence default per-request access logging

    worker = threading.Thread(target=server._worker_loop, args=(dry_run_label,), daemon=True)
    worker.start()

    # 127.0.0.1 only: external access now goes through the nginx gateway
    # (see nginx.conf.template), which runs in this same container and
    # reaches this server over loopback too. This server is never published
    # or reachable directly — nginx is the sole ingress and adds the
    # /api/<model>/ API-key check this port intentionally skips.
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[{server.name}] Serving on port {port} (loading model in background)...", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
