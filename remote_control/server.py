"""Small authenticated LAN HTTP service and isolated 50 Hz UDP publisher."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import threading
import time
from urllib.parse import urlsplit

from .control import ControlError, ControlState


STATIC = Path(__file__).with_name("static")
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8")}
ZERO = b'{"vx":0.0,"vy":0.0,"wz":0.0,"qr":-1}'


class RemoteHandler(BaseHTTPRequestHandler):
    # One bounded request per connection also limits slow/partial clients.
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(1.)

    def log_message(self, _format, *args):
        pass  # Avoid access-token or request-body logging.

    def reply(self, code, data, mime="application/json; charset=utf-8"):
        payload = json.dumps(data, allow_nan=False).encode() if isinstance(data, dict) else data
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                         "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authorized(self):
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.reply(403, {"error": "Cross-origin control is not allowed"})
            return False
        provided = self.headers.get("Authorization", "").encode("utf-8")
        expected = f"Bearer {self.server.service.token}".encode("utf-8")
        if not hmac.compare_digest(provided, expected):
            self.reply(401, {"error": "Incorrect access token"})
            return False
        if self.server.service.stopped.is_set():
            self.reply(503, {"error": "Command publisher is unavailable"})
            return False
        return True

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            if self.authorized():
                self.reply(200, self.server.service.state.snapshot())
        elif path in ASSETS:
            name, mime = ASSETS[path]
            try:
                self.reply(200, (STATIC / name).read_bytes(), mime)
            except FileNotFoundError:
                self.reply(404, {"error": "Page asset is missing"})
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authorized():
            return
        path = urlsplit(self.path).path
        if path not in ("/api/arm", "/api/command", "/api/stop"):
            self.reply(404, {"error": "Not found"})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ControlError("Transfer-Encoding is not supported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048:
                raise ControlError("JSON body must be 1..2048 bytes", 413)
            if self.headers.get_content_type() != "application/json":
                raise ControlError("Content-Type must be application/json", 415)
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ControlError("Incomplete request")
            def invalid_constant(value):
                raise ValueError(f"Invalid number: {value}")
            message = json.loads(payload, parse_constant=invalid_constant)
            if not isinstance(message, dict):
                raise ControlError("JSON object required")
            state = self.server.service.state
            if path == "/api/arm":
                result = state.arm()
            elif path == "/api/command":
                result = state.command(message["session"], message["seq"], message["vx"], message["wz"])
            else:
                result = state.stop(message["session"])
            self.reply(200, result)
        except ControlError as exc:
            self.reply(exc.status, {"error": str(exc)})
        except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
            self.reply(400, {"error": "Invalid command JSON or missing fields"})
        except (TimeoutError, ConnectionError):
            self.close_connection = True


class RemoteService:
    def __init__(self, host, port, token, policy_port=5007, *, state=None):
        if not token or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError("Access token must be nonempty ASCII without whitespace")
        if not 1 <= policy_port <= 65535:
            raise ValueError("policy_port must be 1..65535")
        self.token = token
        self.state = state or ControlState()
        self.destination = ("127.0.0.1", policy_port)
        self.stopped = threading.Event()
        self.failure = None
        self.http = ThreadingHTTPServer((host, port), RemoteHandler)
        self.http.daemon_threads = True
        self.http.service = self
        self.address = self.http.server_address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.publisher_thread = threading.Thread(target=self._publish, daemon=True)
        self.http_thread = threading.Thread(target=self.http.serve_forever,
                                            kwargs={"poll_interval": .05}, daemon=True)
        self.started = False

    def start(self):
        self.started = True
        self.publisher_thread.start()
        self.http_thread.start()

    def _publish(self):
        previous = time.monotonic()
        try:
            while not self.stopped.is_set():
                now = time.monotonic()
                # Serialize send with stop/command so an old tick cannot follow a newer stop.
                with self.state.lock:
                    status = self.state.tick(now - previous)
                    payload = {"vx": status["vx"], "vy": 0., "wz": status["wz"], "qr": -1}
                    self.sock.sendto(json.dumps(payload, allow_nan=False).encode(), self.destination)
                previous = now
                self.stopped.wait(max(0., .02 - (time.monotonic() - now)))
        except Exception as exc:
            self.failure = str(exc)
            self.stopped.set()
        finally:
            self.state.force_stop()
            for _ in range(3):
                try:
                    self.sock.sendto(ZERO, self.destination)
                except OSError:
                    break

    def close(self):
        self.stopped.set()
        if self.started:
            self.publisher_thread.join(timeout=2)
            self.http.shutdown()
            self.http_thread.join(timeout=2)
        self.http.server_close()
        self.sock.close()
        self.state.force_stop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LAN dual-stick browser remote (no motor/serial access)")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policy-port", type=int, default=5007)
    parser.add_argument("--max-vx", type=float, default=.4)
    parser.add_argument("--max-wz", type=float, default=.5)
    parser.add_argument("--timeout", type=float, default=.3)
    parser.add_argument("--vx-accel", type=float, default=1.)
    parser.add_argument("--wz-accel", type=float, default=2.)
    parser.add_argument("--yaw-sign", type=int, choices=(-1, 1), default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be 1..65535")
    token = os.environ.get("ROBOT_REMOTE_TOKEN") or secrets.token_urlsafe(18)
    try:
        state = ControlState(max_vx=args.max_vx, max_wz=args.max_wz, timeout=args.timeout,
                             vx_accel=args.vx_accel, wz_accel=args.wz_accel, yaw_sign=args.yaw_sign)
        service = RemoteService(args.bind, args.port, token, args.policy_port, state=state)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    def shutdown(_signum, _frame):
        service.stopped.set()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Remote page: http://<Jetson-LAN-IP>:{args.port} (bind {args.bind})", flush=True)
    print(f"Local preview: http://127.0.0.1:{args.port}", flush=True)
    print(f"Access token: {token}", flush=True)
    print(f"Command output: 127.0.0.1:{args.policy_port}; no robot feedback; motor enable stays in main.py", flush=True)
    service.start()
    try:
        while not service.stopped.wait(.2):
            pass
    finally:
        service.close()
    if service.failure:
        print(f"Command publisher stopped: {service.failure}")
        return 1
    print("Remote stopped; zero commands sent")
    return 0
