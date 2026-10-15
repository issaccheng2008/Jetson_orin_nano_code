#!/usr/bin/env python3
"""Local web console / SSH command client for the dedicated teaching runtime."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import signal
import sys
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import numpy as np

from protocol import (COMMAND_ENABLE, COMMAND_ESTOP, STATE_ENCODERS_VALID,
                      STATE_MOTORS_ENABLED, STATE_TEACHING_READY, StatePacket)
from teaching import TeachingController, TeachingStore

WEB = Path(__file__).with_name("teaching_web")
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "records" / "teaching"
ALIASES = {"开始记录": "start-recording", "停止记录": "stop-recording",
           "记录checkpoint": "checkpoint", "锁定": "hold", "松弛": "release", "急停": "estop"}


class SimulatedLink:
    """Explicitly offline; free joints move slowly to demonstrate teaching."""
    state_queue_drops = 0

    def __init__(self):
        self.q = np.zeros(12)
        self.target = self.q.copy()
        self.enabled, self.free, self.sequence = False, False, 0
        self.started = time.monotonic()

    def drain_states(self):
        now = time.monotonic()
        if self.free and self.enabled:
            self.q = 0.12 * np.sin((now-self.started) * 0.6 + np.arange(12) * 0.4)
        elif self.enabled:
            self.q += (self.target-self.q) * 0.75
        self.sequence = (self.sequence + 1) & 0xffff
        flags = STATE_TEACHING_READY | STATE_ENCODERS_VALID
        if self.enabled:
            flags |= STATE_MOTORS_ENABLED
        return [(StatePacket(self.sequence, int((now-self.started)*1e6) & 0xffffffff,
                             self.q.copy(), np.zeros(12), np.zeros(3), np.zeros(3),
                             np.array([1., 0, 0, 0]), flags), now)]

    def send_command(self, timestamp_us, joint_target, kp_scale, kd_scale, command_flags, **kwargs):
        self.target = np.asarray(joint_target).copy()
        self.enabled = bool(command_flags & COMMAND_ENABLE and not command_flags & COMMAND_ESTOP)
        self.free = kp_scale == kd_scale == 0
        return True

    def close(self):
        pass


def make_handler(controller, token, simulated=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _trusted_host(self):
            return urlsplit("http://" + self.headers.get("Host", "")).hostname in ("127.0.0.1", "localhost", "::1")

        def _reply(self, code, data, mime="application/json; charset=utf-8", attachment=None):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            if attachment:
                from urllib.parse import quote
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(attachment))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError):
                pass

        def do_GET(self):
            if not self._trusted_host():
                self._reply(403, dict(error="只允许 localhost / 127.0.0.1"))
                return
            url = urlsplit(self.path)
            try:
                if url.path == "/api/state":
                    self._reply(200, dict(controller.snapshot(), simulated=simulated))
                elif url.path == "/api/session":
                    self._reply(200, dict(token=token))
                elif url.path == "/api/files":
                    self._reply(200, controller.store.files())
                elif url.path == "/api/download":
                    name = parse_qs(url.query)["name"][0]
                    path = controller.store.path(name)
                    if path.suffix not in (".jsonl", ".json"):
                        raise ValueError("不支持此文件类型")
                    # Recording files can be large; stream without holding the control lock.
                    size = path.stat().st_size
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Cache-Control", "no-store")
                    from urllib.parse import quote
                    self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(name))
                    self.end_headers()
                    with path.open("rb") as f:
                        remaining = size
                        while remaining:
                            chunk = f.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                else:
                    assets = {"/": ("index.html", "text/html; charset=utf-8"),
                              "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                              "/style.css": ("style.css", "text/css; charset=utf-8")}
                    if url.path not in assets:
                        self._reply(404, dict(error="不存在"))
                        return
                    name, mime = assets[url.path]
                    self._reply(200, (WEB/name).read_bytes(), mime)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                self._reply(400, dict(error=str(exc)))

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (not self._trusted_host()
                    or not secrets.compare_digest(self.headers.get("X-Teaching-Token", ""), token)
                    or (origin and urlsplit(origin).netloc != self.headers.get("Host"))):
                self._reply(403, dict(error="控制请求验证失败；请刷新页面"))
                return
            if self.path != "/api/command":
                self._reply(404, dict(error="不存在"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024*1024:
                    raise ValueError("请求大小不合法")
                self.connection.settimeout(3)
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("请求必须为 JSON 对象")
                self._reply(200, controller.execute(data["action"], data))
            except (ValueError, TypeError, KeyError, OSError, StopIteration) as exc:
                self._reply(400, dict(error=str(exc)))
    return Handler


def assert_port_unused(port):
    """Detect older processes that opened the serial device without exclusivity."""
    import os
    if sys.platform != "linux":
        return
    device = os.path.realpath(port)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for fd in (process / "fd").iterdir():
                if os.path.realpath(fd) == device:
                    raise RuntimeError(f"串口 {device} 已被 PID {process.name} 占用；先停止行走/按钮自启动程序")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue


def serve(args):
    store = TeachingStore(args.data_dir)
    # Bind before opening serial: a second server must not touch motor state.
    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), BaseHTTPRequestHandler)
    controller = None
    link = None
    try:
        if args.simulate:
            link = SimulatedLink()
        else:
            assert_port_unused(args.port)
            from serial_link import SerialLink
            link = SerialLink(args.port, args.baud, queue_states=True)
        controller = TeachingController(link, store, enable_motors=args.enable_motors or args.simulate,
                                        max_speed=args.max_speed_rad_s)
        server.RequestHandlerClass = make_handler(controller, secrets.token_urlsafe(32), args.simulate)
        controller.start()
        print(f"{'SIMULATION — ' if args.simulate else ''}http://127.0.0.1:{args.http_port}", flush=True)
        print(f"记录目录: {store.root}\n启动后仅监控；点击松弛后进入示教。Ctrl+C 退出。", flush=True)
        def terminate(*_args):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, terminate)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
    finally:
        server.server_close()
        if controller:
            controller.close()
        elif link:
            link.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    run = sub.add_parser("serve", help="启动 Jetson 本地示教服务")
    run.add_argument("--port", default="/dev/ttyACM0")
    run.add_argument("--baud", type=int, default=921600)
    run.add_argument("--http-port", type=int, default=8765)
    run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    run.add_argument("--enable-motors", action="store_true")
    run.add_argument("--simulate", action="store_true", help="完全离线演示，不打开串口")
    run.add_argument("--max-speed-rad-s", type=float, default=3.0)
    cli = sub.add_parser("command", help="通过 SSH 向已启动服务发送命令")
    cli.add_argument("action", choices=["status", "files", "release", "hold", "disable", "estop", "reset",
                                       "start-recording", "stop-recording", "checkpoint", "new-checkpoints",
                                       "load-checkpoints", "play-log", "play-checkpoints", "stop-playback"] + list(ALIASES))
    cli.add_argument("--http-port", type=int, default=8765)
    cli.add_argument("--name")
    cli.add_argument("--label", default="")
    cli.add_argument("--approach-s", type=float, default=2)
    cli.add_argument("--speed", type=float, default=1)
    args = parser.parse_args()
    if args.operation == "serve":
        serve(args)
    else:
        args.action = ALIASES.get(args.action, args.action)
        base = f"http://127.0.0.1:{args.http_port}"
        try:
            if args.action in ("status", "files"):
                with urlopen(base + "/api/" + ("state" if args.action == "status" else "files"), timeout=5) as r:
                    result = json.load(r)
            else:
                with urlopen(base + "/api/session", timeout=5) as r:
                    token = json.load(r)["token"]
                data = dict(action=args.action, name=args.name, label=args.label,
                            approach_s=args.approach_s, speed=args.speed)
                request = Request(base+"/api/command", json.dumps(data).encode(),
                                  headers={"Content-Type": "application/json", "X-Teaching-Token": token})
                with urlopen(request, timeout=5) as r:
                    result = json.load(r)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            from urllib.error import HTTPError
            message = exc.read().decode() if isinstance(exc, HTTPError) else str(exc)
            parser.exit(1, f"命令失败: {message}\n")


if __name__ == "__main__":
    main()
