"""Explicit browser-triggered runtime lifecycle; boot never launches a model."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from .control import ControlError, finite_number


class PolicyProcess:
    def __init__(self, python, model, serial_port, command_port, *,
                 startup_timeout=20., shutdown_timeout=5., step_cm=None,
                 kp_scale=1., kd_scale=1., popen=subprocess.Popen, clock=time.monotonic):
        self.python = Path(python).expanduser().resolve()
        self.model = Path(model).expanduser().resolve()
        self.main = Path(__file__).resolve().parents[1] / "humanoid_jetson_deploy" / "main.py"
        if not self.python.is_file() or not self.model.is_file() or not self.main.is_file():
            raise ValueError("Policy Python, model and main.py must exist")
        if not serial_port or not 1 <= command_port <= 65535:
            raise ValueError("Valid policy serial port and UDP port required")
        for name, value in (("startup_timeout", startup_timeout), ("shutdown_timeout", shutdown_timeout)):
            if finite_number(value, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (("kp_scale", kp_scale), ("kd_scale", kd_scale)):
            if not 0 <= finite_number(value, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if step_cm is not None and finite_number(step_cm, "step_cm") <= 0:
            raise ValueError("step_cm must be positive")
        self.serial_port, self.command_port = serial_port, command_port
        self.startup_timeout, self.shutdown_timeout = startup_timeout, shutdown_timeout
        self.step_cm, self.kp_scale, self.kd_scale = step_cm, kp_scale, kd_scale
        self.popen, self.clock = popen, clock
        self.lock = threading.RLock()
        self.directory = tempfile.TemporaryDirectory(prefix="humanoid-remote-ready-")
        self.process = None
        self.phase, self.error = "stopped", None
        self.ready_path = None
        self.started_at = self.stopping_at = 0.
        self.closed = False

    def status(self):
        with self.lock:
            return {"policy_managed": True, "policy_state": self.phase,
                    "policy_ready": self.phase == "ready", "policy_error": self.error}

    def start(self):
        with self.lock:
            if self.closed:
                raise ControlError("Remote service is shutting down", 503)
            self.tick()
            if self.process is not None:
                if self.phase in ("starting", "ready"):
                    return
                raise ControlError("Previous walking run is stopping; try enabling again shortly", 503)
            self.ready_path = Path(self.directory.name) / f"{uuid.uuid4().hex}.ready"
            command = [str(self.python), "-u", str(self.main), "--model", str(self.model),
                       "--command-source", "remote", "--udp-command-bind", "127.0.0.1",
                       "--udp-command-port", str(self.command_port), "--port", self.serial_port,
                       "--startup-ready-file", str(self.ready_path), "--no-plot", "--enable-motors",
                       "--kp-scale", str(self.kp_scale), "--kd-scale", str(self.kd_scale)]
            if self.step_cm is not None:
                command.extend(["--step-cm", str(self.step_cm)])
            try:
                # No shell, detached daemon or stdout pipe: systemd owns the full group/log.
                self.process = self.popen(command, cwd=str(self.main.parent), stdin=subprocess.DEVNULL)
            except OSError as exc:
                self.phase, self.error = "failed", str(exc)
                raise ControlError(f"Could not start walking runtime: {exc}", 503) from exc
            self.started_at = self.clock()
            self.phase, self.error = "starting", None

    def cancel(self, error=None):
        with self.lock:
            if self.process is None or self.phase == "stopping":
                return
            self.error = error
            self.phase, self.stopping_at = "stopping", self.clock()
            if self.process.poll() is None:
                try:
                    if os.name == "nt":
                        self.process.terminate()  # Hardware deployment is Linux (SIGINT below).
                    else:
                        self.process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass

    def tick(self):
        with self.lock:
            if self.process is None:
                return
            code = self.process.poll()
            if code is not None:
                intentional = self.phase == "stopping"
                self.process = None
                if intentional and self.error is None:
                    self.phase = "stopped"
                else:
                    self.phase = "failed"
                    self.error = self.error or f"Walking runtime exited (code {code}); enable again"
                return
            if self.phase == "starting":
                if self.clock() - self.started_at >= self.startup_timeout:
                    self.cancel("Walking runtime readiness timed out; check model, serial and STM32 state")
                elif self.ready_path.is_file() and self.ready_path.read_text(encoding="ascii").strip() == "ready":
                    self.phase = "ready"
            elif self.phase == "stopping" and self.clock() - self.stopping_at >= self.shutdown_timeout:
                self.process.kill()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.cancel()
            self.closed = True
            process = self.process
        if process is not None:
            try:
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.)
        with self.lock:
            self.process = None
            self.phase = "stopped"
            self.directory.cleanup()
