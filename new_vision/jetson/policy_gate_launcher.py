"""Start the walking policy once the competition start gate has passed."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]


class PolicyGateLauncher:
    def __init__(self, model: str, port: str, max_seconds: float,
                 ready_timeout_s: float = 30.0,
                 policy_python: str | None = None) -> None:
        self.model = model
        self.port = port
        self.max_seconds = max_seconds
        self.ready_timeout_s = ready_timeout_s
        preferred_python = (Path(policy_python).expanduser() if policy_python
                            else REPO_ROOT / ".venv/bin/python")
        preflight_runtime = bool(policy_python) or preferred_python.is_file()
        if not policy_python and not preferred_python.is_file():
            preferred_python = Path(sys.executable)
        if not preferred_python.is_file() or not os.access(preferred_python, os.X_OK):
            raise RuntimeError(f"policy Python is not executable: {preferred_python}")
        self.policy_python = str(preferred_python)
        self.onnxruntime_version = "not preflighted"
        if preflight_runtime:
            probe = subprocess.run(
                [self.policy_python, "-c",
                 "import onnxruntime; print(onnxruntime.__version__)"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=False)
            if probe.returncode != 0:
                detail = (probe.stderr or probe.stdout).strip()
                raise RuntimeError(f"policy Python cannot import onnxruntime: "
                                   f"{self.policy_python}: {detail}")
            self.onnxruntime_version = probe.stdout.strip()
        self.process: subprocess.Popen | None = None
        self.tee: subprocess.Popen | None = None
        self.ready_file: Path | None = None
        self.started_at = 0.0
        self.log_path: Path | None = None
        self.live_log_path: Path | None = None

    @property
    def started(self) -> bool:
        return self.process is not None

    def start(self) -> None:
        if self.started:
            return
        model = REPO_ROOT / self.model
        one_foot = REPO_ROOT / "humanoid_jetson_deploy/policy-one-foot-standing.onnx"
        if not model.is_file() or not one_foot.is_file():
            raise RuntimeError(f"policy model missing: {model if not model.is_file() else one_foot}")
        records = REPO_ROOT / "records"
        records.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = records / f"main_{stamp}.log"
        self.live_log_path = records / "main_live.log"
        self.ready_file = Path(tempfile.gettempdir()) / f"policy_ready_{uuid.uuid4().hex}"
        command = [
            self.policy_python, "-u", "humanoid_jetson_deploy/main.py",
            "--policy", "walking", "--model", self.model,
            "--one-foot-model", "humanoid_jetson_deploy/policy-one-foot-standing.onnx",
            "--port", self.port, "--command-source", "vision",
            "--udp-command-port", "5005",
            "--diagnostic-log-dir", "records/control_diagnostics",
            "--no-plot", "--max-seconds", f"{self.max_seconds:g}",
            "--enable-motors", "--startup-ready-file", str(self.ready_file),
        ]
        self.process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
        assert self.process.stdout is not None
        try:
            self.tee = subprocess.Popen(["tee", str(self.log_path),
                                         str(self.live_log_path)], cwd=REPO_ROOT,
                                        stdin=self.process.stdout)
        except Exception:
            self.process.terminate()
            self.process.wait()
            raise
        finally:
            self.process.stdout.close()
        self.started_at = time.monotonic()
        print(f"[start-gate] C started pid={self.process.pid}; python={self.policy_python}; "
              f"onnxruntime={self.onnxruntime_version}; log={self.log_path}; "
              f"live={self.live_log_path}", flush=True)

    def exit_report(self, status: int, phase: str = "") -> str:
        """Include C's last output in B's exception even if C's tee races its exit."""
        if self.tee is not None:
            try:
                self.tee.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        label = f" {phase}" if phase else ""
        report = f"policy exited{label} (code {status}); C log: {self.log_path}"
        if self.log_path is None:
            return report
        try:
            with self.log_path.open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 8192))
                recent = log.read().decode("utf-8", errors="replace").splitlines()[-20:]
        except OSError:
            return report
        if recent:
            report += "\n[C recent output]\n" + "\n".join(recent)
        return report

    def ready(self) -> bool:
        if self.process is None:
            return False
        status = self.process.poll()
        if status is not None:
            raise RuntimeError(self.exit_report(status, "before start release"))
        if self.ready_file is not None and self.ready_file.is_file():
            return True
        if time.monotonic() - self.started_at > self.ready_timeout_s:
            raise RuntimeError(f"policy did not become ready within {self.ready_timeout_s:g}s; "
                               f"see {self.log_path}")
        return False

    def close(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.tee is not None:
            try:
                self.tee.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.tee.terminate()
                self.tee.wait()
        if self.ready_file is not None:
            try:
                self.ready_file.unlink()
            except FileNotFoundError:
                pass
