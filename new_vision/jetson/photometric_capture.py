"""Safe V4L2 control and paired-image capture helpers for calibration."""
from __future__ import annotations

import datetime as _datetime
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from photometric_lut import gray_from_bgr, write_png


UNKNOWN = "unknown"
EXPOSURE_ALIASES = ("auto_exposure", "exposure_auto")
EXPOSURE_VALUE_ALIASES = ("exposure_time_absolute", "exposure_absolute")
WB_AUTO_ALIASES = ("white_balance_automatic", "white_balance_temperature_auto",
                   "auto_white_balance")
WB_VALUE_ALIASES = ("white_balance_temperature",)


class CalibrationError(RuntimeError):
    pass


def utc_now():
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def create_session_directory(root, session_id=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stem = session_id or _datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    for suffix in range(10000):
        name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
        path = root / name
        try:
            path.mkdir()
            (path / "auto").mkdir()
            (path / "fixed").mkdir()
            return path
        except FileExistsError:
            continue
    raise OSError(f"could not create a unique session directory under {root}")


def parse_v4l2_catalog(text):
    controls, current = {}, None
    for line in text.splitlines():
        match = re.match(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\(([^)]+)\)\s*:\s*(.*)$", line)
        if match:
            name, kind, description = match.groups()
            current = {"kind": kind, "description": description, "menu": {}}
            for key, value in re.findall(r"\b(min|max|step|default|value)=(-?\d+)", description):
                current[key] = int(value)
            current["flags"] = [s.strip() for s in description.split("flags=", 1)[1].split()
                                if s.strip()] if "flags=" in description else []
            controls[name] = current
            continue
        menu = re.match(r"^\s+(-?\d+):\s*(.*)$", line)
        if menu and current is not None:
            current["menu"][int(menu[1])] = menu[2].strip()
    return controls


class V4L2Controller:
    """Query, set and restore controls only after reading the device menu."""
    def __init__(self, device, runner=None, platform=None):
        self.device = str(device)
        self.runner = runner or subprocess.run
        self.platform = platform or sys.platform
        self.catalog = {}
        self.initial = {}
        self.initial_snapshot = {}

    def _call(self, flag):
        result = self.runner(["v4l2-ctl", "-d", self.device, flag],
                             capture_output=True, text=True, timeout=4, check=False)
        if result.returncode:
            raise CalibrationError((result.stderr or result.stdout or
                                     f"v4l2-ctl {flag} failed").strip())
        return result.stdout

    def query_initial_state(self):
        if not self.platform.startswith("linux"):
            raise CalibrationError("camera control query requires Linux V4L2")
        try:
            self.catalog = parse_v4l2_catalog(self._call("--list-ctrls-menus"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CalibrationError(f"cannot query camera control menu: {exc}") from exc
        controls = {}
        self.initial = {}
        for name, info in self.catalog.items():
            actual = UNKNOWN
            if "read-only" not in info["description"] and "read_only" not in info["description"]:
                try:
                    actual = self.read(name)
                except CalibrationError:
                    pass
            elif "value" in info:
                actual = info["value"]
            self.initial[name] = actual
            controls[name] = {**info, "actual": actual}
        self.initial_snapshot = {"device": self.device, "controls": controls,
                                 "captured_at": utc_now()}
        if not self.catalog:
            raise CalibrationError("camera reports no readable V4L2 controls")
        return self.initial_snapshot

    def read(self, name):
        try:
            output = self._call(f"--get-ctrl={name}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CalibrationError(f"cannot read {name}: {exc}") from exc
        match = re.search(r"^\s*" + re.escape(name) + r":\s*(-?\d+)", output, re.MULTILINE)
        if not match:
            raise CalibrationError(f"driver did not return a numeric value for {name}")
        return int(match.group(1))

    def read_all(self):
        values = {}
        for name in self.catalog:
            try:
                values[name] = self.read(name)
            except CalibrationError:
                values[name] = UNKNOWN
        return values

    def _find(self, aliases):
        return next((name for name in aliases if name in self.catalog), None)

    def _menu_value(self, aliases, semantic):
        name = self._find(aliases)
        if name is None:
            raise CalibrationError(f"required V4L2 control missing: {', '.join(aliases)}")
        info = self.catalog[name]
        if not info["menu"]:
            raise CalibrationError(f"{name} has no readable menu; refusing to guess {semantic} value")
        def matches(label):
            label = label.lower()
            if semantic == "manual":
                return "manual" in label
            return ("auto" in label and "manual" not in label) or "aperture priority" in label
        options = [(value, label) for value, label in info["menu"].items() if matches(label)]
        if not options:
            raise CalibrationError(f"{name} menu has no identifiable {semantic} entry: {info['menu']}")
        current = self.initial.get(name, UNKNOWN)
        return name, next((value for value, _ in options if value == current), options[0][0])

    def _wb_auto_value(self, want_auto):
        name = self._find(WB_AUTO_ALIASES)
        if name is None:
            raise CalibrationError(f"required V4L2 control missing: {', '.join(WB_AUTO_ALIASES)}")
        info = self.catalog[name]
        if info["menu"]:
            target = "auto" if want_auto else "manual"
            def is_target(label):
                label = label.lower()
                if want_auto:
                    return ("auto" in label or "automatic" in label) and "manual" not in label
                return any(word in label for word in ("manual", "disabled", "disable", "off"))
            choices = [(v, label) for v, label in info["menu"].items() if is_target(label)]
            if not choices:
                raise CalibrationError(f"{name} menu has no identifiable {target} entry: {info['menu']}")
            current = self.initial.get(name, UNKNOWN)
            return name, next((v for v, _ in choices if v == current), choices[0][0])
        if info["kind"] == "bool" and ("auto" in name or "automatic" in name):
            low, high = info.get("min", 0), info.get("max", 1)
            value = high if want_auto else low
            if not low <= value <= high:
                raise CalibrationError(f"{name} bool range cannot represent automatic/manual")
            return name, value
        raise CalibrationError(f"{name} does not expose a menu or bool that identifies auto/manual")

    def _check_range(self, name, value):
        info = self.catalog[name]
        if "read-only" in info["description"] or "read_only" in info["description"]:
            raise CalibrationError(f"{name} is read-only")
        low, high = info.get("min"), info.get("max")
        step = info.get("step", 1)
        if low is None or high is None:
            raise CalibrationError(f"{name} has no queryable range; refusing to guess")
        if value < low or value > high:
            raise CalibrationError(f"{name} request {value} outside driver range {low}..{high}")
        if step > 0 and (value - low) % step:
            raise CalibrationError(f"{name} request {value} violates driver step {step} from {low}")
        return value

    def _set_verify(self, name, value, semantic=None):
        self._check_range(name, value)
        try:
            self._call(f"--set-ctrl={name}={value}")
            actual = self.read(name)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CalibrationError(f"setting {name} failed: {exc}") from exc
        record = {"control": name, "requested_value": int(value), "actual_value": actual,
                  "semantic": semantic}
        if actual != value:
            record["error"] = "driver readback differs from requested value"
            raise CalibrationError(f"{name} requested {value}, read back {actual}")
        return record

    def apply_group(self, mode, fixed_exposure_ms=None, fixed_wb_k=None):
        if not self.catalog:
            raise CalibrationError("initial control menu has not been queried")
        if mode not in ("auto", "fixed"):
            raise ValueError("mode must be auto or fixed")
        # Resolve and validate every control before changing any camera state.
        if mode == "auto":
            exp_name, exp_value = self._menu_value(EXPOSURE_ALIASES, "auto")
            wb_name, wb_value = self._wb_auto_value(True)
            operations = [(exp_name, exp_value, "auto_exposure"),
                          (wb_name, wb_value, "auto_white_balance")]
        else:
            if fixed_exposure_ms is None:
                raise CalibrationError("fixed exposure value is required")
            if (not math.isfinite(float(fixed_exposure_ms)) or fixed_exposure_ms <= 0
                    or not math.isclose(float(fixed_exposure_ms) * 10,
                                        round(float(fixed_exposure_ms) * 10),
                                        rel_tol=0, abs_tol=1e-7)):
                raise CalibrationError("fixed exposure must be positive in 0.1 ms steps")
            exp_mode_name, exp_mode_value = self._menu_value(EXPOSURE_ALIASES, "manual")
            wb_mode_name, wb_mode_value = self._wb_auto_value(True)
            exp_name = self._find(EXPOSURE_VALUE_ALIASES)
            if exp_name is None:
                raise CalibrationError("manual exposure-time control is missing")
            exp_value = int(round(float(fixed_exposure_ms) * 10))
            self._check_range(exp_name, exp_value)
            # Only exposure is fixed; both groups use verified automatic WB.
            operations = [(exp_mode_name, exp_mode_value, "manual_exposure"),
                          (wb_mode_name, wb_mode_value, "auto_white_balance"),
                          (exp_name, exp_value, "fixed_exposure_ms")]
        report = {"mode": mode, "requested": {"exposure_ms": fixed_exposure_ms,
                                                  "white_balance": "auto"},
                  "operations": []}
        for name, value, semantic in operations:
            item = self._set_verify(name, value, semantic)
            if semantic == "fixed_exposure_ms":
                item["requested_ms"] = float(fixed_exposure_ms)
                item["actual_ms"] = item["actual_value"] / 10.0
            if semantic == "auto_exposure" or semantic == "manual_exposure":
                item["menu_label"] = self.catalog[name]["menu"].get(value, UNKNOWN)
            report["operations"].append(item)
        report["verified"] = True
        return report

    def restore_initial(self):
        """Restore changed exposure/WB controls and detectable auto side effects."""
        records = []
        try:
            exp_mode = self._find(EXPOSURE_ALIASES)
            exp_value_name = self._find(EXPOSURE_VALUE_ALIASES)
            wb_auto = self._find(WB_AUTO_ALIASES)
            wb_value_name = self._find(WB_VALUE_ALIASES)
            for name, aliases, kind in ((exp_mode, EXPOSURE_ALIASES, "exposure"),
                                        (wb_auto, WB_AUTO_ALIASES, "white_balance")):
                if name is None:
                    continue
                original = self.initial.get(name, UNKNOWN)
                if original == UNKNOWN:
                    records.append({"control": name, "status": "unknown_original_value"})
                    continue
                try:
                    records.append(self._set_verify(name, int(original), f"restore_{kind}_mode"))
                except Exception as exc:
                    records.append({"control": name, "status": "failed", "error": str(exc)})
            # Restore manual values only when their original mode was manual.
            for mode_name, value_name, aliases, kind in (
                    (exp_mode, exp_value_name, EXPOSURE_ALIASES, "exposure"),
                    (wb_auto, wb_value_name, WB_AUTO_ALIASES, "white_balance")):
                if mode_name is None or value_name is None:
                    continue
                mode_value = self.initial.get(mode_name, UNKNOWN)
                label = self.catalog.get(mode_name, {}).get("menu", {}).get(mode_value, "").lower()
                if kind == "exposure":
                    original_manual = "manual" in label
                elif label:
                    original_manual = any(word in label for word in
                                          ("manual", "disabled", "disable", "off"))
                else:
                    original_manual = (mode_value == self.catalog.get(mode_name, {}).get("min", 0))
                original_value = self.initial.get(value_name, UNKNOWN)
                if original_manual and original_value != UNKNOWN:
                    try:
                        records.append(self._set_verify(value_name, int(original_value),
                                                        f"restore_{kind}_value"))
                    except Exception as exc:
                        records.append({"control": value_name, "status": "failed", "error": str(exc)})
            # Automatic modes may cause a driver to move gain or other controls
            # even though this tool never writes them. Restore any readable,
            # writable companion control that actually changed from the initial
            # snapshot; leave read-only/inactive controls untouched and report it.
            mode_controls = {n for n in (exp_mode, exp_value_name, wb_auto, wb_value_name) if n}
            for name, original in self.initial.items():
                if name in mode_controls or original == UNKNOWN:
                    continue
                info = self.catalog.get(name, {})
                if ("read-only" in info.get("description", "")
                        or "read_only" in info.get("description", "")):
                    continue
                try:
                    current = self.read(name)
                except CalibrationError:
                    continue
                if current == original:
                    continue
                if "inactive" in info.get("description", ""):
                    records.append({"control": name, "status": "changed_but_inactive",
                                    "initial": original, "current": current})
                    continue
                try:
                    records.append(self._set_verify(name, int(original), "restore_changed_control"))
                except Exception as exc:
                    records.append({"control": name, "status": "failed", "error": str(exc),
                                    "initial": original, "current": current})
        except Exception as exc:
            records.append({"status": "failed", "error": str(exc)})
        failed = any(r.get("status") == "failed" or r.get("error") for r in records)
        unknown = any(r.get("status") in ("unknown_original_value", "changed_but_inactive")
                      for r in records)
        return {"status": "failed" if failed else "partial" if unknown else "restored",
                "operations": records, "finished_at": utc_now()}


@dataclass(frozen=True)
class CaptureStabilityConfig:
    warmup_seconds: float = 3.0
    stability_window_seconds: float = 1.5
    stability_timeout_seconds: float = 25.0
    check_interval_seconds: float = 0.15
    brightness_relative_range: float = 0.02
    channel_ratio_range: float = 0.012
    sample_interval_seconds: float = 0.5
    group_brightness_relative_range: float = 0.03
    group_channel_ratio_range: float = 0.015
    group_exposure_relative_range: float = 0.05
    group_white_balance_range_k: float = 150.0
    count: int = 10

    def __post_init__(self):
        if self.warmup_seconds < 3.0:
            raise ValueError("warmup_seconds must be at least 3 seconds")
        if self.stability_window_seconds <= 0 or self.stability_timeout_seconds <= self.warmup_seconds:
            raise ValueError("invalid stability window/timeout")
        if self.check_interval_seconds <= 0 or self.sample_interval_seconds <= 0:
            raise ValueError("capture intervals must be positive")
        if self.count != 10:
            raise ValueError("calibration groups contain exactly 10 images")


class CameraReader:
    """Background capture thread; terminal input cannot leave a stale frame queued."""
    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.sequence = 0
        self.latest = None
        self.timestamp = None
        self.failed = None
        self.thread = threading.Thread(target=self._run, name="calibration-camera", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    if not self.stop_event.is_set():
                        with self.condition:
                            self.failed = "camera read returned no frame"
                            self.condition.notify_all()
                    return
                if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                    with self.condition:
                        self.failed = "camera returned a non-uint8 frame"
                        self.condition.notify_all()
                    return
                with self.condition:
                    self.sequence += 1
                    self.latest = frame.copy()
                    self.timestamp = time.monotonic()
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.failed = str(exc)
                self.condition.notify_all()

    def read_after(self, sequence, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while (self.latest is None or self.sequence <= sequence) and self.failed is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CalibrationError("timed out waiting for a fresh camera frame")
                self.condition.wait(remaining)
            if self.failed:
                raise CalibrationError(self.failed)
            return self.sequence, self.latest.copy(), self.timestamp

    def stop(self, timeout=3.0):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def close(self):
        self.stop_event.set()
        try:
            self.capture.release()
        finally:
            if self.thread.is_alive():
                self.thread.join(timeout=3.0)
        return not self.thread.is_alive()


def frame_metrics(frame, roi=(0.25, 0.35, 0.75, 0.75)):
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("camera frame must be BGR")
    h, w = image.shape[:2]
    x0, y0 = int(w * roi[0]), int(h * roi[1])
    x1, y1 = max(x0 + 1, int(w * roi[2])), max(y0 + 1, int(h * roi[3]))
    patch = image[y0:y1, x0:x1, :3]
    med = np.median(patch.reshape(-1, 3), axis=0)
    gray = gray_from_bgr(patch)
    return {"roi_xyxy_fraction": list(roi), "brightness_gray_median": float(np.median(gray)),
            "bgr_median": med.astype(float).tolist(),
            "r_over_g": float((med[2] + 1.0) / (med[1] + 1.0)),
            "b_over_g": float((med[0] + 1.0) / (med[1] + 1.0))}


def _percentile_range(values):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def wait_for_stability(reader, config: CaptureStabilityConfig, *, on_progress=None):
    """Discard transition frames and require a stable rolling ROI window."""
    started = time.monotonic()
    sequence, _, _ = reader.read_after(-1, timeout=5.0)
    warm_until = started + config.warmup_seconds
    samples = deque()
    last_check = 0.0
    while True:
        now = time.monotonic()
        if now - started >= config.stability_timeout_seconds:
            return {"stable": False, "reason": "stability_timeout",
                    "elapsed_seconds": now-started, "samples": len(samples)}
        try:
            sequence, frame, captured_at = reader.read_after(sequence, timeout=1.0)
        except CalibrationError as exc:
            return {"stable": False, "reason": "camera_read_failed", "error": str(exc),
                    "elapsed_seconds": time.monotonic()-started, "samples": len(samples)}
        now = time.monotonic()
        if now < warm_until:
            continue
        if now - last_check >= config.check_interval_seconds:
            metrics = frame_metrics(frame)
            samples.append((now, metrics))
            last_check = now
        while samples and now - samples[0][0] > config.stability_window_seconds * 1.25:
            samples.popleft()
        if on_progress is not None and now - started >= config.warmup_seconds:
            on_progress({"elapsed_seconds": now-started, "samples": len(samples)})
        if not samples or now - samples[0][0] < config.stability_window_seconds:
            continue
        brightness = [s[1]["brightness_gray_median"] for s in samples]
        rg = [s[1]["r_over_g"] for s in samples]
        bg = [s[1]["b_over_g"] for s in samples]
        bmed = max(float(np.median(brightness)), 1.0)
        stable = (_percentile_range(brightness) / bmed <= config.brightness_relative_range
                  and _percentile_range(rg) <= config.channel_ratio_range
                  and _percentile_range(bg) <= config.channel_ratio_range)
        if stable:
            return {"stable": True, "elapsed_seconds": now-started,
                    "window_seconds": now-samples[0][0], "sample_count": len(samples),
                    "brightness_relative_p90_p10": _percentile_range(brightness)/bmed,
                    "r_over_g_p90_p10": _percentile_range(rg),
                    "b_over_g_p90_p10": _percentile_range(bg),
                    "last_frame_sequence": sequence}


def _range_relative(values):
    values = [float(v) for v in values if isinstance(v, (int, float))]
    if len(values) < 2:
        return None
    med = max(abs(float(np.median(values))), 1.0)
    return float((max(values)-min(values))/med)


def assess_group_drift(samples, config, mode, exposure_control=None, wb_control=None):
    reasons = []
    metrics = [s["roi_metrics"] for s in samples]
    bright_rel = _range_relative([m["brightness_gray_median"] for m in metrics])
    rg_range = _percentile_range([m["r_over_g"] for m in metrics]) if metrics else None
    bg_range = _percentile_range([m["b_over_g"] for m in metrics]) if metrics else None
    if bright_rel is not None and bright_rel > config.group_brightness_relative_range:
        reasons.append("brightness_drift")
    if rg_range is not None and rg_range > config.group_channel_ratio_range:
        reasons.append("r_over_g_drift")
    if bg_range is not None and bg_range > config.group_channel_ratio_range:
        reasons.append("b_over_g_drift")
    control_ranges = {}
    for name, limit, kind in ((exposure_control, config.group_exposure_relative_range, "exposure"),
                              (wb_control, config.group_white_balance_range_k, "white_balance")):
        if not name:
            continue
        values = [s["controls"].get(name) for s in samples]
        values = [float(v) for v in values if isinstance(v, (int, float))]
        if len(values) >= 2:
            spread = max(values)-min(values)
            if kind == "exposure":
                control_ranges[name] = {"min": min(values), "max": max(values),
                                        "relative_range": _range_relative(values)}
                if _range_relative(values) is not None and _range_relative(values) > limit:
                    reasons.append("auto_exposure_parameter_drift" if mode == "auto" else "exposure_parameter_drift")
            else:
                control_ranges[name] = {"min": min(values), "max": max(values), "range": spread}
                if spread > limit:
                    reasons.append("auto_white_balance_parameter_drift")
        else:
            control_ranges[name] = {"status": "unknown_or_not_readable"}
    return {"qualified": not reasons, "reasons": reasons,
            "brightness_relative_range": bright_rel,
            "r_over_g_p90_p10": rg_range, "b_over_g_p90_p10": bg_range,
            "parameter_ranges": control_ranges}


def report_unrequested_control_changes(samples, controller, settings):
    """Report gain/gamma/etc side effects without pretending they were written."""
    deliberately_touched = {item.get("control") for item in settings.get("operations", [])}
    for aliases in (EXPOSURE_ALIASES, EXPOSURE_VALUE_ALIASES,
                    WB_AUTO_ALIASES, WB_VALUE_ALIASES):
        found = controller._find(aliases)
        if found:
            deliberately_touched.add(found)
    report = {}
    for name, initial in controller.initial.items():
        if name in deliberately_touched:
            continue
        values = [sample["controls"].get(name) for sample in samples]
        numeric = [int(v) for v in values if isinstance(v, (int, float))]
        if not numeric:
            continue
        changed_from_initial = (initial == UNKNOWN or any(v != initial for v in numeric))
        drifted_during_group = len(set(numeric)) > 1
        if changed_from_initial or drifted_during_group:
            report[name] = {"initial": initial, "sample_min": min(numeric),
                            "sample_max": max(numeric),
                            "changed_from_initial": bool(changed_from_initial),
                            "varied_during_group": bool(drifted_during_group)}
    return report


def capture_group(group, reader: CameraReader, controller: V4L2Controller,
                  directory, config=CaptureStabilityConfig(), *,
                  fixed_exposure_ms=None, fixed_wb_k=None, metadata_callback=None,
                  on_progress=None):
    """Apply/verify a complete settings mode before any PNG is written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("*.png")):
        raise CalibrationError(f"refusing to overwrite existing captures in {directory}")
    result = {"mode": group, "started_at": utc_now(), "status": "control_failed",
              "settings": None, "stability": None, "frames": [], "drift": None,
              "capture_policy": "quality_checks_advisory", "warnings": []}
    try:
        result["settings"] = controller.apply_group(group, fixed_exposure_ms, fixed_wb_k)
    except Exception as exc:
        result["control_error"] = str(exc)
        if metadata_callback:
            metadata_callback(result)
        return result
    if not result["settings"].get("verified", False):
        result["control_error"] = "settings did not pass readback verification"
        if metadata_callback:
            metadata_callback(result)
        return result
    result["status"] = "waiting_for_stability"
    if metadata_callback:
        metadata_callback(result)
    result["stability"] = wait_for_stability(reader, config, on_progress=on_progress)
    if metadata_callback:
        metadata_callback(result)
    if not result["stability"].get("stable"):
        if result["stability"].get("reason") == "stability_timeout":
            result["warnings"].append("stability_timeout")
            print("\n稳定性尚未达标，记录提示并继续拍摄 10 张。")
        else:
            result["status"] = "stability_failed"
            if metadata_callback:
                metadata_callback(result)
            return result
    sequence = result["stability"].pop("last_frame_sequence", -1)
    for index in range(config.count):
        target_time = time.monotonic() if index == 0 else next_sample_time
        try:
            # The sampling deadline is not a camera read deadline. Near the
            # deadline the remaining interval may be shorter than one frame;
            # keep the full read timeout and discard frames until their capture
            # timestamp reaches the scheduled time.
            while True:
                sequence, frame, captured_at = reader.read_after(sequence, timeout=2.0)
                if captured_at >= target_time:
                    break
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise CalibrationError("camera frame is not a BGR image")
            path = directory / f"{index:03d}.png"
            write_png(path, frame)
            controls = controller.read_all()
            sample = {"index": index, "file": path.name, "captured_at": utc_now(),
                      "frame_sequence": sequence, "frame_age_at_save_ms": max(0.0, (time.monotonic()-captured_at)*1000),
                      "resolution": [int(frame.shape[1]), int(frame.shape[0])],
                      "roi_metrics": frame_metrics(frame), "controls": controls}
            result["frames"].append(sample)
            if metadata_callback:
                metadata_callback(result)
        except Exception as exc:
            result["status"] = "incomplete"
            result["capture_error"] = str(exc)
            if metadata_callback:
                metadata_callback(result)
            return result
        next_sample_time = time.monotonic() + config.sample_interval_seconds
    exposure_name = controller._find(EXPOSURE_VALUE_ALIASES)
    wb_name = controller._find(WB_VALUE_ALIASES)
    result["drift"] = assess_group_drift(result["frames"], config, group,
                                         exposure_name, wb_name)
    result["unrequested_control_changes"] = report_unrequested_control_changes(
        result["frames"], controller, result["settings"])
    result["status"] = "complete" if result["drift"]["qualified"] else "unqualified_drift"
    result["warnings"].extend(result["drift"]["reasons"])
    result["capture_complete"] = True
    result["quality_qualified"] = bool(result["stability"].get("stable")
                                       and result["drift"]["qualified"])
    result["finished_at"] = utc_now()
    if metadata_callback:
        metadata_callback(result)
    return result

