#!/usr/bin/env python3
"""Offline, deterministic CPU/CUDA replay; never opens a camera or robot client.

Timing includes required downloads/synchronization and ordinary debug rendering.
Video sampling is a device comparison, not an accuracy or robot-policy evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision" / "jetson"))
import camera_config
import cuda_vision
import line_detector_v1_warp as line_module
import shape_detector as shape_module

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SHAPE_FIELDS = ("shape", "presence", "card_found", "quad", "presence_box",
                "presence_box_work", "presence_cy_frac", "cue_candidate_cy_frac",
                "presence_cue", "cue_presence_path", "box_top_work")
LINE_FIELDS = ("base_err_px", "base_err_cm", "heading_deg", "heading_valid",
               "preview_error_cm", "preview_valid", "measurement_valid",
               "measurement_quality", "measurement_stale", "near_z_cm",
               "near_error_cm", "fused_err_cm", "n_roi_results", "single_line",
               "single_side", "left_seen", "right_seen", "fit_ok", "fit_near_px",
               "fit_far_px", "fit_curve_px", "bottom_lock_valid", "bottom_fit_rmse_px",
               "heading_fit_rmse_px", "band_mask", "narrow_gate_detected")
SHAPE_STAGES = ("shape_preprocess_ms", "shape_proposals_ms", "shape_verification_ms",
                "shape_classify_ms", "shape_presence_ms", "shape_total_ms")


@dataclass
class Sample:
    source: str
    frame: int | None
    timestamp_s: float
    bgr: np.ndarray

    def metadata(self):
        return {"source": self.source, "frame": self.frame,
                "timestamp_s": self.timestamp_s, "width": self.bgr.shape[1],
                "height": self.bgr.shape[0],
                "pixels_sha256": hashlib.sha256(self.bgr.tobytes()).hexdigest()}


def load_samples(input_dir, max_images):
    """Share the budget across files; spread each video's allocation over its duration."""
    directory = Path(input_dir)
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    paths = sorted(p for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    if not paths:
        raise ValueError("Input directory contains no supported image/video files")
    if len(paths) > max_images:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, max_images, dtype=int)]
    quota = {path: 1 for path in paths}
    videos = [path for path in paths if path.suffix.lower() in VIDEO_EXTENSIONS]
    for index in range(max_images - len(paths)):
        if not videos:
            break
        quota[videos[index % len(videos)]] += 1

    def frames(path):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode image: {path}")
            yield Sample(str(path), None, 0., bgr)
            return
        # Only a verified regular filename reaches VideoCapture; never an index/URL.
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot decode video: {path}")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if count < 1 or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Video needs valid frame count and FPS: {path}")
            indices = np.unique(np.linspace(0, count - 1, min(quota[path], count), dtype=int))
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode {path}, frame {index}")
                yield Sample(str(path), int(index), float(index / fps), bgr)
        finally:
            capture.release()

    samples = [sample for path in paths for sample in frames(path)]
    return sorted(samples, key=lambda s: (s.source, s.timestamp_s))


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def binary_difference(cpu, other):
    if cpu is None or other is None:
        return {"available": False, "reason": "binary missing"}
    if cpu.shape != other.shape:
        return {"available": False, "reason": "shape mismatch",
                "cpu_shape": list(cpu.shape), "other_shape": list(other.shape)}
    count = int(np.count_nonzero(cpu != other))
    return {"available": True, "mismatched_pixels": count,
            "pixels": int(cpu.size), "mismatch_fraction": count / cpu.size,
            "foreground_mismatch_fraction": float(np.mean((cpu != 0) != (other != 0)))}


def field_changes(cpu, other, tolerance=1e-6):
    changes = {}
    for key in sorted(cpu.keys() | other.keys()):
        a, b = cpu.get(key), other.get(key)
        if a == b:
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not isinstance(a, bool) and not isinstance(b, bool) and abs(a - b) <= tolerance:
                continue
        changes[key] = {"cpu": a, "other": b}
        if a is not None and b is not None and not isinstance(a, bool) and not isinstance(b, bool):
            try:
                aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                if aa.shape == bb.shape and aa.size and np.all(np.isfinite(aa)) and np.all(np.isfinite(bb)):
                    changes[key]["max_abs_difference"] = float(np.max(np.abs(aa - bb)))
            except (TypeError, ValueError):
                pass
    return changes


class ReplayClock:
    def __init__(self, original):
        self.original = original
        self.seconds = 1000.

    def time(self):
        return self.seconds

    def __getattr__(self, key):
        return getattr(self.original, key)


class TimedBackend:
    """Outer wall timers include transfers; do not pretend kernels run asynchronously."""
    def __init__(self, backend):
        self.backend = backend
        self.timings = {}
        self.prefix = ""

    def __getattr__(self, key):
        value = getattr(self.backend, key)
        if key not in {"gray", "warp", "resize", "morphology"}:
            return value

        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = value(*args, **kwargs)
            stage = self.prefix + ".backend." + key + "_ms"
            self.timings[stage] = self.timings.get(stage, 0.) + (time.perf_counter() - start) * 1000.
            return result
        return timed


@contextmanager
def instrument_line_preprocess(timings):
    original = line_module.extract_lane_candidates

    def timed(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        timings["line_preprocess_ms"] = (time.perf_counter() - start) * 1000.
        return result
    line_module.extract_lane_candidates = timed
    try:
        yield
    finally:
        line_module.extract_lane_candidates = original


def make_detectors(sample, backend, config, line_mode, shape_mode, photo_mode):
    calibration = config["distance_calib"]
    line = line_module.LineDetector(sample.bgr.shape[1], sample.bgr.shape[0],
        cam_height_cm=config["mount_height_cm"], cam_pitch_deg=config["pitch_deg"],
        cam_vfov_deg=config["vfov_deg"], lane_width_cm=config["lane_width_cm"],
        z_calib=(calibration["a"], calibration["b"]))
    shape = shape_module.ShapeDetector(debug=True)
    line.vision_backend = shape.vision_backend = backend
    line.preprocess_mode, shape.preprocess_mode = line_mode, shape_mode
    line.photometric_mode = shape.photometric_mode = photo_mode
    return line, shape


def run_device(samples, backend, config, repeats=5, warmup=1,
               line_mode="legacy", shape_mode="selective", photo_mode="normalize"):
    """Reinitialize temporal state every round/source; retain warmed backend filters."""
    timer = TimedBackend(backend)
    clock = ReplayClock(shape_module.time)
    original_clock = shape_module.time
    rows, snapshots = [], {}
    shape_module.time = clock
    try:
        for round_index in range(-warmup, repeats):
            previous_source = None
            previous_timestamp = None
            for index, sample in enumerate(samples):
                if sample.source != previous_source:
                    line, shape = make_detectors(sample, timer, config, line_mode, shape_mode, photo_mode)
                    previous_timestamp = None
                dt = .1 if previous_timestamp is None else sample.timestamp_s - previous_timestamp
                clock.seconds = 1000. + sample.timestamp_s
                previous_source, previous_timestamp = sample.source, sample.timestamp_s
                timer.timings = {}
                before = backend.diagnostics()
                timer.prefix = "line"
                start = time.perf_counter()
                with instrument_line_preprocess(timer.timings):
                    dev_px, heading, conf, _, line_debug = line.process(sample.bgr, dt=dt)
                timer.timings["line_total_ms"] = (time.perf_counter() - start) * 1000.
                # Remaining CPU work includes geometry, photometry and visualization.
                timer.timings["line_remainder_ms"] = max(0., timer.timings["line_total_ms"]
                    - timer.timings.get("line.backend.warp_ms", 0.)
                    - timer.timings.get("line.backend.gray_ms", 0.)
                    - timer.timings.get("line_preprocess_ms", 0.))
                timer.prefix = "shape"
                start = time.perf_counter()
                action, shape_debug = shape.update(sample.bgr)
                timer.timings["shape_call_ms"] = (time.perf_counter() - start) * 1000.
                for stage in SHAPE_STAGES:
                    timer.timings[stage] = float(shape_debug.get(stage, 0.))
                after = backend.diagnostics()
                after.setdefault("vision_cuda_retained_cpu", "")
                if round_index < 0:
                    continue
                line_values = {key: json_value(line_debug.get(key)) for key in LINE_FIELDS}
                line_values.update(deviation_px=float(dev_px), output_heading_deg=float(heading),
                                   confidence=float(conf))
                shape_values = {key: json_value(shape_debug.get(key)) for key in SHAPE_FIELDS}
                shape_values["action"] = json_value(action)
                row = {"sample_index": index, "repeat": round_index,
                       "timings_ms": dict(timer.timings), "line": line_values,
                       "shape": shape_values, "backend": after,
                       "cuda_calls_delta": after.get("vision_cuda_calls_total", 0)
                           - before.get("vision_cuda_calls_total", 0),
                       "cuda_transfer_ms_delta": after.get("vision_cuda_transfer_ms_total", 0.)
                           - before.get("vision_cuda_transfer_ms_total", 0.)}
                rows.append(row)
                snapshots[index, round_index] = {
                    "line.binary": line_debug.get("binary"),
                    "shape.binary": shape_debug.get("binary"),
                    "shape.ink": shape_debug.get("ink")}
    finally:
        shape_module.time = original_clock
    return rows, snapshots


def timing_summary(rows):
    stages = sorted({key for row in rows for key in row["timings_ms"]})
    return {key: {"n": len(rows), "p50_ms": float(np.percentile(
                    [row["timings_ms"].get(key, 0.) for row in rows], 50)),
                  "p95_ms": float(np.percentile(
                    [row["timings_ms"].get(key, 0.) for row in rows], 95))}
            for key in stages} if rows else {}


def compare_runs(cpu_rows, cpu_arrays, other_rows, other_arrays):
    cpu_by_key = {(row["sample_index"], row["repeat"]): row for row in cpu_rows}
    comparisons = []
    for other in other_rows:
        key = other["sample_index"], other["repeat"]
        cpu = cpu_by_key[key]
        comparisons.append({"sample_index": key[0], "repeat": key[1],
            "binary": {name: binary_difference(cpu_arrays[key][name], other_arrays[key][name])
                       for name in cpu_arrays[key]},
            "line_changes": field_changes(cpu["line"], other["line"]),
            "shape_changes": field_changes(cpu["shape"], other["shape"])})
    return comparisons


def environment_report():
    device = {"count": int(cv2.cuda.getCudaEnabledDeviceCount())}
    if device["count"] > 0 and hasattr(cv2.cuda, "DeviceInfo"):
        try:
            info = cv2.cuda.DeviceInfo(0)
            device.update(name=info.name(), major=int(info.majorVersion()), minor=int(info.minorVersion()))
        except Exception as exc:
            device["info_error"] = str(exc)
    sources = list((ROOT / "new_vision" / "jetson").glob("*.py"))
    sources.append(ROOT / "new_vision" / "config" / "cameras.json")
    return {"python": sys.version, "platform": platform.platform(),
            "opencv": cv2.__version__, "opencv_file": cv2.__file__, "numpy": np.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cuda_device": json_value(device), "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources if path.is_file()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--cuda-fallback", action="store_true")
    parser.add_argument("--line-preprocess", choices=("legacy", "contrast", "canny"), default="legacy")
    parser.add_argument("--shape-preprocess", choices=("selective", "canny"), default="selective")
    parser.add_argument("--photometric-mode", choices=("legacy", "normalize"), default="normalize")
    args = parser.parse_args(argv)
    if args.max_images < 1 or args.repeats < 1 or args.warmup < 1:
        parser.error("max-images, repeats and warmup must be >= 1")
    if args.require_cuda and "cuda" not in args.devices:
        parser.error("--require-cuda requires cuda in --devices")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.input_dir, args.max_images)
    config = camera_config.load()
    if config.get("profile") == "default":
        raise RuntimeError("Camera config silently fell back to defaults; restore cameras.json before comparison")
    report = {"environment": environment_report(), "arguments": vars(args), "camera_config": config,
              "sampling": "Shared file budget; uniform video frames. Sorted per-file replay; reset each source/round.",
              "timing_note": "Warmup excluded; synchronous wall time includes transfers and rendering. Backend stages overlap detector stages; do not add them together. line_remainder includes geometry, photometry and rendering.",
              "samples": [sample.metadata() for sample in samples], "devices": {}, "comparisons": []}
    runs = {}
    # CPU reference always executes first, even when devices contains only cuda.
    for device in dict.fromkeys(["cpu"] + args.devices):
        try:
            backend = cuda_vision.create_vision_backend(device, allow_fallback=args.cuda_fallback)
            rows, arrays = run_device(samples, backend, config, args.repeats, args.warmup,
                                     args.line_preprocess, args.shape_preprocess, args.photometric_mode)
            actual = backend.diagnostics().get("vision_device_actual")
            calls = sum(row["cuda_calls_delta"] for row in rows)
            status = "ok" if device == "cpu" or (actual == "cuda" and calls > 0) else "fallback"
            report["devices"][device] = {"status": status, "backend": backend.diagnostics(),
                "measured_cuda_calls": calls,
                "measured_cuda_transfer_ms": sum(row["cuda_transfer_ms_delta"] for row in rows),
                "timings": timing_summary(rows), "rows": rows}
            runs[device] = rows, arrays
        except Exception as exc:
            report["devices"][device] = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
            print(f"{device}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if "cuda" in runs and "cpu" in runs:
        report["comparisons"] = compare_runs(*runs["cpu"], *runs["cuda"])
    report["comparison_summary"] = {
        "calls_compared": len(report["comparisons"]),
        "line_changed_calls": sum(bool(row["line_changes"]) for row in report["comparisons"]),
        "shape_changed_calls": sum(bool(row["shape_changes"]) for row in report["comparisons"]),
        "classification_changed_calls": sum(any(key in row["shape_changes"] for key in ("shape", "action"))
                                            for row in report["comparisons"]),
        "presence_changed_calls": sum("presence" in row["shape_changes"] for row in report["comparisons"]),
        "geometry_changed_calls": sum(bool(row["line_changes"]) or any(key in row["shape_changes"]
            for key in ("quad", "presence_box", "presence_box_work", "presence_cy_frac", "box_top_work"))
            for row in report["comparisons"]),
        "binary_changed_calls": sum(any(not value["available"] or value["mismatched_pixels"] > 0
                                        for value in row["binary"].values()) for row in report["comparisons"])}
    report["binary_summary"] = {}
    for name in ("line.binary", "shape.binary", "shape.ink"):
        values = [row["binary"][name] for row in report["comparisons"] if row["binary"][name]["available"]]
        pixels = sum(value["pixels"] for value in values)
        mismatches = sum(value["mismatched_pixels"] for value in values)
        report["binary_summary"][name] = {"calls_compared": len(values), "pixels": pixels,
            "mismatched_pixels": mismatches, "mismatch_fraction": mismatches / pixels if pixels else None,
            "max_call_mismatch_fraction": max((value["mismatch_fraction"] for value in values), default=None)}
    (output / "report.json").write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "opencv_build.txt").write_text(cv2.getBuildInformation(), encoding="utf-8")
    lines = ["# Offline vision device benchmark", "", report["timing_note"], "",
             f"Samples: {len(samples)}; repeats: {args.repeats}; warmup rounds: {args.warmup}.", "",
             "| Device | Status | Stage | p50 ms | p95 ms |", "|---|---|---|---:|---:|"]
    for device, result in report["devices"].items():
        for stage, stats in result.get("timings", {}).items():
            lines.append(f"| {device} | {result['status']} | {stage} | {stats['p50_ms']:.3f} | {stats['p95_ms']:.3f} |")
        if result["status"] == "error":
            lines.append(f"| {device} | error: {result['error']} | — | — | — |")
    lines += ["", "Comparison summary: `" + json.dumps(report["comparison_summary"]) + "`.", "",
              "See report.json for every input pixel hash, source hash, backend retained CPU reasons and per-call changes. CPU-only/fallback measurements provide no CUDA speed evidence."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "devices": {k: v["status"] for k, v in report["devices"].items()},
                      **report["comparison_summary"]}, ensure_ascii=False))
    if report["devices"].get("cpu", {}).get("status") != "ok":
        return 2
    if args.require_cuda and report["devices"].get("cuda", {}).get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
