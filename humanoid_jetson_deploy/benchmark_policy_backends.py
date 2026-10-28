"""Offline recorded-observation CPU/TensorRT comparison; never opens serial.

Replays exact POLICY_OBS_CSV columns. It does not reconstruct observations from
motor-position logs or feed predicted actions back into later recorded rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

import config
from policy_runner import observation_columns
from tensorrt_backend import TensorRTSession, sha256


def load_observations(path, obs_dim):
    columns = observation_columns(obs_dim)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in columns):
            raise ValueError("Input must contain the exact POLICY_OBS_CSV observation schema")
        # Reject mixed walking/one-foot schemas instead of dropping extra fields.
        other = {"lift_command"} if obs_dim == 49 else {"cmd_vx", "cmd_wz", "step_distance", "crossing"}
        if other.intersection(reader.fieldnames):
            raise ValueError("Observation CSV policy schema does not match --obs-dim")
        rows = []
        for index, row in enumerate(reader, 2):
            try:
                values = [float(row[name]) for name in columns]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid observation at CSV line {index}") from exc
            rows.append(values)
    data = np.asarray(rows, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != obs_dim or not np.isfinite(data).all():
        raise ValueError("Observation CSV must contain nonempty finite float32 rows")
    return data


def percentiles(values):
    array = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array))}


def synthetic_smoke_observations(obs_dim=49, count=256, seed=20261010):
    """Deterministic numerical deployment probes, never labelled real data."""
    if obs_dim not in (46,49) or count < 2:
        raise ValueError("Synthetic smoke requires obs_dim 46/49 and at least two rows")
    rng = np.random.default_rng(seed)
    values = np.zeros((count, obs_dim), dtype=np.float32)
    values[:, :3] = rng.uniform(-.3, .3, (count,3))
    values[:, 2] -= .981
    values[:, 3:6] = rng.uniform(-.5, .5, (count,3))
    gravity = np.column_stack((rng.uniform(-.3,.3,(count,2)), -np.ones(count)))
    values[:, 6:9] = gravity / np.linalg.norm(gravity,axis=1)[:,None]
    if obs_dim == 49:
        values[:, 9] = rng.choice([0., .1, .2], count)
        values[:, 10] = rng.choice([-.25, 0., .25], count)
        values[:, 11] = np.where(values[:, 9] > 0, rng.choice([.02,.05,.12], count), 0)
        start = 13
    else:
        values[:,9] = rng.choice([0.,1.],count)
        start = 10
    values[:, start:start+12] = rng.uniform(-.25,.25,(count,12))
    values[:, start+12:start+24] = rng.uniform(-2.,2.,(count,12))
    values[:, start+24:] = rng.uniform(-1.,1.,(count,12))
    values[0] = 0
    values[1] = 0
    values[1,2], values[1,8] = -.981,-1
    return values


def benchmark(model, engine, observations, *, obs_dim=49, warmup=100, repeats=3,
              atol=1e-5, rtol=1e-5, support_foot="right"):
    import onnxruntime as ort
    cpu = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    if len(cpu.get_inputs()) != 1 or len(cpu.get_outputs()) != 1:
        raise ValueError("CPU policy must have one input/output")
    inp, out = cpu.get_inputs()[0], cpu.get_outputs()[0]
    if (inp.type != "tensor(float)" or len(inp.shape) != 2 or inp.shape[1] != obs_dim
            or (isinstance(inp.shape[0], int) and inp.shape[0] != 1)):
        raise ValueError("CPU model observation contract mismatch")
    gpu = TensorRTSession(model, engine, obs_dim, profile=True)
    samples, failures = [], 0
    worst_abs, worst_target = 0.0, 0.0
    try:
        gpu.profile = False
        for i in range(warmup):
            obs = observations[i % len(observations):i % len(observations) + 1]
            cpu.run([out.name], {inp.name: obs})
            gpu.run([gpu.output_name], {gpu.input_name: obs})
        for repeat in range(repeats):
            for index, row in enumerate(observations):
                obs = row.reshape(1, obs_dim)
                start = time.perf_counter_ns()
                expected = np.asarray(cpu.run([out.name], {inp.name: obs})[0])
                cpu_ms = (time.perf_counter_ns() - start) / 1e6
                start = time.perf_counter_ns()
                actual = gpu.run([gpu.output_name], {gpu.input_name: obs})[0]
                gpu_ms = (time.perf_counter_ns() - start) / 1e6
                normal_timing = gpu.last_timing.copy()
                # Measure real control-boundary latency without CUDA events.
                # A second replay of the same observation measures GPU stages.
                gpu.profile = True
                gpu.run([gpu.output_name], {gpu.input_name: obs})
                stage_timing = gpu.last_timing.copy()
                gpu.profile = False
                if expected.shape != (1, 12) or not np.isfinite(expected).all():
                    raise ValueError("Invalid CPU model output")
                difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
                matched = bool(np.allclose(expected, actual, atol=atol, rtol=rtol))
                failures += int(not matched)
                error = float(difference.max())
                # Only policy affine conversion; downstream clipping/transport
                # stays in main. Mirror matches OneFootPolicy left support.
                cpu_action, gpu_action = expected[0], actual[0]
                if obs_dim == 46 and support_foot == "left":
                    cpu_action = -np.concatenate((cpu_action[6:], cpu_action[:6]))
                    gpu_action = -np.concatenate((gpu_action[6:], gpu_action[:6]))
                target_error = float(np.abs(
                    (config.Q_DEFAULT + config.ACTION_SCALE * cpu_action).astype(np.float64)
                    - (config.Q_DEFAULT + config.ACTION_SCALE * gpu_action).astype(np.float64)).max())
                worst_abs, worst_target = max(worst_abs, error), max(worst_target, target_error)
                samples.append({"repeat": repeat, "row": index, "cpu_ms": cpu_ms,
                                "gpu_ms": gpu_ms, **normal_timing,
                                **{key: stage_timing[key] for key in ("h2d_ms", "compute_ms", "d2h_ms")},
                                "profiled_total_ms": stage_timing["total_ms"],
                                "action_max_abs": error, "target_max_abs_rad": target_error,
                                "outputs_match": matched})
        timings = {key: percentiles([row[key] for row in samples])
                   for key in ("cpu_ms", "gpu_ms", "total_ms", "host_sync_ms", "h2d_ms", "compute_ms", "d2h_ms", "profiled_total_ms")}
        return {"schema": 1, "model_sha256": sha256(model), "engine": gpu.metadata,
                "obs_dim": obs_dim, "observation_rows": len(observations),
                "warmup": warmup, "repeats": repeats, "samples": len(samples),
                "atol": atol, "rtol": rtol, "mismatch_samples": failures,
                "max_action_abs_error": worst_abs, "max_target_abs_error_rad": worst_target,
                "support_foot": support_foot, "timing_ms": timings,
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "onnxruntime": ort.__version__, "platform": platform.platform()},
                "timing_notes": "cpu_ms/gpu_ms/total_ms/host_sync_ms are unprofiled warm host timings. H2D/compute/D2H CUDA events come from a separate replay of the same observation (profiled_total_ms); host_sync overlaps GPU work and must not be added to stage intervals. gpu_ms includes validation and returning a synchronized host copy. Recorded observations replay independently; no closed-loop or motor validation."}, samples
    finally:
        gpu.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--obs-csv", help="Real recorded POLICY_OBS_CSV")
    source.add_argument("--synthetic-smoke", action="store_true",
                        help="Explicit deterministic numerical/performance smoke; not real observations")
    parser.add_argument("--synthetic-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--out", required=True, help="JSON report; per-row CSV saved next to it")
    parser.add_argument("--obs-dim", type=int, choices=(46, 49), default=49)
    parser.add_argument("--support-foot", choices=("right", "left"), default="right")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("warmup>=0, repeats>=1 and finite nonnegative tolerances required")
    if args.synthetic_smoke:
        if args.synthetic_rows < 2 or args.seed < 0:
            parser.error("synthetic rows>=2 and seed>=0 required")
        data = synthetic_smoke_observations(args.obs_dim, args.synthetic_rows, args.seed)
    else:
        data = load_observations(args.obs_csv, args.obs_dim)
    report, samples = benchmark(args.model, args.engine, data, obs_dim=args.obs_dim,
                                support_foot=args.support_foot, warmup=args.warmup,
                                repeats=args.repeats, atol=args.atol, rtol=args.rtol)
    if args.synthetic_smoke:
        report["observation_source"] = {"kind": "synthetic_smoke", "seed": args.seed,
            "rows": args.synthetic_rows,
            "limits": "Numerical execution/performance smoke only; does not establish real-robot error, closed-loop behavior, training equivalence or coverage of the training distribution.",
            "generator": "synthetic_smoke_observations_v1",
            "ranges": "scaled accel~[-.3,.3], z=-.981+noise; gyro[-.5,.5]; unit gravity tilted<=.3; vx 0/.1/.2,wz0/+/-.25,step0/.02/.05/.12,crossing0 OR lift0/1; qrel[-.25,.25],qd[-2,2],last_action[-1,1]; first two rows zero/standing probes"}
    else:
        report["observation_source"] = {"kind": "recorded_csv", "sha256": sha256(args.obs_csv)}
    report["observation_float32_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
    source_dir = Path(__file__).resolve().parent
    report["source_sha256"] = {name: sha256(source_dir / name) for name in (
        "benchmark_policy_backends.py", "tensorrt_backend.py", "policy_runner.py",
        "one_foot_policy.py", "config.py")}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    if args.synthetic_smoke:
        with path.with_suffix(".observations.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(observation_columns(args.obs_dim))
            writer.writerows(data)
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps(report, indent=2))
    return int(report["mismatch_samples"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
