# Optional TensorRT policy backend

CPU ONNX Runtime remains the default. TensorRT is selected explicitly with
`--inference-backend tensorrt --trt-engine PATH`. In vision mode, an optional
one-foot model needs its own `--one-foot-trt-engine PATH`; standalone one-foot
uses the primary `--trt-engine`. Fixed-frame playback does not load TensorRT.

The backend changes model execution only. The existing observation order,
raw last-action feedback, one-foot left-side mirroring, action scale/default
pose, target clipping, diagnostics and motor transport remain in their existing
modules. The host receives a completed float32 action after stream synchronization.
Missing CUDA/TensorRT, an incompatible engine, or invalid output fails explicitly.

## Offline build and recorded-observation check

These two tools do not import serial/control code or open a motor connection.
Use the Python environment with the board's installed TensorRT 10 and CUDA runtime.
Build on the destination GPU. Engines are bound to the ONNX SHA-256, engine SHA-256,
TensorRT version, CUDA runtime version and GPU UUID in a mandatory JSON sidecar.

```sh
cd humanoid_jetson_deploy
python build_policy_engine.py --model policy_49_v22.onnx --engine /tmp/policy49.engine --obs-dim 49
python benchmark_policy_backends.py --model policy_49_v22.onnx --engine /tmp/policy49.engine --obs-csv /path/to/recorded_policy_obs.csv --out /tmp/policy49_benchmark.json
```

For one-foot recordings use `--obs-dim 46`; add `--support-foot left` to compare
the mirrored physical target. No new motor run is needed for this benchmark.
Default input must be a real `POLICY_OBS_CSV` recording with the named observation columns.
Motor-position CSV files do not contain the model observation and are rejected.
The six-significant-digit observation dump is replayed as recorded; it does not
recreate pre-rounding live observations. Each recorded row is independent, with
its recorded last-action block; predictions are not fed into later observations.

If a recording is unavailable, explicitly replace `--obs-csv PATH` with
`--synthetic-smoke`. This generates 256 deterministic float32 probes (seed
20261010, both configurable), saves the exact probes as `.observations.csv`, and
labels the JSON `observation_source.kind=synthetic_smoke`. This checks numerical
execution and performance only. It does not establish real-robot error, gait,
training equivalence or coverage of the training distribution. Real recordings
and synthetic input are mutually exclusive; the tool never silently substitutes
generated data for a missing recording.

Default FP32 construction uses strong typing and disables TF32. No FP16, INT8,
automatic engine build or CPU fallback occurs. The builder's workspace limit
defaults to 256 MiB and does not constrain total engine/runtime memory.

The benchmark defaults to 100 warmups and three full passes. JSON and per-row CSV
include max action error and affine target error in radians; exit status is 1 if
any row fails `allclose(atol=1e-5, rtol=1e-5)`. It compares raw model outputs and
affine targets before the existing downstream safety clipping. It does not
demonstrate closed-loop gait or hardware behavior.

`cpu_ms` and `gpu_ms` report warm, unprofiled wall times (p50/p95/max). GPU wall time
includes input validation, copying to pinned memory, kernel submission, host
synchronization and copying the action to an independent NumPy array. GPU stage
times use CUDA events in a separate replay of the same row; the corresponding
`profiled_total_ms` includes event overhead. `host_sync_ms` overlaps GPU work and
must not be added to H2D/compute/D2H. Small policies can be faster on CPU; choose
the backend from the destination-board report.

API references: [TensorRT 10 Python execution](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html),
[strong typing and TF32](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html),
[CUDA runtime](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/index.html).
