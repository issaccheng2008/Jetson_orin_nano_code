# Findings

- Current workspace has substantial pre-existing modifications and untracked files. Treat them as user work and leave untouched unless explicitly part of this task.
- Existing camera utilities are in `new_vision/jetson/camera_config.py` and `camera_controls.py`; tests include `tests/test_camera_controls.py` and `tests/test_latest_camera.py`.
- No current production-call-chain changes are authorized; only standalone calibration tools and small shared utility/test modules are in scope.
- `camera_config.load()` provides active camera index and dimensions, but no exposure/WB calibration parameters. Therefore the capture tool should require CLI values or prompt interactively; it must not substitute camera-control defaults.
- `camera_controls.py` configures startup controls but does not provide a complete snapshot/restore API, and it uses fixed V4L2 menu choices in its own path. The new tool will query the active device menu and derive mode values from labels before writing controls.
- Existing tests use `unittest`; offline hardware simulation can follow that convention. Camera capture can use a background reader thread so terminal prompts do not block frame acquisition.
- Implemented standalone capture control with V4L2 menu-derived auto/manual modes, prevalidated writes, exact readback, unknown-safe snapshots, and best-effort restoration. A capture group returns before writing any image if settings verification fails.
- Implemented a NumPy/stdlib image/LUT core with PNG I/O, projective board sampling, same-cell pairing, bounded monotonic fit, conflict reporting, and `apply_mapping`. The fitter GUI imports OpenCV only when interactive selection or video/JPEG support is needed.
- Capture stability defaults: discard at least 3 seconds, then require a 1.5-second rolling stable ROI window (central x=25–75%, y=35–75%), timeout at 25 seconds, and save ten fresh PNGs 0.5 seconds apart. These values are configurable in one dataclass/CLI.

