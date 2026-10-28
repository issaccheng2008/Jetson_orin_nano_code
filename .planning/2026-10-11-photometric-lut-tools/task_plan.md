# Photometric calibration tools

## Goal
Implement independent camera-pair capture and fixed-to-auto LUT fitting tools in the current workspace. Do not modify the production vision call chain.

## Phases
- [x] Inspect camera configuration/control APIs and existing tests; document constraints.
- [x] Implement shared mapping, sample extraction/fitting, metadata and apply helper.
- [x] Implement safe interactive camera capture with readback, stability checks, ten PNGs per mode, and restoration.
- [x] Implement interactive/JSON corner selection, perspective sampling, fit/export and comparison outputs.
- [x] Add offline tests for camera failure, exact capture count, no overwrite, fitting quality and apply contract.
- [x] Run relevant offline tests and report hardware capture status and production integration point.

## Constraints
- Preserve all pre-existing user modifications; do not stage or revert unrelated files.
- No camera, robot, serial, motor or production pipeline execution.
- The LUT maps fixed-exposure grayscale-board responses to measurements from auto-reference photos; it is not a color calibration.

## Errors encountered
| Error | Attempt | Resolution |
|---|---:|---|
| Host Python has no `cv2` module | 1 | Keep capture/GUI/video entry points OpenCV-dependent at runtime; implement offline LUT math, perspective sampling, and PNG I/O with NumPy/stdlib so required tests can still run here. |
| `CameraReader.read_after(-1)` treated the initial empty slot as ready | 1 | Wait until a real frame is present as well as newer than the requested sequence; added a background-reader regression test. |
| Follow-up fit report refactor left an obsolete holdout-count variable and a test fixture lacked new patch-variance fields | 1 | Use the held-out list length directly and update the synthetic fixture to the expanded sample schema. |
| A cleanup patch did not match the current `finally` block context | 1 | Re-read the exact block; its current `finally` already restores controls/resolution and releases the reader on normal and handled-exception paths, so no cleanup edit was needed. |

