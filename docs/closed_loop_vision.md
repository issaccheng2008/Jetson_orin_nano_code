# Closed-loop walking with new_vision

This guide is for the Jetson Orin Nano. Run the commands on the Jetson (locally
or over SSH), from the repository root, with the same Python environment in
each terminal. The Windows computer can provide SSH terminals; it does not
need to forward the camera or the UDP ports.

## What runs, and why this is closed loop

1. The robot-mounted USB camera observes its current position relative to the track.
2. `new_vision/jetson/run_policy_vision.py` uses the new CPU `LineDetector`,
   the dual straight/curve PID and one-step preview from `run_robot.py`, and
   converts steering error to a yaw-rate command in rad/s.
3. `connector.py` holds and forwards the latest command at 50 Hz.
4. `humanoid_jetson_deploy/main.py --command-source vision` puts the live
   forward/yaw commands into the 49-input walking policy along with STM32
   IMU and encoder feedback. The policy sends joint targets back to STM32.
5. Robot motion changes the next camera image, closing the track-following loop.

The communication is the earlier working UDP JSON path (see historical commit
`17e4667f29962215e01bfdb2dd205d24a4157979`):

| Link | Address | Payload |
|---|---|---|
| Vision → connector | `127.0.0.1:5006` | UTF-8 JSON: `{"vx":0.4,"vy":0.0,"wz":-0.1,"qr":-1}` |
| Connector → policy | `127.0.0.1:5005` | Same fields; republished at 50 Hz |
| Policy ↔ STM32 | e.g. `/dev/ttyACM0` | Existing binary state/joint-command protocol |

`vx` is m/s; `wz` is rad/s; `vy` is always zero. `qr=-1` preserves the old
schema; it does not represent a geometric-shape detection. Both UDP stages
retain the existing `vx=0..1` and `wz=-0.5..0.5` clamps.

This entry point is for line-following walking. It does not execute shape-card
actions, stop for cards, or trigger bar crossing. Use a track section without
those tasks for the integration test. The original CPU/GPU `run_robot.py`
entry points still produce V2 serial messages and are not the policy bridge.
Only the policy process should open the STM32 serial device.

## 1. Check out the draft branch and prepare the environment

For an existing clone:

```bash
cd ~/Jetson_orin_nano_code
git fetch origin
git switch --track origin/integrate-new-vision-policy
```

If the local branch already exists, use `git switch integrate-new-vision-policy`.
Keep local calibration/model changes when switching; resolve any Git warning
instead of discarding those files. If needed, clone the repository first:

```bash
git clone https://github.com/issaccheng2008/Jetson_orin_nano_code.git
```

Use the existing environment that successfully runs your ONNX policy and camera.
For a fresh environment, one possible setup is:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r humanoid_jetson_deploy/requirements.txt
python -c 'import cv2, numpy, onnxruntime, serial; print("imports OK")'
```

If `cv2` is absent, install OpenCV for your Jetson environment (for Ubuntu's
system Python: `sudo apt install python3-opencv`). The new entry point uses
CPU OpenCV; CUDA OpenCV is not required. Do not replace a working Jetson CUDA
OpenCV installation just for this test. `--headless` works without GUI windows.
An ONNX Runtime wheel must match the Jetson's Python version and aarch64 platform;
reuse your proven runtime if pip cannot supply a compatible wheel.

Export a **49-input, 12-output walking policy trained with turning and stop
commands**, including its observation normalizer if training used one. Put it
at `humanoid_jetson_deploy/models/current_walking.onnx`. A straight-only policy
does not acquire turning skills by receiving nonzero yaw commands. Old 47/48-input
models and the 46-input one-foot model are incompatible with this interface.
No bundled ONNX model has been replaced or certified by this change.

Check your existing motor/IMU calibration in `humanoid_jetson_deploy/config.py`.
The normal walking step distance remains `DEFAULT_STEP_DISTANCE` (currently
0.08 m). The existing observation builder uses step distance 0 when all velocity
commands are zero. Crossing command remains 0.

## 2. Verify the camera and steering without enabling motors

Camera defaults come from `new_vision/config/cameras.json`: index 0, 1280×720,
height 40 cm, pitch down 45°, vertical FOV 56.2°. Set the actual mounting values
using that file or `--camera-height-cm`, `--camera-pitch-deg`, and
`--camera-vfov-deg`. `CAMERA_PROFILE` selects a configured camera profile.
The CPU detector currently has its own algorithm defaults; editing
`new_vision/line_follow_params.json` does not automatically tune this detector.

Start the connector in terminal A:

```bash
python -u connector.py --vision-port 5006 --policy-port 5005
```

Start vision in terminal B:

```bash
python -u new_vision/jetson/run_policy_vision.py \
  --camera 0 --vx 0.4 --max-wz 0.2 --headless
```

With a local desktop, omit `--headless` to see the camera and debug windows.
Use Q in a window, or Ctrl+C in the terminal, to stop vision. Do not run another
camera program at the same time.

Move the camera/track relative to each other and inspect both terminal logs:

| Track position/detection | Expected command |
|---|---|
| Centred, aligned straight track | `wz` near 0 |
| Track to the camera's right | Positive `wz` (turns right) |
| Track to the camera's left | Negative `wz` (turns left) |
| Lost/invalid line detection | target `vx=0`, `wz=0` after a 0.2 s hold; published value ramps down |
| Vision stopped/frozen | Connector prints `fresh=False`; target zero after 0.25 s, published value ramps down |

Default `--yaw-sign 1` maps positive image-right steering to positive policy yaw.
An earlier version of this document claimed the policy frame made that `-1`; on
the robot `-1` turned the wrong way, so the polarity in the table above is the
one that holds. If a different camera or policy flips it again, restart vision
with `--yaw-sign -1`. Verify physical turning direction in the supported test
below.

## 3. Verify that live commands reach the walking policy (motors disabled)

Leave A and B running. Connect STM32 and identify its device, for example with
`ls /dev/serial/by-id/`. In terminal C, from the repository root:

```bash
python -u humanoid_jetson_deploy/main.py \
  --policy walking \
  --model humanoid_jetson_deploy/models/current_walking.onnx \
  --port /dev/ttyACM0 \
  --command-source vision --udp-command-port 5005 \
  --no-plot --max-seconds 30
```

No `--enable-motors` means outgoing command enable flags remain off. STM32
must still stream valid IMU/encoder data; this is not a serial-free dry run.
Expect `DRY RUN`, a command-source line naming the UDP receiver, and
`policy_target_velocity` following vision/connector changes, including after
five seconds. These are the actual values passed to the policy; `--vx` and
`--wz` on the policy command line do not override vision mode.

Check `crc_errors=0`, finite observations/actions, and inference/loop timing.
Move the camera right/left and confirm the sign changes in terminal C.
While motors are disabled, test both watchdogs:

1. Stop terminal B: the connector prints `fresh=False`, the target goes to zero
   after 0.25 s, and the published value slews down over a further 0.40 s
   (`--max-vx-accel`). `qr` drops to `-1` immediately; it is not smoothed.
2. Restart B: live commands resume automatically when detection is valid.
3. Stop terminal A while B continues: the policy's independent UDP timeout
   produces zero velocity within about 0.25 s plus scheduling delay. That hop is
   a hard step - `command_source.py` has no smoothing and is deliberately not
   touched; it only fires once the connector process itself has died.
4. Restart A: live commands resume automatically.

These are **zero-velocity policy commands**, not motor-disable commands and not
a guarantee of a physically stationary stance. Ctrl+C in the policy process
disables motors; loss of valid STM32 state also invokes existing fault handling.
Keep the robot supported when checking motor-disable behavior.

## 4. Run the physical closed-loop test

After the disabled-motor checks, secure the robot with your normal support/fall
protection, start with a short clear track, and keep the physical stop available.
Keep A and B running with `--max-wz 0.2`. Restart C as:

```bash
python -u humanoid_jetson_deploy/main.py \
  --policy walking \
  --model humanoid_jetson_deploy/models/current_walking.onnx \
  --port /dev/ttyACM0 --command-source vision \
  --enable-motors --no-plot --max-seconds 10
```

It starts walking as soon as valid STM32 state and live vision commands are
available. If vision/connector is absent at startup, the policy receives zeros.
Check that the robot physically turns toward the track and that the observed
track error decreases over successive images. Changing log values alone proves
communication, not successful physical tracking.

After the short test, omit `--max-seconds 10` for continuous operation. Vision
mode has no five-second walking cutoff. Stop using Ctrl+C in terminal C, then
stop vision and the connector. CSV motor/IMU logs are written under
`logs/motor_positions/` relative to the directory where you launched the policy.
Command values appear in terminal logs; they are not additional CSV columns.

## 5. Tune the angular command

The mapping is:

```text
steer_cm = PID(fused_err_cm) + preview_gain * step_len_cm * sin(heading_error)
wz = yaw_sign * clip(clip(steer_cm, -50, 50) / steer_full_scale_cm, -1, 1) * max_wz
```

The D term is filtered: `median(err, 3)` then a first-order IIR
(`--deriv-pole`, default 0.78) over a fixed 0.05 s nominal period. The detector
already EMA-smooths the error, so differentiating it mostly amplified residual
noise - on real line video the raw term averaged 0.84 cm but swung with a
5.49 cm standard deviation and caused 5.93 of the 7.56 cm frame-to-frame steer
jitter. Filtering took that jitter to 3.31 cm and the worst single step from
35.05 to 12.61 cm, while keeping the damping. `--deriv-pole 0` disables the IIR
and leaves the median.

This is a commanded yaw rate derived from image error, not a measured angular
velocity. With the defaults, 10 cm of final PID steering gives -0.1 rad/s.

- `--max-wz`: begin at 0.2; increase toward 0.5 only after verifying tracking.
- `--steer-full-scale-cm`: default 10; reducing it strengthens correction.
- `--vx`: sets forward speed in the vision process; choose a speed your policy
  can track reliably (repository default 0.4 m/s).
- `--step-len-cm`: default 8; keep consistent with your intended walking stride.
- `--preview-gain`: default 1; reduce if anticipatory steering causes oscillation.
- `--lost-hold-s`: default 0.2; how long to hold the last command before
  stopping on line loss.
- `--deriv-pole`: default 0.78; raise for a smoother D term, lower for faster.
- Existing `JETSON_PID_STRAIGHT_KP/KI/KD`, `JETSON_PID_CURVE_KP/KI/KD`,
  `JETSON_PID_I_CLAMP`, `JETSON_PID_D_FILTER`, `STEP_LEN_CM`, and `PREVIEW_GAIN`
  environment overrides work.

The 0.5 rad/s cap is retained from the old communication implementation. Raising
it would require coordinated edits to the mapper, connector, and policy receiver
and validation against the trained policy's turning range.

## Troubleshooting and fixed-mode fallback

- **Vision shows turns, policy does not:** confirm `--command-source vision`,
  matching ports 5006/5005, and only one receiver per port. Fixed is still the
  default for backwards-compatible standalone policy tests.
- **`fresh=False` while vision runs:** check frame processing time and dropped
  camera reads. The watchdog covers missing publications, not the age of a
  frame buffered inside the camera driver. Tune capture latency if needed.
  Increase `connector.py --vision-timeout` only after measuring the frame period;
  a larger timeout also means a longer stale-command hold.
- **Alternating move/stop:** inspect lost-frame/confidence logs, camera geometry,
  illumination and view of the track. `lost_frames` has no hysteresis in the
  detector, so a single missed frame used to be enough to stop the robot; the
  bridge now holds the last command for `lost_hold_s` (0.2 s) before stopping,
  and the connector ramps the result. It still never blindly searches with the
  humanoid. If the stutter survives the hold window, the detection itself is the
  problem, not the command path.
- **Robot turns away:** verify the image and policy yaw sign; reverse `--yaw-sign`
  only after confirming the mismatch.
- **Serial busy:** stop the V2 vision controller, serial monitor, or any other
  program opening the STM32 device.
- **Policy shape error:** re-export the current 49-input walking policy.
- **No display over SSH:** use vision `--headless` and policy `--no-plot`.

For fixed walking without a camera, select it explicitly:

```bash
python humanoid_jetson_deploy/main.py \
  --model humanoid_jetson_deploy/models/current_walking.onnx \
  --port /dev/ttyACM0 --command-source fixed --vx 0.4 --wz 0 \
  --walk-seconds 5 --no-plot
```

`--walk-seconds 5` preserves the current five-second test stop; 0 makes fixed
walking continuous. `--max-seconds` instead exits the process and disables motors.
One-foot mode does not open a vision receiver.

## Automated verification

```bash
python -m unittest discover -s tests -v
cd humanoid_jetson_deploy
python -m unittest discover -s tests -v
```

The integration test uses real local UDP sockets and a connector subprocess to
check new-controller output, historical JSON encoding, policy observation slots,
and both watchdogs. Runtime tests check that vision commands survive beyond five
seconds. Camera, ONNX checkpoint performance, and physical walking require the
Jetson procedure above.

At the base revision `3ca3d9968370a5ab187f41dd938d31ed90ea02d2`, two existing
one-foot tests expect the leg to lower again, but `OneFootCommand.get()` keeps
the lift command active. Those same two tests fail before and after this change;
the one-foot schedule is not modified here.
