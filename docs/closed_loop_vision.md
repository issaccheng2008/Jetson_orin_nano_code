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
3. `humanoid_jetson_deploy/main.py --command-source vision` receives the UDP
   JSON and puts the live
   forward/yaw commands into the 49-input walking policy along with STM32
   IMU and encoder feedback. The policy sends joint targets back to STM32.
4. Robot motion changes the next camera image, closing the track-following loop.

The communication is the earlier working UDP JSON path (see historical commit
`17e4667f29962215e01bfdb2dd205d24a4157979`):

| Link | Address | Payload |
|---|---|---|
| Vision → policy | `127.0.0.1:5005` | UTF-8 JSON: `{"vx":0.4,"vy":0.0,"wz":-0.1,"qr":-1}` |
| Policy ↔ STM32 | e.g. `/dev/ttyACM0` | Existing binary state/joint-command protocol |

`vx` is m/s; `wz` is rad/s; `vy` is always zero. `qr=-1` preserves the old
schema; it does not represent a geometric-shape detection. The receiver retains
the existing `vx=0..1` and `wz=-0.5..0.5` clamps. It holds the latest command
between camera frames and returns zero when no valid packet arrives for 0.25 s.

`connector.py` is optional. It supports the older three-process layout where
vision sends to 5006 and the connector republishes to 5005, but it is not needed
for normal closed-loop operation.

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
git switch --track origin/fix-direct-vision-policy-udp
```

If the local branch already exists, use `git switch fix-direct-vision-policy-udp`.
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

Start vision in terminal A. Its default destination is the policy receiver on
port 5005:

```bash
python -u new_vision/jetson/run_policy_vision.py \
  --camera 0 --vx 0.4 --max-wz 0.2 --headless
```

Do not leave `connector.py` running in this direct mode: an idle old connector
publishes stale zero commands to the same policy port. Check with
`pgrep -af connector.py`; stop the exact process with Ctrl+C in its terminal.

With a local desktop, omit `--headless` to see the camera and debug windows.
Use Q in a window, or Ctrl+C in the terminal, to stop vision. Do not run another
camera program at the same time.

For compatibility with the older relay layout, explicitly start
`python -u connector.py --vision-port 5006 --policy-port 5005` and add
`--command-port 5006` to the vision command. Do not use port 5006 without the
connector: UDP does not report that no receiver exists.

Move the camera/track relative to each other and inspect its log:

| Track position/detection | Expected command |
|---|---|
| Centred, aligned straight track | `wz` near 0 |
| Track to the camera's right | Negative `wz` (right turn in the policy frame) |
| Track to the camera's left | Positive `wz` (left turn in the policy frame) |
| Lost/invalid line detection | `vx=0`, `wz=0` |
| Vision stopped/frozen | Policy prints `udp_fresh=False` and uses zero after 0.25 s |

Default `--yaw-sign -1` converts positive image-right steering into negative
policy yaw. Confirm this against the real camera orientation and your policy's
yaw convention. If the sign is reversed, stop the run and restart vision with
`--yaw-sign 1`. Verify physical turning direction in the supported test below.

## 3. Verify that live commands reach the walking policy (motors disabled)

Leave terminal A running. Connect STM32 and identify its device, for example
with `ls /dev/serial/by-id/`. In terminal B, from the repository root:

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
`policy_target_velocity` following vision changes, including after five seconds.
Healthy logs include `udp_fresh=True`, an increasing `udp_valid` count, and a
small `udp_age`. `udp_fresh=False udp_age=never udp_valid=0` means no valid
packet has reached port 5005. These are the actual values passed to the policy;
`--vx` and `--wz` on the policy command line do not override vision mode.

Check `crc_errors=0`, finite observations/actions, and inference/loop timing.
Move the camera right/left and confirm the sign changes in terminal C.
While motors are disabled, test the receiver watchdog:

1. Stop terminal A: the policy produces zero velocity within about 0.25 s plus
   scheduling delay and prints `udp_fresh=False`.
2. Restart A: live commands resume automatically when detection is valid.

These are **zero-velocity policy commands**, not motor-disable commands and not
a guarantee of a physically stationary stance. Ctrl+C in the policy process
disables motors; loss of valid STM32 state also invokes existing fault handling.
Keep the robot supported when checking motor-disable behavior.

## 4. Run the physical closed-loop test

After the disabled-motor checks, secure the robot with your normal support/fall
protection, start with a short clear track, and keep the physical stop available.
Keep terminal A running with `--max-wz 0.2`. Restart terminal B as:

```bash
python -u humanoid_jetson_deploy/main.py \
  --policy walking \
  --model humanoid_jetson_deploy/models/current_walking.onnx \
  --port /dev/ttyACM0 --command-source vision \
  --enable-motors --no-plot --max-seconds 10
```

It starts walking as soon as valid STM32 state and live vision commands are
available. If vision is absent at startup, the policy receives zeros.
Check that the robot physically turns toward the track and that the observed
track error decreases over successive images. Changing log values alone proves
communication, not successful physical tracking.

After the short test, omit `--max-seconds 10` for continuous operation. Vision
mode has no five-second walking cutoff. Stop using Ctrl+C in terminal B, then
stop vision. CSV motor/IMU logs are written under
`logs/motor_positions/` relative to the directory where you launched the policy.
Command values appear in terminal logs; they are not additional CSV columns.

## 5. Tune the angular command

The mapping is:

```text
steer_cm = PID(fused_err_cm) + preview_gain * step_len_cm * sin(heading_error)
wz = yaw_sign * clip(clip(steer_cm, -50, 50) / steer_full_scale_cm, -1, 1) * max_wz
```

This is a commanded yaw rate derived from image error, not a measured angular
velocity. With the defaults, 10 cm of final PID steering gives -0.1 rad/s.

- `--max-wz`: begin at 0.2; increase toward 0.5 only after verifying tracking.
- `--steer-full-scale-cm`: default 50; reducing it strengthens correction.
- `--vx`: sets forward speed in the vision process; choose a speed your policy
  can track reliably (repository default 0.4 m/s).
- `--step-len-cm`: default 8; keep consistent with your intended walking stride.
- `--preview-gain`: default 1; reduce if anticipatory steering causes oscillation.
- Existing `JETSON_PID_STRAIGHT_KP/KI/KD`, `JETSON_PID_CURVE_KP/KI/KD`,
  `JETSON_PID_I_CLAMP`, `STEP_LEN_CM`, and `PREVIEW_GAIN` environment overrides work.

The 0.5 rad/s cap is retained from the old communication implementation. Raising
it would require coordinated edits to the mapper and policy receiver
and validation against the trained policy's turning range.

## Troubleshooting and fixed-mode fallback

- **Vision shows turns, policy does not:** confirm the vision startup line says
  `UDP -> policy 127.0.0.1:5005`, the policy says `udp_fresh=True`, and only one
  receiver owns port 5005. A log saying `[vision -> connector]` identifies the
  older build, which sends to 5006 and requires `connector.py`.
- **`fresh=False` while vision runs:** check frame processing time and dropped
  camera reads. The watchdog covers missing publications, not the age of a
  frame buffered inside the camera driver. Tune capture latency if needed.
  Increase policy `--command-timeout` only after measuring the frame period; a
  larger timeout also means a longer stale-command hold.
- **Alternating move/stop:** inspect lost-frame/confidence logs, camera geometry,
  illumination and view of the track. The bridge stops immediately on reported
  line loss instead of blindly searching with the humanoid.
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

The integration tests use real local UDP sockets to check both the default direct
path and optional connector relay, historical JSON encoding, policy observation
slots, watchdogs, and UDP health counters. Runtime tests check that vision
commands survive beyond five seconds. Camera, ONNX checkpoint performance, and
physical walking require the Jetson procedure above.

At the base revision `3ca3d9968370a5ab187f41dd938d31ed90ea02d2`, two existing
one-foot tests expect the leg to lower again, but `OneFootCommand.get()` keeps
the lift command active. Those same two tests fail before and after this change;
the one-foot schedule is not modified here.
