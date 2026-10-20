# Jetson Orin Nano humanoid runtime

## Phone / desktop remote control
Run `python3 -m remote_control` from the repository root on the Jetson, then
open `http://<Jetson-LAN-IP>:8080` from a phone or computer on the same Wi-Fi.
The browser provides independent forward and yaw joysticks, keyboard controls,
an explicit stop, and an expiring single-controller lease. Run the walking
runtime with `--command-source remote` (UDP port 5007). Existing motor enable
and model requirements still apply. See the [Chinese setup guide](docs/web_remote_zh.md)
for complete commands and a hardware-free test. The page displays commands,
not measured robot motion; no Internet connection or frontend packages are needed.

Moving card presence now uses only the corrected-gray cue detector. It skips
card-box binarization, line/corner proposals and box verification. Walking stops
only when current cue geometry passes the configured height/distance gate and
the existing presence-history and cooldown gates. Startup and stopped shape
classification retain full card-box detection. Legacy box-stop log counters
remain zero for log-reader compatibility; `card_presence_mode` is `cue_only`.
Numbered recording sessions and detailed performance timing remain enabled.

The runtime supports standard forward walking from `Humanoid_Robot_RSL_RL`
and one-foot standing from `Humanoid_Robot_One_Foot_Standing`. Select the
interface with `--policy walking` (default) or `--policy one-foot`.
Walking can receive live `new_vision` commands with `--command-source vision`.
See the [closed-loop setup and testing guide](docs/closed_loop_vision.md).
Shape-card actions in that vision mode are described in the
[shape action deployment guide](docs/shape_action_deployment.md).
Fixed-command walking remains the default. Standalone `--policy one-foot` does
not use vision; shape actions switch to the one-foot model from walking vision mode.

Walking defaults:

- 49 observations, including step distance and crossing command.
- Fixed mode: **0.4 m/s** forward by default; lateral velocity and yaw rate: **0**.
- Vision mode: live forward/yaw commands over the original UDP JSON protocol.
- Step distance: **scales with the commanded speed** — 0.08 m at the default 0.3 m/s, 0 at a standstill. The calibration pair is `--max-vx` / `--max-step-cm`; crossing: **0**.
- 50 Hz inference, 12 joint-position actions, action scale 0.25.
- Fixed mode commands zero after 5 seconds (`--walk-seconds 0` makes it continuous).
- Vision mode runs continuously with upstream-command watchdogs; no bar-crossing commands.
- Existing motor enable, state watchdog, joint limits, and emergency shutdown remain active.

These values match training commit `4eb3d5b4d72a792c610ad46f0a8c65b931ed3b22`.
The joint order, default pose, and IMU acceleration scaling are unchanged.

## Run walking

Export the current walking/stepping checkpoint to ONNX and copy it to
`humanoid_jetson_deploy/models/current_walking.onnx`. Include its trained
observation normalizer in the export if normalization was enabled.
Existing model binaries are not updated in this change. Legacy 47/48-input
walking-test models are rejected; a 49-input model must also have the documented
observation semantics.

```bash
cd humanoid_jetson_deploy
python main.py --model models/current_walking.onnx --port /dev/ttyACM0
```

Motor output remains disabled unless `--enable-motors` is passed.
Follow the calibration and motor-test procedure in
[the deployment guide](humanoid_jetson_deploy/README.md).
Add `--no-plot` for headless operation; CSV position and IMU logging stays enabled.
`--max-seconds` ends a timed run and disables motors as before.

`--vx` selects a constant positive speed (at most 1 m/s), defaulting to 0.4.
`--command-source fixed` uses `--vx` and `--wz` (yaw range ±0.5 rad/s).
`--command-source vision` receives both values from the connector instead.
`--walk-seconds` applies only to fixed mode; its default is 5 seconds.

## Run one-foot standing

The repository includes `humanoid_jetson_deploy/policy-one-foot-standing.onnx`.
Use a checkpoint with the same 46-input interface if replacing it.
From the repository root:

```bash
cd humanoid_jetson_deploy
python main.py --policy one-foot --model policy-one-foot-standing.onnx --port /dev/ttyACM0 --enable-motors --max-seconds 8
```

The default sequence is command **0** for 1 second, **1** for 4 seconds,
then **0** until exit. This commands the left foot to lift with right-foot
support. Use `--stand-seconds` and `--lift-seconds` to change the durations.
There is one sequence per process run; it does not repeat or reset the physical
robot at 8 seconds. `--max-seconds 8` ends this example after the lowering phase
and disables motors. Omit it to continue the standing command until Ctrl+C.

`--support-foot left` selects left support/right lift with the training code's
observation and action mirroring. Right support is the default because current
training disables left/right switching. The side remains fixed throughout a run.
`--vx`, `--wz`, and `--command-source` do not supply observations to this mode.
Add `--no-plot` for headless operation.

The 46-input layout and side transforms match training commit
`c1b4e8c8bdedafc8c7fd4c162a7c3c9e0a28df9f`; see
[the one-foot interface](humanoid_jetson_deploy/README.md#one-foot-standing-interface)
for the exact order. The runtime rejects a walking model selected for one-foot
mode before opening the serial port.

Hardware testing is complete as confirmed by the owner:
`CALIBRATION_CONFIRMED = True` and `IMU_CALIBRATION_CONFIRMED = True` are
already set in `config.py` and remain so. `--enable-motors` enables motor
output for the requested run; omitting it keeps the existing dry-run behavior.

## Vision code

Use `new_vision/jetson/run_policy_vision.py` for the walking-policy integration.
It uses the new line detector and PID steering, then sends
`{vx, vy: 0, wz, qr}` to `connector.py` on port 5006. Confirmed cards also carry
`event_id` and `event_action`; `qr=-1` denotes no current recognition. The connector forwards
commands to the policy on port 5005, slew-rate limiting `vx` and `wz` so the
policy never receives a velocity step. Only the policy opens the STM32 serial port.
The V2 CPU/GPU `run_robot.py` scripts are separate serial-controller entry points.

See [the detailed guide](docs/closed_loop_vision.md) for camera setup, exact
three-terminal commands, dry runs, physical tests, yaw-sign tuning and watchdogs.
Add `--one-foot-model humanoid_jetson_deploy/policy-one-foot-standing.onnx` to
the walking vision command to enable shape actions. Bar crossing remains outside
this integration.

## Tests

Offline vision logs can be opened with `python new_vision/scripts/view_vision_log.py`
or `new_vision/scripts/view_vision_log.bat` on Windows. `LOG="vision"` records
the exact Close+Fit `rowhdg` value and validity, plus fresh policy-adopted yaw
feedback. See [the viewer guide](docs/vision_log_viewer_zh.md).

```bash
python -m unittest discover -s tests -v
cd humanoid_jetson_deploy
python -m unittest discover -s tests -v
```


