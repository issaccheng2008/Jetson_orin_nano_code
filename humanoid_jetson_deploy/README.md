# Humanoid Robot: Jetson Orin Nano ONNX Deployment

## Fixed joint-angle frames (without ONNX)

Use `examples/fixed_joint_frames.json` as a template. `format` must be
`joint_frames_v1`, `hz` must be `50`, and `joint_names` must match
`config.JOINT_NAMES` exactly. Each entry of `frames` has 12 **policy-frame
radian** targets in that order: six right-leg joints, then six left-leg joints.
One entry is consumed per 20 ms control tick. To hold a pose, repeat the same
entry for as many ticks as needed. Playback stops and sends disable packets
after the last entry; it does not loop.

```bash
python main.py --fixed-policy examples/fixed_joint_frames.json --no-plot
python main.py --fixed-policy examples/fixed_joint_frames.json --enable-motors --no-plot
```

The first command is a dry run. The second can move the robot. Use overhead
support and verify joint order on the physical machine before enabling motors.
The existing joint-limit, target-speed and encoder-relative target limits still
apply, so a large jump in file values is sent as a limited target rather than
as an instantaneous jump. Every fixed-policy command requires a new STM32
state sequence in response; missing or stale feedback triggers a fault and
disable packets. Fixed playback does not require an ONNX model or
`onnxruntime`. The existing `--model` mode is unchanged.

This package runs the current `Humanoid_Robot_RSL_RL` walking policy or the
`Humanoid_Robot_One_Foot_Standing` policy on a Jetson Orin Nano and exchanges state/target data with an STM32. It is intentionally split into:

- **Jetson, 50 Hz:** observation construction, quaternion-derived projected gravity, ONNX inference, action scaling, joint limits, target transmission.
- **STM32, 1 kHz:** encoders, IMU acquisition, motor position/PD control, current limits, communications watchdog, emergency stop.

The FK723M1-ZGT6 implementation uses native USB CDC through the board's USB-C connector, with binary framing, CRC-16, sequence IDs, status flags, and a 100 ms command watchdog.

## Wire protocol

All multi-byte values are little-endian. Floating-point fields are IEEE-754 `float32`.

| Frame field | Bytes | Description |
|---|---:|---|
| Magic | 2 | `0xA55A` (`5A A5` on the wire) |
| Version | 1 | Protocol version `2` |
| Message type | 1 | `1=state`, `2=command` |
| Payload length | 2 | Number of payload bytes |
| Sequence | 2 | Wraparound packet counter |
| Payload | variable | Packed state or command structure |
| CRC | 2 | CRC-16/CCITT-FALSE over version through the end of payload |

The state payload is 144 bytes and its complete frame is 154 bytes. The command payload is 64 bytes and its complete frame is 74 bytes. The Python and STM32 implementations use the same packed layouts and CRC algorithm.

At 200 state frames/s and 50 command frames/s, the total framed traffic is approximately 34.5 kB/s, comfortably within USB full-speed CDC capacity.

## Walking model compatibility

The interface matches `Humanoid_Robot_RSL_RL` commit
`4eb3d5b4d72a792c610ad46f0a8c65b931ed3b22`: 49 observations and 12 actions.
Use an ONNX export from that walking/stepping policy with its trained observation
normalizer included if normalization was enabled. Existing 47/48-input models
are incompatible; the runtime rejects them before opening the serial link.
Model files are not replaced by this change. Input size alone does not establish
compatibility: the observation order and training configuration must also match.

## Walking policy interface

The policy period is `0.005 s * decimation 4 = 0.020 s`, or 50 Hz.

| Observation indices | Size | Value sent to ONNX |
|---:|---:|---|
| 0:3 | 3 | IMU acceleration in policy frame, m/s², multiplied by `0.1` |
| 3:6 | 3 | IMU angular velocity in policy frame, rad/s |
| 6:9 | 3 | Projected gravity: world-down unit vector in body/IMU frame |
| 9:11 | 2 | Command `[vx, wz]` |
| 11:12 | 1 | Default step distance: 0.08 m; 0 for all-zero velocity |
| 12:13 | 1 | Crossing command: 0 (normal walking) |
| 13:25 | 12 | Joint position minus Isaac default position, radians |
| 25:37 | 12 | Joint velocity, rad/s |
| 37:49 | 12 | Previous raw ONNX action |

The ONNX output is converted to the Isaac joint target using:

```text
q_target_policy = q_default + 0.25 * action
```

Observation noise used during training is **not** added during deployment.

## One-foot standing interface

Select `--policy one-foot` with an ONNX model exported from
`Humanoid_Robot_One_Foot_Standing` commit
`c1b4e8c8bdedafc8c7fd4c162a7c3c9e0a28df9f` (or a checkpoint with this exact
interface). The model must accept one float32 `[1, 46]` input (dynamic batch
allowed) and produce `[1, 12]` actions. The checked training configuration has
actor observation normalization disabled; if your checkpoint enabled it,
include that trained normalizer in the ONNX export.

| Observation indices | Size | Value sent to ONNX |
|---:|---:|---|
| 0:3 | 3 | Canonical IMU acceleration, m/s² × 0.1 |
| 3:6 | 3 | Canonical IMU angular velocity, rad/s |
| 6:9 | 3 | Canonical projected gravity |
| 9:10 | 1 | Binary `lift_one_foot_in_the_air` command |
| 10:22 | 12 | Canonical joint position minus default pose, radians |
| 22:34 | 12 | Canonical joint velocity, rad/s (default velocity is zero) |
| 34:46 | 12 | Previous raw canonical ONNX action |

There are no velocity, step-distance, or crossing-command observations and no
history stacking. Joint order, default pose, acceleration scale, action scale
0.25, and 50 Hz policy rate match the shared deployment constants.

Right support/left lift uses physical policy-frame values directly. With
`--support-foot left`, acceleration and gravity transform as `[x, -y, z]`,
angular velocity as `[-x, y, -z]`, and joint offsets/velocities swap the six-joint
right/left blocks and negate every coordinate. Actions undergo the same joint
transformation before calculating `q_default + 0.25 * physical_action`.
The previous-action observation always stores the raw canonical network output,
before mirroring, scaling, or target limiting. Mirroring remains active for
left support during command-zero phases too; side selection is not another
network input.

From this directory, after copying your exported model to `models/one_foot.onnx`:

```bash
python main.py --policy one-foot --model models/one_foot.onnx --enable-motors --max-seconds 8
```

Defaults are right support, 1 second standing, 4 seconds lifting, then the
standing command until exit. Training currently uses a fixed 1-second initial
stand and a random 3–5-second lift; deployment uses a reproducible 4-second lift.
Change `--stand-seconds` and `--lift-seconds` as needed. Both must be finite and
positive. The timer starts after valid startup state and IMU checks. Command zero
after lifting asks the policy to lower the foot; it does not abruptly replace
policy output with the default joint pose. No new episode or automatic repeat is
introduced, and action history is retained across command changes. Use
`--max-seconds 8` for an 8-second run or omit it to keep commanding standing.

`--support-foot left` uses the opposite physical support side. The current
training configuration has `ENABLE_LEFT_RIGHT_SWITCHING = False`, hence the
right-support default. `--no-plot`, logging, gain scales, and serial options
work in both modes. Console status shows the lift command and selected support
foot. The existing target limits, slew/deviation limits, stale-state checks,
fault shutdown, and STM32 protocol are shared by both modes.

Both hardware calibration confirmations in `config.py` are already `True`, as
requested after completed hardware testing. Motor output is enabled per run
with `--enable-motors`. No new calibration gate is added. The training repository
references a local `v3.1.usd`; this change retains the deployment's calibrated
hardware mapping and existing limits documented as `v2.4.1.urdf`, and does not
replace the robot asset or any ONNX binaries. Software tests use mocked model
outputs and serial state; they do not validate a particular trained checkpoint
or physical balance on the Jetson/robot.

## Files

```text
humanoid_jetson_deploy/
├── README.md
├── requirements.txt
├── config.py                    robot constants, limits, motor/IMU calibration
├── protocol.py                  shared wire format implemented in Python
├── serial_link.py               background serial receiver and state freshness checks
├── imu_filter.py                quaternion-derived projected gravity
├── policy_runner.py             shared ONNX inference and walking observations
├── one_foot_policy.py           46-value standing observations, mirroring, lift sequence
├── command_source.py            fixed or local UDP velocity command
├── main.py                      50 Hz deployment program
├── STM32H723_CubeMX_CubeIDE_Guide.md
├── tools/
│   ├── inspect_onnx.py
│   ├── protocol_demo.py
│   ├── send_velocity_command.py
│   └── stm32_link_test.py
├── tests/
│   ├── test_protocol.py
│   └── test_imu_filter.py
└── stm32_demo/
    ├── README.md
    ├── jetson_protocol.h
    ├── jetson_protocol.c
    ├── jetson_usb_cdc.h/.c
    ├── fk723_robot_app.h/.c
    ├── fk723_hardware.h
    └── fk723_hardware_stub.c
```

## Step 1: export the trained model

On the Isaac Lab training computer:

```bash
cd /home/tt/Humanoid_Robot_Policy_RSL_RL/humanoid_robot_policy_rsl_rl

CHECKPOINT=/absolute/path/to/logs/rsl_rl/humanoid_robot_rsl_rl_rough/YOUR_RUN/model_XXXX.pt

python scripts/rsl_rl/play.py \
  --task Humanoid-Robot-RSLRL-Play-v0 \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT" \
  --headless
```

The normal Isaac Lab RSL-RL play script creates:

```text
logs/rsl_rl/humanoid_robot_rsl_rl_rough/YOUR_RUN/exported/policy.onnx
```

Stop the play process after it reports the export. Verify the model:

```bash
python -m pip install onnxruntime numpy
python tools/inspect_onnx.py /path/to/policy.onnx
```

Expected model dimensions are one `[1, 49]` input and one `[1, 12]` output. The input/output names are detected automatically.

## Step 2: copy the package and policy to Jetson

From another computer, assuming the Jetson account is `isaac`:

```bash
scp -r humanoid_jetson_deploy isaac@JETSON_IP:/home/isaac/
scp /path/to/policy.onnx isaac@JETSON_IP:/home/isaac/humanoid_jetson_deploy/
```

Check file integrity if desired:

```bash
sha256sum /path/to/policy.onnx
ssh isaac@JETSON_IP sha256sum /home/isaac/humanoid_jetson_deploy/policy.onnx
```

## Step 3: install the Jetson environment

First confirm that the board has a Jetson Linux/JetPack installation:

```bash
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core 2>/dev/null
```

Create a virtual environment:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

python3 -m venv --system-site-packages /home/isaac/venvs/humanoid_policy
source /home/isaac/venvs/humanoid_policy/bin/activate

cd /home/isaac/humanoid_jetson_deploy
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This first implementation deliberately uses ONNX Runtime on the CPU. The actor is a small MLP, and the included inspection tool measures its actual inference time. TensorRT can be added later, but it should not be introduced before the state/action pipeline is validated.

Test the model:

```bash
python tools/inspect_onnx.py policy.onnx
```

The average latency must be comfortably below the 20 ms policy period.

## Step 4: connect Jetson and STM32

Use the FK723M1-ZGT6 board's USB-C port. It is connected to the STM32H723 native USB FS interface on PA11/PA12. After the USB CDC firmware is flashed, Jetson should expose it as `/dev/ttyACM0`.

The pyserial code still supplies a baud value, but USB CDC treats that as virtual line coding; it is not a physical UART baud rate.

Program/debug the board separately through ST-Link V2 connected to P1 SWDIO, SWCLK, GND, VTref, and preferably NRST.

Locate the port:

```bash
ls -l /dev/ttyACM* 2>/dev/null
dmesg --follow
```

Give the current user temporary access:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing the group. Avoid running the motor controller as root.

## Step 5: add the STM32 code

Follow [STM32H723_CubeMX_CubeIDE_Guide.md](STM32H723_CubeMX_CubeIDE_Guide.md). It gives the exact STM32H723ZGT6 selection, 25 MHz HSE, PG7 LED, USB CDC, TIM6, ST-Link wiring, CubeIDE integration, and test procedure.

Start with `fk723_hardware_stub.c`, which never drives a motor. After the link test passes, replace its functions with real implementations:

```c
Hardware_GetJointPositionRad(...)
Hardware_GetJointVelocityRadS(...)
Hardware_GetAccelerationMS2(...)
Hardware_GetGyroRadS(...)
Hardware_MotorsSetPositionTargets(...)
Hardware_MotorsDisable()
Hardware_Micros32()
```

STM32 transmits at 200 Hz:

```text
timestamp_us
12 joint positions in motor/joint coordinates, rad
12 joint velocities, rad/s
3 accelerations, m/s²
3 angular velocities, rad/s
orientation quaternion in `[w, x, y, z]` order
status flags
```

Jetson transmits at 50 Hz:

```text
timestamp_us
12 motor-coordinate target positions, rad
Kp scale
Kd scale
enable / e-stop flags
```

The MCU must continue running its motor loop at approximately 1 kHz. It must disable output if commands are stale for more than 100 ms.

## Step 6: test the protocol without motors

Run the pure Python framing test:

```bash
cd /home/isaac/humanoid_jetson_deploy
source /home/isaac/venvs/humanoid_policy/bin/activate

python tools/protocol_demo.py
python -m unittest discover -s tests -v
```

Flash the STM32 application, but keep motor power disabled. Run:

```bash
python tools/stm32_link_test.py --port /dev/ttyACM0 --seconds 10
```

This test does not load ONNX. It checks both communication directions while always leaving the motor-enable flag clear. Only after it passes should you run `main.py` without `--enable-motors`.

A healthy log resembles:

```text
PASS
  observed state rate: approximately 200 Hz
  sequence drops: 0
  CRC errors: 0
  STM32 received disabled Jetson commands: yes
```

Check that:

- `crc_errors` remains zero.
- Inference stays below 20 ms.
- Upright stationary IMU acceleration becomes approximately `[0, 0, +9.81]` after the configured axis transform.
- Projected gravity is approximately `[0, 0, -1]` upright.
- All observations and actions are finite.
- STM32 keeps motors disabled because the enable flag is clear.

## Step 7: calibrate joint coordinates

`config.py` intentionally starts with:

```python
CALIBRATION_CONFIRMED = False
```

The program refuses to enable the motors until this is changed.

For every joint, determine:

1. The physical encoder reading that corresponds to Isaac joint angle zero.
2. Whether increasing physical encoder angle increases or decreases the Isaac coordinate.
3. Whether the measurement is motor-shaft or joint-side radians.

Enter the results into:

```python
MOTOR_SIGN = np.array([...], dtype=np.float32)       # each entry +1 or -1
MOTOR_ZERO_RAD = np.array([...], dtype=np.float32)   # physical encoder zero offsets
```

The conversion is:

```text
q_policy = sign * (q_motor - motor_zero)
q_motor_target = motor_zero + sign * q_policy_target
```

With the robot manually placed in its default crouched pose, converted joint angles should be close to:

```text
right: [+0.15, 0, 0, +0.30, -0.15, 0]
left:  [-0.15, 0, 0, -0.30, +0.15, 0]
```

Only then set `CALIBRATION_CONFIRMED = True`.

## Step 8: calibrate the IMU frame

Acceleration, gyro, and quaternion-derived gravity must use the same frame as the simulated IMU. Configure:

```python
IMU_TO_POLICY = np.array([...], dtype=np.float32)
```

This is the fixed sensor-to-policy mounting rotation. It may be a signed
permutation for an exactly aligned installation or a general orthonormal
rotation if the sensor has a small mounting tilt. For example, if sensor X
corresponds to policy Y and sensor Y corresponds to negative policy X:

```python
IMU_TO_POLICY = np.array([
    [ 0, -1,  0],
    [ 1,  0,  0],
    [ 0,  0,  1],
], dtype=np.float32)
```

Set `IMU_CALIBRATION_CONFIRMED = True` only after measuring and saving this
matrix. The calibration remains in `config.py`, so the robot does not need to
be perfectly level at every startup. The STM32 intentionally does not issue
`imu_set_zero()` during boot: the DM-IMU-L1 EKF quaternion retains its
gravity-referenced roll and pitch, while yaw is irrelevant to projected gravity.

Required checks:

- Upright and stationary: transformed acceleration approximately `[0, 0, +9.81]` m/s².
- Roll robot right: projected-gravity Y changes in the same direction as Isaac playback.
- Pitch robot forward: projected-gravity X changes in the same direction as Isaac playback.
- Positive yaw rotation: transformed gyro Z has the same sign as Isaac.
- Stationary at any tilt: normalized `-acceleration` agrees with projected gravity.

The current simulation multiplies acceleration by `0.1` before it reaches the network. `policy_runner.py` applies that same scaling exactly once.

## Step 9: walking with vision or fixed commands

For closed-loop walking, follow the [new_vision integration guide](../docs/closed_loop_vision.md).
Run this receiver from the deployment directory:

```bash
python main.py --model models/current_walking.onnx --port /dev/ttyACM0 \
  --command-source vision --no-plot
```

It receives live forward/yaw commands from `connector.py` on UDP port 5005.
Missing connector data for 0.25 seconds commands zero velocity. The connector
also has a 0.25-second vision watchdog. There is no timed walking cutoff in vision
mode. Step distance is the configured default while moving and zero for an
all-zero velocity command; crossing remains zero.

Standalone fixed tests remain available:

```bash
python main.py --model models/current_walking.onnx --port /dev/ttyACM0 \
  --command-source fixed --vx 0.4 --wz 0 --walk-seconds 5
```

Fixed mode commands zero after five seconds by default; `--walk-seconds 0`
keeps it continuous. `--wz` accepts values within ±0.5 rad/s. Motor enable remains
opt-in. Ctrl+C, `--max-seconds` completion, invalid/stale STM32 state and faults
still disable motors. A zero velocity command itself does not disable motors.

## Live motor-position/IMU monitor and CSV log

Every normal `main.py` run opens a separate motor-position window. The solid
lines are the final targets sent to STM32 and the dashed lines are the actual
positions returned by STM32, both in motor coordinates and radians. The right
and left knee motors are selected initially. Use the checkboxes on the right to
show or hide any of the 12 motors while the policy is running.

The same window includes separate plots for policy-frame IMU acceleration
(`x`, `y`, `z`, in m/s²) and fused orientation (`roll`, `pitch`, `yaw`, in
radians). Use the `IMU acceleration` and `IMU orientation` checkboxes to show
or hide those two groups while the policy is running.

The plot runs in a separate process so drawing and window interaction do not
block the 50 Hz policy loop. By default it refreshes every five policy steps
and shows a rolling 10-second history. Adjust these settings if needed:

```bash
python main.py --model policy.onnx \
  --plot-every 5 \
  --plot-history-seconds 10
```

All 12 target positions, all 12 measured positions, policy-frame acceleration,
and policy-frame roll/pitch/yaw are recorded at every policy step, regardless
of which data is visible. Each run creates a new CSV under
`logs/motor_positions/`; the program prints the exact path at startup. Change
the directory with `--position-log-dir PATH`.

On a headless session without a desktop display, use `--no-plot`. This
disables only the window; CSV logging remains active.

When a timed run such as `--max-seconds 30` finishes normally, motor output is
disabled and the STM32 link is closed immediately, but the completed plot stays
open. Close the plot window when you have finished inspecting it. A fault or
Ctrl+C shutdown still closes the window without waiting.

To inspect an earlier run, use the history viewer. With no CSV argument it
opens the newest log in `logs/motor_positions/`:

```bash
python tools/view_position_log.py
```

Pass a file to open a specific run:

```bash
python tools/view_position_log.py \
  logs/motor_positions/motor_positions_YYYYMMDD_HHMMSS_ffffff.csv
```

The history viewer loads the full run and blocks until its window is closed.
It uses the same solid-target/dashed-actual lines, 12-motor checkboxes, and IMU
group checkboxes as the live plot, with both knee motors and both IMU groups
selected by default. Older motor-only CSV files remain viewable. The standard
Matplotlib toolbar can zoom and pan through the saved data.

## Step 10: first motor-enabled tests

Use a physical emergency stop and overhead support. First command the default pose without ONNX and confirm all PD loops, limits, signs, and current limits. Then run the policy at reduced hardware gain scales:

```bash
python main.py \
  --model policy.onnx \
  --port /dev/ttyACM0 \
  --vx 0.4 \
  --wz 0.0 \
  --kp-scale 0.2 \
  --kd-scale 0.3 \
  --enable-motors
```

The reduced scales are only a suspended-test starting point, not final gains. Increase them gradually while comparing physical response to Isaac Sim. A policy trained with the repository's implicit actuator gains expects approximately the same closed-loop stiffness and damping, but motor-controller gain numbers may not use SI units.

Recommended progression:

1. Motors unpowered, communications test.
2. One joint at a time, direction and zero verification.
3. Default pose controller without ONNX.
4. Policy while suspended, constant forward command.
5. Feet lightly contacting the floor with overhead support.
6. Small `vx`, approximately 0.15–0.25 m/s.
7. Continue straight walking with zero yaw command.
8. Unsupported operation only after reliable fault handling.

## Safety behavior

After ONNX inference, the Jetson applies the absolute URDF joint limits, the existing target slew-rate limit, and a final encoder-relative position window. Each transmitted target must remain within `MAX_TARGET_DEVIATION_DEG` of that joint's latest measured position. The default is 5 degrees; adjust this value in `config.py` only after suspended testing.

The Jetson program sends an e-stop command after any exception involving stale state, invalid IMU/encoder flags, STM32 fault state, NaN/Inf, or serial failure. It also sends disable frames during normal shutdown.

The MCU remains the final safety authority. It disables motor output when:

- The Jetson command is more than 100 ms old.
- The enable flag is absent.
- The e-stop flag is present.
- A motor fault is present.

Add independent MCU-side joint, speed, current, voltage, temperature, and tilt limits. A Jetson process is not a substitute for a physical emergency stop.

## Troubleshooting

### No STM32 packet received

```bash
ls -l /dev/ttyACM0
groups
```

Check the USB data cable, USB 48 MHz clock, USB Device CDC middleware, PA11/PA12 configuration, `MX_USB_DEVICE_Init()`, permissions, and whether STM32 set both `STATE_IMU_VALID` and `STATE_ENCODERS_VALID`.

### CRC errors increase

Check both sides use the supplied protocol version and packed structures unchanged. Rebuild both sides, try another USB data cable/port, and verify no other program has opened `/dev/ttyACM0`.

### Robot moves in the wrong direction

Stop immediately. Correct `MOTOR_SIGN`, encoder order, `MOTOR_ZERO_RAD`, or `IMU_TO_POLICY`. Do not compensate by reordering the ONNX output.

### Policy output is finite but behavior is nonsensical

The most likely causes are:

- ONNX was exported from the older base-linear-velocity policy.
- Joint order/sign/default offsets do not match Isaac Lab.
- Accelerometer was not multiplied by `0.1`, or was scaled twice.
- Acceleration/gyro units are not m/s² and rad/s.
- Projected gravity sign is reversed.
- Real actuator response differs greatly from the simulated PD controller.

### `onnxruntime` will not install on Jetson

Confirm the Python and aarch64 environment. Do not install an x86 wheel. As an alternative, use the TensorRT packages supplied by the board's matching JetPack installation and build the engine on the Jetson, not on the training computer.

## Before real walking

Resolve the mass discrepancy in the provided robot files. The supplied URDF totals approximately 4.19 kg, while the supplied CSV totals approximately 1.16 kg. Confirm which values match the built robot and the USD used for training. Also add sim-to-real randomization for actuator strength, gains, delay, joint zero error, sensor bias, mass/COM, and battery effects before expecting robust unsupported walking.



