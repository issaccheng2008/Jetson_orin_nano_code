# Humanoid Robot: Jetson Orin Nano ONNX Deployment

This package runs the current `Humanoid_Robot_RSL_RL` policy on a Jetson Orin Nano and exchanges state/target data with an STM32. It is intentionally split into:

- **Jetson, 50 Hz:** observation construction, projected-gravity estimation, ONNX inference, action scaling, joint limits, target transmission.
- **STM32, 1 kHz:** encoders, IMU acquisition, motor position/PD control, current limits, communications watchdog, emergency stop.

The FK723M1-ZGT6 implementation uses native USB CDC through the board's USB-C connector, with binary framing, CRC-16, sequence IDs, status flags, and a 100 ms command watchdog.

## Wire protocol

All multi-byte values are little-endian. Floating-point fields are IEEE-754 `float32`.

| Frame field | Bytes | Description |
|---|---:|---|
| Magic | 2 | `0xA55A` (`5A A5` on the wire) |
| Version | 1 | Protocol version `1` |
| Message type | 1 | `1=state`, `2=command` |
| Payload length | 2 | Number of payload bytes |
| Sequence | 2 | Wraparound packet counter |
| Payload | variable | Packed state or command structure |
| CRC | 2 | CRC-16/CCITT-FALSE over version through the end of payload |

The state payload is 128 bytes and its complete frame is 138 bytes. The command payload is 64 bytes and its complete frame is 74 bytes. The Python and STM32 implementations use the same packed layouts and CRC algorithm.

At 200 state frames/s and 50 command frames/s, the total payload traffic is approximately 31.3 kB/s, comfortably within USB full-speed CDC capacity.

## Critical model-version warning

The repository revision inspected for this package is commit `0de3d9b5b11af011eceefc1cc33c72c3d077acc4`. Its first observation is **IMU linear acceleration**, scaled by `0.1`. An older policy used base linear velocity instead.

Both versions have 48 inputs, so ONNX shape inspection cannot distinguish them. Use a checkpoint trained after changing the observation to acceleration. An old checkpoint will execute successfully but receive the wrong data.

## Policy interface

The policy period is `0.005 s * decimation 4 = 0.020 s`, or 50 Hz.

| Observation indices | Size | Value sent to ONNX |
|---:|---:|---|
| 0:3 | 3 | IMU acceleration in policy frame, m/s², multiplied by `0.1` |
| 3:6 | 3 | IMU angular velocity in policy frame, rad/s |
| 6:9 | 3 | Projected gravity: world-down unit vector in body/IMU frame |
| 9:12 | 3 | Command `[vx, vy, wz]` |
| 12:24 | 12 | Joint position minus Isaac default position, radians |
| 24:36 | 12 | Joint velocity, rad/s |
| 36:48 | 12 | Previous raw ONNX action |

The ONNX output is converted to the Isaac joint target using:

```text
q_target_policy = q_default + 0.25 * action
```

Observation noise used during training is **not** added during deployment.

## Files

```text
humanoid_jetson_deploy/
├── README.md
├── requirements.txt
├── config.py                    robot constants, limits, motor/IMU calibration
├── protocol.py                  shared wire format implemented in Python
├── serial_link.py               background serial receiver and state freshness checks
├── imu_filter.py                six-axis projected-gravity estimator
├── policy_runner.py             ONNX loading and exact 48-value observation layout
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

Expected model dimensions are one `[1, 48]` input and one `[1, 12]` output. The input/output names are detected automatically.

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

Both acceleration and gyro must use the same frame as the simulated IMU. Configure:

```python
IMU_TO_POLICY = np.array([...], dtype=np.float32)
```

This should normally be a signed permutation matrix describing the sensor mounting. For example, if sensor X corresponds to policy Y and sensor Y corresponds to negative policy X:

```python
IMU_TO_POLICY = np.array([
    [ 0, -1,  0],
    [ 1,  0,  0],
    [ 0,  0,  1],
], dtype=np.float32)
```

Required checks:

- Upright and stationary: transformed acceleration approximately `[0, 0, +9.81]` m/s².
- Roll robot right: projected-gravity Y changes in the same direction as Isaac playback.
- Pitch robot forward: projected-gravity X changes in the same direction as Isaac playback.
- Positive yaw rotation: transformed gyro Z has the same sign as Isaac.

The current simulation multiplies acceleration by `0.1` before it reaches the network. `policy_runner.py` applies that same scaling exactly once.

## Step 9: fixed-speed policy test without vision

The UDP command receiver is currently commented out in `main.py`. Run only the
policy process to use its default fixed command of `vx=0.5 m/s`, `vy=0.0 m/s`,
and `wz=0.0 rad/s`:

```bash
python main.py --model policy.onnx --port /dev/ttyACM0
```

To restore vision integration later, uncomment the UDP-related lines in
`main.py`, then run the policy with a local UDP command receiver:

```bash
python main.py \
  --model policy.onnx \
  --port /dev/ttyACM0 \
  --udp-command-port 5005
```

From the repository root, run the connector in a second terminal:

```bash
python connector.py --vision-port 5006 --policy-port 5005
```

Then start the integrated vision producer in a third terminal:

```bash
python vision/run_real_car.py
```

Vision sends `{vx, vy: 0, wz, qr}` to UDP port 5006. The connector validates and processes that output, then sends it to this policy receiver on port 5005. The current example processing function is in `connector.py`; it clamps the training ranges, forces `vy=0`, and forwards QR values `1`–`6` or `-1` when none is visible.

Both connector and policy receiver independently force the velocity command to zero if new upstream messages stop for 250 ms. Commands are clamped to the training range: `vx=0..1 m/s`, `vy=0`, and `wz=-0.5..0.5 rad/s`.

## Step 10: first motor-enabled tests

Use a physical emergency stop and overhead support. First command the default pose without ONNX and confirm all PD loops, limits, signs, and current limits. Then run the policy at reduced hardware gain scales:

```bash
python main.py \
  --model policy.onnx \
  --port /dev/ttyACM0 \
  --vx 0.0 \
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
4. Policy while suspended, zero command.
5. Feet lightly contacting the floor with overhead support.
6. Small `vx`, approximately 0.15–0.25 m/s.
7. Turning commands.
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
