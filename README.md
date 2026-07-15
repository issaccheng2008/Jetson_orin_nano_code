# Jetson Orin Nano humanoid runtime

This repository contains the vision/navigation code and the ONNX walking-policy
runtime. `connector.py` links them while keeping camera processing and the 50 Hz
policy loop in separate processes.

## Data flow

```text
vision/run_real_car.py
  -> UDP 5006: {vx, vy: 0, wz, qr}
connector.py
  -> validate/process, force vy=0, apply stale-input stop
  -> UDP 5005: {vx, vy: 0, wz, qr}
humanoid_jetson_deploy/main.py
  -> command observation [vx, 0, wz]
  -> ONNX policy -> STM32 motor targets
```

The QR value is `1` through `6` while a valid code is visible and `-1` when no
QR code is visible. The current example connector forwards the QR value but the
policy consumes only `[vx, vy, wz]`. Add future QR behavior in
`process_vision_output()` in `connector.py`.

## Run all three processes

Open three terminals in the repository root and activate the same Python
environment in each.

1. Start the policy receiver:

   ```bash
   cd humanoid_jetson_deploy
   python main.py --model policy.onnx --port /dev/ttyACM0 --udp-command-port 5005
   ```

   Omit `--enable-motors` until the complete dry-run and calibration procedure
   in `humanoid_jetson_deploy/README.md` has passed.

2. Start the connector:

   ```bash
   cd ..
   python connector.py --vision-port 5006 --policy-port 5005
   ```

3. Start vision:

   ```bash
   python vision/run_real_car.py
   ```

The connector publishes at 50 Hz independently of the camera rate. It uses a
zero-order hold: if vision produces one command every 0.1 seconds (10 Hz), the
connector republishes that same command about five times so the policy receives
a target on every 50 Hz inference step. If vision messages stop for more than
250 ms, it publishes `[0, 0, 0]`; the policy receiver also has its own 250 ms
UDP watchdog.

## Debug the target velocity

All three stages print clearly labeled target values:

```text
vision_target_velocity=[vx=+0.150 m/s, vy=+0.000 m/s, wz=-0.200 rad/s]
[connector -> policy] ... target_velocity=[vx=+0.150 m/s, vy=+0.000 m/s, wz=-0.200 rad/s]
policy_target_velocity=[vx=+0.150 m/s, vy=+0.000 m/s, wz=-0.200 rad/s]
```

By default, connector and policy logs appear every 25 steps (twice per second at
50 Hz). For short debugging runs, print every 50 Hz step with:

```bash
python connector.py --vision-port 5006 --policy-port 5005 --log-every 1

cd humanoid_jetson_deploy
python main.py --model policy.onnx --port /dev/ttyACM0 \
  --udp-command-port 5005 --log-every 1
```

Printing at 50 Hz adds terminal overhead, so use the default interval for normal
operation.

## Command conversion and calibration

`run_real_car.py` converts its existing forward target from cm/s to m/s. Its
steering output is not a measured angular velocity, so it is normalized and
mapped into the policy range. Configure the mapping with:

```bash
VISION_MAX_WZ=0.3 VISION_WZ_SIGN=1 python vision/run_real_car.py
```

Use `VISION_WZ_SIGN=-1` if the robot turns opposite to the detected direction.
Calibrate this with the robot supported and motors at reduced gains.

Useful network settings are `CONNECTOR_ENABLED`, `CONNECTOR_HOST`, and
`CONNECTOR_PORT`. All sockets bind to localhost by default, so no external
network access or ROS installation is required.

## Test the connector logic

```bash
python -m unittest discover -s tests -v
python -m compileall -q connector.py vision humanoid_jetson_deploy
```
