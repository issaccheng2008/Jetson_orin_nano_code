# Jetson Orin Nano Vision

Computer-vision code for a USB camera connected to a Jetson Orin Nano. The repository currently provides:

- line/track detection using inverse perspective mapping (IPM) and bird's-eye processing;
- lateral path-deviation, heading, confidence, curve, and lost-line outputs;
- a PID-based differential-wheel command generator with optional serial output;
- QR-code recognition for action numbers `1`–`6`;
- UDP publication of humanoid commands and the currently visible QR value to
  the repository-level `connector.py` process;
- red-region detection in the visualization demo; and
- a Webots bridge for testing parts of the same vision pipeline in simulation.

> **Compatibility note:** `run_real_car.py` retains the legacy four-wheel
> serial output, but it also publishes the humanoid `[vx, 0, wz]` command and
> current QR value to the repository-level connector.

## Processing pipeline

```text
USB camera
    |
    +--> LineDetector
    |      +--> lateral deviation (px and debug cm value)
    |      +--> path heading (deg)
    |      +--> confidence, curve mode, lost-frame count
    |
    +--> QRDetector
    |      +--> action number 1-6
    |
    +--> red HSV mask (vision demo)
           +--> red-pixel ratio and detected regions
```

In `run_real_car.py`, the line detector's fused tracking error is passed through a straight/curve PID controller. The resulting steering value is combined with a configured target speed to produce four commanded wheel angular velocities. These commands can be sent to a microcontroller over a serial port.

## Repository files

| File | Purpose |
|---|---|
| `line_detector_v1_warp.py` | Main line detector. Warps the camera image to a 320×400 bird's-eye view, enhances dark lines with black-hat morphology, applies adaptive thresholding and connected-component filtering, scans near/mid bands, and returns path deviation, heading and confidence. |
| `run_real_car.py` | Vision command producer plus the legacy four-wheel differential-drive controller. Runs line and QR detection, publishes `[vx, 0, wz]` plus QR to `connector.py`, and retains optional legacy serial transmission. |
| `qr_detector.py` | Reusable OpenCV QR detector. Tries the raw grayscale image and then a 2× upscaled image. Only payloads `1` through `6` are accepted. |
| `vision_main.py` | GUI visualization demo for line tracking, QR recognition and red-region detection. Displays the original image, bird's-eye view, binary image and annotated fit. |
| `usb_cam_qr_test.py` | Minimal USB-camera/QR test. It automatically runs without a preview window when no desktop display is available. |
| `webots_controller.py` | Webots/e-puck test controller using related line and QR processing. Its UART output is still marked as TODO. |

## Jetson GPU acceleration

The real-robot path now accelerates the parts that OpenCV CUDA supports:

- UVC MJPEG decode uses Jetson's `nvv4l2decoder` through GStreamer when
  available;
- full-frame inverse-perspective warp, channel reduction, black-hat filtering,
  thresholding and morphology run with `cv2.cuda`;
- QR grayscale conversion, resize and 2x upscale run with `cv2.cuda`;
- QR decode defaults to every third frame, while line tracking and command
  publication still run on every frame; and
- headless mode skips all copies, overlays, window resizes and `imshow` calls.

QR decoding itself (`QRCodeDetector.detectAndDecode`), connected-component
filtering, scan-line heuristics and PID control remain on CPU because OpenCV
does not expose CUDA implementations for them. The program therefore uses both
GPU and CPU; it does not falsely claim that every instruction runs on GPU.

`VISION_DEVICE=auto` safely uses CUDA when available and falls back to CPU.
`VISION_DEVICE=cuda` is the recommended deployment setting because it fails at
startup if Python loaded a non-CUDA OpenCV build.

## Jetson Orin Nano setup (Ubuntu 24.04 / JetPack 7.2)

The supported Ubuntu 24.04 route is NVIDIA's official JetPack 7.x / Jetson
Linux image. A generic Ubuntu ARM image does not include the Jetson GPU driver,
multimedia plugins or CUDA integration.

### 1. Verify the base system

```bash
uname -m
cat /etc/os-release
dpkg-query -W nvidia-l4t-core 2>/dev/null || true
```

For the current Ubuntu 24.04 path, expect `aarch64`, Ubuntu 24.04, and an
NVIDIA L4T/Jetson Linux package. If `nvidia-l4t-core` is absent, install the
official JetPack image before continuing.

### 2. Install the JetPack components

```bash
sudo apt update
sudo apt install -y nvidia-jetpack
/usr/local/cuda/bin/nvcc --version
```

The normal Ubuntu `python3-opencv` package and PyPI `opencv-python` wheels are
CPU-only for this purpose. Do not install either inside the vision environment.

### 3. Clone this repository

```bash
git clone https://github.com/issaccheng2008/Jetson_orin_nano_code.git
cd Jetson_orin_nano_code
```

### 4. Build CUDA-enabled OpenCV

The included script builds OpenCV 4.13 in an isolated environment with CUDA
architecture 8.7 (Jetson Orin), GStreamer, V4L2 and the Python bindings:

```bash
chmod +x vision/build_opencv_cuda.sh
VISION_VENV="$HOME/.venvs/jetson-vision" BUILD_JOBS=2 \
  ./vision/build_opencv_cuda.sh
source "$HOME/.venvs/jetson-vision/bin/activate"
```

The source build is large and can take a long time on an Orin Nano. It needs
roughly 10 GB of free storage. Keep `BUILD_JOBS=2` on an 8 GB board; use `4`
only if the board has enough RAM/swap and cooling.

Verify that Python loaded this build rather than a CPU wheel:

```bash
python - <<'PY'
import cv2
print("cv2:", cv2.__version__, cv2.__file__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
for line in cv2.getBuildInformation().splitlines():
    if "NVIDIA CUDA:" in line or "GStreamer:" in line:
        print(line.strip())
PY
```

`CUDA devices` must be at least `1`, and build information must show CUDA and
GStreamer as `YES`.

### 5. Check the camera and Jetson decoder

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
gst-inspect-1.0 nvv4l2decoder
```

Confirm that the camera lists MJPG at `1280x720` and the desired frame rate.
The requested resolution is 1280x720, not 1080x720. Test hardware decode:

```bash
gst-launch-1.0 -v \
  v4l2src device=/dev/video0 io-mode=2 ! \
  'image/jpeg,width=1280,height=720,framerate=30/1' ! \
  jpegparse ! nvv4l2decoder mjpeg=true ! fakesink sync=false
```

Stop this test with Ctrl+C before starting Python.

### 6. Run the accelerated vision process

For robot deployment over SSH/headless:

```bash
source "$HOME/.venvs/jetson-vision/bin/activate"
SHOW_WINDOW=0 VISION_DEVICE=cuda CAMERA_BACKEND=auto \
QR_INTERVAL=3 QR_MAX_PROCESS_WIDTH=960 \
CAM_IDX=0 CAM_W=1280 CAM_H=720 CAM_FPS=30 \
python vision/run_real_car.py
```

Startup must print something similar to:

```text
Camera 0: requested 1280x720, got 1280x720, backend=gstreamer-nvv4l2decoder
Vision backend: line=cuda, qr_preprocess=cuda, qr_every=3 frame(s), display=False
```

The periodic command line now includes measured processing FPS and the active
line backend. Use `sudo tegrastats` in another terminal to confirm GPU load;
on Jetson, `tegrastats` is more useful than desktop `nvidia-smi`.

### 7. Tune latency if necessary

- Keep `SHOW_WINDOW=0` during autonomous operation. Preview windows force
  device-to-host copies and GUI work.
- If QR detection is still the bottleneck, try `QR_INTERVAL=5` or
  `QR_MAX_PROCESS_WIDTH=640`, then re-test recognition distance.
- If the NVIDIA MJPEG pipeline does not accept the camera stream, use
  `CAMERA_BACKEND=v4l2`; CUDA image processing remains enabled.
- For comparison only, force the old path with `VISION_DEVICE=cpu`.
- Use a proper heatsink/fan. `sudo jetson_clocks` can hold high clocks but also
  increases power and heat; check the selected power mode and temperature with
  `sudo nvpmodel -q` and `sudo tegrastats`.

## Quick start

Activate the environment from the setup above:

```bash
cd Jetson_orin_nano_code
source "$HOME/.venvs/jetson-vision/bin/activate"
```

### 1. Test the USB camera and QR recognition

```bash
CAM_IDX=0 CAM_W=1280 CAM_H=720 CAM_FPS=30 python usb_cam_qr_test.py
```

This script can run headlessly. To explicitly disable or enable the preview:

```bash
SHOW_WINDOW=0 python usb_cam_qr_test.py
SHOW_WINDOW=1 python usb_cam_qr_test.py
```

`SHOW_WINDOW=1` still requires a valid `DISPLAY` or `WAYLAND_DISPLAY`.

### 2. Run the visualization demo

```bash
CAM_IDX=0 CAM_W=1280 CAM_H=720 CAM_FPS=30 python vision_main.py
```

Press `Esc` to exit. The demo displays:

1. the camera image with line, QR and red-region status;
2. the IPM bird's-eye grayscale image;
3. the thresholded binary image; and
4. the annotated line-detection image.

### 3. Run the real-car controller without serial output

```bash
SERIAL_ENABLED=0 \
CAM_IDX=0 CAM_W=1280 CAM_H=720 CAM_FPS=30 \
python run_real_car.py
```

Press `q` to quit or `s` to toggle serial transmission.

### 4. Run with serial output to an MCU

First identify the serial device:

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

Then run, replacing the port if needed:

```bash
SERIAL_ENABLED=1 \
SERIAL_PORT=/dev/ttyUSB0 SERIAL_BAUD=115200 \
CAM_IDX=0 CAM_W=1280 CAM_H=720 CAM_FPS=30 \
python run_real_car.py
```

If the user does not have serial-port permission:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing the group.

## Line-detector output

`LineDetector.process(frame)` returns:

```python
dev_px, heading_deg, confidence, visualization, debug = detector.process(frame)
```

| Output | Meaning |
|---|---|
| `dev_px` | Estimated lateral displacement of the path center in the bird's-eye image. Positive means the path center is to the robot's right. |
| `heading_deg` | Estimated path-heading angle in degrees. |
| `confidence` | Detection confidence from 0 to 1. |
| `visualization` | Annotated 320×400 bird's-eye BGR image. |
| `debug` | Intermediate images and values such as `base_err_cm`, `far_dist_cm`, `fused_err`, `curve_mode`, `lost_frames` and bottom-lock diagnostics. |

The centimeter values depend on the configured camera height, pitch and vertical field of view. They should be treated as geometric approximations until the camera is calibrated and the IPM result is checked against measured distances on the real course.

## QR and red-region output

`QRDetector.update(frame)` returns an integer event from `1` to `6` after the configured stability/cooldown checks, or `None` when there is no new event. `QRDetector.current_qr` separately reports the code visible in the current frame, or `-1`; this is the value published to `connector.py`.

The red detector in `vision_main.py` uses two HSV hue ranges and reports a red region when red pixels exceed 5% of the image. This is a color-area threshold, not a distance estimate or a full semantic obstacle detector.

## Serial frame used by `run_real_car.py`

The controller sends one six-byte frame for every processed camera frame:

| Byte | Meaning |
|---:|---|
| 0 | Header `0xFF` |
| 1 | Front-right commanded wheel speed |
| 2 | Front-left commanded wheel speed |
| 3 | Rear-right commanded wheel speed |
| 4 | Rear-left commanded wheel speed |
| 5 | Footer `0xEE` |

Each speed is encoded as a signed value in units of 0.1 rad/s, clamped to `[-127, 127]`, and stored as an unsigned byte. On the MCU:

```c
int8_t raw = (int8_t)received_byte;
float commanded_rad_s = raw / 10.0f;
```

The current frame has no checksum, sequence number, timestamp or acknowledgement. Add these before relying on it as a safety-critical robot-control link.

## Configuration

Most real-car settings can be overridden with environment variables.

| Variable | Default | Meaning |
|---|---:|---|
| `CAM_IDX` | `0` | OpenCV camera index |
| `CAM_W`, `CAM_H` | `1280`, `720` | Requested camera resolution |
| `CAM_FPS` | `30` | Requested frame rate |
| `CAMERA_BACKEND` | `auto` | `auto`, Jetson `gstreamer`, or normal `v4l2` capture |
| `VISION_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` image preprocessing |
| `SHOW_WINDOW` | `auto` | `0` disables GUI work; `1` requires a desktop display |
| `QR_INTERVAL` | `3` | Decode QR once per this many camera frames |
| `QR_MAX_PROCESS_WIDTH` | `960` | QR working width; `0` keeps full camera width |
| `CAM_HEIGHT_CM` | `40.0` | Camera height used by IPM |
| `CAM_PITCH_DEG` | `45.0` | Downward camera pitch |
| `CAM_VFOV_DEG` | `49.0` | Vertical field of view |
| `REAL_CAR_SPEED` | `15.0` | Configured target forward speed in cm/s |
| `REAL_CAR_MIN_SPEED` | `5.0` | Minimum commanded speed |
| `REAL_CAR_LOST_SCALE` | `0.92` | Speed multiplier after the line is lost |
| `REAL_CAR_ST2WHL` | `0.1` | Steering-to-left/right speed-difference gain |
| `REAL_CAR_WHEEL_RADIUS` | `3.0` | Wheel radius in cm |
| `SERIAL_ENABLED` | `0` | Set to `1` to enable serial output |
| `SERIAL_PORT` | `COM10` | Serial port; override with a Linux device on Jetson |
| `SERIAL_BAUD` | `115200` | Serial baud rate |

The PID and lost-line recovery parameters are also configurable through the `JETSON_PID_*`, `JETSON_STEER_*` and `LOST_*` environment variables defined near the top of `run_real_car.py`.

## Does this code estimate the robot's current velocity?

**No.** The displayed `spd` value is a **commanded target speed**, not a measured or observed velocity.

The controller calculates it as follows:

1. Start with `REAL_CAR_SPEED` (default 15 cm/s).
2. Multiply it by `REAL_CAR_LOST_SCALE` if the line is lost.
3. Reduce it when line-detection confidence is below 0.5.
4. Clamp it to at least `REAL_CAR_MIN_SPEED`.
5. Combine it with the steering command and wheel radius to calculate four commanded wheel angular velocities.

The code does not read motor encoders, IMU data, joint states or odometry. It also does not calculate optical flow or track ground features between frames. Therefore:

- `spd` is not the current base velocity;
- `FL/FR/RL/RR` are requested wheel speeds, not measured wheel speeds;
- `FPS` is camera processing rate, not robot speed;
- `far_dist_cm` is a geometric path look-ahead distance, not velocity; and
- the derivative inside the PID is the time derivative of line-tracking error, not robot velocity.

For a humanoid robot, estimate actual base motion separately by fusing the STM32 IMU, joint encoders and foot-contact/kinematic constraints. Camera visual odometry can be added as another measurement. A robust estimator should provide body-frame linear velocity and angular velocity to the deployed walking policy, while this repository's line detector should provide navigation commands such as desired forward velocity and desired yaw rate.

## Safety notes

- Test with the robot supported or wheels off the ground before enabling serial commands.
- Add an MCU watchdog that stops the actuators if valid commands stop arriving.
- Validate the frame header/footer and reject malformed values.
- Add an emergency-stop path independent of the vision process.
- Clamp commands again on the MCU.
- Recalibrate the IPM parameters after changing camera height, angle, lens or resolution.

## License

No license file is currently included. Add a license before third parties reuse or redistribute the code.
