#!/usr/bin/env bash
set -euo pipefail

# Build a CUDA/GStreamer-enabled OpenCV inside a dedicated virtual environment.
# Defaults target Jetson Orin (compute capability 8.7) and JetPack 7.2.
OPENCV_VERSION="${OPENCV_VERSION:-4.13.0}"
VISION_VENV="${VISION_VENV:-$HOME/.venvs/jetson-vision}"
SOURCE_DIR="${OPENCV_SOURCE_DIR:-$HOME/src}"
BUILD_JOBS="${BUILD_JOBS:-2}"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: run this script on the Jetson Orin Nano (aarch64)." >&2
    exit 1
fi

if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    echo "ERROR: CUDA toolkit not found at /usr/local/cuda." >&2
    echo "Install the official JetPack components first: sudo apt install nvidia-jetpack" >&2
    exit 1
fi

sudo apt update
sudo apt install -y \
    build-essential cmake git ninja-build pkg-config \
    python3-dev python3-venv python3-pip \
    libgtk-3-dev libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    libjpeg-dev libpng-dev libtiff-dev libopenexr-dev \
    libtbb-dev libeigen3-dev v4l-utils

python3 -m venv --system-site-packages "$VISION_VENV"
# shellcheck disable=SC1091
source "$VISION_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$(dirname "$0")/requirements-jetson.txt"

mkdir -p "$SOURCE_DIR"
if [[ ! -d "$SOURCE_DIR/opencv-$OPENCV_VERSION/.git" ]]; then
    git clone --depth 1 --branch "$OPENCV_VERSION" \
        https://github.com/opencv/opencv.git \
        "$SOURCE_DIR/opencv-$OPENCV_VERSION"
fi
if [[ ! -d "$SOURCE_DIR/opencv_contrib-$OPENCV_VERSION/.git" ]]; then
    git clone --depth 1 --branch "$OPENCV_VERSION" \
        https://github.com/opencv/opencv_contrib.git \
        "$SOURCE_DIR/opencv_contrib-$OPENCV_VERSION"
fi

PYTHON_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
BUILD_DIR="$SOURCE_DIR/opencv-$OPENCV_VERSION/build-jetson"
cmake -S "$SOURCE_DIR/opencv-$OPENCV_VERSION" -B "$BUILD_DIR" -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$VISION_VENV" \
    -D CMAKE_INSTALL_RPATH="$VISION_VENV/lib" \
    -D OPENCV_EXTRA_MODULES_PATH="$SOURCE_DIR/opencv_contrib-$OPENCV_VERSION/modules" \
    -D BUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,objdetect,python3,cudev,cudaarithm,cudaimgproc,cudawarping,cudafilters \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=8.7 \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_CUDNN=OFF \
    -D OPENCV_DNN_CUDA=OFF \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D WITH_FFMPEG=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE="$(command -v python)" \
    -D PYTHON3_PACKAGES_PATH="$PYTHON_SITE"

cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"
cmake --install "$BUILD_DIR"

python - <<'PY'
import cv2

info = cv2.getBuildInformation()
cuda_lines = [line for line in info.splitlines() if "NVIDIA CUDA:" in line]
assert cuda_lines and "YES" in cuda_lines[0], info
assert cv2.cuda.getCudaEnabledDeviceCount() > 0, "No CUDA device visible to OpenCV"
print("OpenCV:", cv2.__version__)
print("cv2 path:", cv2.__file__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
print(next(line.strip() for line in info.splitlines() if "GStreamer:" in line))
PY

echo "Done. Activate with: source $VISION_VENV/bin/activate"
