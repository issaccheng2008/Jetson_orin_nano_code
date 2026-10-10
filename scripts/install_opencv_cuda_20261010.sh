#!/usr/bin/env bash
# Run on the Orin board, after installing the build dependencies in the guide.
# Installs only into the dedicated GPU environment and private prefix.
set -euo pipefail

gpu_root="$HOME/src/robot_gpu_20261010"
gpu_env="$HOME/venvs/robot_gpu_20261010"
gpu_prefix="$HOME/opt/opencv-cuda-4.14.0-20261010"
gpu_python="$gpu_env/bin/python"
test -x "$gpu_python"
test -x /usr/local/cuda/bin/nvcc
test "$(uname -m)" = aarch64
mkdir -p "$gpu_root" "$gpu_prefix"

fetch_source() {
    local repo="$1" commit="$2" dest="$gpu_root/$1"
    if [ ! -e "$dest" ]; then
        git clone --depth 1 --branch 4.14.0 "https://github.com/opencv/$repo.git" "$dest"
    fi
    test "$(git -C "$dest" rev-parse HEAD)" = "$commit"
    test -z "$(git -C "$dest" status --porcelain)"
}
fetch_source opencv 0654a42e19215ef25b1d367d822f3c630447e7c7
fetch_source opencv_contrib a8e9acd62cabd30419dba83007f2ac0d07de5e2c

gpu_numpy_include="$($gpu_python -c 'import numpy; print(numpy.get_include())')"
gpu_site="$($gpu_python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
cmake -S "$gpu_root/opencv" -B "$gpu_root/build" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$gpu_prefix" \
    -D CMAKE_INSTALL_RPATH="$gpu_prefix/lib" \
    -D OPENCV_EXTRA_MODULES_PATH="$gpu_root/opencv_contrib/modules" \
    -D BUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,objdetect,calib3d,features2d,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping \
    -D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D CUDA_ARCH_BIN=8.7 -D CUDA_ARCH_PTX= \
    -D CMAKE_C_COMPILER=/usr/bin/gcc-13 \
    -D CMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -D CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -D WITH_CUDNN=OFF -D OPENCV_DNN_CUDA=OFF -D WITH_CUBLAS=ON \
    -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON -D WITH_V4L=ON -D WITH_GTK=ON \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE="$gpu_python" \
    -D PYTHON3_NUMPY_INCLUDE_DIRS="$gpu_numpy_include" \
    -D PYTHON3_PACKAGES_PATH="$gpu_site" \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
    -D BUILD_JAVA=OFF -D BUILD_opencv_python2=OFF
cmake --build "$gpu_root/build" --parallel 2
cmake --install "$gpu_root/build"

"$gpu_python" - <<'PY'
import cv2
import numpy as np
print('OpenCV:', cv2.__version__, cv2.__file__)
print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())
assert cv2.cuda.getCudaEnabledDeviceCount() > 0
assert hasattr(cv2, 'QRCodeDetector'), 'QR support missing'
image = np.zeros((64, 64), np.uint8)
image[20:40, 20:40] = 255
kernel = np.ones((3, 3), np.uint8)
gpu = cv2.cuda_GpuMat()
gpu.upload(image)
filter_ = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel)
actual = filter_.apply(gpu).download()
np.testing.assert_array_equal(actual, cv2.dilate(image, kernel))
print('CUDA morphology smoke PASS; project replay validation still required')
PY
