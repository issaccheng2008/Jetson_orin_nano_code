# Nano GPU 环境：实际状态、后续安装与验证

核实日期：2026-10-10；板子 `isaac@192.168.1.131`。全程使用新环境，保留项目 `.venv`、`humanoid_policy` 环境及系统 OpenCV。

## 已确认的状态

| 项目 | 现场结果 |
| --- | --- |
| 系统 | Orin Nano，L4T 39.2，Python 3.12，CUDA 13.2，SM 8.7 |
| 新环境 | `/home/isaac/venvs/robot_gpu_20261010` |
| PyTorch | `2.12.1+cu132` 已安装；GPU 矩阵乘法、卷积、同步取回和 NumPy 桥接通过 |
| PyTorch 限制 | 初始化明确警告 wheel 支持列表排除 SM 8.7。基础测试通过不代表全部算子兼容，不作为正式控制环境直接切换依据 |
| OpenCV | 新环境目前仍继承系统 `4.6.0`，路径 `/usr/lib/python3/dist-packages/cv2.cpython-312-aarch64-linux-gnu.so`；CUDA 设备数为 0 |
| CUDA OpenCV | **尚未安装**；以下编译命令尚未在板上完成验证 |
| TensorRT | 系统已有 10.16；新增模型后端尚未完成实板验证 |

因此，不能说所有依赖已经安装完成。Torch 不需要重复下载；接下来先安装 CUDA OpenCV。

## 1. 登录并检查新环境

```bash
ssh isaac@192.168.1.131
source /home/isaac/venvs/robot_gpu_20261010/bin/activate
python -m pip show torch
python -m pip check
python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('available:', torch.cuda.is_available())
x = torch.ones((32, 32), device='cuda')
y = x @ x
torch.cuda.synchronize()
assert torch.all(y == 32).item()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('GPU matmul PASS')
PY
```

SM 8.7 警告需保留。当前无需为它卸载原有环境；后续用实际模型完整验证，必要时另建明确包含 SM 8.7 的 PyTorch 源码构建。现有 ONNX 模型的 TensorRT 路径并不依赖 PyTorch。

## 2. 安装 OpenCV 构建依赖

在板子执行以下安装；没有卸载或清理旧包的步骤：

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
  build-essential cmake git pkg-config python3-dev \
  libjpeg-dev libpng-dev libtiff-dev libgtk-3-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

不要用 `pip install opencv-python` 代替下面的 CUDA 源码构建。该命令不能完成这里要求的 CUDA OpenCV 安装。

## 3. 上传脚本，在独立目录编译

脚本位于本地仓库 `scripts/install_opencv_cuda_20261010.sh`。从本地 WSL 的仓库根目录上传，无需切换或覆盖板上的工作分支：

```bash
ssh isaac@192.168.1.131 'mkdir -p /home/isaac/gpu_setup_20261010'
scp scripts/install_opencv_cuda_20261010.sh \
  isaac@192.168.1.131:/home/isaac/gpu_setup_20261010/
```

然后在板子启动一次后台编译，并实时看日志：

```bash
mkdir -p /home/isaac/records/gpu_setup_20261010
nohup bash /home/isaac/gpu_setup_20261010/install_opencv_cuda_20261010.sh \
  > /home/isaac/records/gpu_setup_20261010/opencv_build.log 2>&1 &
echo $!
tail -f /home/isaac/records/gpu_setup_20261010/opencv_build.log
```

`Ctrl+C` 退出的是 `tail`，后台编译继续。不要同时启动第二份编译。

源码是 OpenCV/contrib 4.14.0 的固定提交；脚本检查源码提交及工作区，遇到已有不同源码时退出，不重置。采用 SM 8.7、两线程编译，保留二维码、相机、视频和 GUI 模块。目录为：

- 源码和构建：`/home/isaac/src/robot_gpu_20261010/`
- OpenCV 安装：`/home/isaac/opt/opencv-cuda-4.14.0-20261010/`
- Python 绑定：只安装进 `/home/isaac/venvs/robot_gpu_20261010/`

成功时日志最后显示 `CUDA morphology smoke PASS`。如果报错，应从最早的编译错误处理；不能把下载完成或 CMake 配置成功当成安装成功。

## 4. 验证导入的确是新 OpenCV

```bash
/home/isaac/venvs/robot_gpu_20261010/bin/python - <<'PY'
import cv2
print('version:', cv2.__version__)
print('path:', cv2.__file__)
print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())
print('morphology:', hasattr(cv2.cuda, 'createMorphologyFilter'))
print(cv2.getBuildInformation())
PY
```

要求版本为 4.14、导入路径属于新环境/独立安装目录、CUDA 设备数大于 0、形态学工厂存在，并且第 3 步 GPU 实际运算成功。还应检查构建信息中的 GStreamer、FFmpeg、V4L 和 GUI 与原使用方式相符。

验证旧环境仍能导入原 OpenCV：

```bash
/home/isaac/jetson_orin_code/.venv/bin/python -c \
  'import cv2; print(cv2.__version__, cv2.__file__)'
```

## 5. 用实拍离线验证算子改写

以下需要包含新版 `cuda_vision.py` 和 `benchmark_vision_devices.py` 的代码快照。当前不能假设板上原工作区已经拥有这些文件；应上传独立代码目录后，在该目录执行。`/path/to/...` 必须换成实际目录。

```bash
source /home/isaac/venvs/robot_gpu_20261010/bin/activate
cd /path/to/isolated-code
python -c 'from new_vision.jetson.cuda_vision import create_vision_backend; print(create_vision_backend("cuda").diagnostics())'
python scripts/benchmark_vision_devices.py \
  --input-dir /path/to/real-photos-or-videos \
  --output-dir /home/isaac/records/gpu_setup_20261010/vision_report \
  --max-images 20 --repeats 5 --warmup 1 --require-cuda
```

脚本默认巡线 `legacy`；若实际比赛配置是 `contrast`，加 `--line-preprocess contrast`。报告检查二值像素、图卡种类、模糊 presence、巡线几何及耗时。CUDA 调用必须真实发生；有舍入差异的灰度/缩放/透视算子可能明确保留 CPU。所有上传、下载和同步耗时都要计入比较。

这些检查不连接串口、不使能电机。验证通过后，终端 B 才使用新 Python，并在原视觉命令增加 `--vision-device cuda`。终端 C 可继续使用现有 CPU 模型环境，不需要同时切换。

## 6. 模型 GPU 是另一步

在同一独立代码目录、具备 CPU ONNX Runtime 和系统 TensorRT 的环境中，先离线构建和对照：

```bash
mkdir -p /home/isaac/records/gpu_setup_20261010/engines
python humanoid_jetson_deploy/build_policy_engine.py \
  --model /home/isaac/jetson_orin_code/humanoid_jetson_deploy/policy_49_v22.onnx \
  --engine /home/isaac/records/gpu_setup_20261010/engines/policy49.engine \
  --obs-dim 49
python humanoid_jetson_deploy/benchmark_policy_backends.py \
  --model /home/isaac/jetson_orin_code/humanoid_jetson_deploy/policy_49_v22.onnx \
  --engine /home/isaac/records/gpu_setup_20261010/engines/policy49.engine \
  --synthetic-smoke \
  --out /home/isaac/records/gpu_setup_20261010/policy_benchmark.json
```

`policy_49_v22.onnx` 是板上实际存在的候选文件，不代表已确认是当前比赛模型；需要与实际启动命令对应。新 GPU 环境尚未核验 ONNX Runtime 导入，执行前检查 `python -c 'import onnxruntime, tensorrt; print(onnxruntime.__version__, tensorrt.__version__)'`。缺失时先补该独立环境依赖，不能直接运行主程序来试。

`--synthetic-smoke` 是明确标注的合成输入，仅检查数值和性能。拿到真实模型观测 CSV 后改用 `--obs-csv 文件`。小模型 GPU 可能比 CPU 慢，最终以后端对照报告决定，不默认启用。
