# Orin / R39.2 独立 GPU 环境兼容性核对

核对日期：2026-10-10。只读查询官方索引、wheel 元数据、源码和 GitHub 合并记录；没有安装软件、改板端或修改视觉运行代码。板端 R39.2 / Ubuntu24.04 / Python3.12 / CUDA13.2 / cuDNN9.20 / TensorRT10.16 来自主任务现场记录，本研究没有再次连接板端确认。

## 可冻结的组合

| 部件 | 冻结建议 | 已确认的依据 | 仍需板端确认 |
| --- | --- | --- | --- |
| GPU | Orin，计算能力 8.7 / SM87 | NVIDIA GPU 表列出 Orin Nano / NX / AGX 为 8.7 | 设备确实是该板，驱动可用 |
| Python / 平台 | CPython3.12，aarch64，glibc >=2.28 | Torch wheel 是 cp312 / manylinux_2_28_aarch64 | 实际解释器和 glibc |
| PyTorch | `2.12.1+cu132` | 官方 CUDA13.2 索引提供真实 wheel；对应 CI 构建保留 SM80 | CPU 后端、CUDA 张量和 cuDNN 算子执行 |
| CUDA Python bindings | `13.2.0` | PyPI 有 cp312 aarch64 wheel；满足 Torch 的 >=13.0.3,<14 | 动态库发现和驱动加载 |
| NumPy | `2.2.6`，在新独立环境冻结 | 官方发行有 cp312 manylinux2014_aarch64 wheel；Python >=3.10 | Torch NumPy 桥接、OpenCV 编译及导入 |
| OpenCV | `4.14.0`，源码编译 | 包含 CUDA13.2 的 contrib namespace 修复及 CUDA 检测修复 | CUDA 构建、GPU 设备计数和真实算子 |
| opencv_contrib | 同为 `4.14.0` | 与核心 tag 配对；修复提交是 tag 的祖先 | cudev / CUDA 模块实际编译 |
| TensorRT | 现有 `10.16` | 本建议的 Torch 与最小 OpenCV 不使用 TRT | 如后续加 Torch-TensorRT，另作兼容核对 |

GPU 依据：[NVIDIA CUDA GPU 表](https://developer.nvidia.com/cuda-gpus)。NumPy 版本是隔离环境的建议冻结值，不是 Torch wheel 的强制依赖；该 Torch METADATA 没有 NumPy 约束。[NumPy2.2.6 官方发行元数据](https://pypi.org/pypi/numpy/2.2.6/json)。

## PyTorch 真实索引、文件与依赖

官方索引：[CUDA13.2 Torch wheel 列表](https://download.pytorch.org/whl/cu132/torch/)。

建议冻结文件：

```text
https://download.pytorch.org/whl/cu132/torch-2.12.1%2Bcu132-cp312-cp312-manylinux_2_28_aarch64.whl
SHA256: 8139af68deafa649e0f0313684791e93ab4269ecf550535284dbfb549fdac41a
HEAD: 200
Content-Length: 434571466 bytes
官方索引上传时间: 2026-06-18T02:55:38Z
```

[该 wheel 的官方 METADATA](https://download.pytorch.org/whl/cu132/torch-2.12.1%2Bcu132-cp312-cp312-manylinux_2_28_aarch64.whl.metadata) 已读到下列精确 CUDA 依赖：

```text
cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.2.1
cuda-bindings>=13.0.3,<14
nvidia-cudnn-cu13==9.20.0.48
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvshmem-cu13==3.4.5
triton==3.7.1
```

另需 `filelock`、`typing-extensions>=4.10.0`、`setuptools<82`、`sympy>=1.13.3`、`networkx>=2.5.1`、`jinja2`、`fsspec>=0.8.5`。cuDNN/cuSparseLt/NCCL/NVSHMEM 有 aarch64 wheel，Triton3.7.1 与 cuda-bindings13.2.0 有 cp312 aarch64 wheel，cuda-toolkit 是纯 Python 元包并按平台声明各 CUDA 库。`cuda-bindings==13.2.1` 不存在，不能按 toolkit 的同名版本硬凑。

直接文件 URL 已 HEAD 核对成功。索引 href 的 `download-r2.pytorch.org` 变体 `.metadata` 本次返回403，而上面 `download.pytorch.org` 的文件与 `.metadata` 正常；若 pip 在镜像元数据路径失败，可使用已核对的直接文件及 SHA。不能把镜像403解释成不存在 ARM wheel。

SM87 兼容是有官方依据的推断，尚不是这块板上的执行结果：PyTorch v2.12.1 [manywheel 构建脚本](https://github.com/pytorch/pytorch/blob/v2.12.1/.ci/manywheel/build_cuda.sh) 的 CUDA13.2 / aarch64 列表为 `8.0;9.0;10.0;11.0;12.0`，保留 SM80。NVIDIA [Ampere 兼容性说明](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html) 规定同 major 的 cubin 可运行于相同或更高 minor，因此 SM80 cubin 可以在 SM87 执行；`torch.cuda.get_arch_list()` 没有字面 `sm_87` 不能单独作为不兼容判据。CUDA/cuDNN/CPU 后端的实际加载仍须测试。

这条路线使用 PyTorch 官方通用 ARM wheel。NVIDIA [Jetson PyTorch 发布矩阵](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html) 在 25.10–26.09 的 standalone wheel 栏均为 `-`，当前列出的 JetPack 为7.0/7.1；没有查到可诚实推荐的 R39.2 / cp312 / CUDA13.2 Jetson 专用 wheel 链接。不能照旧 v61/v62 安装链接猜一个 v7*/v39 链接。

## OpenCV 确切源码与已合并修复

两仓库配对 tag `4.14.0`，固定 **commit**，不是 annotated tag object：

```text
opencv:         0654a42e19215ef25b1d367d822f3c630447e7c7
opencv_contrib: a8e9acd62cabd30419dba83007f2ac0d07de5e2c
```

官方 GitHub compare 核对：核心 tag 相对修复是 ahead233 / behind0；contrib tag 是 ahead23 / behind0，均包含相应修复。不能只因 tag 日期较新就假设包含。

| 修复 | 合并提交 | 实际修复范围 |
| --- | --- | --- |
| [opencv_contrib #4097](https://github.com/opencv/opencv_contrib/pull/4097) | `f2854f4f5e7b67d4e073ea002ae0174d437e2962`，2026-03-30 | cudev `ptr2d/zip.hpp` 在 CUDA>=13.2 使用 `_CCCL_BEGIN/END_NAMESPACE_CUDA_STD`，替代旧 libcudacxx 宏 |
| [opencv #28954](https://github.com/opencv/opencv/pull/28954) | `4b031345f0c49b183d5e2b849188572984d44e10`，2026-05-14 | arch 自动检测为空时给出有意义错误，并传递 `CMAKE_CUDA_HOST_COMPILER`；不是 GPU 算子或不受支持 GCC 的通用修复 |

建议最小模块列表：

```text
core,imgproc,imgcodecs,videoio,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping
```

需要现场 `imshow` 时在列表加 `highgui` 并使用实际可用 GTK/Qt；无显示需求才关闭 GUI。暂不纳入 dnn、cudacodec、Java、Python2、样例、测试和性能测试。它们不是当前图像预处理需要的模块。

必要 CMake 参数（由主任务按独立 prefix / venv 路径填充）：

```text
CMAKE_BUILD_TYPE=Release
OPENCV_EXTRA_MODULES_PATH=<配对contrib>/modules
BUILD_LIST=<上面的模块列表>
WITH_CUDA=ON
CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.2
CUDA_ARCH_BIN=8.7
CUDA_ARCH_PTX=
CUDA_HOST_COMPILER=/usr/bin/g++-13
CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
CMAKE_C_COMPILER=/usr/bin/gcc-13
CMAKE_CXX_COMPILER=/usr/bin/g++-13
WITH_CUDNN=OFF
OPENCV_DNN_CUDA=OFF
WITH_CUBLAS=ON
WITH_FFMPEG=ON
WITH_GSTREAMER=ON
WITH_V4L=ON
BUILD_opencv_python3=ON
PYTHON3_EXECUTABLE=<独立venv>/bin/python
PYTHON3_NUMPY_INCLUDE_DIRS=<该venv的numpy.get_include()>
PYTHON3_PACKAGES_PATH=<该venv的site-packages>
BUILD_TESTS=OFF
BUILD_PERF_TESTS=OFF
BUILD_EXAMPLES=OFF
BUILD_JAVA=OFF
BUILD_opencv_python2=OFF
```

使用 legacy FindCUDA 路径时重点是 `CUDA_ARCH_BIN=8.7`。若明确启用 first-class CUDA language，则同时设置 `CMAKE_CUDA_ARCHITECTURES=87` 并核对最终 nvcc 命令；不要只设置一个不会被所选路径消费的变量。GCC13 位于 [CUDA13.2 Update1 官方支持范围](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-installation-guide-linux/index.html) 的 Arm64 GCC6–15 之内，不需要用忽略编译器检查来绕过。

## 仍需满足的验收条件

1. 板端确认 aarch64 / Python3.12 / GPU `(8,7)`，驱动能执行现有 CUDA13.2；没有核对到厂商给 R39.2 的专用 Torch wheel，不把通用包称为已获 Jetson 专用认证。
2. 在独立环境完整解析依赖，记录版本与 hash；Torch METADATA 的 CUDA 库不能仅凭系统目录同样写13.2就用 `--no-deps` 跳过。该方案不会要求升级 BSP/驱动。
3. Torch 执行 CPU 张量运算、NumPy 双向桥接、CUDA elementwise/matmul、cuDNN convolution，并 `torch.cuda.synchronize()` 读取错误。导入成功或 `is_available()` 单独不足以验收；实际部署模型还要做无执行器输出的离线加载/推理。
4. OpenCV build info 必须实际包含 CUDA 与所需模块；Python 导入路径必须属于独立 prefix。检查 GPU device count、上传/下载、cvtColor、resize、Gaussian、阈值/形态学等需要的 GPU API和 CPU 对照。需要的 Python API 不存在时不能把 C++ 模块构建成功视为可用。
5. 编译需可用的 GCC13、Python3.12 开发头、NumPy头、CUDA13.2 headers/NPP/CUBLAS、相应摄像/视频开发库、足够存储和可控并行度。缺库时先定位实际缺失依赖；本研究未做安装或全模块编译，因此不能声称 CUDA13.2 下全部 OpenCV 模块都已验证。

研究结论：存在真实官方 cp312+aarch64 CUDA13.2 Torch 路线，SM80→SM87有官方兼容依据；OpenCV4.14配对tag已包含指定两修复。最小构建与板端算子/模型执行检查仍是完成安装的必要步骤。
