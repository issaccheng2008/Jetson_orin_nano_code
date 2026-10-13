# 分支差异与双摄像头合并（2026-10-13）

## 1. c7ad975 是共同基线

`c7ad975224e9714c8b862d0cb860788cc6c7e597` 在两条分支历史中都存在，同一提交没有两种代码版本。该提交时间为 2026-10-09 20:32:43 +0800，只更新文档和计划记录，没有修改运行代码。

之后首先出现的不同改动：

- flitter：`fa9793b`（21:09）直走补偿 wz=0.1；`7016d44`（21:55）将远处航向观测改为可配置的 29 cm。直走补偿后来又改回零，不能用这次历史改动代表当前默认。
- VI：`dd8228a`（22:07）在原巡线预处理上对 adaptive C 做归一化，随后加入连续边界追踪、度量鸟瞰等。
- 后续有选择性同步和回退，两条分支不是从此完全独立；例如 `44cf839` 的阶段时序/亮度匹配同步也进入了 VI。

下面比较的是本次通过 SSH 获取的远端：
`policy49flitter=f046a0c`，`policy49flitter-vi=0834aed`。
本地 flitter 引用原先停在 `3139bbf`，因此没有把它当作远端最新版。

## 2. 当前主要功能差异

| 环节 | policy49flitter | policy49flitter-vi |
|---|---|---|
| 鸟瞰几何 | 320×400 旧鸟瞰，地面坐标 LUT 随行变化，像素横纵比例不同 | X/Z 使用相同 cm/px 的度量 IPM，向原相机底部射线延伸近端，处理有效地面掩膜；像素尺寸由几何和栅格构造确定 |
| 路径来源 | 原扫描带和配对几何 | 增加连续边界追踪、左右身份管理及持续单边法向偏移目标路径，提供统一位置与航向 |
| 光度处理 | 原均值/标准差图像匹配及旧阈值归一化语义 | raw、threshold、受保护 match、标定 LUT 互斥；legacy 对应 match，normalize 对应 LUT，匹配不叠加阈值补偿 |
| 预处理配置模板 | LINE_PREPROCESS=contrast | LINE_PREPROCESS=legacy、LINE_ADAPTIVE_C=-12；contrast 路径另有细线过滤。模板不等于板子当前实际配置 |
| 图卡检测 | 较早的检测和分类链 | 独立存在性/局部灰度辅助处理；原灰度分类失败才回退旧分类；起点投票及 DONE+0.1 秒后的限时投票等修正 |
| 图卡 3/4 动作 | Nano 调用 OneFootPolicy；新增全部 12 关节目标幅值倍率，默认配置 0.9，并分别记录目标/反馈角度 | 视觉图卡 3/4 请求 STM32 固定动作；视觉分支不调用 one-foot 模型。独立 one-foot 运行模式仍保留 |
| 通信和诊断 | 原对应实现 | connector 延长低头请求的失联宽限；可读的四类证据计数；修正过期宽度对单边的否决，以及右单边强左转的几何条件 |
| CPU 处理 | 旧对应实现 | 增加 CPU 算子优化、最新帧异步读取及相关诊断 |

它们的共同控制框架、模型输入/串口协议仍有共享历史。不要只按分支名字推断板子正在使用哪种配置。

## 3. 本次合并

来源：`card_second_camera_20261012=f112344`，基于 VI 的 `0834aed`。
采用保留两个父提交的普通 merge，合入：

- `policy49flitter-vi-gap-straight_20261012`，原基线 `cd71c88`。
- `lane_loss_recovery_20261012`，原基线 `8e39d71`。

冲突发生在入口配置校验、源文件哈希清单、启动脚本和配置末尾；均保留目标分支巡线设置，同时接入副相机设置。
保留 gap-straight 的断线连接、严格直道判定和按配置档位限制。
保留 lane 的 `LANE_STRICTNESS_PCT=50`、`LANE_VISIBLE_DIRECTION_FALLBACK=1`、断线连接和现有直道保护。
未从 flitter 另行合入 Nano 抬腿模型和关节缩放功能。

## 4. 两台相机如何工作

- 主相机：巡线、运动中图卡接近/停车判定、二维码和原录像。
- 起点等待图卡及正常停车精准投票：主副相机都分类，每个检测周期只形成一票。
- 两者一致：使用共同类别；仅一者有类别：使用该类别；两者冲突：弃本次融合票，起点重置冲突证据。
- 副相机仅在起点图卡门控或停车精准识别准备阶段打开，离开相应阶段释放；普通行走不常开副相机。
- 正常停车投票使用 DONE+稳定等待之后捕获的新副相机帧；启动图卡没有低头 DONE，使用启动阶段新帧。
- 副相机默认 raw，可独立配置安装高度、俯角、视场角，不能直接假定主副标定相同。

模板默认启用副相机，稳定名称配置为：

```bash
CAMERA_DEVICE='/dev/robot_main_camera'
CARD_CAMERA_DEVICE='/dev/robot_card_camera'
CARD_CAMERA=1
```

规则 `config/udev/99-robot-cameras.rules` 将主 USB 2.1 绑定为 robot_main_camera，副 USB 2.3 绑定为 robot_card_camera。
两相机序列号相同，按已确认的固定 USB 接口命名，不修改硬件序列号。
模板不会覆盖现有按钮配置。稳定名称需要先安装 udev 规则；未配置稳定路径时，数字 0/1 只是兼容选择，不保证插拔后角色。
关闭副相机用 `CARD_CAMERA=-1`；手动入口用 `--card-camera -1`。
配置或打开副相机失败明确报错，不会静默交换设备。

安装说明参见 [dual_camera_devices.md](dual_camera_devices.md)。本次未在板子安装规则、更新配置或重启服务。

## 5. 验证结果

- gap-straight：87 项相关测试通过。
- lane：119 项相关测试通过。
- 覆盖双相机类别融合、默认开关、稳定路径/同设备拒绝、帧时间、起点与停车实际入口路由、单相机兼容、按钮启动和目标分支巡线回归。
- 原单相机按钮测试只模拟一个检测器，已显式添加 `--card-camera -1`；双相机专用测试仍检查默认开启。
- Python 编译、Bash 语法、git diff 空白检查通过。
- 尚未实机验证；本次没有打开串口、启动电机或部署板子。

测试命令（在各自工作树，使用已有 OpenCV 依赖）：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=/tmp/robocup_photo_deps:tests
python -m unittest test_card_vote_camera test_start_gate_vote \
  test_button_start test_button_autostart \
  test_policy_vision.VisionEntryPointTests \
  test_policy_vision.StartGateTests test_policy_vision.StartGateIntegrationTests \
  test_straight_road_guard test_vi_gap_completion -q
```

lane 将最后的 `test_vi_gap_completion` 换为：
`test_lane_strictness test_lane_recovery test_continuous_lane_heading test_sustained_boundary`。
