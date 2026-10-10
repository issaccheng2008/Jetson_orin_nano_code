# Jetson 开机自启与 PC2 按钮起步

当前巡线的新丢线策略与 robust 强滤波设置见 [丢线补偿与强滤波](丢线补偿与强滤波.md)。已有本地参数文件需要显式启用 robust。

heading/segments 的近端位置回正参数、默认值及回退方式见 [近端位置纠偏](position_steering.md)。已有配置缺少位置字段时使用默认值，可通过启动日志 `[steering-position]` 核对。

启动顺序是 `connector → vision → 等待 STM32 PC2 按钮 → gait`。两个 systemd 服务只启动 connector 和带 `--start-gate button --start-policy-on-gate` 的 vision；不直接启动 `main.py`。Vision 收到本次运行的新按钮事件后，才通过策略 Python 启动 gait 并等待 STM32 就绪。PC2 接线与固件按当前版本说明完成；安装服务本身不会启动电机。

## 在 Jetson 上安装

以平时运行机器人程序的普通用户登录，仓库放在 `~/jetson_orin_code`。不要给整个安装命令加 `sudo`。

```bash
cd ~/jetson_orin_code
bash scripts/install_button_autostart.sh --dry-run
bash scripts/install_button_autostart.sh
```

首次安装创建 `config/button_start.env`，默认视觉 Python 为仓库的 `.venv/bin/python`，策略 Python 为 `~/venvs/humanoid_policy/bin/python`。两个环境可以不同。安装先检查脚本、两份模型及视觉环境的 `numpy/cv2/serial`、策略环境的 `onnxruntime/serial`，再用 `systemd-analyze verify` 校验生成的服务文件，成功后才写系统服务并设为开机启用。配置生成与程序运行使用当前用户；只有系统文件安装和 `systemctl` 使用 sudo。缺依赖或路径错误时，编辑配置后重跑安装，已有配置保留。更新安装脚本后也需重跑安装；已有配置不会被覆盖。

```bash
vi config/button_start.env
bash scripts/install_button_autostart.sh --check
```

vi 的编辑、保存退出和参数生效步骤见 [参数修改与 vi 保存速查](Jetson参数修改与vi保存速查.md)。

配置使用 Bash 语法，带空格的路径要加引号。默认串口 `/dev/ttyACM0`、摄像头 0；用户须已有串口和摄像头权限（通常为 `dialout`、`video` 组；权限修改后重新登录或重启）。不使用 root 跑视觉或步态。不要把此用户可编辑配置交给不可信账户修改。

默认模型是 `humanoid_jetson_deploy/policy_49_max.onnx` 和 `policy-one-foot-standing_old.onnx`，可以分别修改 `WALKING_MODEL`、`ONE_FOOT_MODEL`。配置还保留当前视觉命令：heading 模式、vx 0.2、max-wz 0.5、wz-step 0.3、lookahead 50 cm、corridor 5 cm、右容差 9°、左容差 4°、full scale 15°、左转档位 0.3/0.4/0.5、卡片触发距离 30 cm、shape-every 2。策略单次最长运行默认 1200 秒，可修改 `POLICY_MAX_SECONDS`。UDP 端口固定 connector 输入 5006、策略输入 5005；connector 使用 `--max-vx-accel 1 --max-wz-accel 0`。

转向模式由 `WZ_MODE` 设置：`WZ_MODE=heading` 使用原航向控制，`WZ_MODE=segments` 使用分段控制（自动启用分段观测）。已有 `config/button_start.env` 不会被更新覆盖；缺少此项时默认 heading，需要切换就手动添加 `WZ_MODE=segments` 并重启 vision 服务。可用 `bash scripts/run_button_vision.sh --dry-run` 核对实际传入的模式，无需重新安装服务。

角度查表与选点距离可在 `config/button_start.env` 中添加以下配置，保存后重启 vision 服务生效：

```bash
# 每行：[起始角度°，结束角度°，输出角速度rad/s]，区间左闭右开。
STEERING_ANGLE_WZ_TABLE='[[0,10,0],[10,30,0.3],[30,90,0.3]]'
HEADING_NEAR_CM=25.07
HEADING_FAR_CM=29
SEGMENT_REGIONS_CM='[[20,32],[32,44],[44,56]]'
```

查表输入是当前最终决策角的绝对值，即滤波后的目标方位角加近端位置纠偏；正负转向和 `yaw-sign` 自动处理。例如 10°进入 0.3 rad/s 档，30°进入下一档。`STEERING_HYSTERESIS_DEG` 在边界附近保留当前相邻档位（设 0 可严格按边界切换）。启用表后，旧角度容差、full-scale、左右档位和 enter/exit 参数不参与普通选档；走廊保护、位置恢复、趋势减速、丢线处理和停车仍可改变最终命令。日志中的 `steering_combined_demand_deg`、`steering_table_wz`、`steering_applied_wz` 可分别核对输入、普通查表输出、实际输出。

角度区间需要连续覆盖 0～90°，第一档输出 0，其余输出是非负幅值且至少包含一个转向档。超过 90°的最终决策角使用最后一档。每段宽度必须大于两倍迟滞值。表中速度必须不超过 `WZ_STEP`、`MAX_WZ` 和右转上限（接口最大 0.5 rad/s）；例如使用 0.4/0.5 前先将 `WZ_STEP=0.5`。不合法配置启动时报错；表留空或缺少时保持原选档逻辑。

heading 近远端距离表示实际观测带中心，按当前相机标定换算，实际值会因像素取整略有差异。近端留空或缺少时保留原始约 25.07cm；远端默认 29cm。两带仍各扫描 8 行。设置近端时，底部锁定的扫描位置和厘米比例尺一起调整；窄门的独立参考位置保持原值。远端至少比近端远 3cm，近端须小于 `HEADING_LOOKAHEAD_CM`；完整扫描带必须落在鸟瞰图中且相隔至少 8 行，否则报错。`HEADING_LOOKAHEAD_CM=50` 是投影目标距离，与实际近远观测点不同。

segment 数量由 `SEGMENT_REGIONS_CM` 数组长度决定，支持 2～5 段，不需要另写数量。区域须由近到远连续、不重叠，每段至少 6cm，整体在 20～70cm 内，第一段要包含近端观测中心。可改成两段 `'[[20,32],[32,44]]'` 或四段 `'[[20,32],[32,44],[44,56],[56,68]]'`。只在现有广域扫描数据中分段，不增加扫描次数。原有配对、至少 5 个点、至少 4cm 实测跨度、宽度、残差、近端锚定及连续 3 帧确认要求继续有效；缺段后停止向远端连接，前两段不可靠时回退 heading。区域太窄、距离太远或近远点太接近会减少可靠观测，合法配置也可能更频繁回退；推荐每段约 10～12cm。

Python 直接启动时对应参数为 `--steering-angle-wz-table`、`--heading-near-cm`、`--heading-far-cm`、`--segment-regions-cm`。录像运行清单记录这些配置，离线回放会使用相同的表和区域。

短暂方向丢失时的命令保持由 `LOST_HOLD_S` 设置，未写时仍默认 0.2 秒；过期观测仍会取消角速度。在 heading/segments 中，`WZ_STEP=0.3` 是档位上限，会挡住配置的更高档位；改为 `0.5` 才允许现有 0.37/0.4/0.5 档位参与选择。关于转弯偏少的原因、调参顺序和逐帧统计命令，见 [转弯占空比分析与测试方案](转弯占空比分析与测试方案.md)。

## 启动、停止与检查

### 可调指令保持与视觉滤波

`COMMAND_MIN_HOLD_S` 控制步态模型入口正常指令的最短持续时间，0 关闭，0.5 恢复半秒保持。`STEERING_FILTER_MODE` 可选 legacy/shadow/active，分别为原视觉控制、只记录对照、实际使用新滤波与滞回。已有本机配置缺少这两项时默认 0 和 legacy。详细启用、调强度、换公式、查看日志及回退方法见 [policy49flitter：滤波与指令保持操作](policy49flitter_滤波与指令保持操作.md)。

### 角速度偏置

在 `config/button_start.env` 中添加或修改 `WZ_BIAS=0.1`，connector 会给每条运动指令的 `wz` 加 `0.1 rad/s` 后发给 Nano；`WZ_BIAS=-0.1` 表示减 `0.1`，`WZ_BIAS=0` 关闭补偿。已有配置未写此项时默认 0。该设置只调整角速度，不改变前进速度的处理。

前进时，视觉发 `wz=0`，偏置 `0.1` 后得到 `0.1`；`0.2` 得到 `0.3`，`-0.2` 得到 `-0.1`。加完偏置后限制在 `[-0.5, 0.5]`，所以 `0.45 + 0.1` 最终发 `0.5`。原地转向指令也会补偿；完整停车 (`vx=0, wz=0`)、读卡站稳/倾斜请求、视觉通信超时及 connector 退出时仍发零速度。偏置只在接收时加一次，50 Hz 转发不会重复累加。

修改后需重启 connector。它重启时会停止依赖它的 vision 和步态进程，因此再启动 vision，并重新按 PC2：

```bash
bash scripts/run_button_connector.sh --dry-run # 确认输出含 --wz-bias 0.1
sudo systemctl restart humanoid-button-connector.service
sudo systemctl start humanoid-button-vision.service
```

手动运行时，在原 connector 命令后增加参数，例如：

```bash
python -u connector.py --vision-port 5006 --policy-port 5005 --max-wz-accel 0 --wz-bias 0.1
```

启动日志会显示 `wz_bias=+0.1 rad/s`。这是新增的偏置参数；视觉的 `heading-left-wz` / `heading-right-wz` 档位校验仍要求正数、递增且不超过 0.5，偏置不能解决不合法档位导致的启动错误。

首次安装只启用下次开机启动。台架准备好后，可以手动启动 vision，systemd 会先启动 connector：

```bash
sudo systemctl start humanoid-button-vision.service
systemctl status humanoid-button-connector.service humanoid-button-vision.service
journalctl -u humanoid-button-connector.service -u humanoid-button-vision.service -b -f
```

确认日志显示等待按钮，且按按钮前没有 C 步态进程或上位机运动 COMMAND；再按 PC2 验证 gait 就绪后起步。STM32 原有上电电机初始化及 `Action_Goto` 站姿仍会执行，按钮只控制上位机步态程序的启动。避免同时运行原来的手动 connector、vision 或策略命令，否则 UDP、摄像头或串口会冲突。

```bash
# 停止视觉及其 gait 子进程，再停止 connector
sudo systemctl stop humanoid-button-vision.service humanoid-button-connector.service
# 修改配置后重新启动；每次重新启动都要重新按按钮
sudo systemctl restart humanoid-button-vision.service
# 关闭开机启动并停止当前运行
sudo systemctl disable --now humanoid-button-vision.service humanoid-button-connector.service
# 恢复开机启动；当前运行仍需手动 start 或等待重启
sudo systemctl enable humanoid-button-connector.service humanoid-button-vision.service
```

Vision 异常退出后，systemd 在 5 秒后重启并重新等待本次运行的新按钮事件；不会自动恢复步态。正常结束（包括 gait 到达运行时限）不自动重启。服务停止使用 `KillMode=control-group`、先发 SIGINT，20 秒后仍未退出则由 systemd 强制清理，包括 gait 子进程。Connector 停止会连带停止依赖它的 vision；恢复后用 `sudo systemctl start humanoid-button-vision.service` 重新进入等待按钮状态。

起点首卡已经识别并锁存时，按按钮启动后先直行，再停车执行首卡。原来写死为 1.0 秒，现在默认 0.5 秒；在 `config/button_start.env` 添加 `STARTUP_FIRST_WALK_S=0.3` 可改成 0.3 秒（须大于 0），`1.0` 恢复原时长。它与 `COMMAND_MIN_HOLD_S` 独立，停车执行动作仍会打断指令保持。未识别首卡时按按钮直接进入普通巡线，不使用这段首卡直行。手动运行参数为 `--startup-first-walk-s 0.3`。修改后重启视觉服务；没有此字段的旧按钮配置也默认使用 0.5 秒。计时从首个直行帧开始，在后续视觉帧检查结束，因此实际发送时长受处理周期及耗时波动影响，可能与配置值有一个周期左右偏差。

需要组合指令时，在同一配置文件中增加 `STARTUP_SEQUENCE`。详见 [启动指令序列](启动指令序列.md)：每段分别指定时间、VX、WZ，序列结束再停车执行首卡；非空序列优先于 `STARTUP_FIRST_WALK_S`。

相机曝光和画质也在同一配置文件调整。固定 5 ms 使用 `CAMERA_EXPOSURE_MODE=manual`、`CAMERA_EXPOSURE_MS=5`；亮度、对比度、饱和度、锐度、白平衡和抗闪烁设置详见 [相机曝光与画质参数](相机曝光与画质参数.md)。默认 keep/空不会改变现有相机状态。

每次测试统一写入 `records/tests/日期/test_时间_唯一编号/`，包括视觉、控制指令、关节角、IMU 和图片；详见 [测试数据每日归档](测试数据每日归档.md)。查看本次启动全部日志：

```bash
journalctl -u humanoid-button-connector.service -u humanoid-button-vision.service -b --no-pager
bash scripts/run_button_connector.sh --dry-run
bash scripts/run_button_vision.sh --dry-run
```

`--dry-run` 只显示配置、服务或命令，不创建配置、不安装服务、不运行 Python、不打开硬件；`--check` 会运行依赖导入检查，但不启动服务或策略。Windows 工作区仅验证脚本语法和 dry-run 输出，尚未在 Jetson 上部署，也未验证摄像头、串口、按钮或电机。
