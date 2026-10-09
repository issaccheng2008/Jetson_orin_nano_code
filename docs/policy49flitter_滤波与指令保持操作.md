# policy49flitter：滤波与指令保持操作

## 先改哪里

按钮自启动使用 `config/button_start.env`。这是本机配置，不随 git pull 覆盖；已有配置缺少新参数时，默认仍是不保持指令、不启用新滤波。

在 Jetson 拉取分支：

```bash
cd ~/jetson_orin_code
git fetch origin
git switch policy49flitter
git pull --ff-only
```

如尚无本地分支，使用 `git switch --track origin/policy49flitter`。

编辑配置：

```bash
vi config/button_start.env
```

按 `i` 编辑，完成后按 `Esc`，输入 `:wq` 并回车保存。

## 每条指令至少持续多久

```bash
COMMAND_MIN_HOLD_S=0.5
```

- `0.5`：正常 walking 模型实际使用的 `(vx, vy, wz)` 至少保持半秒。
- `0.2`：至少保持 0.2 秒。
- `0`：关闭最短保持，使用最新请求。

此参数与滤波开关独立。它只在步态模型入口执行，视觉仍逐帧计算和发送，没有第二层固定保持。到期后采用最新请求，不排队执行中间积压指令；重复相同指令不会续期。

全零停止、通信超时产生的停止、图卡动作接管、单脚模型接管、故障及退出仍可以打断保持。普通 `vx>0,wz=0` 是直走指令，不是全零停止，仍受保持时间约束。视觉失线只取消 WZ 而仍保留 VX 时，也属于普通指令更新，可能在步态端等待当前保持期结束。

0.5 秒保持能限制普通指令更新频率，但也可能延迟加大转向、减小转向及正常换向。它与滤波分别调节，不要将长保持造成的入弯延迟归因于滤波。

## 滤波开关

建议先采集对照日志：

```bash
STEERING_FILTER_MODE=shadow
STEERING_FILTER_ALGORITHM=one-euro
```

三种模式：

| 值 | 实际执行 | 用途 |
|---|---|---|
| `legacy` | 原视觉控制 | 默认值、回退 |
| `shadow` | 原视觉控制 | 新方案独立计算，只写对照日志 |
| `active` | 新滤波与滞回 | 实机试验 |

准备实际启用时改为：

```bash
STEERING_FILTER_MODE=active
```

只适用于 `WZ_MODE=heading` 或 `WZ_MODE=segments`。保持当前 WZ_MODE、速度、档位、WZ_BIAS、模型及 KP/KD，先只比较滤波开关。若想先复现旧半秒保持，设置 `COMMAND_MIN_HOLD_S=0.5`；若要隔离滤波的影响，则保持这个参数在两轮测试中相同。

## 初始可调参数

```bash
STEERING_FILTER_MIN_HZ=1.5
STEERING_FILTER_MAX_HZ=4
STEERING_FILTER_BETA=0.03
STEERING_FILTER_DERIVATIVE_HZ=1
STEERING_FILTER_POSITION_TAU_S=0.1
STEERING_HYSTERESIS_DEG=1
STEERING_ENTER_DEG=2
STEERING_EXIT_DEG=1
```

常用调整方法：

| 想改变什么 | 修改哪项 | 影响 |
|---|---|---|
| 小波动仍太多 | 降低 `STEERING_FILTER_MIN_HZ`，例如 1.5 → 1.0 | 平滑更强，也会增加延迟 |
| 真实入弯跟随偏慢 | 增大 `STEERING_FILTER_BETA`，例如 0.03 → 0.05 | 快速变化时更跟手，也更容易跟随尖峰 |
| 横向偏移抖动太大 | 增大 `STEERING_FILTER_POSITION_TAU_S` | 位置变化更平滑、响应更慢 |
| 同方向档位来回跳 | 增大 `STEERING_HYSTERESIS_DEG` | 升档需更大需求，降档需更小需求 |
| 小角度频繁直行/转向切换 | 调整 `STEERING_ENTER_DEG` / `STEERING_EXIT_DEG` | 普通目标跟随的进入/退出门槛；既有边界保护可覆盖 |

这些是未完成实机验证的初始参数，先一次调整一项。阈值使用综合目标需求角，不是终端里的像素偏差角 `ang`。

One Euro 会依据观测变化速度在最低和最高截止频率之间调整。截止频率并不能恢复低帧率下没有采集到的细节。

同方向档位滞回门槛由现有档位、full-scale、上限及容忍区自动计算；滞回设置太大导致相邻门槛重叠时启动会报配置错误，减小它或拉开档位间隔即可。

## 只试其中一部分或更换公式

只试档位滞回、保留原三样本中值处理：

```bash
STEERING_FILTER_MODE=active
STEERING_FILTER_ALGORITHM=none
```

只试新滤波，不增加新滞回和新直行/转向门槛：

```bash
STEERING_FILTER_MODE=active
STEERING_FILTER_ALGORITHM=one-euro
STEERING_HYSTERESIS_DEG=0
STEERING_ENTER_DEG=0
STEERING_EXIT_DEG=0
```

更换为普通固定截止频率低通：

```bash
STEERING_FILTER_ALGORITHM=ema
```

EMA 使用 `STEERING_FILTER_MIN_HZ` 调整角度滤波强度，忽略 beta、最高及导数截止频率；位置仍由 POSITION_TAU_S 控制。

未来添加新公式主要修改 `new_vision/jetson/steering_filter.py`：实现 `update(value, dt)`，在 `make_angle_filter()` 注册并扩展配置选项。模块仅使用 Python 标准库，不增加相机、模型、串口或网络依赖。

## 保存、检查与启动

先检查传入的参数：

```bash
bash scripts/run_button_vision.sh --dry-run
```

确认输出中有 `--command-min-hold-s 0.5` 和选定的 `--steering-filter-mode`。然后重启视觉服务；它会重新启动由自己管理的步态程序，按钮起跑流程沿用原流程：

```bash
sudo systemctl restart humanoid-button-vision.service
```

另开终端看视觉日志：

```bash
sudo journalctl -u humanoid-button-vision.service -f -o short-precise
```

步态启动后另看实际入口的设置：

```bash
tail -f records/main_live.log
```

应有 `Walking command minimum hold: 0.5 s at model input`。手动单独启动步态时，需要自己在 main.py 命令后加 `--command-min-hold-s 0.5`；视觉进程的参数只会传给它自动启动的子进程，不会更改单独运行的步态进程。

退回原视觉控制：设 `STEERING_FILTER_MODE=legacy`，重启视觉服务。若连新增的半秒保持也要关闭，同时设 `COMMAND_MIN_HOLD_S=0`。此次无需重新烧录下位机。

## 日志和离线回放

`line_frames.jsonl` 保留原始观测，并增加：

- `steering_filter_heading_deg` / `steering_filter_demand_deg` / `steering_filter_near_cm`：实际用于新决策的处理值。
- `steering_filter_shadow_wz`：shadow 新方案对照指令，实际发送的仍是顶层 `wz`。
- `steering_braked`、`steering_decision`：现有预测刹车和位置保护的影响。

shadow 模式中新滤波诊断位于 `steering_filter_shadow_` 前缀下；停止或无有效几何时不强行填入零值作为观测。

步态 CSV 的 `requested_cmd_wz` 是接收到的请求，`cmd_wz` 是经过最短保持和本地接管后实际使用的指令；manifest 记录实际保持时间。因此判断半秒保持是否生效，检查 cmd_wz，不只看视觉日志。

在有 NumPy 的视觉 Python 环境中执行：

```bash
python new_vision/jetson/replay_steering_filter.py \
  records/line_telemetry/你的运行目录/line_frames.jsonl \
  --output-dir records/filter_replay
```

默认读取 JSONL 同目录的 `run_manifest.json`；文件分开存放时用 `--manifest 路径` 指定。回放输出四份 CSV 和 summary.json，分别比较原方案、仅滞回、仅滤波、两者结合。可以通过同名 `--steering-filter-*` 参数试新数值。安装了 matplotlib 时加 `--plot` 输出 comparison.png；在线控制不需要 matplotlib。

回放需要 legacy/shadow 记录。它不启动硬件、不预测新轨迹、不模拟 connector 的偏置或通信超时，也不模拟模型入口的最短保持。legacy 与原记录不一致时会报错，先核对来源版本与参数，再解释其他方案。精确控制保持时长仍看实机控制 CSV。

滤波继续保留既有的预测刹车和位置释放判断，因此不保证转向比例增加，也不保证减少出界。新方案是否合适，需要在相同设置下对照实机结果。

## 本次验证

Windows 离线环境中，视觉相关 347 项测试、步态相关 172 项测试通过。包括参数传递、模型入口保持时间、segments 目标保持、滤波复位、shadow 不改变实际输出、按钮启动脚本和回放工具。未进行 Jetson 实机测试。

使用此前 20:51 的 523 帧记录回放，legacy 与原记录逐帧一致。原方案切换 114 次，仅滞回 111 次，仅滤波 120 次，滤波加滞回 115 次；这组默认参数没有明显减少整条控制链的切换。滤波加滞回的单帧计算中位数约 75 微秒、95 分位约 150 微秒，属于此电脑上的离线测量，不能作为 Jetson 性能保证。回放不包含新增指令保持，实机效果需结合控制 CSV 检查。
