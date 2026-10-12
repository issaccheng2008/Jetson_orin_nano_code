# 起步到首卡动作结束：暂停视觉计算实验

分支 `startup_vision_pause_20261012`，从 `lane_loss_recovery_20261012` 的 `ef60348` 建立。没有部署板子或连接电机，尚未实机验证。

## 实际执行顺序

1. 起点等待时正常巡线、识别首卡和二维码/按钮。首卡仍由原有起点逻辑确认和锁存。
2. 起步条件满足、策略就绪后，进入实验分支。**第一条起步运动命令发出前停止图像运算**。按原有 `STARTUP_FIRST_WALK_S` 或 `STARTUP_SEQUENCE` 发送启动命令；默认是 `vx=VX`、`wz=0`。
3. 启动序列按单调时钟到期后，立即发送 `vx=0,wz=0` 和已经锁存的首卡动作 1–6。保持同一个 `event_id`，后续重复传递该事件由原接收端去重，不能生成第二次动作。
4. 沿用 `card_action_stop_s()` 的动作保持时间：类别 3/4 至少 6 秒，其他类别至少 3 秒，`--card-hold-ms` 更长时取更长值。**这是原来的定时等待，不是新增对物理动作 DONE 的等待**。首卡原本就不走普通图卡“发7低头、投票、发8”的读卡流程，本实验保留这个区别。
5. 等待结束时先发 `vx=VX,wz=0`（`VX=0.2` 时就是 0.2），再恢复视觉。丢弃曝光开始于恢复时刻之前的在途帧，用新的采集帧重新交给原控制器；如果新帧取不到，仍按原相机失效逻辑发送停止。不会把暂停的几秒累加成巡线滤波的 dt。

首卡防重复、下一卡停车间隔及后续普通巡线/停车流程沿用现有状态。暂停时不调用巡线、鸟瞰、LUT、二值化、图卡、二维码、显示、截图或录像提交。图像录像编码在整个实验运行中关闭，避免后台队列继续编码和恢复时补写旧画面；标量日志及主程序的模型/关节诊断继续记录。

异步相机采集线程继续取图、解码和保留最新槽位，以免恢复时读到暂停期间积压的旧帧。因此本实验减少的是视觉处理负载，**没有移除相机驱动、解码和步态推理负载**。

## 使用与对照

按钮配置 `config/button_start.env`：

```bash
STARTUP_VISION_PAUSE=1
VX=0.2
# 保留正在使用的 STARTUP_FIRST_WALK_S / STARTUP_SEQUENCE 和其他配置。
```

本实验分支的按钮脚本在旧配置缺少 `STARTUP_VISION_PAUSE` 时也默认启用；示例配置显式写了 1。值为 0 恢复正常图像处理。用 `bash scripts/run_button_vision.sh --dry-run` 只查看参数，不启动模型、电机或相机。

手动 B 终端在原来的完整视觉命令中加：

```bash
--start-gate both --startup-vision-pause
```

按钮启动入口自带 `--start-gate button`。实验要求起点首卡已经锁存、使用异步采集；手动命令未指定实验参数时保持正常运行。按钮若在首卡未确认时提前启动，原程序会跳过首卡特殊策略，这时也不会进入暂停实验。

对照恢复：

```bash
STARTUP_VISION_PAUSE=0
RECORD_VIDEO=0
```

手动对照使用 `--no-startup-vision-pause --no-record-video`。两组使用相同模型、短走时长、速度、相机配置和首卡。保持录像关闭一致，避免把录像负载也作为组间差异。

暂停期间 B 的命令按 20 ms 周期发送，正常对照仍按现有视觉帧节奏发送。因此若实验起步/停车更稳定，证据涉及**视觉负载减少与命令/停车调度更及时两种变化**，不能据此单独断定 GPU/CPU 竞争或 UDP 心跳哪一个是唯一原因。要区分两者，可进一步让两组都使用相同的独立命令节奏再对照。

## 日志怎么看

控制台打印 `[startup-vision-pause] armed`，然后 `WALK → ACTION → RESUME`，每个阶段有速度、事件号、累计命令数、发送平均频率和最大间隔。50Hz 是目标周期，实际频率以日志为准。

原 `line_frames.jsonl` 新增暂停期间的**命令行**，它们不是相机帧：

| 字段 | 含义 |
|---|---|
| `observation_processed=false` | 本行没有新图像计算，不能纳入视觉识别率/丢线率 |
| `startup_vision_paused` | 当前是否处于暂停视觉阶段 |
| `startup_pause_phase` | WALK、ACTION 或 RESUME |
| `startup_pause_command_tick` | 独立命令发送累计数 |
| `startup_pause_publish_hz` | 暂停实验中实际发送平均频率 |
| `startup_pause_max_publish_gap_ms` | 实际发包最大间隔 |
| `vx/wz/event_id/event_action` | 此次真正发出的命令和事件 |

暂停行保持上一张真实图像的 `frame` 编号，不伪造相机帧或更新巡线四计数；`measurement.tracking_mode=PAUSED` 不应统计为 LOST。正常图像行写 `observation_processed=true`。日志具体目录沿用运行时打印的 `[vision] per-frame log:` 路径；Nano 主程序和 connector 的日志位置沿用现有配置。

## 离线验证

```bash
PYTHONPATH=/tmp/robocup_photo_deps:tests python -m unittest \
  test_startup_vision_pause test_startup_sequence \
  test_policy_vision.VisionEntryPointTests -q
```

使用模拟时钟、相机和客户端。检查真实入口在第一条前进命令至恢复命令之间不调用相机取帧/巡线/图卡，不构造录像编码器；发包间隔不超过 21ms；0.5 秒短走与 3 秒动作等待无多余视觉周期；同一首卡只有一个事件；恢复时丢弃旧曝光并读取新帧。另覆盖 6 秒动作等待和多段起步命令。模拟通过不能证明 Nano 实际发包达到 50Hz，也不能证明真实停车距离改善。
