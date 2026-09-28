# 图形卡任务：Jetson 与 STM32 部署说明

摄像头识别六种边长为 10 cm 的图形卡。确认图形后，视觉程序通过现有的“视觉 → connector → 策略程序”UDP 通路发送结果。`qr` 表示当前识别结果；没有识别到图形时为 `-1`。`event_id` 和 `event_action` 标识一次已确认的动作事件，并在图形卡停留窗口内持续发送。策略程序按 `event_id` 去重，因此 connector 以 50 Hz 重复转发时不会重复执行动作。同一张卡持续留在画面中，也不会在识别器冷却结束后再次触发。摄像头读取失败时，视觉程序发送 `qr=-1` 和零速度。

| 图形卡 | 编号 | 控制方 | 动作 |
|---|---:|---|---|
| 圆形 | 1 | STM32 | 举左手 |
| 五角星 | 2 | STM32 | 举右手 |
| 正方形 | 3 | Jetson 单腿 ONNX 模型 | 抬左腿，右脚支撑 |
| 菱形 | 4 | Jetson 单腿 ONNX 模型 | 抬右腿，左脚支撑 |
| 十字形 | 5 | STM32 | 举双手 |
| 三角形 | 6 | STM32 | 摇头 |

STM32 只接收编号 1、2、5、6 的上半身动作请求，并拒绝编号 3、4。包括 STM32 控制手臂或头部的期间，12 个腿部关节目标仍只由 Jetson 下发。`--fixed-policy` 继续使用原有的固定关节角播放路径，不处理图形事件。

## 启动方式

在 Jetson 的仓库根目录，分别启动 connector 和视觉程序：

```bash
python -u connector.py --vision-port 5006 --policy-port 5005
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless
```

在第三个终端启动策略程序。`--model` 应填写**已在这台机器人上验证过的 49 维观测行走模型**；`--one-foot-model` 指向 46 维观测的单腿站立模型：

```bash
python -u humanoid_jetson_deploy/main.py \
  --policy walking --command-source vision \
  --model /path/to/verified-49-input-walking.onnx \
  --one-foot-model humanoid_jetson_deploy/policy-one-foot-standing.onnx \
  --port /dev/ttyACM0 --no-plot
```

上述命令未启用电机。先在机器人受到支撑的条件下，检查摄像头分类、事件编号、模型输入输出维度、CRC 错误计数和策略输出；进行实体动作测试时，再添加 `--enable-motors`。同一时间只能有一个策略进程打开 `/dev/ttyACM0`。

单腿动作先短暂站稳，再默认保持抬腿命令 4 秒（可用 `--shape-lift-seconds` 设置），随后下发落腿命令，最后恢复行走。需要通过录像确认实体腿部实际抬起至少 3 秒；仅凭命令持续时间不能证明实体动作达标。

## USB CDC 协议扩展

现有的第 2 版帧头、CRC-16、154 字节状态帧和 74 字节腿部指令帧均未改变。新增两种消息类型，沿用相同的封帧方式：

| 类型 | 方向 | 紧凑载荷格式 | 含义 |
|---:|---|---|---|
| 3 | Jetson → STM32 | `<IB`：`event_id`、`action_id` | 上半身动作请求，仅使用编号 1/2/5/6 |
| 4 | STM32 → Jetson | `<IBB`：`event_id`、`action_id`、`status` | 状态：1 已接收，2 已完成，3 忙碌，4 无效，5 失败 |

在确认动作完成前，Jetson 每 100 ms 使用同一个 `event_id` 重发请求。STM32 对重复事件返回当前状态，不会重新执行动作。原有的非阻塞舵机状态机完成后，STM32 返回完成状态。如果腿部指令看门狗超时，STM32 中止正在执行的上半身动作并返回状态 5。执行手臂或头部动作时，正常的腿部指令处理和状态反馈仍持续运行。
