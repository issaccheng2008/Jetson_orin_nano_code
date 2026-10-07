# STM32 命令接收与控制循环计数诊断

STM32 的 `diag/usb-command-counters` 分支从 `main` 的 `920e98a` 建立。状态帧尾部增加两个 32 位计数，原有字段顺序、命令帧、动作帧和电机控制逻辑不变。Nano 的 `diag/stm32-command-counters` 分支同时能解析原来 144 字节的状态载荷和新增计数的 152 字节状态载荷。

## 部署顺序

1. 先部署 Nano 分支，使用原 STM32 固件确认能正常接收状态帧。此时下述两个 CSV 列为空。
2. 再将 STM32 诊断分支编译并刷入板子。新增状态帧会被新 Nano 解析；旧 Nano 解析器不接受 152 字节状态载荷，不要反过来部署。
3. 原来的 `main.py --diagnostic-log-dir records/control_diagnostics` 命令不需增加参数。每次运行的 `control_trace_*.csv` 末尾会有以下两列。

| CSV 列 | STM32 来源 | 含义 |
| --- | --- | --- |
| `stm32_command_rx_count` | `g_debug_command_count` | CRC 正确、长度正确的 COMMAND 帧进入 STM32 USB 接收解析器的累计次数；即使尚未应用或没有使能，也会增加。 |
| `stm32_system_control_cycle` | `system_control_cycle` | STM32 调用腿部目标下发路径、向 12 个电机提交目标的累计次数；不代表电机已经达到目标。 |

两者是 32 位无符号数，会回绕；STM32 重启后会重新从零计数。现有 `main.c` 的模式切换路径也可能清零 `system_control_cycle`，分析时要同时看状态序号、时间和内核 USB 日志。两列来自收到的状态包；**状态包停止后，CSV 无法再看到 STM32 内部计数是否继续增加**。

## 断联现场

运行前可在独立终端记录 USB 内核事件：

```bash
cd ~/jetson_orin_code
journalctl -k -f -o short-iso-precise | tee "records/kernel_$(date +%Y%m%d_%H%M%S).log"
```

主程序报 `Write timeout` 后，先保留现场，立即执行：

```bash
date --iso-8601=ns
lsusb -d 0483:5740
ls -l /dev/ttyACM*
journalctl -k --since "2 minutes ago" --no-pager | grep -Ei 'ttyACM|0483|5740|usb 1-2'
```

若 `stm32_command_rx_count` 仍增加而 `stm32_system_control_cycle` 停止，说明 USB 接收解析器还收到了 COMMAND，主循环没有继续下发电机目标。若两者同时停止，需要再结合主机写入记录和内核日志，区分 USB 接收停滞与整机复位；单靠最后一帧不能断定哪一侧先停。
