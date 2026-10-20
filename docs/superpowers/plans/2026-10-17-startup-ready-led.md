# PA1 启动成功灯实施计划

> 用户已于 2026-10-17 缩小需求并授权直接提交、推送；按本计划顺序在当前会话实施。

**Goal:** 在上位机提交 `47d7cc2fabb0aa49df72e5faf75f1177e10d7ed6` 上增加 vision + connector 启动成功的 PA1 指示，并提交下位机 `抬脚` 分支。

**Architecture:** 原有 UDP 5006 增加一次启动探测/回复；vision 收到 connector 回复后，在现有类型 5 的单字节 STARTUP_CONTROL 中设置 `STARTUP_HOST_READY = 1 << 2`。下位机接受此位并写 PA1 RESET。成功指示保持到 MCU 复位。没有持续健康监测，没有新服务、线程、端口、串口拥有者或协议帧类型。原有按钮复位、ARM、首卡 PA3、运动包、PA2 和电机看门狗行为保持。

**Tech Stack:** Python unittest、原有 UDP/串口协议、STM32 HAL。

早按 PC2 不等待点灯回执。vision 交接包保留已确认的 HOST_READY；若交接早于回执，由原策略进程在收到 connector 的原 UDP 数据后补发一次 HOST_READY。该路径仅在原有 `--startup-ready-file` 自动启动模式启用，写灯失败不影响运动处理，串口重连后允许重发。

本计划取代 2026-10-16 设计文档中的故障灯和完整健康监测设想。

## 实施顺序

- [x] 测试先行：`tests/test_startup_ready.py` 覆盖真实 UDP 探测/回复、无回复不就绪、探测不产生运动指令、协议位编解码、按钮复位/ARM/首卡不变。
- [x] 上位机：`connector.py` 识别独立探测并向来源回复，跳过运动处理；`policy_bridge.py` 中 ConnectorClient 非阻塞轮询回复，最多每秒重试一次，确认后不再探测。
- [x] 上位机：`protocol.py` 接受新位；`startup_button_link.py` 在现有发送节拍中携带标志；`run_policy_vision.py` 在成功读到相机画面后的按钮等待路径查询 connector 状态；`main.py` 处理早按 PC2 的单次补发，不增加起步条件。
- [x] 下位机：`Core/Inc/jetson_protocol.h` 定义新位，`Core/Src/jetson_protocol.c` 接受新位，`Core/Src/jetson_robot_bridge.c` 收到该位后点亮 PA1。只改这三处运行代码。
- [x] 验证：先观察新增测试因缺功能失败，再实现并运行相关启动、connector、协议和视觉回归；完整 Release 固件已编译、链接成功。
- [x] 更新 `docs/button_autostart.md` 与下位机简短说明，明确这是启动成功指示而非实时故障灯。
- [x] 代码审查：修正回执影响按钮起步与交接覆盖点灯包两项问题，复查无阻断问题。

## 验证记录

软件回归收集 227 项，20 项 Bash 自启动测试因 Windows 环境按原规则跳过。206 项首轮通过；另 1 项因测试进程 PATH 缺少原有 `tee` 命令失败，补入 `F:\Git\usr\bin` 后单独复跑通过，合计 207 项通过。新增 5 项启动灯测试及新增无回执不阻止按钮的整流程测试均通过。

下位机 `cmake --preset Release`、`cmake --build --preset Release --parallel 4` 成功，FLASH 使用 53816 B、DTCMRAM 使用 13160 B。编译器仍报告原有未使用函数、LCD 格式/别名和链接 RWX 等警告；未为本任务修改这些无关代码。未连接 Jetson、相机、STM32 或电机，板上灯与按钮仍需部署后核对。

交付时仅提交本任务文件，保留原有 pyc 和无关文档；推送上位机当前基线对应的 `card_recognition_debug_20261016` 与下位机 `抬脚` 分支，并核对远端哈希。
