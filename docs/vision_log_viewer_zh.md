# Vision 日志桌面查看器

只需要 Python 3.10 或更新版本与 Tkinter。工具使用标准库，可以在 Windows 电脑离线查看从 Jetson 拷来的日志，无需安装 OpenCV、模型或连接机器人。

## 启动与导入

Windows 双击 `new_vision/scripts/view_vision_log.bat`，然后点击 **导入时间文件夹**，选择当次日志目录。例如：

```text
2026-10-17/
└── test_101620_123456_abcd1234/    ← 选择这个当次目录
    ├── recording_manifest.json
    └── vision/
        └── run_.../
            └── line_frames.jsonl
```

也支持直接选择 `vision`、`run_...` 目录，或用“打开 JSONL”打开 `line_frames.jsonl`。一个目录里有多段运行时，在“运行记录”下拉框切换，各段分别计时、分别浏览。

从仓库根目录运行：

```bash
python new_vision/scripts/view_vision_log.py
python new_vision/scripts/view_vision_log.py "你的当次时间文件夹"
```

Linux 桌面若缺少 Tkinter，安装系统包 `python3-tk`。此工具是桌面窗口，需要图形桌面；可以把日志复制到 Windows 后使用。

## 操作

- 拖动底部时间轴，按实际日志时间选择最近一帧。摄像头帧间隔不均匀也按时间定位。
- **D：后退 5 帧；F：前进 5 帧**，支持大小写，也可以点击对应按钮。这里的“帧”指 vision JSONL 的一条有效帧记录，不是固定秒数或视频重复帧。
- 上方显示所选帧前 10 帧、后 9 帧，共 20 帧。首尾自动补足；总记录不足 20 帧时显示全部。当前行高亮，可以点击表格行定位。
- 时间同时提供录制时间、相对本段起点的秒数和当前帧序号。绝对时间按查看电脑的本地时区显示，当前帧提示中显示日期及时区偏移；相对时间优先使用录制时的单调时钟，不受系统对时跳变影响。
- 展示八列，左侧的行标题为原始视觉帧号。窗口较小时可以横向滚动表格；D/F 和时间轴仍可使用。

## 八列含义

| 列 | 日志来源/含义 |
| --- | --- |
| 时间 | `host_time_ns`，精确到毫秒；缺少绝对时间时显示相对秒数 |
| rowhdg 原始角（°） | `measurement.rowhdg_deg`，与实时 Close+Fit 的 `rowhdg` 使用同一个逐行拟合值，日志保存未取整的精度；`rowhdg_valid=false` 时仍显示原数值并标注“无效” |
| 偏转角（°） | `measurement.steering_heading_deg`，回退到有效 `heading_control_deg` 或有效单边 `single_edge_heading_deg`；正值表示前方线路向左 |
| 滤波后偏转角（°） | `measurement.steering_filtered_heading_deg`，兼容 `steering_filter_heading_deg`；直接显示录制值，不重新计算滤波 |
| 查表目标角（°） | `measurement.steering_demand_deg`，由原始航向与横向位置计算的前视目标方向；保留日志原始精度，UI 显示三位小数。缺失时显示“未记录”，丢线/起步等待时不显示残留角度 |
| 识别线的情况 | 双线、左单线、右单线、单线、丢线、起步等待或未记录 |
| vision 输出角速度（rad/s） | 顶层 `wz`，vision 发给 connector 的指令 |
| 实机采用角速度（rad/s） | `robot_cmd_wz`，来自策略进程反馈的实际采用指令；经过 connector、指令保持、动作接管等环节，并确认串口写出且电机使能。它不是 IMU 测得的身体旋转速度 |

没有记录、未知/过期、未发送、电机未使能分别显示对应文字，不用 0 或 vision 输出代替。丢线或起步未巡线时不显示可能残留的控制偏转角。`rowhdg` 列保留本帧原始拟合值及无效标记；方向兜底成功时，“偏转角”可以与无效的 `rowhdg` 不同。旧日志缺少 `rowhdg_deg` 时显示“未记录”，不会用控制角度代替。

## 只录制 vision

保持运行配置 `LOG="vision"` 即可。更新后的 `run_policy_vision.py` 会把现有姿态/策略反馈通道里的实机指令快照一并写入 vision JSONL，不需要启用 control 或 video 日志。新增字段为 `robot_cmd_wz`、`robot_cmd_status`、反馈时间、策略步号、模式及使能状态；`rowhdg_deg`、`rowhdg_valid` 记录在 `measurement` 中。headless 和实时窗口模式均记录相同字段。

`measurement.steering_demand_deg` 同样包含在 vision 日志中，无有效方向时记录 null。配置 `STEERING_ANGLE_WZ_TABLE` 时，普通有效帧直接以此角度查表：已删除目标方向历史滤波、额外位置修正及其 `POSITION_GAIN`、`POSITION_DEAD_CM`、`POSITION_LOOKAHEAD_CM`、`POSITION_MAX_DEG` 参数。查表结果不再被角度滞回、预测刹车或位置分支改写；航向滤波仍单独记录。外部停机、丢线补偿、直道限制、配置的指令保持及 connector 的中位数窗口仍生效，实机反馈可能晚于视觉候选变化。

旧日志仍展示原来记录的目标角和命令，不会按新规则重新计算。因此旧日志的“查表目标角”可能与当时经过滤波和位置修正后生成的命令不直接对应。

实机数值是该视觉记录时最新且未过期的反馈，反馈通道最多允许 0.5 秒延迟，并不是该视觉帧发送后立即产生的机器人响应。必须保持现有姿态反馈端口可用；反馈缺失或过期会显示“未知/过期”。

**以前已经录好的 vision 日志没有保存实机反馈，无法恢复这一列，界面会显示“未记录”。** `steering_applied_wz` 仍然属于视觉控制器内部值，不能代表实机采用值。其它列可以正常查看。只复制 GUI 到电脑不会补回旧数据；要记录新增字段，需把更新后的 `line_telemetry.py` 和 `run_policy_vision.py` 同步到 Jetson 后重新录制。

读取过程中会跳过损坏或末尾未写完的行，并在窗口下方显示跳过数量。没有有效帧时弹出错误，保留先前已打开的数据。
