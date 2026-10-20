# 目标方向查表与近端位置保护

2026-10-17 起，已删除目标方向历史滤波和额外位置修正。

配置 `STEERING_ANGLE_WZ_TABLE` 时，`WZ_MODE=heading` / `segments` 的普通有效帧直接使用 `steering_demand_deg` 查表。这个目标方向已经包含原始航向、横向位置及前视目标几何，不再叠加第二次位置修正。查表模式不再使用角度滞回、预测刹车或近端位置分支改写结果。

`POSITION_GAIN`、`POSITION_DEAD_CM`、`POSITION_LOOKAHEAD_CM`、`POSITION_MAX_DEG` 及对应 CLI 参数已删除，启动脚本也不再传递它们。旧本机配置若仍有这些变量，可以删除；它们已无作用。

## 未配置查表时的位置保护

以下参数仅控制未配置角度表时的独立近端位置保护，不是位置叠加角：

| 配置项 | 默认 | 含义 |
|---|---:|---|
| POSITION_RECOVERY_CM | 8 | 同侧近端偏移达到此阈值后优先回正；0 关闭 |
| POSITION_RECOVERY_FULL_SCALE_CM | 12 | 超出阈值多少 cm 达到满幅需求 |
| POSITION_CONFIRM_FRAMES | 2 | 同侧新鲜有效观测的确认次数 |

无效观测、外部停车或偏移换侧会清除确认。起步闸、图卡停车、人工停车仍优先。查表模式下这些位置保护参数不改写查表候选。

## 日志与检查

```bash
bash scripts/run_button_vision.sh --dry-run
```

启动命令包含上述三项 `--position-*` 参数，启动日志显示实际设置及 `input=steering_demand_deg`。`LOG="vision"` 自动记录：

- `measurement.steering_demand_deg`：当前未经历史滤波的目标方向，度；无有效方向时为 null。
- `steering_table_input_deg` / `steering_table_wz`：配置表的输入及候选输出。
- `steering_heading_deg` / `steering_filtered_heading_deg`：原始/滤波航向，仍独立保留。
- `steering_position_recovery` / `steering_position_confirm_frames`：未配置表时的位置保护状态。

不再产生 `steering_filtered_demand_deg`、`steering_filter_demand_deg`、`steering_position_correction_deg`、`steering_combined_demand_deg`。桌面查看器显示记录的目标角，不重新计算旧日志的命令。

外部停机、丢线补偿、直道限制、配置的命令保持以及 connector 中位数窗口继续生效。机器人实际采用的指令以 `robot_cmd_*` 反馈为准，可能晚于本帧候选。

回放工具使用当前控制逻辑处理旧观测；旧日志曾启用已删除的两层处理时，回放结果与原命令不同是预期行为，不能声称逐帧复现旧控制器。
