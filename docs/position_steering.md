# policy49flitter：近端位置纠偏

适用于 `WZ_MODE=heading` 和 `WZ_MODE=segments`。在原有方向/远端选点需求上叠加近端中心偏移修正，避免只取消内转后仍偏在赛道内侧。

## 参数与启用

修改 Jetson 的 `config/button_start.env`。已有配置缺少这些字段时，按钮脚本会补上表中的默认值；拉取代码后重新启动 vision 才会生效。

| 配置项 | 默认 | 含义 |
|---|---:|---|
| POSITION_GAIN | 1 | 普通位置修正增益；0 关闭叠加项 |
| POSITION_DEAD_CM | 2 | 近端偏移死区，cm |
| POSITION_LOOKAHEAD_CM | 50 | 位置项的距离尺度；减小会增强纠偏，不改变赛道选点距离 |
| POSITION_MAX_DEG | 12 | 叠加修正角的幅度上限，度 |
| POSITION_RECOVERY_CM | 8 | 优先回正阈值，cm；0 关闭优先回正 |
| POSITION_RECOVERY_FULL_SCALE_CM | 12 | 超出阈值多少 cm 时达到满幅需求 |
| POSITION_CONFIRM_FRAMES | 2 | 同侧新鲜有效观测的确认次数 |

CLI 为配置名转小写、下划线换短横线并加 `--`，例如 `POSITION_GAIN` 对应 `--position-gain`。

```text
有效偏移 = sign(near) × max(|near| - POSITION_DEAD_CM, 0)
位置修正角 = clamp(-POSITION_GAIN × atan2(有效偏移, POSITION_LOOKAHEAD_CM), ±POSITION_MAX_DEG)
综合需求角 = 原有滤波目标角 + 位置修正角
```

角度单位为度。`near > 0` 表示近端赛道中心在相机右侧，位置项要求几何右转；`near < 0` 镜像处理。最终经 `yaw_sign` 映射到策略输入。这是相机前方近端的偏移，不是机器人脚下的实际横向坐标；弯道形状、相机安装与地面映射也会影响它。

同侧偏移连续达到阈值后，优先回正覆盖相反的远端目标、角度滤波延迟、视觉控制器的普通指令保持和预测减档。输出仍服从左右档位、是否允许右转及角速度上限。步态模型入口的 `COMMAND_MIN_HOLD_S` 仍按原合同处理运动指令，图卡/起步闸/人工停车仍优先。

无效/丢线观测、外部停车或偏移换侧会清除位置确认。单边线跟随仍使用队友新增的有效单边几何。丢线后的默认动作沿用当前分支的继续前进并向左搜线；位置否决会清理旧转向历史，但没有取消这个默认搜线动作。

## 检查与回退

```bash
bash scripts/run_button_vision.sh --dry-run
```

输出应包含七项 `--position-*` 参数。启动日志的 `[steering-position]` 显示实际设置。`line_frames.jsonl` 的 measurement 包含：

- `steering_near_cm`：当前近端偏移。
- `steering_position_correction_deg`：普通位置叠加角。
- `steering_combined_demand_deg`：叠加后的转向需求。
- `steering_position_recovery` / `steering_position_confirm_frames`：优先回正状态。
- `steering_decision=position_recovery_left/right`：采用了位置优先级。

先固定方向与滤波参数，检查偏移符号与实际左右位置是否一致，再根据一趟录像调整位置参数。默认参数已经过离线回归，仍需要实车复测。

关闭本次位置增强：

```bash
POSITION_GAIN=0
POSITION_RECOVERY_CM=0
```

回放读取记录 manifest 中的位置参数、丢线模式（`history-turn` 与兼容别名 `history-stop`）和分段兜底设置。旧记录没有位置参数时按关闭位置增强回放；显式消融参数仍可关闭丢线恢复或分段兜底。
