# F/C 查表角加权

本分支在远端 `5212d70` 基础上，为 `heading` 模式的角度映射表增加 F/C 加权。这里的 F 是滤波后的**目标方位角**，不是日志查看器的“滤波后偏转角”（后者是线路方向）。

## 计算顺序

1. 原始线路方向、近端横向位置和前视距离计算原始目标方位角 `demand`。这一步本身已经包含横向位置。
2. 对目标方位角作因果滤波，得到 F。
3. 根据近端横向位置，计算额外位置修正 P；完整位置修正后的目标角 C = F + P。
4. 用 `(1-w) × F + w × C` 查表得到角速度。因此也等价于 `F + w × P`。
5. 视觉候选继续经过既有直道限制、connector 时间窗口和模型指令保持。这里混合的是角度（°），不是查表后的角速度（rad/s）。

默认 `w=0.3`，即 F 占 0.7、C 占 0.3。例如 F=30°、P=-10°、C=20°，新查表角为 27°。减弱的是第 3 步的额外修正，不会消除第 1 步的位置影响。

## 配置

在 Jetson 的 `config/button_start.env` 中设置：

```bash
STEERING_FILTER_MODE=active
STEERING_POSITION_BLEND_WEIGHT=0.3
```

需要 `WZ_MODE=heading` 且配置了 `STEERING_ANGLE_WZ_TABLE`。旧配置缺少权重时，启动脚本默认使用 0.3。权重必须是 0～1 的有限数，非法值会在打开相机前报错。

| w | F 权重 | C 权重 | 实际查表输入 |
| --- | --- | --- | --- |
| 0 | 1 | 0 | F，仍然有目标角滤波 |
| 0.3 | 0.7 | 0.3 | F + 0.3P |
| 1 | 0 | 1 | C = F + P |

`legacy` 模式保留原始目标角查表行为；`shadow` 模式只在试验控制器计算新结果，实际输出沿用基线；非角度表和分段控制沿用原逻辑。

额外位置修正使用旧 F/C 阶段的定义：横向偏移减去死区后，计算 `-POSITION_GAIN × atan(偏移 / POSITION_LOOKAHEAD_CM)`，转为角度并按 `POSITION_MAX_DEG` 限幅。默认增益 1、死区 2 cm、位置前视 50 cm、上限 12°。只有方向、没有可靠位置的观测不追加 P。

修改配置后可先查看生成参数，再重启视觉服务：

```bash
bash scripts/run_button_vision.sh --dry-run
sudo systemctl restart humanoid-button-vision.service
```

生成参数应含 `--steering-position-blend-weight 0.3`。`STEERING_COMMAND_WINDOW_S` 和 `STEERING_COMMAND_MEDIAN` 不随本功能改变；它们仍控制查表后角速度的时间窗口取样。

## 日志核对

| measurement 字段 | 含义 |
| --- | --- |
| `steering_demand_deg` | 原始目标方位角 |
| `steering_filtered_demand_deg` | F |
| `steering_position_correction_deg` | P |
| `steering_combined_demand_deg` | C = F + P |
| `steering_position_blend_weight` | C 的权重 w |
| `steering_blended_demand_deg` | 加权后的目标角 |
| `steering_table_input_deg` | 实际送入映射表的角度 |
| `steering_table_wz` | 查表结果，单位 rad/s |

日志查看器“查表目标角”优先显示 `steering_table_input_deg`，缺失时兼容显示旧日志的 `steering_demand_deg`，不重算旧命令。实际采用角速度仍读取策略反馈，不能用加权角或视觉候选代替。
