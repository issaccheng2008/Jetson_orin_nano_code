# 实车视觉与 standalone debug 对齐

本分支从 `af53f1a054111c1c95a25c58c557ab93784ed5fc` 创建。对齐参照为 2026-10-17 16:16 的 standalone debug 测试：主相机、关闭第二相机、无动态姿态几何、默认巡线及图卡处理，保留图卡检测。

## 启用

`config/button_start.env` 不被 Git 跟踪，本分支不修改或覆盖它。可以在文件最后手动添加：

```bash
VISION_PROFILE=debug
LOG="vision"
```

本分支的按钮脚本在缺少 `VISION_PROFILE` 时也默认 debug。配置项重复时以最后一处为准。切换回逐项调参时设置 `VISION_PROFILE=configured`。

## 有效行为

| 项目 | debug profile |
|---|---|
| 巡线候选提取 | legacy |
| 光度处理 / 图卡预处理 | lut / selective |
| 持续单边跟踪 | 开启 |
| 航向近远采样 | 默认近端约 25.07 cm、远端 29 cm，清除自定义区间 |
| 角度查表 | 清空，使用 debug 的默认航向控制规则 |
| 转向档位 | 左 0.37/0.43/0.5，右 -0.3/-0.5，直行 0 |
| 转向上限 / 走廊 / full-scale | 0.5 rad/s / 8 cm / 20° |
| 正负方向容差 | +12° / -4° |
| 视觉滤波模式 | legacy |
| 丢线策略 / 默认搜线 | history-turn / +0.3 rad/s |
| 图卡触发距离 | 43 cm |
| 第二相机 | 禁用 |
| 图卡姿态几何 | 静态安装角 45° |
| 相机硬件控制 | keep，不重新设置曝光、白平衡、对比度等 |

旧配置里的 `LINE_PREPROCESS=contrast`、角度表、采样区间、滤波、图卡距离和曝光等逐项设置仍可以保留，但在 debug profile 下会被覆盖，不参与有效调参。profile 与普通 debug 共用同一份解析器默认值；环境敏感项目固定为记录中的 debug 值。`keep` 仅表示不写相机硬件控制，不会重置相机上一次保留的设置，也不保证两次拍摄亮度完全一样。

运行期间同时固定 `CAMERA_PROFILE=usb_main`、图卡检测距离上限 80 cm 和仓库内的光度 LUT，避免旧环境变量绕过 CLI 改变距离标定、图卡面积门槛或 LUT。退出时恢复调用者环境。

按钮起步、策略模型/Python/串口、实际 connector 端口、启动序列、日志选择、运行时长和 `--hold-still` 不被覆盖。服务仍等待按钮后启动策略，并不改用 debug 的 5999 端口。

姿态通道继续接收动作完成、起步准备及 `robot_cmd_*` 反馈；仅将 `attitude_pitch_correction` 关闭，使图卡几何与 debug 的 `--attitude-port 0` 一样使用静态安装角。不能把实车的姿态端口改成 0，否则会同时失去动作确认和执行反馈。

## 检查与生效

```bash
bash scripts/run_button_vision.sh --dry-run
```

输出应含 `--vision-profile debug`。dry-run 展示的是原始启动参数，旧调参仍可能出现在命令中；profile 会在 Python 解析后、验证和启动硬件前覆盖它们。最终设置以 `[vision-profile] debug` 启动提示和新日志 `run_manifest.json` 的 `arguments` 为准，包含 `vision_profile=debug`、`line_preprocess=legacy`、`steering_angle_wz_table=null`、`heading_regions_cm=null`、`attitude_pitch_correction=false`。

更新分支/配置后，下一次启动 vision 生效。不要在机器人运动中切换；本次开发不自动切换 Jetson 的运行分支，也不重启服务。独立 SSH 测试仍应使用 `--headless`，避免 Qt 无显示环境崩溃。

对齐参数不等于承诺提高 FPS。图卡检测仍逐帧执行，实车还存在步态负载和画面变化。需在一致画面和运行条件下查看 `loop_period_ms`、`line_process_ms`、`shape_update_ms` 评估性能。
