# 固定关节角测试

两个文件都使用 `joint_frames_v1`、50 Hz、`config.JOINT_NAMES` 顺序，角度单位为弧度。以此前已验证的全零目标作为起点与终点；当前实测角若偏离零位，程序现有的速度和实测角窗口限制仍会生效。

## 测试一：12 个关节小幅逐个运动

文件：`test_all_joints_small.json`。12 个关节按文件中的顺序逐个从 0 平滑到 +0.05 rad（约 2.9°），短暂保持后回零。每次只有一个关节偏离零位；共 530 帧，约 10.6 秒。

## 测试二：左腿、右腿依次抬高

文件：`test_high_knee_within_limits.json`。左侧 `l_leg_pitch_joint=-1.5` rad、`l_ankle_pitch_joint=+0.4` rad；左腿降回零后，右侧 `r_leg_pitch_joint=+1.5` rad、`r_ankle_pitch_joint=-0.4` rad。左右符号依据当前配置中的默认姿态推定，首次使能前应核实实物方向。其余关节目标为零。

每侧用 3 秒平滑抬起，在目标姿态保持 **5 秒（250 帧）**，再用 3 秒降回零；两侧之间停 1 秒，最后零位停 1 秒。共 1200 帧，24 秒。

原要求的 leg pitch 和 ankle pitch 均为 90°，但现有软件校验限位（含 0.05 rad 余量）分别约为 **87.1°** 和 **25.8°**。因此本文件明确采用约 **85.9°** 和 **22.9°**；它不是 90° 踝关节测试。未修改 `config.py` 或绕过限位。

## 运行

从 `humanoid_jetson_deploy/` 目录运行。先用不使能模式确认串口与状态，再在机器人受到可靠支撑、方向核对后执行使能测试：

```bash
python main.py --fixed-policy examples/test_all_joints_small.json --no-plot
python main.py --fixed-policy examples/test_all_joints_small.json --no-plot --enable-motors

python main.py --fixed-policy examples/test_high_knee_within_limits.json --no-plot
python main.py --fixed-policy examples/test_high_knee_within_limits.json --no-plot --enable-motors
```

`--port` 的默认值为 `/dev/ttyACM0`。每次运行的位置目标与反馈角度会写入 `logs/motor_positions/` 下的 CSV。
