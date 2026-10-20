# p3 相位时钟跨棍：上位机模型导入交接

本版执行方案 A：机器人恢复微屈膝初始姿态后，当**最前脚尖到木棍近侧边缘的净距离为 8 cm**，外部只给一次启动信号。随后上位机自行按时钟生成指令，每 **20 ms** 读取 IMU、关节编码器并运行策略。普通步长输入 **10 cm**，跨前腿步长输入 **23 cm**。后续无需木棍位置、足底触地信息或人工“开始跨棍”命令。

项目目录：`F:\RobotProject\cross_stick_p3`。机器人资产为 `v3.2/v3.2.usd`。木棍尺寸：前进方向宽 3 cm、横向长 80 cm、高 3 cm。策略输入/输出仍为 **49 / 12**。

## 1. 应交给同伴的文件

准备好的原型包位于 `exported/phase_clock_model_850/`，可整目录复制。其权重来自 `logs/rsl_rl/fixed_stick_stage1_v32/2026-10-02_17-07-45-315815_finetune/model_850.pt`，该模型原先以触地事件切换指令。现在已验证它在候选时钟下的单次仿真回放；**它还不是充分完成时钟微调、经过真机验证的最终模型**。短程 smoke 模型仅供检查训练流程，不应替换这个原型作为性能更好的模型。

| 文件 | 用途 | 真机是否需要 |
|---|---|---|
| `policy.onnx` | 确定性 actor，输入 obs，输出 action | 是 |
| `phase_clock.json` | 控制周期、相位边界、速度与步长参数 | 是，务必实际读取 |
| `deployment/phase_clock.py` | 与仿真共享的相位调度、一次启动及重置逻辑 | Python上位机可直接用；其他语言照此移植 |
| `deployment/policy_interface.py` | IMU/编码器拼成49维输入、动作换算、ONNX加载接口 | 同上 |
| `deployment/__init__.py` | Python包入口 | Python上位机需要 |
| `policy_contract.json` | 模型校验值、接口、训练模式和部署模式说明 | 作为接口核对依据 |
| `export_validation.json` | 导出数值校验记录 | 留档 |
| `training_reference/` | 下表中的训练源码快照 | 供阅读，推理端不导入 |
| 本 MD | 控制流程与训练代码参考 | 给同伴 |

Python推理端依赖 `numpy` 和 `onnxruntime`。运行已导出的ONNX不需要Isaac、IsaacLab、Torch或训练奖励代码。ONNX中**没有相位时钟**，49维观测的指令字段仍由上位机填写。

## 2. 哪些训练文件作为参考

以下路径均相对于 `cross_stick_p3` 项目根目录。部署代码在包的 `deployment/` 中；下表主要训练文件的副本也放在包的 `training_reference/` 中，保留原目录结构，供同伴阅读。建议按表中顺序阅读。

| 文件 | 同伴应参考的内容 |
|---|---|
| `deployment/phase_clock.py` | WALK/LEAD/FOLLOW边界，时钟量化，启动信号不能反复归零，完成后的交接信号 |
| `deployment/policy_interface.py` | `JOINT_NAMES`、`DEFAULT_JOINT_POS`、`build_observation()`、`PolicyController.tick()`、`from_onnx()` |
| `tasks/manager_based/humanoid_robot_policy_rsl_rl/humanoid_robot_policy_rsl_rl_env_cfg.py` | `LEG_JOINT_NAMES`、`ActionsCfg`、`ObservationsCfg.PolicyCfg` 和IMU安装配置；原始49/12接口依据 |
| `tasks/manager_based/humanoid_robot_policy_rsl_rl/humanoid_robot.py` | 初始微屈膝角度、关节轴符号、仿真驱动参数 |
| `tasks/manager_based/humanoid_robot_policy_rsl_rl/fixed_stick_env_cfg.py` | 当前8cm间隙、10cm普通步长、23cm跨步、20ms任务配置、两阶段碰撞配置 |
| `hpc/fixed_stick_control.py` | 训练/回放从参数和checkpoint恢复同一套时钟设置，修改模式后应暖启动而不是续恢复 |
| `hpc/train_walk_stop_cross.py`、`hpc/fixed_stick_runner.py` | 权重暖启动、训练入口、保存 `infos.fixed_stick_control` |
| `hpc/play_walk_stop_cross.py` | 确定性推理、录像、指令相位与物理结果分别记录 |
| `hpc/export_fixed_stick_onnx.py` | 导出actor及其配套时钟，检查49/12和ONNX数值一致性 |
| 每次训练的 `fine_tune.json`、`params/env.yaml` | 核对该模型实际使用的模式、时刻、距离、动作比例及关节名 |

`mdp/fixed_stick_state.py` 和 `mdp/fixed_stick.py` 还含仿真足底几何、触地、奖励和成功判定。**上位机无需移植这些仿真评分逻辑**：本版指令由共享时钟生成，不使用其中的触地结果或 `expected_foot` 来切换。仿真的足部监控顺序是左、右；策略关节输入/输出顺序是右腿6个、左腿6个，两者不能混用。

## 3. 一次启动后的指令表

时间从第一次有效启动归零开始，使用单调时钟，例如 Python `time.monotonic()`。边界是候选值，之后通过仿真和微调修改；训练、回放、导出和上位机应使用同一份配置。

| 相位 | 区间，秒 | 控制tick | 前进速度m/s | 偏航rad/s | 步长m | 跨棍标志 |
|---|---|---|---:|---:|---:|---:|
| WALK | `0 ≤ t < 0.14` | 0～6 | 0.20 | 0 | 0.10 | 0 |
| LEAD | `0.14 ≤ t < 0.44` | 7～21 | 0.20 | 0 | 0.23 | 1 |
| FOLLOW | `0.44 ≤ t < 0.76` | 22～37 | 0.20 | 0 | 0 | 1 |
| DONE | `t ≥ 0.76` | 38起 | 0 | 0 | 0 | 0 |

在0.14秒开始的那次策略推理中，输入23cm及跨棍标志1；无需等待脚落地。FOLLOW的0cm是跟进/并齐目标，速度仍是0.20m/s。10cm和23cm是策略训练时的双脚脚尖相对纵向落脚目标，不保证真实脚精确移动该距离；8cm则是开始时脚尖到木棍近边的净间隙。

非20ms整倍数的边界向上取整到下一控制tick，例如0.145秒在0.16秒生效。不要靠睡眠次数累计时间；按单调时间与共享量化规则计算。策略每帧仍读取本体传感器闭环控制关节，但相位切换由时间决定。

`sequence_finished=True` 只表示**指令序列结束**。此时适配器停止调用跨棍actor，不再返回关节目标。上位机须向现有控制层发送“序列结束”，并由现有站立控制器接管。它不等于检测到双脚站稳或跨棍成功；本仓库没有真机电机通信、站立控制器或接管API实现。

## 4. 49维观测：顺序、单位和坐标系

ONNX输入名 `obs`，类型 `float32`，真机单机器人形状 **`[1,49]`**。下表索引从0开始，范围包含两端。没有额外观测归一化器，推理时不添加训练噪声。

| 索引 | 维数 | 内容 | 上位机处理 |
|---|---:|---|---|
| 0～2 | 3 | IMU加速度x/y/z | 机身坐标系、m/s²，乘0.1；包含重力对应的加速度计比力 |
| 3～5 | 3 | IMU角速度x/y/z | 机身坐标系、rad/s；如设备输出°/s，先转换 |
| 6～8 | 3 | 机身坐标系下单位重力方向 | 用IMU融合姿态求 `R_body_to_world.T @ [0,0,-1]` |
| 9～10 | 2 | 前进速度、偏航角速度指令 | 来自相位时钟；m/s、rad/s |
| 11 | 1 | 步长指令 | 来自时钟；米，不能输入10或23 |
| 12 | 1 | 跨棍标志 | 来自时钟；float32的0或1 |
| 13～24 | 12 | `q - q_default` | 编码器关节角，rad，按下面顺序排列 |
| 25～36 | 12 | 关节速度 | rad/s，相同顺序；仿真默认关节速度为0 |
| 37～48 | 12 | 上一次actor原始输出 | 未乘0.25、未加默认角度；开始新序列时置0 |

仿真IMU挂在 `base_link`，安装偏移为0、安装旋转为单位四元数，使用机身x向前、y向左、z向上的轴约定。真机先将IMU安装坐标变换到对应机身坐标，关节编码器则转换成USD同样的零点和正方向。左右腿pitch轴符号是镜像的，不能把左腿符号直接照搬右腿。

IsaacLab 3 PhysX IMU的加速度含重力比力：静止直立时应近似 `[0,0,+9.81] m/s²`，网络输入近似 `[0,0,+0.981]`。不要将设备已经去重力的“线性加速度”直接填进去。重力方向观测在同一姿态下是 `[0,0,-1]`，不是加速度计数值，也不是9.81。

`projected_gravity()` 接收**机身到世界旋转的XYZW四元数**；若IMU输出WXYZ、世界到机身四元数或NED坐标系，先转换。姿态必须使用融合输出，动态摆腿时不能简单把原始加速度归一化当重力方向。如果编码器只给位置，需要在控制层估计关节速度并保持采样时间与单位一致。

## 5. 12维动作及初始姿态

ONNX输出名 `action`，类型 `float32`，形状 **`[1,12]`**。输出是相对动作，转换公式：

```text
q_target_rad[i] = q_default_rad[i] + 0.25 * action[i]
```

| 索引 | 关节 | q_default，rad |
|---:|---|---:|
| 0 | r_leg_pitch_joint | 0.15 |
| 1 | r_leg_roll_joint | 0 |
| 2 | r_leg_yaw_joint | 0 |
| 3 | r_knee_pitch_joint | 0.30 |
| 4 | r_ankle_pitch_joint | −0.15 |
| 5 | r_ankle_roll_joint | 0 |
| 6 | l_leg_pitch_joint | −0.15 |
| 7 | l_leg_roll_joint | 0 |
| 8 | l_leg_yaw_joint | 0 |
| 9 | l_knee_pitch_joint | −0.30 |
| 10 | l_ankle_pitch_joint | 0.15 |
| 11 | l_ankle_roll_joint | 0 |

以上角度构成原微屈膝初始姿态。仿真root高度0.32m只是资产复位配置，不作为真机策略输入。启动前由现有控制层恢复此姿态、完成8cm定位，并让第一次推理及时执行。

本模型没有训练包装器动作裁剪，不能擅自按 `[-1,1]` 裁剪后当成训练原动作。若现有电机层有目标过滤或关节限位，`last_action` 仍记录actor原始12维输出；这些额外处理及电机延迟会改变执行效果，应与部署验证一起检查。

仿真关节位置驱动参数仅供对照，原值见 `humanoid_robot.py`；上位机继续调用现有电机控制层。不是向电机发送动作向量当力矩。

| 关节组 | stiffness | damping |
|---|---:|---:|
| leg_pitch / knee_pitch | 35 | 1.5 |
| leg_roll | 30 | 1.2 |
| leg_yaw | 20 | 1.0 |
| ankle_pitch | 15 | 0.8 |
| ankle_roll | 12 | 0.7 |

## 6. 上位机最小接入示例

在原型包所在目录导入以下接口。`controller`不负责定时线程或电机通信；下面三个回调由已有上位机程序调用。

```python
import json
import time
from pathlib import Path
from deployment.phase_clock import PhaseClockConfig
from deployment.policy_interface import PolicyController, projected_gravity

bundle = Path(".")  # policy.onnx、phase_clock.json所在目录
config = PhaseClockConfig(**json.loads((bundle / "phase_clock.json").read_text()))
controller = PolicyController.from_onnx(bundle / "policy.onnx", config)

def prepare_new_sequence():
    # 由上层先完成初始姿态、定位，再调用；不能在每次距离通知里反复reset。
    controller.reset()

def on_start_cue():
    # 最好与第一次20ms策略控制帧对齐；重复通知会被忽略。
    return controller.start(time.monotonic())

def on_control_frame(acc_body, gyro_body, base_to_world_xyzw, q_rad, qd_rad_s):
    sample = controller.tick(
        acc=acc_body, gyro=gyro_body,
        gravity=projected_gravity(base_to_world_xyzw),
        q=q_rad, qd=qd_rad_s, now=time.monotonic(),
    )
    if sample.sequence_finished:
        # 通知现有上层“序列结束”、接管到站立控制，然后结束本次跨棍循环。
        return "sequence_finished", None
    if sample.active:
        # 按12个关节的既定映射发送位置目标（单位rad）。
        return "joint_targets", sample.joint_targets
    return "waiting_for_start", None
```

上层安排20ms控制周期，并记录实际IMU/编码器采样时间、推理完成时间与目标发送时间。执行过慢时不能补发一串过去帧动作；先检查真实延迟与相位配置。正常结束后不要自动重新启动，新动作需要重新准备初始姿态、定位并显式reset。重复距离触发不能让正在执行的时钟重新归零。

## 7. 训练、微调、回放和导出

### Windows

在 `F:\RobotProject\cross_stick_p3` 执行。下面第一个checkpoint路径是已有模型，后续替换为自己的新训练模型。

```powershell
$python = 'F:\isaacsim\env_isaacsim\Scripts\python.exe'
$checkpoint = 'F:\RobotProject\cross_stick_p3\logs\rsl_rl\fixed_stick_stage1_v32\2026-10-02_17-07-45-315815_finetune\model_850.pt'

# 第一次改用时钟：暖启动，不加--resume。
& $python -X utf8 -B hpc/train_walk_stop_cross.py --checkpoint $checkpoint --stage 1 --command-mode phase_clock --num-envs 1024 --max-iterations 3000 --headless

# 用同一候选时钟做真实Kit 3D回放。
& $python -X utf8 -B hpc/play_walk_stop_cross.py --checkpoint $checkpoint --command-mode phase_clock --headless --video-backend kit --video-stride 4

# 重新导出actor及配套JSON和便携代码；之后用新模型替换$checkpoint。
& $python -X utf8 -B hpc/export_fixed_stick_onnx.py --checkpoint $checkpoint --command-mode phase_clock --output-dir exported/phase_clock_model_850
```

默认候选边界为0.14/0.44/0.76秒。可追加 `--walk-end-s 0.16 --lead-end-s 0.46 --sequence-end-s 0.80` 评估另一组边界；这只是参数写法示例，不表示该组更好。导出时也传入相同参数，或直接导出已经保存该组时钟的新检查点。

时钟训练日志写到 `logs/rsl_rl/fixed_stick_stage1_v32_phase_clock/` 和 `fixed_stick_stage2_v32_phase_clock/`，与旧触地调度日志分开。checkpoint新增 `infos.fixed_stick_control`，包含实际模式及全部时钟参数。

同一模式、同一边界、同一碰撞阶段续训可加 `--resume`，恢复优化器、迭代和纯净成功统计。**旧触地模型改为时钟，或修改边界时，使用不带 `--resume` 的暖启动**；否则入口会拒绝混用旧优化器/统计。回放和导出默认采用新checkpoint内的控制配置；没有该配置的旧模型默认采用本版候选时钟。旧触地方式仅供比较，可显式使用 `--command-mode touchdown`。

Stage 1仍是无实体碰撞、碰棍软罚；但计时到结束时必须真实完成全序列、两脚完整足底过棍并连续双脚支撑才能记成功。序列结束却未完成则记失败。时钟不会因为机器人提前落地就提前结束。纯净成功仍要求整个回合没有hit。

Stage 1在新时钟下达到准出后，用其新checkpoint `--stage 2` 暖启动实体木棍微调；不加 `--resume`。Stage 2恢复碰撞及严格碰棍终止。Stage 1旧模式的纯净率不能作为新时钟准出依据，`stage2_ready`只记录准出，不会在同一运行自动切换物理碰撞。

### HPC

沿用本项目容器、Isaac版本检查及脚本格式：Python3.12.x、IsaacSim6.0.x、IsaacLab框架3.0.x、RSL-RL5.4.x。新增 `deployment/` 目录也要上传，否则仿真无法导入共享时钟。

```bash
# 换成HPC中的checkpoint实际路径。
sbatch hpc/train_walk_stop_cross.sh --stage 1 --command-mode phase_clock \
  --checkpoint /workspace/cross_stick_p3/logs/rsl_rl/fixed_stick_stage1_v32/RUN/model_850.pt

# 新时钟模型在Stage 1达标后，明确切Stage 2暖启动。
sbatch hpc/train_walk_stop_cross.sh --stage 2 --command-mode phase_clock \
  --checkpoint /workspace/cross_stick_p3/logs/rsl_rl/fixed_stick_stage1_v32_phase_clock/RUN/model_N.pt
```

## 8. 本次验证记录

以下结果针对代码接口和候选时钟，不代表真机表现：

- 新增时钟测试检查同一tick命令不受触地/障碍几何影响、启动只锁存一次、部分环境重置、49维拼接、12维换算、计时结束但未越障不记成功，以及模式/边界改变的resume保护。
- `model_850` 的Stage 1真实Kit回放：净距0.079999998m，指令在第7/22/38帧切换，最终全序列完成、`hit=False`、`clean_success=True`。回放目录 `recordings/phase_clock_validation/model850_default/`，包含 `summary.json`、`trajectory.csv`、`obstacle-3d.mp4`。这是一回合确定性结果，不能当成批量成功率。
- 同一模型的Stage 2实体木棍Kit回放也在第38帧完成且无碰棍，目录 `recordings/phase_clock_validation/model850_stage2/`。这是单次实体仿真评估，模型权重仍来自Stage 1。
- ONNX checker通过；64组输入、batch1及batch64与原生actor比对，最大绝对误差约 `2.86e-6`。验证后端是ONNX ReferenceEvaluator；目标上位机仍需实际验证自己的ONNX Runtime与控制周期。
- 4环境、6次PPO更新的训练smoke已跑通，能保存时钟检查点。它不是充分微调，也不足以评价是否达标。
- 新时钟checkpoint再续训2次更新已通过，恢复了时钟配置、优化器和统计窗口。新旧自动测试共129项通过；包含自动复位首帧速度和较大单调时间原点的边界回归。

后续验证结果以本目录配套的 `summary.json`、训练检查点和 `fine_tune.json` 为准。最终交接模型应来自新时钟任务的充分训练/实体木棍评估，导出时保持其实际时钟配置。
