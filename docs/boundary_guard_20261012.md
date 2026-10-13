# 左弯内侧越界：阶段一逻辑修复

分支：boundary_guard_20261012，基于巡线放宽版本 5461b83 独立创建。
依据：用户提供的《左弯内侧越界_代码问题与修改建议_20261012.txt》。本提交完成其阶段一的确定性逻辑问题；保留阶段二、三的测量前提，没有宣称足部保护已经标定。未部署、未启动电机，尚未实机验证。按该文档末尾要求，本分支不自动推送。

## 已实现

1. 确认位置回正后，RIGHT_ONLY 左转增强不再拥有覆盖权。最终还有左右对称的方向否决，放在普通规则之后。near 为正时不能重新变成正左转；near 为负时不能重新变成负右转。
2. 明确零偏航为 0；禁止右转的配置遇到右回正时，不再用可能为正的 nominal straight 档冒充零偏航。
3. 控制意图为 normal、boundary_guard、controlled_stop。保护立即穿过视觉中位数窗口，清除旧样本；解除后重新开始窗口，旧左转不能复活。图卡、启动、人工停止仍在外部拥有更高优先级。
4. 位置保护保持到新的位置证据解除。释放需要连续 POSITION_CONFIRM_FRAMES 帧，abs(near) 小于 POSITION_RECOVERY_CM 的 75%。出现另一侧达到确认条件时可以切换保护侧。仅方向、缺帧、丢线不构成释放证据；已有保护且位置不明时请求 vx=0,wz=0，保留电机/站立链路。这是停步请求，不是瞬时物理停止。
5. 实际入口开启发布反馈：client.publish 后，将窗口后的命令反馈给控制器及历史记录。候选被窗口保持替换时清除跨命令趋势样本。日志区分 candidate、window、published。该反馈是本层发布值，不能视为模型采用值或真实运动。
6. 不改 near 的旧含义。新增 body_cross_track_valid=false、foot_clearance_valid=false，相关身体位置/航向/脚余量字段为空，原因为 body_and_foot_transform_unverified。没有从前方 near 伪造脚下位置。

修改文件：heading_steering.py、steering_command_window.py、run_policy_vision.py、对应测试。segments 继承同一最终控制逻辑。没有改图卡时序、UDP格式、模型、标定或曝光。

## 配置及日志

继续使用已有 POSITION_RECOVERY_CM（默认 8 cm）、POSITION_CONFIRM_FRAMES（默认 2）。这些是在前方参考 Z 的位置代理门槛，不是距左脚边界距离。没有增加未经验证的增益默认值。

关键日志：steering_control_intent、steering_guard_side、steering_guard_source、steering_guard_reason、steering_position_candidate_wz、steering_auxiliary_candidate_wz、steering_guard_candidate_wz、steering_window_input_wz、steering_window_output_wz、steering_final_wz、steering_published_vx/wz、steering_publish_time_ns。保护侧 +1 表示 near 指示目标路径在右边，-1 对称。

解除迟滞只表示 near 代理量回到范围内，不能解释为脚部已经安全。未曾建立位置保护时，原先的丢线策略保留；没有声称解决未知风险下的全部动作选择。

## 下游接管边界

自动启动策略时，视觉窗口开启会通过 effective_policy_hold 把 C 的最小命令保持设为 0。connector 的 held 模式透传命令，不施加普通斜率限制。

但是 connector 仍可配置 --wz-bias；它会改变非全零命令，包括视觉明确请求的零偏航。因此本层的偏航符号保护只有在偏置未反向改变结果时才能成立。建议本分支验证时 connector 使用 --wz-bias 0，并核对实际配置。手动启动 C 时须自行使用 --command-min-hold-s 0 才能保证非零回正请求不额外等待。全零停步会抢占现有 WalkingCommandHold。

本次不更改 UDP 协议或偷偷绕过下游设置，不能把视觉发布反馈当作模型反馈。真实模型采用值应继续查看 C 的诊断记录。

## 测试结果

129 项通过：新边界保护 5、位置控制与指令窗口 26、持续单边 23、视觉入口 18、heading/segments 57。修正了两条原测试的旧预期：此前测试竟期望禁止右转后的“直走”是正 0.05，以及之后丢线回到左转；现在分别要求明确零偏航与保护未解除时停步。

覆盖报告 RIGHT_ONLY/near=+10cm/左向35° 的覆盖反例、左右镜像和三种来源、窗口立即接管及释放清理、丢线和仅方向不解除保护、发布历史记录窗口输出、外部停止及原入口回归。

圆弧公式测试说明 R=77.6cm、机身内偏5cm 时，前方25cm near进入死区、30cm符号反转。这里只验证“不能冒充身体位置”的事实，没有实现或验证身体坐标估计。

```bash
PYTHONPATH=tests python -m unittest test_boundary_guard test_position_steering test_steering_command_window test_sustained_boundary test_policy_vision.VisionEntryPointTests test_heading_steering test_segment_steering -q
```

本机还通过 PYTHONPATH=/tmp/robocup_photo_deps 提供 OpenCV。

## 尚不能完成的测量相关部分

文档建议的脚部扫掠包络、下一步余量、身体局部切线与停步能力，需要相机到身体/双脚变换、脚尺寸/站距/摆幅、模型实际转向和停步延迟。当前任务未获得可信标定，不能编造尺寸，也不能靠提高位置增益补上。这些字段显式无效。阶段二的足部/曲线风险状态机和阶段三的增益定值尚未实施；当前只是 near 代理保护。

保护会在已确认偏移后遇到低质量/仅方向时增加停步，这是本阶段的明确行为变化。先离线及支撑条件验证，再决定实车使用；不能据此保证脚不越界。

回滚完整控制修复：回到 5461b83；该版本仍保留巡线接纳放宽。加强补线独立留在 lane_loss_recovery_20261012，未混入本分支。

## 2026-10-13 合并 lane 更新

合入 lane_loss_recovery_20261012 的 e9edb82，保留 boundary_guard 原有保护控制与高角度 +0.5 档位配置。
新增内容包括数值严格度默认50%、独立仅方向回退开关、直道识别/档位限制、双摄像头起点及停车投票、对应离线对照记录。
前文“加强补线未混入本分支”描述的是先前版本；当前合并已纳入 lane 最新实现。

唯一入口冲突按以下顺序解决：现有命令窗口 → 直道限档 → 最终输出记录/发布反馈。
边界保护 boundary_guard 和明确停步 controlled_stop 跳过直道限档，以保留本分支已确认位置保护的优先级；普通巡线与丢线补偿仍受直道限档。
外部图卡停车、启动门控、hold-still 的接管保持不变。未改步态模型、串口和电机逻辑。

208 项相关离线测试通过；包含原边界保护、位置控制、指令窗口、单双线、阈值、heading/segments、按钮入口及双摄像头。
新增集成用例用合成鸟瞰真实黑线确认直道，然后注入现有控制器的保护意图，验证入口最终发布：普通丢线高档被限制，边界保护高档保留，明确停车仍为零。该用例验证合并后的优先级，不能替代位置估计实车验证。
Python 编译、Bash 语法、diff检查通过；历史CSV保留CRLF，diff检查采用cr-at-eol。
尚未实机验证，本次未部署板子、重启服务或使能电机。
