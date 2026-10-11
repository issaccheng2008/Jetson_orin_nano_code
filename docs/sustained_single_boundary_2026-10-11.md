# 持续单边界巡线：实现与离线验证

## 接入链路

`run_policy_vision.py` 的实际入口仍将相机帧交给 `LineDetector.process()`；新模块 `lane_boundary_path.py` 从现有 contrast/legacy 候选掩膜的真实 run 追踪左右边界。已有双边近带与逐行航向继续优先。它们不足时，只有通过独立单边几何检查的结果才写入原有 `near_error_cm`、`near_z_cm`、`heading_control_deg`、`measurement_valid`、`fused_err_cm`。`HeadingSteeringController` 直接读取这些字段；segments 的双边路径不成立时，回退到相同的单边目标路径。UDP、步态和停车业务未改。

控制接口的地面坐标是 X 向右、Z 向前，单位厘米。内部斜率 `dX/dZ` 向右为正，公开航向仍向左为正。单边先拟合实际胶带中心线，再按法向偏移半个等宽赛道宽度；法向同时改变 X 和 Z，最后在统一的近端 Z 上求交。虚拟目标只以紫色菱形画在调试视图，不作为另一条实测边界，也不更新宽度或横向在线比例尺。

## 接受与拒绝

- 双边逐行追踪仍需当前真实双边观测；它建立左右身份与实测法向宽度。
- 单边需要身份来自可信双边历史并连续关联，或者冷启动显式指定 `--initial-boundary-side left|right`。默认 `unknown`，单线冷启动返回 `side_ambiguous`。
- 每帧单边至少有 6 个真实 run，纵向跨度至少 8 cm，局部拟合残差不超过 2.5 cm；方向绝对斜率不超过 1.2，跨帧方向突变不超过 35°。弯道选择可覆盖近端参考位置的短局部直线，最多允许 3 cm 的参考边界外推。
- 单边依赖配置的等宽赛道宽度。最近真实双边法向宽度与配置偏离超过 18%，或已识别处于窄门区域时，拒绝单边目标。旧双边宽度的年龄单独记录；真实单边每帧仍可见时不会因其超过 1 秒而过期。
- 真实边界消失、横线/孤立黑斑/远处卡框、身份歧义、旧帧重复、参考距离超出可靠支持范围时拒绝。BOTH 与单边目标在同一参考位置冲突超过 8 cm 时，不直接平均；冲突双边需下一帧一致再接管。
- `tracking_mode`、`measurement_source`、左右真实点数及年龄、宽度来源及年龄、路径质量与拒绝原因进入逐帧日志。单边质量低于双边。`paired_rows` 不会被伪造。

这仍然是局部几何估计：只有一侧时无法验证另一边的安全余量，也无法发现没有视觉证据的突然变宽/变窄。宽度配置必须与实际胶带中心线间法向距离一致。

## 细线过滤与录像对照

contrast 的二值候选在连通域筛选前增加 5×5 椭圆开运算，移除长而很细的场地外框，尤其是与粗胶带相连的细枝。`legacy`、LUT、曝光、IPM 不变。调用时需显式选 `--line-preprocess contrast`；当前生产入口默认值仍保留为 `legacy`，不会悄悄切换。

用仓库已有的 `docs/line_preprocess_camera_commands_2026-10-11` 五张录像鸟瞰截帧，在已保存的 contrast 二值结果上重放同一末级宽度筛选：总白像素分别减少 25.3%、8.7%、3.5%、8.6%、10.7%。第 162 帧右侧细外框区域由 408 降至 0，真实粗弯线区域由 4571 降至 4545。该录像左画面已经经过鸟瞰与视频压缩，不能当成未处理原始相机帧。用此帧加**显式左边界身份**和近似比例尺做单帧几何演示，可以输出局部路径；这些数值不用于实车控制评价。

## 配置与回滚

视觉端推荐显式使用 `--line-preprocess contrast`。默认 `--single-boundary-tracking` 已启用，冷启动侧别默认 `--initial-boundary-side unknown`。只有起点朝向和可见边界侧别已独立确定时才选 `left` 或 `right`。

要在同一版本回到旧双边控制条件，在视觉端加 `--no-single-boundary-tracking`。这样双边失败后仍走原来的丢线分支；不必回滚其他工作区已有改动。录像中若相机帧完全重复，入口的帧 CRC 会使启用的单边跟踪拒绝刷新观测。

## 离线测试与边界

`python -m unittest discover -s tests -p test_sustained_boundary.py -q` 覆盖双边、持续左右单边各 5～8 秒、中轴跨越、法向圆弧、缺失边界在画面外、模式切换、错误宽度、卡框/横线/空白、重复旧帧、真实检测器到 heading/segments，以及模拟入口发布。其他聚焦测试：`test_continuous_lane_heading.py`、`test_line_geometry_p1.py`、`test_line_ground_heading.py`、`test_heading_steering.py`、`test_segment_steering.py`、`test_line_preprocess.py`、`test_policy_vision.py` 的 `SingleLineTrackingTests`、`LineDetectorStateTests`、`VisionEntryPointTests`。

上述组合在干净暂存快照上共 152 项通过，包含实际检测器与入口回放测试。完整 `test_policy_vision.py` 的 131 项在允许本机 UDP socket 的离线环境中也全部通过；受限沙箱中有 2 项仅因禁止创建本地 socket 而报错。连续 541 帧旧鸟瞰录像的当前与关闭单边接管对照、14 帧单边接管的原图以及逐帧 CSV，见 [连续录像回放](sustained_single_boundary_video_2026-10-11/README.md)。

没有实机验证。已有加工视频可验证连续状态切换和可见胶带的明显误接，无法证明机器人在真实抖动、曝光变化、卡片遮挡和转弯过度时仍安全。要计算正式误接率，还需要对未叠字原始连续帧逐段标注左右身份和可见性。
