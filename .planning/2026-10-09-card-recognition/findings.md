# 初始证据

- c7ad975默认selective+normalize原31图27正确0错类，仍四张没有有效类别；gain0.6/0.4模拟分别26正确/2错类与26正确/1错类。
- 全图Otsu + selective并集先二值再warp可能改变薄笔画拓扑；需用卡片原灰度局部rectify验证，不用Canny双路替换（上轮6正确/6错类）。
- 当前模糊BH31+close9仍依赖闭合环/孔，框边断开可能完全无候选。
- existing presence独立armed，9窗2命中/recent3，出口cue归一化单位1.2；累计未确认候选只能远处重武装、不能停车/投票；确认漏帧不供旧cy。这些保持。
- run_policy_vision.py模糊presence无quad时退cy触发，对俯仰敏感；先验证检测改善，避免擅自改变距离/停车合同。
- 独立原图人工标注16连续片段/15静态照片，正80calls匹配真实卡presence仅24，负167calls无cue/presence。单看rawpresence40会高估召回（错框）；按真实ROI匹配度量。
- 正确相机标定下WIN20260922三角11图仅2正确/4错类/5未检。数据agent初始git archive只快照jetson导致camera_config缺config fallback，root发现与真实运行不一致，已补new_vision/config并逐项断言，旧WIN4/3/4明确撤回。
- 精准warp固定缩边14%可能裁掉靠外边的图形，旧binary先warp进一步丢细笔画；探索完整原灰度矫正与外框剥离/分割一致性。
