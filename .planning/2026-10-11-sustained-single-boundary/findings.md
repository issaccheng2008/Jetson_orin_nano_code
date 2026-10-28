# 已确认

- `trace_heading` 当前必须有当前帧双边采样；没有双边时无法输出有效航向。
- `_qualified_band` 单边要求最近双边宽度未超过 1 秒，不能持续跟随。
- `LineDetector.process` 的有效观测条件为合格近带与双边航向同时成立。
- `SegmentSteeringController` 可回退到 `HeadingSteeringController` 的统一近端路径，需保证单边通过检测器有效性并不被配对专用门槛挡住。
- 工作区已有多处未提交改动，尤其是 GPU 后端接口；细线过滤必须单独分离提交。当前 Git 写入被沙箱阻止，提交尚未完成。
- 第 162 帧保存的 contrast 二值图：末级 5 px 宽度过滤将外侧细框 408 白像素清零，粗弯线保留 4545/4571 白像素。
- `test_policy_vision.py` 全量 130 项里有两项需要本沙箱禁用的 UDP socket；另一个首卡重新出现的事件编号断言失败，仍需与当前工作区已有卡片改动分开核对，本轮未改图卡状态机。
