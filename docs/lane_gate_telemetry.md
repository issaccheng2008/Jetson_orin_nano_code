# 每帧巡线门槛日志

line_frames.jsonl 的 measurement 内增加标量字段，即使最终 BOTH 或 VISIBLE_ONLY 接管也保留底层拒绝原因。

lane_gate_*：实际点数、跨度、残差、斜率、跨帧/行关联半径、歧义距离差、参考Z与允许外推距离；history/path_jump 的 enforced 明确表示这些阈值是否真的启用。

lane_scan_gate_*：连续扫描配置全部数值；lane_scan_valid/reject_reason/observed_rows/paired_rows/span_cm/residual_cm 为该帧实际扫描结果。

left/right_association_*：最后一次关联比较的最近距离、实际搜索半径与次近候选距离差。不是全帧平均值。

left/right、paired_left/paired_right/paired_center 前缀：input_points、fits_attempted、rejected_原因次数、reference_rejected、fit_reason、selected_points/span_cm/rmse_cm/slope、target_z_min_cm/max_cm。rejected_* 是该帧尝试的局部拟合窗口数，不是丢线帧数；未执行为 not_attempted，未选中值为空。原始底层原因保留在 boundary_tracker_reject_reason，最终接管模式仍看 tracking_mode。

这些字段记录现有门槛，不新增 CLI 调参入口，不改变接受逻辑。strict 双边路径不执行局部拟合的条目标记 not_attempted，不能理解为失败。
