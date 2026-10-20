# Raw demand steering table implementation plan

**Goal:** Use `steering_demand_deg` directly for configured angle tables, record it with `LOG="vision"`, and show it in the desktop viewer.

**Design:** Delete target-demand history smoothing and additive position correction, including their dedicated constructor/CLI/startup settings and obsolete diagnostics. For accepted geometry and a configured angle table, use the current raw target bearing without table hysteresis, prediction braking or position overrides. Keep heading/position observation filters, normal command hold, missing-geometry recovery, external stops, straight-road limits and connector windows. Non-table mode retains its independent protection/level logic with the same two removed layers. Store the exact raw demand in `measurement.steering_demand_deg`; missing geometry is null. Display it independently of heading and output yaw. Existing logs are read without recalculation.

**Scope:** Apply the user's specified control design and UI addition in the existing checkout, preserving unrelated uncommitted work. Deploy to Jetson's existing `card_recognition_debug_20261016` branch. Do not restart robot services or start motion.

## Tasks

- [ ] Add regression tests in `tests/test_raw_demand_table.py`: replay frame 839 geometry, large-to-small demand steps, exact table boundaries, position overrides, command hold and missing geometry.
- [ ] Run `python -m unittest discover -s tests -p test_raw_demand_table.py -v` and confirm the current implementation fails.
- [ ] Update `heading_steering.py` and `steering_filter.py`: bypass demand smoothing for table mode, select raw-demand table values through a shared command adoption helper, and expose table-input diagnostics.
- [ ] Add data/GUI tests for exact raw demand, absent/invalid data and an eight-column viewer. Update `vision_log_data.py` and `view_vision_log.py`.
- [ ] Verify `LOG="vision"` records the unrounded demand without video/control logging; document the field and new control semantics in `docs/vision_log_viewer_zh.md`.
- [ ] Run affected steering, telemetry, integration, GUI tests and the actual log reader. Review the diff and commit only task files and required existing viewer dependencies.
- [ ] Back up affected Jetson files and diffs; push the tested commit to its current branch; synchronize by SSH without discarding local modifications; verify commit, file hashes, config and tests remotely.

**Validation:** The recorded frame-839 geometry must give raw demand 9.327595389 degrees and table output 0.1, independent of demand history or position overrides. Heading smoothing, independent non-table protection and command transport tests must continue to pass. GUI must display the logged demand separately from heading, with missing data clearly marked.
