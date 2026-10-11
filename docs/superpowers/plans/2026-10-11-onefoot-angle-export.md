# One-foot angle export implementation plan

**Goal:** Automatically add separate host-target and STM32-feedback CSVs while one-foot standing is active.

**Design:** Add a focused CSV logger with two files per process run. Reuse the final motor target passed to send_command and the live STATE joint_position sampled for that control tick. Both files share host timestamp, elapsed_s, step and state_sequence; angles use radians and config.JOINT_NAMES order. Existing motor/IMU logs remain available.

Files are created lazily on the first successfully written onefoot46 command, including standalone --policy one-foot and vision card one-foot actions. Walking and fixed playback create no one-foot files. A False send result or a raised send exception creates no exported sample. Feedback is the latest pre-command STATE sample, not an acknowledgement or measured response to that same target. File pairs use a common timestamp and exclusive creation. Line buffering makes completed rows readable during the run. ExitStack closes both handles on constructor failure; main finally closes them on normal exit or fault.

Implementation stays in the primary Jetson_orin_nano_code checkout; its pre-existing tracked bytecode modification is preserved.

- [x] Inspect main.py, position_monitor.py, serial_link.py and existing one-foot/shape tests.
- [x] Add tests/test_one_foot_angle_log.py for paired schemas, distinct runs, immediate reads, invalid dimensions, actual send arguments, raw STATE feedback, mode filtering, failed sends, and fault shutdown.
- [x] Run the focused tests and confirm missing-export failures.
- [x] Implement one_foot_angle_log.py with OneFootAngleCsvLogger(log_dir, joint_names), write(elapsed_s, step, state_sequence, target_motor_rad, actual_motor_rad), and idempotent close().
- [x] Integrate lazy creation/writing for diagnostic_mode == "onefoot46" and send_result is True; close in finally.
- [x] Extend the existing diagnostic fixture that sends one-foot commands to use a temporary position-log directory.
- [x] Document file names, output directory, units and sample timing.
- [x] Run python -B -m unittest discover -s tests -v from humanoid_jetson_deploy, plus git diff --check; inspect the final diff.

No motor or serial hardware is available in this validation. Tests use real CSV files and simulated serial/policy boundaries.

Verification: the initial focused run reproduced missing-module/file failures. After implementation all 5 new tests passed; the complete deployment suite passed 173 tests in 9.550 seconds (exit 0). git diff --check passed. Test-generated protocol bytecode was restored; the pre-existing config bytecode modification was retained.

Cloud target corrected to policy49flitter. Ported the export commit into its existing clean worktree, preserving apply_lift_target_scale and configurable command holding. Baseline: 178 tests passed; after port: 183 tests passed in 8.189 seconds. The original policy49 cloud branch was not updated.
