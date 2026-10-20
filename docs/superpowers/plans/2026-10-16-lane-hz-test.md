# Lane camera and recording changes

Goal: Disable the secondary card camera reversibly on lane_loss_recovery_20261012,
move button-start test sessions to records/hz_test, and prevent epoch-date folders.

- [x] Trace secondary capture lifecycle and recording directory creation.
- [x] Add regression cases for default camera disable/explicit restore, old button
  configurations, and unsynchronized/synchronized clocks; reproduce failures.
- [x] Default manual secondary capture to -1. Add CARD_CAMERA_ENABLED=0 to the
  button wrapper defaults; retain device, calibration, and voting code for restore.
- [x] Replace the wrapper's records/tests destination with records/hz_test.
- [x] Reject pre-2026 dates for folder naming. Use clock_unsynced and a monotonic
  timestamp plus UUID without blocking startup; preserve observed wall time and
  validity in the session manifest. Never split an active session after clock sync.
- [x] Run hardware-free camera, wrapper, recording, and policy regressions.
- [x] Document paths, time fallback, and restoration commands.

Verification: run unittest discovery for the affected modules with a writable
temporary directory. Run Bash syntax and dry-run wrapper checks without hardware.
Do not modify existing local changes or historical recordings.

Result: 185 affected-module tests passed, including real wrapper dry-runs,
loopback UDP and fake-child log propagation. Bash syntax and git diff --check
passed. Read-only code review found no actionable issues. Hardware deployment
and camera/motor testing were not performed.
