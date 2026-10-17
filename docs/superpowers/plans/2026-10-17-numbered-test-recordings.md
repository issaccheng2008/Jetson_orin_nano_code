# Numbered test recordings implementation plan

**Goal:** On branch `codex/numbered-test-recordings`, based on `af53f1a054111c1c95a25c58c557ab93784ed5fc`, create `records/hz_finalltest/1`, `2`, etc., with a persistent `test_counter.txt` and no clock synchronization wait.

**Architecture:** Keep the existing once-per-button recording lifecycle and shared session directory. Allocate IDs under a nonblocking OS file lock, reserve a new directory exclusively, and atomically replace a flushed counter file. Never reuse existing numeric paths; preserve the counter when logs are removed. Invalid counters fail explicitly instead of resetting. Dates remain advisory manifest metadata, with `wall_clock_valid=false` before 2026.

**Tech stack:** Python standard library (`fcntl` on Jetson, `msvcrt` on Windows), unittest, Bash wrappers. No changes to motor, IMU, button, firmware or system clock behavior; no hardware deployment requested.

- [x] Replace clock-wait tests with failing coverage for epoch startup without sleep, persisted sequence across processes/restarts, deleted old directories, stale/missing/invalid counters, exclusive directory reservation, atomic-update failure, and concurrent allocation without overwrite. Add button integration at the epoch, maintaining deferred recording and one directory per launch.
- [x] Implement allocation in `new_vision/jetson/recording_session.py`; preserve manifest recording paths and add numeric `test_number`. Validate all selected log items before allocation. Use a stable hidden `.test_counter.lock`, nonblocking OS locking, unique temporary counter files, `os.replace`, file flush/fsync and parent directory fsync on POSIX. Keep a reserved directory if a write fails so retry skips its ID.
- [x] Change `scripts/run_button_vision.sh` to `$RECORDS_DIR/hz_finalltest`; update CLI help, config example, and shell expectations. Existing local configuration remains valid because the wrapper chooses the subdirectory.
- [x] Update current recording documentation and button instructions. Explain numbering, gaps, counter preservation, lock file, invalid counter recovery, advisory timestamps, unchanged old recordings and `LOG=none` behavior.
- [x] Verify expected tests fail before implementation, then run recording/button/log/video/analysis regressions and Bash tests. Review final diff and verify branch ancestry. Do not change or deploy the user's original dirty working tree.

Validation commands (from this worktree):

```powershell
python -B -m unittest discover -s tests -p test_recording_session.py
python -B -m unittest discover -s tests -p test_button_start.py
```

Run additional suites using `unittest.TestLoader.discover` for `test_log_selection.py`, `test_startup_video.py`, `test_command_video.py`, `test_recording_analysis.py`, `test_button_autostart.py`, `test_policy_gate_launcher.py`, and `test_vision_log_viewer.py`. For Windows Bash checks set `BUTTON_TEST_BASH=F:\Git\bin\bash.exe` if available; include `F:\Git\usr\bin` in PATH for `tee`.

Verification: 127 related tests passed with zero failures, errors or skips using Python UTF-8 mode, including real Git Bash wrapper checks. Independent review found no blocking issue; current operator and analysis documentation paths were updated. After explicit consent to upload the four named files, Linux SSH verification ran in an isolated /tmp/hz-numbered-* directory on Jetson: all 13 recording-session tests passed (0.384 s), including POSIX locking, directory fsync and concurrent processes. SHA-256 matched all four uploaded files to this branch. No running repository, service or motor process was changed. Original working tree remains untouched.
