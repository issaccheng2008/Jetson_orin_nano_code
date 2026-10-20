# F/C Target Angle Blend Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement and review this plan task-by-task. The user approved mixing filtered target demand F with C = F + P, not mixing lane heading with target bearing.

**Goal:** Select heading-table angular velocity from `(1-w)*F + w*C`, default `w=0.3`, on a new branch based on remote commit `5212d7028749422ee9c097d89ab8cd4f8a7ab50d`.

**Architecture:** Latest remote code removed the old F/C stages. Restore only the demand filter and additive position correction required to calculate F and C. Retain the latest direct table lookup, recording, card handling, stop/loss handling and connector scheduling. A single configuration `STEERING_POSITION_BLEND_WEIGHT=0.3` controls C's share; F's share is `1-w`. Keep non-table steering behavior unchanged. The active heading-table path uses the blend, shadow mode records the trial without changing actual commands, and legacy mode remains the baseline.

**Tech Stack:** Python unittest, Python steering controllers, Bash button launchers, JSONL telemetry.

### Task 1: Implement and verify the configurable blend

- [x] Add failing behavior tests for F/C arithmetic, weight endpoints and invalid inputs, a filtered demand step, position sign/deadband/cap, no-position direction-only inputs, reset/loss, direct table lookup, and non-table/legacy/shadow isolation.
- [x] Run `python -B -X utf8 -m unittest discover -s tests -p 'test_steering_position_blend.py' -v` and confirm failures identify the missing feature.
- [x] Update `heading_steering.py` and `steering_filter.py`. Preserve GeometryFilter's default raw-demand behavior for unrelated callers; make filtering target demand an explicit heading-table opt-in. Restore old correction settings `position_gain=1`, `position_dead_cm=2`, `position_lookahead_cm=50`, `position_max_deg=12` as needed for this opt-in path only. Compute `P = clamp(-gain*degrees(atan2(signed_deadband(near), lookahead)), -max_deg, max_deg)` and `blend = F + w*P`. Log F, P, C, w, and the actual table input separately.
- [x] Add CLI `--steering-position-blend-weight` with finite `[0,1]` validation before camera initialization. Forward it through controller construction. Wire the matching env setting through `scripts/button_autostart_common.sh`, `scripts/run_button_vision.sh` and `config/button_start.env.example`. Preserve current window settings.
- [x] Verify with new tests plus `test_steering_filter.py`, `test_steering_config.py`, `test_position_steering.py`, `test_heading_steering.py`, `test_heading_integration.py`, `test_button_autostart.py`, and command-window tests. Update old tests only when they assert behavior intentionally replaced by the new active table feature; retain non-table expectations.
- [x] Review specification compliance and then code quality with independent reviewers; repair actionable findings.

### Task 2: Document, verify, and publish

- [x] Document F/C semantics, default configuration, endpoints, mode isolation, compatibility with old logs, and startup/restart instructions. Make the viewer show the actual blended lookup input for new logs while retaining the raw-demand fallback for old logs.
- [x] Run relevant controller, CLI, launcher, telemetry/viewer, connector, and startup regression suites. Confirm `git diff --check` and inspect the staged scope.
- Publication target: commit on `codex/steering-fc-angle-blend`, push without force, verify remote SHA, and report branch/configuration/test results. Do not merge. The final response records the published commit.

## Baseline verification

The pristine remote base passed all 78 `test_steering*.py` tests on Windows Python 3.12. Sandbox temporary-directory permissions required running the suite outside the sandbox.

## Final verification

- Combined controller/config/entrypoint/replay/Bash launcher suite: 205 tests passed, using `BUTTON_TEST_BASH=F:\Git\bin\bash.exe` with no launcher skips.
- Viewer data and Tk GUI suite: 17 tests passed.
- Connector window integration, executed windows, startup watchdog/sequence and log selection: 33 tests passed.
- Independent specification review approved implementation and source-transition coverage.
- Independent code-quality review approved the complete diff; its 15 blend tests and 12 replay tests passed independently.
- Total related verification: 255 tests passed. Hardware execution is not part of these desktop tests.
