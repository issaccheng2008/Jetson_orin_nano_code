# Command timing experiment implementation plan

**Goal:** Test one timing change on top of policy49-button-test, ce79df7.
**Architecture:** Change the heading/segments applied-command gate and the walking model-input gate together. Keep geometry, three-frame segment confirmation, command levels, camera parameters, PD profiles, models, button startup and stop precedence unchanged. Preserve the original 0.5 s trend window and prediction horizon independently of minimum command hold.
**Tech stack:** Python, pytest/unittest, local UDP tests; no robot hardware.

This worktree implements experiment 1: default normal minimum hold is zero in B and C. An explicit controller constructor option retains the original strict half-second policy for geometry/regression tests. The other experiment is an independent sibling from the same base, not stacked on this branch.

- [x] Check exact requested remote base, preserve existing workspaces, create isolated sibling worktrees.
- [x] Baseline: 305 tests and 758 subtests pass; original Linux tee child-process test excluded on Windows.
- [x] RED: prove default controllers release yaw before half a second, accept increases/reversals immediately after existing measurement filtering, preserve trend history, apply actual model observation values, and keep stops/takeovers immediate.
- [x] GREEN: separate hold from trend/prediction constants; change B/C default timing only. Keep original strict behavior testable explicitly.
- [x] Update historical timing tests to test explicit legacy configuration, and entrypoint timing assertions to match experimental defaults. Add real B -> connector -> C model-input regression.
- [x] Document exact branch behavior, existing button commands, and test limitations; run targeted and complete regression/compile/diff checks.
- [x] Independent review: no blocking findings.

Delivery: commit/push only this experiment to its own branch and verify base ancestry and remote SHA. Do not operate robot motors.
