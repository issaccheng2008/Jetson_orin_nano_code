# Command timing experiment implementation plan

**Goal:** Test one timing change on top of policy49-button-test: initially ce79df7, synchronized to 6201468 before delivery.
**Architecture:** Change the heading/segments applied-command gate and the walking model-input gate together. Keep geometry, three-frame segment confirmation, command levels, camera parameters, PD profiles, models, button startup and stop precedence unchanged. Preserve the original 0.5 s trend window and prediction horizon independently of minimum command hold.
**Tech stack:** Python, pytest/unittest, local UDP tests; no robot hardware.

This worktree implements experiment 2: B/C allow earlier same-sign yaw reductions/zero while linear velocity is unchanged. Increases, reversals and velocity changes retain the 0.5 s minimum. Each accepted change starts a new block; reducing yaw never enables an immediate increase or reversal. An explicit constructor option retains original strict behavior for geometry/regression tests. The other experiment is an independent sibling from the same base, not stacked on this branch.

- [x] Check exact requested remote base, preserve existing workspaces, create isolated sibling worktrees.
- [x] Baseline: 305 tests and 758 subtests pass; original Linux tee child-process test excluded on Windows.
- [x] RED: prove default controllers release yaw before half a second but keep increases/reversals and velocity changes held, preserve trend history, apply actual model observation values, and keep stops/takeovers immediate.
- [x] GREEN: separate hold from trend/prediction constants; change B/C default timing only. Keep original strict behavior testable explicitly.
- [x] Update historical timing tests to test explicit legacy configuration, and entrypoint timing assertions to match experimental defaults. Add real B -> connector -> C model-input regression.
- [x] Document exact branch behavior, existing button commands, and test limitations; run targeted and complete regression/compile/diff checks.
- [x] Independent review: no blocking findings.

Delivery: commit/push only this experiment to its own branch and verify base ancestry and remote SHA. Do not operate robot motors.

- [x] Synchronize both published experiments with button-base 6201468 using non-rewriting merges; full regression rerun: 470 passed, 869 subtests passed, 2 Bash tests skipped, 1 Linux tee test deselected on Windows.
