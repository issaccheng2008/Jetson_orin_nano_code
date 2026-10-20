# Executed yaw windows implementation plan

**Goal:** Collect each vision frame's mapped, filtered steering demand in half-open 0.5 s windows, select the configured lower/upper median, and hold each applied walking yaw for at least that duration. Card stops preempt immediately.

**Architecture:** The existing 50 Hz connector closes windows independently of image processing. Vision sends one candidate and window settings per completed frame; geometric restrictions apply before sampling. The connector continues sending the last applied command between boundaries. The first window sends an explicit stop while collecting. Empty windows reuse the applied command (or the current loss compensation if loss samples exist). Windowed vision has a separate 1 s freshness timeout; stale input, card/posture requests and shutdown stop immediately and discard pending samples. No new worker thread is needed. The window duration also reaches the actual model-input hold through UDP, including independently launched policies. Timer jitter may lengthen a block but must never shorten it or replay missed blocks in a burst.

**Configuration:** Preserve STEERING_COMMAND_WINDOW_S and STEERING_COMMAND_MEDIAN. Zero disables aggregation. Add WINDOW_VISION_TIMEOUT_S (default 1 s) to the connector launcher. Candidate bins use connector monotonic receipt time, after vision processing has completed; a frame received at 0.52 s is never backdated into [0, 0.5).

**Recording:** VISION becomes the per-frame candidate after geometric restrictions; ROBOT CMD remains applied model input after scheduling/holds/takeovers. Record the candidate protocol in run manifests and update video semantics.

- [x] Add failing deterministic tests for first-window collection, the user's .12/.30/.40/.52 example, upper/lower and signed medians, exact boundaries, empty/lost windows, delayed ticks, immediate card stops, watchdog reset and metadata validation.
- [x] Implement a pure connector window scheduler and integrate it into the 50 Hz loop without changing legacy/continuous commands.
- [x] Send candidate/window metadata from vision, move road restrictions before collection, and restore/propagate actual model hold. Add receiver and real model-input regressions, including immediate zero.
- [x] Update configuration/help/recording semantics and affected integration expectations.
- [x] Run targeted and full vision/policy suites, replay the supplied log through the scheduler and real hold, inspect the diff. No hardware motion or deployment is performed.

## Verification

- Policy suite: 177 tests passed.
- Vision suite: 679 run, 660 passed, 17 skipped, one failure and one error. The failure in `test_line_state_snapshot` (tracker-object equality) and error in `test_policy_gate_launcher` (Windows lacks `tee`) also reproduce using the committed original tests and unchanged implementation files.
- New scheduler tests: 14 passed. Real model-input/hold tests: 16 passed.
- Read-only independent code review identified a hold-duration leak across mode changes. Regression tests first reproduced it; per-block duration snapshots now preserve the old contract while restoring the configured duration for future commands. Re-review found no blockers.
- Supplied test replay: 477 vision records / 3065 original control timestamps, no non-stop changes shorter than .5 s. Only initial collection and card-stop intervals remain zero. This validates timing only, not robot motion.
- Test logs and replay report are under ignored `records/yaw_window_*`.

## Configuration and remote integration check

- Fast-forwarded `lane_loss_recovery_20261012` from `6e98a5f` to remote `1a9df08` before reapplying this work. Preserved the remote dual-camera/device-role and numeric lane-threshold features.
- Validated the user's supplied `button_start.env.example` through both real Bash launchers in `--dry-run` mode and their actual Python parsers. Every existing assignment matches the integrated repository template; the only added setting is `WINDOW_VISION_TIMEOUT_S=1`, which is also the backward-compatible default when absent.
- Confirmed configured hold `0` plus window `0.5` produces effective model hold `0.5`, median `lower`, bias `0`, strictness `50`, and both named camera paths. `WZ_STEP=0.3` continues to cap the configured higher yaw levels. Older USB 2.1 comments are superseded by the remote USB 2.3/2.4 main-camera rule.
- All 16 Bash startup/configuration tests passed with Git Bash and UTF-8. All 177 policy tests passed after remote integration.
- Four remaining vision test failures also reproduce with the original remote entrypoint and tests: tracker-object snapshot equality, Windows handling of a mocked `/dev` path, and two older single-camera test fixtures using a plain `Mock` with the new dual-camera default. These baseline issues are recorded separately from this feature's regressions.
- Final integrated vision run with UTF-8 and Git Bash: 696 tests, 691 passed and five skipped (the four verified baseline failures explicitly marked, plus one existing skip). No additional failures. Final policy run: 177 passed. `git diff --check` passed.
