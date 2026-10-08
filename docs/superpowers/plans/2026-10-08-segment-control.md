# Measured lane segment control implementation plan

Goal: connect policy49's measured ground segments to an opt-in `--wz-mode segments` that actually publishes steering commands, with a heading baseline recorded on the same frames.

Architecture: extend the existing heading controller's geometry input. Use accepted paired segments with near anchoring, adjacent continuity and three consecutive stable observations. Choose a target within the fitted segment's observed depth support, capped by `--heading-lookahead-cm`; never extend a segment beyond its observed depth. Preserve one shared half-second command hold, existing levels, near-position protection, geometry-loss behavior and external stops. Missing/untrusted segments use the original heading geometry. A separate heading controller calculates comparison commands without publishing them.

The human explicitly requested implementation after discussion of measured targets, fallback and A/B evaluation, then changed the destination to a direct commit and push to `policy49`. The isolated local worktree remains named `policy49-test`; no remote test branch is needed. Base: latest fetched origin/policy49 (5be871f).

- [x] Baseline: run heading, ground-heading, segment diagnostic, entrypoint, policy-vision and connector tests. Correct one stale card-settle expectation to the existing upstream 100 ms default; card behavior is unchanged.
- [x] RED: add known-geometry tests for depth support, turn entry/exit, mirrored direction, qualification/fallback, temporal confirmation, source switching, hold and stops, plus real entrypoint/UDP integration.
- [x] GREEN: publish observed inlier depth limits from lane_segments; implement SegmentSteeringController and a target-distance hook that leaves heading behavior intact.
- [x] Integrate CLI `segments`, automatically enable measured scanning, retain held command metadata and card/gate stop precedence, log geometry and shadow heading commands.
- [x] Verify new tests and baseline regression. Actual field recordings referenced in the upstream report are absent locally; use synthetic images through the real scanner and local UDP integration. Physical robot improvement requires real A/B runs. Final compile and full regression are recorded in the acceptance report.
- [x] Review implementation independently: no blocking findings. Document robot commands and fallback reasons.

Delivery: commit only task files and fast-forward push HEAD to policy49; verify the remote commit SHA. Do not operate motors. The delivery receipt is reported in the task response.

Acceptance: an observed distant bend must change published wz compared with near straight extrapolation; disappearing/invalid evidence must not invent far targets, reverse signs, bypass the hold or replay commands after an explicit stop. Default heading behavior remains available unchanged.
