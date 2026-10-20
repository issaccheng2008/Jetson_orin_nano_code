# Remote-only boot startup

User selected: boot the webpage; start the walking runtime when the phone's Enable button is pressed. No physical button or vision startup participates.

- [x] Add a managed policy subprocess launched only by an authenticated successful control claim. While STM32/runtime readiness is pending, accept zero heartbeats and suppress motion. Failed launches revoke the lease and require explicit retry.
- [x] Reuse the existing atomic startup-ready marker for remote mode, retaining the motor-enabled/fresh-state confirmation. Vision keeps its existing behavior.
- [x] Stop/revoke/timeout shuts down a managed run; web process restart never automatically launches the policy. Manual two-terminal remote mode remains available.
- [x] Teach the existing page to show readiness and ignore motion until ready; preserve its layout and stop/race protections.
- [x] Provide private independent remote config, installer preflight/dry-run, a single boot service, and persistent disabling/masking with backup of the two legacy services. Retire actual legacy installation on this branch.
- [x] Verify manager transitions, startup failure/stop races, HTTP/UDP zero gating, frontend waiting behavior, shell syntax and generated units, then run regression suites and push both merge and startup commits.

Installation is code-only here; systemd replacement must be executed on the Jetson. Tests must never start a real model or enable motors.

## Verified result

- Root Python suite: 845 run, OK with 1 environment-dependent skip; Git Bash enabled the legacy/new installer tests.
- Runtime Python suite: 184 passed. Frontend state suite: 13 passed. Real browser checks: 21 passed.
- Shell syntax and whitespace checks passed. A real subprocess stub exercised readiness, UDP and shutdown without executing production main.py or hardware code.
- Independent review found incomplete preflight bounds/token validation; exact runtime ranges and token checks were added before migration, with reproducing regression tests. Review then passed.
- No systemd installer, Jetson service, ONNX inference or real motor was run here. The user must install on the Jetson for replacement to take effect.
