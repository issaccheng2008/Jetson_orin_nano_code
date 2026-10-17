# Vision debug parity implementation plan

**Base:** af53f1a054111c1c95a25c58c557ab93784ed5fc, verified latest remote development commit on 2026-10-17.

**Approved goal:** Match visual perception and steering to the recorded standalone debug invocation. Preserve button gate, policy launch, operational command ports, startup lifecycle and execution feedback. Do not edit the ignored installed config or start robot motion.

**Design:** Add a named `debug` vision profile at the entrypoint, reusing the same parser defaults as standalone debug. Explicitly pin its environment-sensitive defaults to the recorded camera/vision settings. Apply the profile before normal validation so manifest arguments contain effective values. Button wrapper selects it with `VISION_PROFILE=debug`; `configured` retains editable per-setting operation. Existing saved parameters may remain but are overridden in debug profile. Keep feedback reception while disabling only IMU-driven camera-pitch correction to match `--attitude-port 0` geometry.

## Tasks

- [x] Write failing parser parity tests against the ordinary debug CLI and tests preserving button/ports/feedback.
- [x] Implement profile selection, shared defaults and explicit static-pitch control; verify matching normalized arguments and unchanged configured mode.
- [x] Update button wrapper and tracked example config, add dry-run tests against old conflicting settings, and document the single manual config switch plus effective differences.
- [ ] Run affected parsing, camera/card/attitude, steering, log and launcher tests. Run Bash dry-run on Jetson in an isolated diagnostic checkout without touching its current checkout/config/services.
- [ ] Independently review, commit and push the new branch; report tests, commit and manual config steps.

**Validation:** A debug-profile button invocation matches standalone debug perception/steering arguments even when old CLI tuning requests contrast, narrow regions, active filtering or an angle table. Effective pose use is static for both while robot feedback remains available. No FPS improvement is promised: debug still includes the same card detector.

Local verification: 318 tests passed with 21 Bash-only skips on Windows. Independent review has no remaining findings. Jetson Bash verification follows publication of the isolated branch; its current checkout and ignored config are preserved.
