# Steering configuration implementation plan

> Execute in the current clean policy49flitter checkout; retain existing user backups.

**Goal:** Configure final decision angle → rad/s and measured observation regions without increasing per-frame scan budgets.

**Architecture:** Parse and validate JSON arrays once at startup. A shared angle-table helper serves heading and segment controllers, including prediction and shadow replay. Region lengths define segment count; continuous observed support and heading fallback remain mandatory.

**Tech stack:** Python standard library, NumPy/OpenCV, Bash env config, unittest.

- [x] Add failing behavior tests: exact table boundaries/sign/caps/hysteresis and position guards; near/far distance ordering and eight-row scans; two/four-region fitting and continuity; Bash-to-CLI and replay options.
- [x] Add `steering_config.py`: normalize contiguous degree/rate intervals covering [0,90], reject rates above caps; normalize 2–5 contiguous ground intervals of at least 6cm within 20–70cm.
- [x] Wire table into nominal filtered/legacy/predicted steering, derive turn magnitudes from the table, preserve corridor/position/loss guards, and retain old behavior when unset.
- [x] Add atomic near/far distance setup, move lock rows and scale anchor with configured near band, preserve legacy near rows when unset, and recalculate on pitch changes.
- [x] Share configured segments between detector/controller/replay; stop at first broken join and retain observation/confirmation gates. Generalize diagnostics beyond three regions.
- [x] Pass optional settings through `run_button_vision.sh`; document configuration and startup validation in existing button documentation.
- [x] Run focused and existing vision/controller/Bash tests and request focused independent review.

Validation: 316 related unittest cases pass with Python UTF-8 and Git Bash; independent review found no blockers and separately passed the 10 new configuration tests. Full-suite connector/command-timing failures (5 subtest failures) reproduce on unchanged `88df0b3`; connector code was not changed. Delivery uses a normal commit and direct push to `policy49flitter`, followed by local/remote head verification.
