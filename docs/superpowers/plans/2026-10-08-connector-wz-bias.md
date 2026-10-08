# Connector yaw bias implementation plan

**Goal:** Add a configurable signed yaw-rate offset in the connector for policy49-exp-no-hold.

**Approved design:** Add `--wz-bias` (rad/s, default 0), apply it once when receiving a moving vision command, then clamp yaw to [-0.5, 0.5]. Keep forward velocity and metadata processing unchanged. Do not offset full stops or posture requests. The watchdog and shutdown still send zero commands. Existing held/continuous smoothing follows the adjusted target.

**Files:** `connector.py`, `scripts/button_autostart_common.sh`, `scripts/run_button_connector.sh`, `config/button_start.env.example`, `docs/button_autostart.md`, `tests/test_connector_bias.py`, `tests/test_button_autostart.py`.

- [x] Add tests for positive/negative offsets, straight walking, limits, stop/posture/watchdog behavior, metadata, finite validation, CLI wiring and actual UDP output. Run them before implementation and confirm failures identify missing bias support.
- [x] Add optional `wz_bias=0.0` to `process_vision_output`. Validate finite bias; offset only a moving command without hold_upright/card_tilt; clamp after addition. Pass `args.wz_bias` from the receive loop and print the bias at startup.
- [x] Add `WZ_BIAS=0` to the config template and the common loader's defaults for existing installed configs. Pass `--wz-bias "$WZ_BIAS"` from the connector wrapper. Check explicit positive/negative values and missing-setting compatibility using real Bash dry runs.
- [x] Document manual and service use, offset examples, stop behavior, and the need to restart the connector (which also stops its dependent vision service).
- [x] Run connector regression tests, new UDP tests and wrapper tests. Review the diff and report local verification separately from Jetson deployment.
