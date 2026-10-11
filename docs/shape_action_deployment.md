# Shape cards: Jetson and STM32 deployment

Vision confirms six 10 cm cards through the existing vision -> connector ->
policy UDP path. `qr` is the current observation (-1 when absent).
`event_id` and `event_action` identify one confirmed action and stay available
during the stop window. The host deduplicates event IDs, and a visible card
cannot restart its action when classifier cooldown expires.

| Card | ID | Controller | Action |
|---|---:|---|---|
| Circle | 1 | STM32 | Raise left hand |
| Star | 2 | STM32 | Raise right hand |
| Square | 3 | STM32 | Lift left leg using the PA4 pose |
| Diamond | 4 | STM32 | Lift right leg using the PC0 pose |
| Cross | 5 | STM32 | Raise both hands |
| Triangle | 6 | STM32 | Shake head |

All six cards use `ACTION_REQ` / `ACTION_STATUS`. The host waits for stopped
joint feedback before requesting an action; leg cards also wait for the card
restore ramp (action 8). STM32 owns the legs during a leg action and suppresses
host leg targets while continuing COMMAND validation, watchdog/ESTOP handling,
and live STATE feedback. The host keeps its 50 Hz COMMAND heartbeat and retries
the same action event at 10 Hz until DONE. Its action timeout is 8 seconds.
`--fixed-policy` keeps its original playback path and does not consume events.

Leg timing is 25 preparation steps at 20 ms, 50 lift steps at 20 ms, a
3-second hold after reaching the lift pose, 50 restore steps at 20 ms, then
300 ms of settling: 5.8 seconds. Restore uses the exact GPIOC13 standing pose.
STM32 holds that final pose for a further 100 ms after successfully reporting
DONE so the host can take over. Vision budgets at least 6.5 seconds for a leg
card, including untilt; host velocity remains zero while busy.

The walking model is paused during local leg execution. Heartbeat targets
remain at the last host hold target, and the feedback deviation limiter uses
that held reference. On DONE, the host reseeds its heartbeat and slew limiter
from returned live feedback, waits for joint velocity to settle, resets the
walking policy history, and starts its return tick at zero velocity.

Before voting, vision requests the card re-pose (action 7). The host reports
its DONE status on the attitude UDP channel (port 5007). Vision waits another
100 ms (`--card-settle-ms`) before taking the first vote. The stop-trigger frame
cannot vote. An older host without completion telemetry uses the legacy
`--card-tilt-ms` wait. Deploy host, vision, and firmware together.

## Run

Start connector and vision from the repository root on Jetson:

```bash
python -u connector.py --vision-port 5006 --policy-port 5005
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless
```

In another terminal use the verified 49-observation walking model:

```bash
python -u humanoid_jetson_deploy/main.py \
  --policy walking --command-source vision \
  --model /path/to/verified-49-input-walking.onnx \
  --port /dev/ttyACM0 --no-plot
```

This command keeps motor enable off. Append `--enable-motors` for a physical
run after checking classification and the action sequence with support.
Only one policy process may open `/dev/ttyACM0`. Verify PA4/PC0 lift sides
and the GPIOC13 restore pose on hardware.

Vision does not load or require a one-foot model. `--one-foot-model`,
`--shape-lift-seconds`, and `--start-policy-one-foot-model` remain accepted
compatibility options and are ignored. Standalone ONNX testing remains
available with `--policy one-foot --model /path/to/one-foot.onnx`.

## USB CDC extension

Version-2 framing and CRC-16 remain unchanged. COMMAND is 162 bytes; STATE is
154 bytes in the legacy layout or 162 bytes with diagnostic counters.

| Type | Direction | Packed payload | Meaning |
|---:|---|---|---|
| 3 | Jetson -> STM32 | `<IB`: event_id, action_id | IDs 1-6; 7/8 are card posture events |
| 4 | STM32 -> Jetson | `<IBB`: event_id, action_id, status | 1 accepted, 2 done, 3 busy, 4 invalid, 5 failed |

STM32 replies with current status for duplicate IDs without restarting the
action. BUSY keeps the host retrying; INVALID and FAILED abort the host action.
The normal COMMAND and STATE service continues during every local action.
