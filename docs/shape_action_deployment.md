# Shape cards: Jetson and STM32 deployment

The camera recognizes six 10 cm cards. The confirmed card is sent through the
existing vision → connector → policy UDP path. `qr` is the current observation
(`-1` when no card is recognized). `event_id` and `event_action` identify one
confirmed action and remain available during the card stop window. The policy
process deduplicates `event_id`; repeated 50 Hz connector packets cannot repeat
the action. A card still in view cannot start another action when the detector's
cooldown expires. Camera read failure sends `qr=-1` and zero velocity.

| Card | ID | Controller | Action |
|---|---:|---|---|
| Circle | 1 | STM32 | Raise left hand |
| Star | 2 | STM32 | Raise right hand |
| Square | 3 | Jetson one-foot ONNX | Lift left leg, right foot supports |
| Diamond | 4 | Jetson one-foot ONNX | Lift right leg, left foot supports |
| Cross | 5 | STM32 | Raise both hands |
| Triangle | 6 | STM32 | Shake head |

The STM32 receives only IDs 1, 2, 5, and 6. It rejects IDs 3 and 4. The
Jetson remains the only writer of 12 leg targets, including while the STM32
controls the arms or head. `--fixed-policy` continues through its original
joint-frame playback path and does not consume shape events.

## Run

Run the connector and vision from the repository root on the Jetson:

```bash
python -u connector.py --vision-port 5006 --policy-port 5005
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless
```

In a third terminal, use the **49-observation walking model that is already
verified on this robot** for `--model`. The specified one-foot file is the
46-observation model:

```bash
python -u humanoid_jetson_deploy/main.py \
  --policy walking --command-source vision \
  --model /path/to/verified-49-input-walking.onnx \
  --one-foot-model humanoid_jetson_deploy/policy-one-foot-standing.onnx \
  --port /dev/ttyACM0 --no-plot
```

The command above keeps motor enable off. After checking camera classification,
event IDs, model input/output dimensions, CRC counts and policy output with the
robot supported, append `--enable-motors` for the physical run. Only one policy
process may open `/dev/ttyACM0`. The one-foot action holds the lift command for
4 seconds by default (`--shape-lift-seconds`) after a short standing phase,
then commands lowering before walking resumes. Video must confirm the physical
leg is raised for at least 3 seconds; command time alone does not prove this.

## USB CDC extension

The existing version-2 header, CRC-16, 154-byte state frame and 74-byte leg
command frame are unchanged. Two new message types use the same framing:

| Type | Direction | Packed payload | Meaning |
|---:|---|---|---|
| 3 | Jetson → STM32 | `<IB`: `event_id`, `action_id` | Upper-body request, IDs 1/2/5/6 |
| 4 | STM32 → Jetson | `<IBB`: `event_id`, `action_id`, `status` | 1 accepted, 2 done, 3 busy, 4 invalid, 5 failed |

Jetson repeats the request every 100 ms until completion is confirmed, using
the same `event_id`. STM32 replies with its current status for duplicate IDs without
restarting the action. It reports completion after the existing nonblocking
servo state machine finishes. If the leg-command watchdog expires, it aborts
an active upper-body action and returns status 5. The normal leg-command and
state-feedback service continues during arm and head actions.
