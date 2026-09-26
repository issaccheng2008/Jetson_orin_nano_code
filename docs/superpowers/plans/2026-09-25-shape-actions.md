# Shape action routing implementation plan

**Goal:** Route six confirmed shape cards to their designated controller while retaining the working fixed joint frame playback.

**Architecture:** Vision publishes a card observation and a uniquely identified event over the existing local UDP path. The walking policy process is the sole USB CDC owner: it runs the one-foot model for cards 3/4 and sends typed upper-body requests for cards 1/2/5/6. STM32 accepts only upper-body IDs and keeps its continuous leg command and state feedback service running during those actions.

**Tech stack:** Python `unittest`, ONNX Runtime, UDP JSON, STM32 HAL/C, USB CDC protocol version 2.

---

### Task 1: Vision event transport

- [x] Add a failing test that a confirmed card carries a stable event ID through vision, connector and UDP source, while repeated publications do not create another event.
- [x] Add a failing test that camera loss reports `qr=-1` and no old card observation.
- [x] Extend `new_vision/jetson/policy_bridge.py`, `run_policy_vision.py`, `connector.py`, and `humanoid_jetson_deploy/command_source.py` without changing the existing velocity ranges.
- [x] Run the targeted vision, connector and command source tests.

### Task 2: Jetson task state machine

- [x] Add failing tests for card mapping, event deduplication, three-second lift, left/right support selection, busy handling and walking resume.
- [x] Add a focused card-action controller and integrate it only into `main.py` walking vision mode with an explicit one-foot model path.
- [x] Keep fixed frame playback branching, state sequencing and motor safety limits unchanged; run its existing tests.

### Task 3: USB event protocol and STM32 execution

- [x] Add failing Python frame tests for upper-body request and status messages, with CRC and unchanged state/command layouts.
- [x] Add STM32 request parsing, event-ID deduplication and status encoding. Wire cards 1/2/5/6 to existing nonblocking arm/head states; reject IDs 3/4.
- [x] Ensure the main loop continues USB joint commands and state feedback in every upper-body action state.
- [x] Build the STM32 project and run Python protocol tests.

### Task 4: Verification and documentation

- [x] Run all relevant Python tests, including fixed joint frame regression.
- [x] Check the STM32 build, inspect the cross-repository diff, and document launch arguments and hardware validation limits.

Hardware validation is pending on the robot: camera/OpenCV, ONNX Runtime inference, motor response and physical three-second holds were not available on this Windows host.
