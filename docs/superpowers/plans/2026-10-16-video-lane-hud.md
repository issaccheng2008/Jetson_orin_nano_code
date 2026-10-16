# Test video lane HUD

Goal: show the current lane deviation angle and Chinese dual/single/lost labels
on lane_loss_recovery_20261012, with no changes to lane detection or control.

Design: extract only scalar observations from the existing per-frame debug data
after command publication. Pass them through the existing bounded video queue.
Render in the encoder thread within the existing 110-pixel footer. Use embedded
16-pixel monochrome Chinese labels so Jetson needs no fonts or extra libraries.
Angle uses valid heading_control_deg, then valid single_edge_heading_deg;
positive means lane ahead left, negative right. Invalid/lost angles show --.
Keep observations attached to source frames during timestamp resampling.

- [x] Add regression coverage for states, invalid angles, repeated frames and
      unchanged source images; observe failures before implementation.
- [x] Extend command_video.py metadata and HUD; pass debug from
      run_policy_vision.py without changing publication or detector logic.
- [x] Run video, recording and steering/publication regressions; inspect an
      actual encoded preview and compare render overhead locally.
- [x] Review diff for isolation, document display/angle semantics and results.

Verification: 102 distinct tests exercised across video, command-window
publication, recording/session analysis, heading integration/controller,
lane recovery and line telemetry. The sandbox run passed 101/102; its one
loopback UDP receive timeout passed on rerunning the entire 16-test heading
integration module outside the network sandbox. All other tests passed inside
the sandbox with workspace-local temporary files. An independent read-only
review also reran the 25 video/publication tests successfully.

Decoded a real three-frame MJPG sample and visually checked all Chinese labels,
angles, existing command arrows and the unchanged footer dimensions. Confirmed
recording enabled/disabled produces identical published command sequences.
Desktop timing is noisy under sandbox scheduling: scalar snapshot was about
1.4 microseconds; alternating render batches showed approximately 0.8 ms/frame
additional median process CPU time at width 960. No Jetson or motor/camera
hardware test was performed. No commits or pushes were made.
