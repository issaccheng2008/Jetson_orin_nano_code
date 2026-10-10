"""Real B entrypoint and wire contract; camera/detector only are synthetic."""
import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
import run_policy_vision
from policy_bridge import ConnectorClient


class HeadingIntegrationTests(unittest.TestCase):
    def test_camera_options_validate_before_hardware(self):
        with patch('sys.argv',['run_policy_vision.py','--camera-exposure-mode','manual',
                               '--camera-exposure-ms','5','--camera-sharpness','3']):
            args=run_policy_vision.parse_args()
        self.assertEqual(args.camera_exposure_ms,5)
        self.assertEqual(args.camera_sharpness,3)
        with patch('sys.argv',['run_policy_vision.py','--camera-exposure-ms','5']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_policy_vision.parse_args()

    def test_camera_controls_applied_before_detection_and_saved_in_manifest(self):
        self.test_default_entrypoint_held_levels_and_auditable_log(camera_settings=True)

    def test_filter_and_policy_hold_interfaces_validate_without_hardware(self):
        argv = ['run_policy_vision.py', '--command-min-hold-s', '.23',
                '--steering-filter-mode', 'active', '--steering-filter-algorithm', 'ema']
        with patch('sys.argv', argv):
            args = run_policy_vision.parse_args()
        self.assertEqual(args.command_min_hold_s, .23)
        self.assertEqual(args.steering_filter_algorithm, 'ema')
        for extra in (['--command-min-hold-s', 'nan'], ['--steering-filter-beta', '-1'],
                      ['--steering-filter-min-hz', '0'], ['--steering-exit-deg', '3']):
            with patch('sys.argv', ['run_policy_vision.py', *extra]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_policy_vision.parse_args()

    def test_video_arguments_validate_without_hardware(self):
        with patch('sys.argv', ['run_policy_vision.py', '--no-record-video', '--video-fps', '8', '--video-width', '640']):
            args = run_policy_vision.parse_args()
        self.assertFalse(args.record_video)
        self.assertEqual(args.video_fps, 8)
        self.assertEqual(args.video_width, 640)
        for extra in (['--video-fps', 'nan'], ['--video-fps', '0'], ['--video-width', '0']):
            with patch('sys.argv', ['run_policy_vision.py', *extra]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_policy_vision.parse_args()

    def test_heading_rejects_amplitude_that_connector_would_clip(self):
        with patch("sys.argv", ["run_policy_vision.py", "--max-wz", "1", "--wz-step", ".6"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_policy_vision.parse_args()

    def test_heading_requires_usable_corridor_and_recovery_caps(self):
        for extra in (("--heading-corridor-cm", "0"), ("--heading-corridor-cm", "nan"),
                      ("--max-wz-right", ".2"), ("--wz-step", ".3")):
            with self.subTest(extra=extra), patch("sys.argv", ["run_policy_vision.py", *extra]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_policy_vision.parse_args()

    def test_client_optional_mode_real_udp(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(1)
            client = ConnectorClient(port=receiver.getsockname()[1])
            try:
                client.publish(.2, .17, command_mode="held")
                packet = json.loads(receiver.recv(4096))
                self.assertEqual(packet["command_mode"], "held")
                self.assertEqual(packet["wz"], .17)
                client.publish(.2, .17)
                self.assertNotIn("command_mode", json.loads(receiver.recv(4096)))
                for bad in (True, 1, [], "bad"):
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        client.publish(.2, .1, command_mode=bad)
            finally:
                client.close()

    def test_active_entrypoint_filter_reaches_publication_and_log(self):
        self.test_default_entrypoint_held_levels_and_auditable_log(filter_mode='active')

    def test_shadow_entrypoint_logs_comparison(self):
        self.test_default_entrypoint_held_levels_and_auditable_log(filter_mode='shadow')

    def test_recording_can_be_disabled(self):
        self.test_default_entrypoint_held_levels_and_auditable_log(record_video=False)

    def test_video_initialization_failure_does_not_stop_publication(self):
        self.test_default_entrypoint_held_levels_and_auditable_log(video_error=True)

    def test_default_entrypoint_held_levels_and_auditable_log(self, filter_mode='legacy', record_video=True, video_error=False, camera_settings=False):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        clock, count, sent, order = [0.0], [0], [], []

        def read():
            count[0] += 1
            clock[0] += .05
            if count[0] > 42:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame

        camera.read.side_effect = read
        detector = Mock()
        def process(_frame, **_kw):
            order.append('detect')
            angle = (20., 40., -20., 0.)[min(3, (count[0]-1)//11)]
            return 0, 0, .9, _frame.copy(), dict(
                fused_err_cm=0., base_err_cm=0., near_error_cm=0., near_z_cm=0.,
                angle_err_deg=0., lost_frames=0, measurement_valid=True,
                heading_control_valid=True, heading_control_deg=angle)
        detector.process.side_effect = process
        camera_extra = ['--camera-exposure-mode','manual','--camera-exposure-ms','5'] if camera_settings else []
        camera_report = dict(status='applied' if camera_settings else 'unchanged',settings=[])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--no-shape-detect",
                               '--steering-filter-mode', filter_mode,
                               '--steering-filter-algorithm', 'one-euro',
                               '--record-video' if record_video else '--no-record-video',
                               "--attitude-port", "0", "--recording-root", tmp, *camera_extra]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client,
            patch('command_video.CommandVideo') as video,
            patch('camera_controls.apply_camera_controls',side_effect=lambda *a: order.append('settings') or camera_report) as camera_controls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            if video_error:
                video.side_effect = MemoryError('recorder unavailable')
            client.return_value.publish.side_effect = lambda vx,wz,*a,**kw: sent.append((clock[0],vx,wz,kw))
            self.assertEqual(run_policy_vision.main(), 0)
            submitted = video.return_value.submit.call_args_list
            self.assertEqual(len(submitted), 42 if record_video and not video_error else 0)
            for submission, publication in zip(submitted, sent):
                self.assertEqual((submission.kwargs['vx'], submission.kwargs['wz']), publication[1:3])
                self.assertFalse(submission.kwargs['lost'])
            if video_error:
                video.assert_called_once()
                video.return_value.close.assert_not_called()
            elif record_video:
                video.return_value.close.assert_called_once()
                self.assertEqual(Path(video.call_args.args[0]).name, 'video')
            else:
                video.assert_not_called()
            rows = [json.loads(line) for p in Path(tmp).rglob("line_frames.jsonl")
                    for line in p.read_text().splitlines()]
            manifests = [json.loads(p.read_text()) for p in Path(tmp).rglob('run_manifest.json')]
            self.assertEqual(manifests[0]['camera_controls'],camera_report)
            self.assertEqual(order[0],'settings')
            self.assertEqual(camera_controls.call_args.args[0],'/dev/video0')
            self.assertEqual(camera_controls.call_args.args[1].camera_exposure_mode, 'manual' if camera_settings else 'keep')
        self.assertEqual(len(rows), 42)
        detector.set_heading_distances.assert_called_once_with(None, 29.0)
        self.assertTrue(all(r['start_gate_mode'] == 'off' for r in rows))
        self.assertTrue(all(r['qr_passed'] is False and r['shape_passed'] is False for r in rows))
        self.assertTrue(all(r['body_track_deviation_valid'] for r in rows))
        self.assertTrue(all(r['body_track_deviation_deg'] == r['measurement']['heading_control_deg'] for r in rows))
        self.assertTrue(all(r["mode"] == "heading" for r in rows))
        self.assertTrue(all("steering_heading_deg" in r["measurement"] for r in rows))
        if filter_mode == 'active':
            self.assertTrue(all('steering_filter_demand_deg' in r['measurement'] for r in rows))
        elif filter_mode == 'shadow':
            self.assertTrue(all('steering_filter_shadow_wz' in r['measurement'] for r in rows))
        walking = [s for s in sent if s[1] != 0]
        self.assertTrue(all(s[3].get("command_mode") == "held" for s in walking))
        levels = {round(s[2],6) for s in walking}
        self.assertGreaterEqual(len(levels), 3, levels)
        self.assertTrue(levels <= {0., .37, .43, .5, -.3, -.5}, levels)
        self.assertTrue(any(wz < 0 for wz in levels), levels)
        start, previous = walking[0][0], walking[0][1:3]
        for now,vx,wz,_ in walking[1:]:
            if (vx,wz) != previous:
                # No minimum delay; sensor filtering and levels still apply.
                self.assertGreater(now-start, 0.)
                start,previous = now,(vx,wz)
        self.assertEqual(sent[-1][1:3], (0.,0.))


if __name__ == "__main__":
    unittest.main()
