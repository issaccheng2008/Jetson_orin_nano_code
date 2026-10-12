"""The optional second camera cannot change walking detection or replay old votes."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision' / 'jetson'))
from card_vote_camera import CardVoteCamera
import run_policy_vision


class CardVoteCameraTests(unittest.TestCase):
    def test_dual_camera_enabled_by_default(self):
        with patch('sys.argv',['vision']):
            self.assertEqual(run_policy_vision.parse_args().card_camera,1)

    def test_stable_paths_and_duplicate_device_rejected(self):
        from camera_devices import resolve_devices
        with patch('camera_devices.Path.exists',return_value=True):
            self.assertEqual(resolve_devices(0,1,'/dev/v4l/by-path/main','/dev/v4l/by-path/card'),
                             ('/dev/v4l/by-path/main','/dev/v4l/by-path/card'))
            with self.assertRaises(ValueError):
                resolve_devices(0,1,'/dev/video0','/dev/video0')

    def test_only_fresh_post_settle_frames_can_vote(self):
        reader = Mock()
        old = np.full((2, 2, 3), 1, np.uint8)
        fresh = np.full((2, 2, 3), 2, np.uint8)
        reader.read.side_effect = [
            (True, old, {'camera_capture_started_s': 9.9}),
            (True, fresh, {'camera_capture_started_s': 10.1}),
            (False, None, {}),
        ]
        camera = CardVoteCamera(reader)
        self.assertEqual(camera.sample(10.0)[2], 'captured_before_vote')
        frame, meta, reason = camera.sample(10.0)
        self.assertIs(frame, fresh)
        self.assertEqual(reason, 'fresh')
        self.assertEqual(camera.sample(10.0)[2], 'read_unavailable')
        self.assertEqual(reader.read.call_count, 3)
        camera.close()
        reader.close.assert_called_once()

    def test_stopped_votes_use_second_frame_while_approach_uses_first(self):
        primary_frame = np.zeros((720, 1280, 3), np.uint8)
        second_frame = np.full((720, 1280, 3), 7, np.uint8)
        primary_cap = Mock()
        primary_cap.isOpened.return_value = True
        primary_cap.get.side_effect = lambda key: 1280 if key == 3 else 720
        second_cap = Mock()
        second_cap.isOpened.return_value = True
        second_cap.get.side_effect = lambda key: 1280 if key == 3 else 720
        clock = [0.0]
        reads = [0]

        def primary_read():
            reads[0] += 1
            clock[0] = round(reads[0] * 0.1, 3)
            if reads[0] > 7:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, primary_frame
            return True, primary_frame

        primary_cap.read.side_effect = primary_read
        capture = Mock()
        capture.start.return_value = capture
        capture.read.side_effect = lambda timeout: (
            True, second_frame, {'camera_capture_started_s': clock[0],
                                 'camera_sequence': reads[0]})
        capture.close.return_value = True
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, {
            'fused_err': 0., 'fused_err_cm': 0., 'angle_err_deg': 0.,
            'lost_frames': 0, 'curve_mode': False, 'bottom_lock_valid': True,
            'curve_px': 0., 'base_err_cm': 0.})
        primary_shape, precision_shape = Mock(), Mock()
        primary_shape.action_map = {'square': 3}
        precision_shape.cfg = {'max_dist_cm': 80.}
        precision_shape._card_area_at_dist.return_value = 100.
        def primary_result(*_args, **_kwargs):
            cy = 0.2 if reads[0] == 1 else 0.9
            return None, {'presence': True, 'card_found': False,
                          'presence_cy_frac': cy, 'shape': None}
        primary_shape.update.side_effect = primary_result
        precision_shape.update.return_value = (None, {
            'presence': True, 'card_found': True, 'shape': 'square',
            'presence_cy_frac': 0.9})
        with (
            patch('sys.argv', ['run_policy_vision.py', '--no-camera-async', '--headless',
                               '--card-camera', '1', '--shape-every', '1',
                               '--card-every-stopped', '1', '--card-tilt-ms', '0',
                               '--card-settle-ms', '0', '--card-vote-frames', '1',
                               '--card-stop-ms', '300', '--photometric-mode', 'raw']),
            patch.object(run_policy_vision.signal, 'signal'),
            patch.object(run_policy_vision, 'ConnectorClient') as connector,
            patch('utils.open_camera', side_effect=[primary_cap, second_cap]) as open_camera,
            patch('latest_camera.LatestCamera', return_value=capture),
            patch('line_detector_v1_warp.LineDetector', return_value=detector),
            patch('shape_detector.ShapeDetector', side_effect=[primary_shape, precision_shape]),
            patch.object(run_policy_vision.time, 'monotonic', lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        self.assertEqual(open_camera.call_count, 2)
        self.assertTrue(all(call.args[0] is primary_frame
                            for call in primary_shape.update.call_args_list))
        self.assertTrue(precision_shape.update.called)
        self.assertTrue(all(call.args[0] is second_frame
                            for call in precision_shape.update.call_args_list))
        self.assertTrue(any(call.kwargs.get('classify') is True
                            for call in precision_shape.update.call_args_list))
        self.assertTrue(any(call.kwargs.get('event_action') == 3
                            for call in connector.return_value.publish.call_args_list))
        capture.close.assert_called_once()
