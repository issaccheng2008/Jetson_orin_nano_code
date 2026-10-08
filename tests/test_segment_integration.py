"""Real vision entrypoint -> UDP packets and per-frame comparison telemetry."""
from copy import deepcopy
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
import run_policy_vision
from policy_bridge import ConnectorClient
from test_segment_steering import detection, bend


class SegmentIntegrationTests(unittest.TestCase):
    def test_segments_send_changed_held_commands_with_original_heading_comparison(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        frame = np.zeros((720, 1280, 3), np.uint8)
        clock, count = [0.], [0]
        detector = Mock()
        curved, straight = detection(bend), detection(heading=30.)

        def read():
            clock[0] += .05
            count[0] += 1
            if count[0] > 56:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame

        camera.read.side_effect = read
        detector.process.side_effect = lambda *a, **kw: (
            0, 0, 1., None, deepcopy(curved if count[0] <= 28 else straight))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(('127.0.0.1', 0))
            receiver.settimeout(.1)
            real_client = ConnectorClient(port=receiver.getsockname()[1])
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch('sys.argv', ['run_policy_vision.py', '--wz-mode', 'segments',
                    '--headless', '--no-shape-detect', '--attitude-port', '0',
                    '--line-log-dir', tmp]),
                patch.object(run_policy_vision.signal, 'signal'),
                patch.object(run_policy_vision, 'ConnectorClient', return_value=real_client),
                patch('utils.open_camera', return_value=camera),
                patch('line_detector_v1_warp.LineDetector', return_value=detector),
                patch.object(run_policy_vision.time, 'monotonic', lambda: clock[0]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run_policy_vision.main(), 0)
                rows = [json.loads(line) for p in Path(tmp).rglob('line_frames.jsonl')
                        for line in p.read_text().splitlines()]
            packets = []
            while True:
                try:
                    packets.append(json.loads(receiver.recv(4096)))
                except socket.timeout:
                    break
        self.assertTrue(detector.lane_fit_enable)
        self.assertTrue(detector.lane_segments_enable)
        self.assertEqual(len(rows), 56)
        self.assertTrue(all(r['mode'] == 'segments' for r in rows))
        self.assertTrue(any(r['wz'] > 0 and r['measurement']['segment_shadow_wz'] == 0
                            for r in rows[:28]))
        self.assertEqual(rows[-1]['wz'], 0.)
        self.assertGreater(rows[-1]['measurement']['segment_shadow_wz'], 0.)
        walking = [p for p in packets if p['vx'] > 0]
        self.assertTrue(all(p['command_mode'] == 'held' for p in walking))
        self.assertTrue(any(p['wz'] > 0 for p in walking))
        self.assertEqual(walking[-1]['wz'], 0.)
        self.assertEqual((packets[-1]['vx'], packets[-1]['wz']), (0., 0.))
        # Published ordinary command transitions retain the training hold.
        start, previous = rows[0]['process_monotonic_s'], (rows[0]['vx'], rows[0]['wz'])
        for row in rows[1:]:
            current = row['vx'], row['wz']
            if current != previous:
                # No minimum delay; sensor filtering and levels still apply.
                self.assertGreater(row['process_monotonic_s'] - start, 0.)
                start, previous = row['process_monotonic_s'], current

    def test_segments_require_valid_levels_and_corridor(self):
        for extra in (('--max-wz', '1', '--wz-step', '.6'),
                      ('--heading-corridor-cm', '0'), ('--max-wz-right', '.2')):
            with self.subTest(extra=extra), patch('sys.argv',
                    ['run_policy_vision.py', '--wz-mode', 'segments', *extra]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_policy_vision.parse_args()

    def test_hold_still_overrides_segment_command_and_comparison_log(self):
        camera, detector = Mock(), Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        clock, count, sent = [0.], [0], []
        frame = np.zeros((720, 1280, 3), np.uint8)

        def read():
            clock[0] += .05
            count[0] += 1
            if count[0] > 20:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame

        camera.read.side_effect = read
        detector.process.side_effect = lambda *a, **kw: (0, 0, 1., None, detection(bend))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch('sys.argv', ['run_policy_vision.py', '--wz-mode', 'segments',
                '--hold-still', '--headless', '--no-shape-detect', '--attitude-port', '0',
                '--line-log-dir', tmp]),
            patch.object(run_policy_vision.signal, 'signal'),
            patch.object(run_policy_vision, 'ConnectorClient') as client,
            patch('utils.open_camera', return_value=camera),
            patch('line_detector_v1_warp.LineDetector', return_value=detector),
            patch.object(run_policy_vision.time, 'monotonic', lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            client.return_value.publish.side_effect = lambda vx,wz,*a,**kw: sent.append((vx,wz,kw))
            self.assertEqual(run_policy_vision.main(), 0)
            rows = [json.loads(line) for p in Path(tmp).rglob('line_frames.jsonl')
                    for line in p.read_text().splitlines()]
        self.assertTrue(all((vx, wz) == (0., 0.) for vx, wz, _ in sent))
        held = [packet for packet in sent if packet[2].get('command_mode') == 'held']
        self.assertEqual(len(held), 20)
        self.assertEqual(len(rows), 20)
        for row, (vx, wz, kw) in zip(rows, held):
            self.assertEqual((vx, wz), (0., 0.))
            self.assertEqual(kw['command_mode'], 'held')
            measurement = row['measurement']
            self.assertFalse(measurement['segment_control_active'])
            self.assertEqual(measurement['segment_applied_wz'], 0.)
            self.assertEqual(measurement['segment_shadow_wz'], 0.)
            self.assertEqual(measurement['segment_gate_reason'], 'external_stop')


if __name__ == '__main__':
    unittest.main()
