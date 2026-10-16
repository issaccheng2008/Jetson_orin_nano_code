"""Replay complete ordinary-card stops through the real vision main loop."""
import contextlib
import io
import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
import run_policy_vision


class CardFlowIntegrationTests(unittest.TestCase):
    def test_cooldown_environment_override_and_invalid_values(self):
        with (patch.dict(os.environ, {'STOP_COOL_DOWN': '3.5'}),
              patch('sys.argv', ['vision', '--card-trigger-dist-cm', '17'])):
            args = run_policy_vision.parse_args()
            self.assertEqual(args.stop_cool_down, 3.5)
            self.assertEqual(args.card_trigger_dist_cm, 17.)
        with patch('sys.argv', ['vision', '--stop-cool-down', '0']):
            self.assertEqual(run_policy_vision.parse_args().stop_cool_down, 0.)
        for value in ('-1', 'nan', 'inf'):
            with (self.subTest(value=value),
                  patch('sys.argv', ['vision', '--stop-cool-down', value]),
                  contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                run_policy_vision.parse_args()

    def replay(self, labels, seconds=6., *, distance=30, cooldown=30, failure=.5,
               pose_delay=.2, action_delay=3., present=lambda t: True,
               quad=None, reach=.9, startup_args=()):
        frame = np.zeros((720, 1280, 3), np.uint8)
        clock = [0.]
        calls, detections = [], []
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [1280, 720]
        def read():
            clock[0] = round(clock[0]+.1, 6)
            if clock[0] > seconds:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame
        cap.read.side_effect = read
        lane = Mock()
        lane.process.return_value = (0, 0, .9, frame, dict(fused_err=0.,
            angle_err_deg=0., lost_frames=0, base_err_cm=0., curve_mode=False,
            bottom_lock_valid=True, curve_px=0.))
        shape = Mock()
        shape.action_map = dict(circle=1, square=3, cross=5)
        results = iter(labels) if not callable(labels) else None
        def update(_frame, **kwargs):
            classify = kwargs['classify']
            detections.append((clock[0], classify))
            name = (labels(clock[0]) if callable(labels) else next(results, None)) if classify else None
            return None, dict(presence=present(clock[0]), presence_cy_frac=reach,
                              card_found=quad is not None, quad_work=quad, shape=name)
        shape.update.side_effect = update
        attitude = SimpleNamespace(value=45., last_packet_s=0., card_tilt_status_seen=True,
                                   card_tilt_event_id=0, card_tilt_done=False)
        state = dict(tilt=False, start=0., restore=0, restored_at=0., action=0, done_at=0.)
        def poll():
            attitude.last_packet_s = clock[0]
            attitude.card_tilt_event_id = state['tilt_id'] if state['tilt'] else 0
            attitude.card_tilt_done = state['tilt'] and clock[0] >= state['start']+pose_delay
        state['tilt_id'] = 0
        attitude.poll = poll
        attitude.executed_command = lambda: None
        attitude.card_task = lambda now: dict(restore_id=state['restore'],
            restore_done=now >= state['restored_at'], action_id=state['action'],
            action_done=state['action'] != 0 and now >= state['done_at'])
        def publish(vx, wz, qr, **kwargs):
            calls.append((clock[0], vx, wz, qr, kwargs))
            tilt = kwargs.get('card_tilt', False)
            if tilt and not state['tilt']:
                state['tilt_id'] += 1
                state['start'] = clock[0]
            if not tilt and state['tilt']:
                state['tilt_id'] += 1
                state['restore'] = state['tilt_id']
                state['restored_at'] = clock[0]+.65
            state['tilt'] = tilt
            if kwargs.get('event_id') and kwargs['event_id'] != state['action']:
                state['action'] = kwargs['event_id']
                state['done_at'] = clock[0]+action_delay
        output = io.StringIO()
        with (patch('sys.argv', ['vision', '--headless', '--no-camera-async',
                  '--card-camera', '-1', '--shape-every', '7',
                  '--card-trigger-dist-cm', str(distance),
                  '--stop-cool-down', str(cooldown),
                  '--card-failure-cooldown-s', str(failure), *startup_args]),
              patch.object(run_policy_vision.signal, 'signal'),
              patch.object(run_policy_vision, 'ConnectorClient') as client,
              patch('utils.open_camera', return_value=cap),
              patch('line_detector_v1_warp.LineDetector', return_value=lane),
              patch('shape_detector.ShapeDetector', return_value=shape),
              patch('attitude_input.AttitudeInput', return_value=attitude),
              patch.object(run_policy_vision.time, 'monotonic', lambda: clock[0]),
              contextlib.redirect_stdout(output)):
            attitude.close = lambda: None
            client.return_value.publish.side_effect = publish
            self.assertEqual(run_policy_vision.main(), 0)
        return [row for row in calls if row[0] <= seconds], detections, output.getvalue()

    def test_every_walking_frame_checks_presence_even_during_cooldown(self):
        calls, detections, _ = self.replay([], seconds=4.)
        self.assertEqual(detections[0], (.1, False))
        self.assertTrue(any(t > 2.5 and not classify for t, classify in detections))
        # Detection starts even though startup sampling was configured to seven.
        presence_times = {t for t, classify in detections if not classify}
        walking_times = {t for t, vx, *_ in calls if vx > 0}
        self.assertTrue(walking_times)
        self.assertTrue(walking_times.issubset(presence_times))

    def test_stops_without_knowing_shape_then_waits_done_and_settle(self):
        calls, detections, _ = self.replay(['circle']*6)
        self.assertEqual(calls[0][1:4], (0., 0., -1))
        self.assertTrue(calls[0][4]['card_tilt'])
        first_vote = next(t for t, classify in detections if classify)
        self.assertGreaterEqual(first_vote, .4)
        events = [row for row in calls if row[4].get('event_id')]
        self.assertTrue(events)
        self.assertGreaterEqual(events[0][0], first_vote+.5-1e-6)
        self.assertFalse(events[0][4]['card_tilt'])
        self.assertEqual(len({row[4]['event_id'] for row in events}), 1)

    def test_success_waits_completion_and_uses_cooldown_independent_of_distance(self):
        calls, _, _ = self.replay(['circle']*6, seconds=10., distance=30,
                                 cooldown=2., action_delay=5.)
        event_at = next(t for t, *_, kw in calls if kw.get('event_id'))
        resumed = next(t for t, vx, *_ in calls if t > event_at and vx > 0)
        self.assertGreaterEqual(resumed, event_at+5.)
        tilts = [t for t, *_, kw in calls if kw.get('card_tilt')]
        self.assertFalse(any(resumed <= t < resumed+2 for t in tilts))
        self.assertTrue(any(t >= resumed+2 for t in tilts))

    def test_failure_restores_then_half_second_cooldown(self):
        calls, _, _ = self.replay([], seconds=4.)
        self.assertFalse(any(row[4].get('event_id') for row in calls))
        resumed = next(t for t, vx, *_ in calls if vx > 0)
        self.assertFalse(any(resumed <= t < resumed+.5-1e-6 and kw.get('card_tilt')
                             for t, *_, kw in calls))
        self.assertTrue(any(t >= resumed+.5 and kw.get('card_tilt')
                            for t, *_, kw in calls))

    def test_two_tied_rounds_restore_twice_then_fail(self):
        calls, _, text = self.replay(itertools.cycle(['circle', 'square']), seconds=5.)
        self.assertFalse(any(row[4].get('event_id') for row in calls))
        self.assertEqual(text.count("'reason': 'tie'"), 2)
        resumed = next(t for t, vx, *_ in calls if vx > 0)
        starts = []
        previous = False
        for t, *_, kw in calls:
            current = kw.get('card_tilt', False)
            if current and not previous and t < resumed:
                starts.append(t)
            previous = current
        self.assertEqual(len(starts), 2)

    def test_timeout_with_single_vote_does_not_execute(self):
        calls, _, _ = self.replay(['circle'], seconds=2.5)
        self.assertFalse(any(row[4].get('event_id') for row in calls))
        self.assertTrue(any(row[1] > 0 for row in calls))

    def test_failed_repose_never_classifies_and_recovers(self):
        calls, detections, _ = self.replay(['circle']*6, seconds=7., pose_delay=100.)
        self.assertFalse(any(classify for _, classify in detections))
        self.assertFalse(any(row[4].get('event_id') for row in calls))
        self.assertTrue(any(row[1] > 0 for row in calls))

    def test_absence_or_far_distance_prevents_stop(self):
        for options in (dict(present=lambda t: False), dict(reach=.2)):
            with self.subTest(options=options):
                calls, detections, _ = self.replay([], seconds=1., **options)
                self.assertTrue(all(row[1] > 0 for row in calls))
                self.assertEqual(len(detections), len(calls))
                self.assertTrue(all(not classify for _, classify in detections))

    def test_verified_box_uses_width_instead_of_centroid(self):
        for width, stopped in ((100., False), (130., True)):
            quad = np.float32([[300, 300], [300+width, 300], [300+width, 400], [300, 400]])
            calls, _, _ = self.replay([], seconds=.1, quad=quad, reach=.9)
            self.assertEqual(calls[0][1] == 0., stopped)
