"""Replay preserves external stops and can reproduce the legacy controller."""
import importlib
import contextlib
import io
import json
import math
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from test_heading_steering import detection
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from steering_recovery import RecoveryConfig


class SteeringFilterReplayTests(unittest.TestCase):
    def test_manifest_table_and_region_settings_reproduce_commands(self):
        from segment_steering import SegmentSteeringController
        from test_steering_config import SteeringConfigTests, TABLE
        regions = ((20, 32), (32, 44), (44, 56), (56, 68))
        for mode, controller_type in (('heading', HeadingSteeringController), ('segments', SegmentSteeringController)):
            options = dict(angle_wz_table=TABLE, position_gain=0, position_recovery_cm=0,
                           lookahead_cm=65, corridor_cm=100)
            if mode == 'segments':
                options['segment_regions_cm'] = regions
            controller = controller_type(SteeringController(), **options)
            frame = detection(40, z=25, fit_seg_anchored=True, **SteeringConfigTests().segments(regions))
            rows = self.recorded_rows(controller, [(frame, 1., .1)]*5)
            arguments = dict(wz_mode=mode, steering_angle_wz_table=TABLE,
                             segment_regions_cm=regions, heading_lookahead_cm=65, heading_corridor_cm=100)
            replayed = importlib.import_module('replay_steering_filter').replay(rows, arguments, None)
            self.assertEqual([r['new_wz'] for r in replayed], [r['wz'] for r in rows])

    def recorded_rows(self, controller, observations):
        rows, now = [], 10.
        for index, (debug, confidence, dt) in enumerate(observations):
            now += dt
            vx, wz = controller.command(debug, confidence, dt)
            rows.append(dict(frame=index, host_time_ns=round(now*1e9),
                process_monotonic_s=now, confidence=confidence, vx=vx, wz=wz,
                measurement=dict(debug, **controller.diagnostics)))
        return rows

    def test_old_manifest_does_not_enable_new_position_control(self):
        controller = HeadingSteeringController(SteeringController(),
            position_gain=0, position_recovery_cm=0)
        frame = detection(math.degrees(math.atan(6/25)), near=6, z=25)
        rows = self.recorded_rows(controller, [(frame, 1., .1)]*2)
        result = importlib.import_module('replay_steering_filter').replay(
            rows, {'wz_mode':'heading'}, None)
        self.assertEqual([r['new_wz'] for r in result], [0., 0.])

    def test_manifest_position_settings_reproduce_real_controller(self):
        for options in (dict(position_gain=0, position_recovery_cm=0),
                dict(position_gain=2, position_dead_cm=1, position_lookahead_cm=20,
                     position_max_deg=9, position_recovery_cm=6,
                     position_recovery_full_scale_cm=2, position_confirm_frames=3)):
            with self.subTest(options=options):
                controller = HeadingSteeringController(SteeringController(), **options)
                frame = detection(35, near=9, z=25)
                rows = self.recorded_rows(controller, [(frame, 1., .1)]*3)
                result = importlib.import_module('replay_steering_filter').replay(
                    rows, dict(wz_mode='heading', **options), None)
                self.assertEqual([(r['new_vx'], r['new_wz']) for r in result],
                                 [(r['vx'], r['wz']) for r in rows])

    def test_manifest_loss_recovery_and_explicit_ablation_override(self):
        controller = HeadingSteeringController(SteeringController(),
            position_gain=0, position_recovery_cm=0, recovery_config=RecoveryConfig())
        rows = self.recorded_rows(controller, [(detection(30), 1., .1),
            (detection(measurement_valid=False), 0., .3)])
        arguments = dict(wz_mode='heading', steering_loss_mode='history-stop',
                         steering_loss_max_s=.8, steering_loss_history_s=.8)
        module = importlib.import_module('replay_steering_filter')
        result = module.replay(rows, arguments, None)
        self.assertEqual(result[-1]['new_wz'], rows[-1]['wz'])
        self.assertEqual(result[-1]['reason'], 'loss_history_turn')
        without_recovery = module.replay(rows, arguments, None, recovery_config=None)
        self.assertEqual(without_recovery[-1]['new_wz'], 0.)
        self.assertEqual(without_recovery[-1]['reason'], 'geometry_lost_yaw_zero')

    def test_manifest_segment_fallback_is_used_without_overriding_ablation(self):
        from segment_steering import SegmentSteeringController
        from test_segment_steering import detection as segment_detection, bend
        controller = SegmentSteeringController(SteeringController(),
            position_gain=0, position_recovery_cm=0, segment_fallback=True)
        frame = segment_detection(bend, heading_control_valid=False,
            near_observation_paired=True, near_observation_quality=.9)
        rows = self.recorded_rows(controller, [(frame, 1., .1)]*5)
        arguments = dict(wz_mode='segments', steering_segment_fallback=True)
        module = importlib.import_module('replay_steering_filter')
        result = module.replay(rows, arguments, None)
        self.assertEqual([r['new_wz'] for r in result], [r['wz'] for r in rows])
        without_fallback = module.replay(rows, arguments, None, segment_fallback=False)
        self.assertEqual(without_fallback[-1]['new_wz'], 0.)

    def test_both_loss_mode_aliases_reproduce_default_search_and_expired_history(self):
        module = importlib.import_module('replay_steering_filter')
        for mode in ('history-turn', 'history-stop'):
            with self.subTest(mode=mode):
                controller = HeadingSteeringController(SteeringController(),
                    position_gain=0, position_recovery_cm=0, recovery_config=RecoveryConfig())
                bad = detection(measurement_valid=False)
                rows = self.recorded_rows(controller, [(detection(-30), 1., .1),
                    (bad, 0., .2), (bad, 0., .7), (bad, 0., .1)])
                arguments = dict(wz_mode='heading', steering_loss_mode=mode,
                    steering_loss_max_s=.8, steering_loss_history_s=.8)
                result = module.replay(rows, arguments, None)
                self.assertEqual([(r['new_vx'], r['new_wz']) for r in result],
                                 [(r['vx'], r['wz']) for r in rows])
                self.assertEqual(result[-1]['reason'], 'loss_default_left')

    def test_shadow_recording_is_accepted_as_legacy_reference(self):
        module = importlib.import_module('replay_steering_filter')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(frame=i, host_time_ns=round((100+i*.15)*1e9),
                process_monotonic_s=10+i*.15, confidence=1., vx=.2, wz=.5,
                measurement=dict(detection(30.), steering_reason='new_block')) for i in range(2)]
            (root/'line_frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (root/'run_manifest.json').write_text(json.dumps({'arguments':{
                'wz_mode':'heading', 'steering_filter_mode':'shadow'}}))
            with patch('sys.argv', ['replay', str(root/'line_frames.jsonl'),
                                   '--output-dir',str(root/'out')]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                module.main()
            self.assertTrue((root/'out/summary.json').is_file())

    def test_legacy_replay_and_event_stop(self):
        self.assertIsNotNone(importlib.util.find_spec('replay_steering_filter'))
        module = importlib.import_module('replay_steering_filter')
        readings = []
        for index, angle in enumerate((30., 30., 0., -30.)):
            debug = detection(angle)
            debug['steering_reason'] = 'external_stop' if index == 2 else 'new_block'
            readings.append(dict(frame=index, host_time_ns=round((100+index*.15)*1e9),
                process_monotonic_s=10+index*.15, confidence=1., vx=0. if index==2 else .2,
                wz=0. if index==2 else (.5 if angle>0 else -.5), measurement=debug))
        result = module.replay(readings, {'wz_mode':'heading', 'heading_right_tolerance_deg':0.,
            'heading_left_tolerance_deg':0.}, None)
        self.assertEqual(result[2]['new_wz'], 0.)
        self.assertEqual(result[2]['new_vx'], 0.)
        self.assertEqual(result[3]['new_wz'], -.5)


if __name__ == '__main__':
    unittest.main()
