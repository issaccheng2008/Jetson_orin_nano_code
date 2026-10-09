"""Replay preserves external stops and can reproduce the legacy controller."""
import importlib
import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from test_heading_steering import detection


class SteeringFilterReplayTests(unittest.TestCase):
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
