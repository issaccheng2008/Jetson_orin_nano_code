import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'new_vision/scripts/analyze_turn_duty.py'


def frame(t, vx=.2, wz=0., **measurement):
    return json.dumps(dict(schema='line_frames_v1', process_monotonic_s=t,
                           vx=vx, wz=wz, measurement=measurement))


class TurnDutyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'turn duty analyzer has not been implemented')
        spec = importlib.util.spec_from_file_location('turn_duty_analysis', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_irregular_frame_times_weight_duration_and_exclude_stops(self):
        report = self.module.analyze_jsonl([
            frame(10, wz=.3, steering_reason='new_block', segment_control_active=True,
                  segment_gate_reason='qualified'),
            frame(10.1, steering_reason='geometry_lost_yaw_zero', segment_control_active=False,
                  segment_gate_reason='measurement_unavailable'),
            frame(10.5, vx=0, steering_reason='external_stop'), frame(11, wz=.3)])
        self.assertAlmostEqual(report['duration_s']['turn'], .1)
        self.assertAlmostEqual(report['duration_s']['straight'], .4)
        self.assertAlmostEqual(report['duration_s']['stop'], .5)
        self.assertAlmostEqual(report['moving_turn_pct'], 20)
        self.assertAlmostEqual(report['moving_reason_duration_s']['geometry_lost_yaw_zero'], .4)
        self.assertAlmostEqual(report['moving_segment_active_duration_s']['true'], .1)
        self.assertAlmostEqual(report['moving_gate_duration_s']['measurement_unavailable'], .4)

    def test_entire_long_gap_is_unknown_and_threshold_is_configurable(self):
        rows = [frame(0, wz=.3), frame(2), frame(2.2)]
        report = self.module.analyze_jsonl(rows)
        self.assertAlmostEqual(report['duration_s']['unknown'], 2)
        self.assertAlmostEqual(report['duration_s']['turn'], 0)
        self.assertAlmostEqual(report['moving_turn_pct'], 0)
        relaxed = self.module.analyze_jsonl(rows, max_gap=2)
        self.assertAlmostEqual(relaxed['duration_s']['turn'], 2)

    def test_invalid_command_and_malformed_record_prevent_bridging(self):
        bad = json.dumps(dict(schema='line_frames_v1', process_monotonic_s=.1, vx=.2))
        report = self.module.analyze_jsonl([frame(0, wz=.3), bad,
                                          frame(.2), '{broken', frame(.4)])
        self.assertAlmostEqual(report['duration_s']['turn'], .1)
        self.assertAlmostEqual(report['duration_s']['unknown'], .3)
        self.assertEqual(len(report['issues']), 2)

    def test_run_changes_clock_regression_and_clock_type_change_do_not_integrate(self):
        rows = [json.dumps(dict(schema='line_frames_v1', run_id=run,
                                process_monotonic_s=t, vx=.2, wz=.3))
                for run, t in [('a', 1), ('b', 1.2), ('b', .8), ('b', 1)]]
        report = self.module.analyze_jsonl(rows)
        self.assertAlmostEqual(report['duration_s']['turn'], .2)
        self.assertEqual(report['discontinuities']['run_change'], 1)
        self.assertEqual(report['discontinuities']['non_increasing_time'], 1)
        report = self.module.analyze_jsonl([
            frame(100), json.dumps(dict(schema='line_frames_v1', host_time_ns=100200000000,
                                       vx=.2, wz=.3))])
        self.assertAlmostEqual(report['duration_s']['straight'], 0)
        self.assertEqual(report['discontinuities']['clock_source_change'], 1)

    def test_alias_clocks_and_empty_or_all_stopped_input(self):
        report = self.module.analyze_jsonl([
            json.dumps(dict(schema='line_frames_v1', timestamp=1, vx=0, wz=0)),
            json.dumps(dict(schema='line_frames_v1', timestamp=1.3, vx=0, wz=0))])
        self.assertAlmostEqual(report['duration_s']['stop'], .3)
        self.assertIsNone(report['moving_turn_pct'])
        self.assertIsNone(self.module.analyze_jsonl([])['moving_turn_pct'])
        self.assertEqual(len(self.module.analyze_jsonl(['{}'])['issues']), 1)

    def test_journal_keeps_B_and_C_snapshots_separate_and_pairs_segment_diagnostics(self):
        prefix = 'Oct 08 18:31:22 host bash[1]: '
        rows = [prefix + '[vision] 6Hz vx=+0.200 wz=+0.300 reason=new_block',
                prefix + 'step=25 state_seq=20 policy_target_velocity=[vx=+0.200 m/s, vy=+0.000 m/s, wz=+0.000 rad/s]',
                prefix + '[lane-segments] anchored=0 segments=0 pattern=insufficient_support',
                prefix + '[segment-control] active=0 actual=+0.300 heading_shadow=+0.300 gate=confirming',
                prefix + '[vision] 6Hz vx=+0.000 wz=+0.000 reason=external_stop',
                prefix + '[vision] 6Hz vx=+0.200 wz=+0.000 reason=geometry_lost_yaw_zero',
                prefix + 'step=50 state_seq=45 policy_target_velocity=[vx=+0.200 m/s, vy=+0.000 m/s, wz=+0.300 rad/s]',
                prefix + '[vision] card window closed']
        report = self.module.analyze_journal(rows)
        self.assertEqual(report['B_vision']['counts'], dict(turn=1, straight=1, stop=1, other=0))
        self.assertEqual(report['C_policy']['counts'], dict(turn=1, straight=1, stop=0, other=0))
        self.assertEqual(report['B_vision']['moving_turn_pct'], 50)
        self.assertEqual(report['B_vision']['gate_counts']['confirming'], 1)
        self.assertIn('snapshot', report['timing_caveat'])
        self.assertNotIn('duration_s', report['B_vision'])

    def test_journal_reports_malformed_command_and_missing_segment_fields(self):
        report = self.module.analyze_journal([
            '[vision] 6Hz vx=+0.200 wz=+0.000 reason=continue_block',
            '[vision] 6Hz vx=oops wz=+0.300 reason=new_block'])
        self.assertEqual(report['B_vision']['samples'], 1)
        self.assertEqual(report['B_vision']['gate_counts']['missing'], 1)
        self.assertEqual(len(report['issues']), 1)

    def test_cli_auto_detects_jsonl_and_emits_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'frames.jsonl'
            path.write_text(frame(0, wz=.3) + '\n' + frame(.2), encoding='utf-8')
            proc = subprocess.run([sys.executable, str(SCRIPT), str(path), '--json'],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertAlmostEqual(json.loads(proc.stdout)['moving_turn_pct'], 100)


if __name__ == '__main__':
    unittest.main()
