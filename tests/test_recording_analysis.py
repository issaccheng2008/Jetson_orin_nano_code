import sys
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_recording import find_start, crop_data, orientation_degrees, main, unique_file
import analyze_recording


def frame(t, **fields):
    return dict(host_time_ns=int(t*1e9), **fields)


class RecordingAnalysisTests(unittest.TestCase):
    def test_display_angles_wrap_negatives_without_changing_quaternion_math(self):
        values = analyze_recording.display_orientation_degrees([1, 0, 0, -1])
        self.assertAlmostEqual(values[2], 270.)
        self.assertEqual(values[:2], (0., 0.))

    def test_filtered_angle_is_logged_output_and_never_filled_from_raw(self):
        row = dict(body_track_deviation_deg=30., body_track_deviation_valid=True,
                   measurement=dict(steering_filter_heading_deg=20.))
        self.assertEqual(analyze_recording.deviation_values(row), (30., 20.))
        row['measurement'] = {}
        self.assertTrue(math.isnan(analyze_recording.deviation_values(row)[1]))

    def test_missing_control_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'columns'):
            analyze_recording.validate_control_columns(['host_unix_s', 'cmd_wz'])

    def test_both_valves_must_pass_even_if_robot_is_already_moving(self):
        rows = [frame(1, start_gate_mode='both', qr_passed=False, shape_passed=True, wz=.5),
                frame(2, start_gate_mode='both', qr_passed=True, shape_passed=True)]
        self.assertEqual(find_start(rows), (2., 'both_valves'))

    def test_button_requires_no_shape_but_waits_for_release(self):
        rows = [frame(1, start_gate_mode='button', button_passed=True, start_released=False),
                frame(2, start_gate_mode='button', button_passed=True, start_released=True)]
        self.assertEqual(find_start(rows), (2., 'button_release'))

    def test_missing_gate_is_not_guessed_from_speed(self):
        with self.assertRaisesRegex(ValueError, 'start'):
            find_start([frame(1, vx=.2, wz=.5)])

    def test_common_clock_crop_preserves_independent_sample_times(self):
        visual = [frame(9), frame(10), frame(10.2)]
        control = [dict(host_unix_s=9.9), dict(host_unix_s=10.1)]
        v, c = crop_data(visual, control, 10)
        self.assertEqual(len(v), 2)
        self.assertAlmostEqual(v[1]['t_s'], .2)
        self.assertAlmostEqual(c[0]['t_s'], .1)

    def test_quaternion_normalization_and_invalid_values(self):
        self.assertEqual(orientation_degrees([2, 0, 0, 0]), (0., 0., 0.))
        self.assertAlmostEqual(orientation_degrees([1, 0, 0, 1])[2], 90.)
        self.assertTrue(all(x != x for x in orientation_degrees([0, 0, 0, 0])))

    def test_end_to_end_plots_and_crop_keep_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visual = root/'line_frames.jsonl'
            rows = [frame(t, start_gate_mode='both', qr_passed=t>=2, shape_passed=True,
                          body_track_deviation_deg=15., body_track_deviation_valid=True, wz=.3, vx=.2)
                    for t in (1, 2, 3)]
            original = '\n'.join(json.dumps(row) for row in rows)
            visual.write_text(original, encoding='utf-8')
            control = root/'control_trace_test.csv'
            with control.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['host_unix_s', 'cmd_wz',
                    *['received_orientation_'+a for a in 'wxyz'],
                    *['received_'+group+'_'+a for group in ('gyro', 'accel') for a in 'xyz']])
                writer.writeheader()
                writer.writerows(dict(host_unix_s=t, cmd_wz=.3, received_orientation_w=1,
                                      received_orientation_x=0, received_orientation_y=0, received_orientation_z=0)
                                 for t in (1, 2, 3))
            main([str(root)])
            summary = json.loads((root/'analysis/summary.json').read_text())
            self.assertEqual(summary['visual_rows_after'], 2)
            self.assertEqual(summary['control_rows_after'], 2)
            for name in ('steering.png', 'imu_0_360.png'):
                self.assertGreater((root/'analysis'/name).stat().st_size, 1000)
            self.assertEqual(visual.read_text(encoding='utf-8'), original)
            self.assertEqual(unique_file(root, 'line_frames.jsonl'), visual)
