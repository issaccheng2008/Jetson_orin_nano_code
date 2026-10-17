"""Offline viewer must preserve recorded geometry and command provenance."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))


class VisionLogDataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('vision_log_data'), 'viewer data module missing')
        self.data = importlib.import_module('vision_log_data')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def write_log(self, rows, relative='vision/run_a/line_frames.jsonl'):
        path = self.folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        return path

    def row(self, frame=1, **changes):
        return dict(schema='line_frames_v1', frame=frame,
                    host_time_ns=1_790_000_000_000_000_000 + frame * 100_000_000,
                    process_monotonic_s=frame / 10, wz=0., **changes)

    def test_six_columns_read_recorded_heading_not_demand_or_controller_wz(self):
        row = self.row(measurement=dict(measurement_valid=True, left_seen=True,
            right_seen=True, steering_heading_deg=12.5, heading_control_deg=99.,
            steering_filtered_heading_deg=8.25, steering_filtered_demand_deg=42.,
            steering_applied_wz=.3))
        row['wz'] = -.2
        log = self.data.load_log(self.write_log([row]))
        frame = log.frames[0]
        self.assertEqual(frame.heading_deg, 12.5)
        self.assertEqual(frame.filtered_heading_deg, 8.25)
        self.assertEqual(frame.line_state, '双线')
        self.assertEqual(frame.vision_wz, -.2)
        self.assertIsNone(frame.robot_wz)
        self.assertEqual(frame.robot_status, '未记录')
        self.assertEqual(log.frames[0].elapsed_s, 0.)

    def test_line_states_and_missing_or_invalid_geometry(self):
        cases = [
            (dict(startup_card_only=True), '起步等待'),
            (dict(measurement_valid=False, heading_control_deg=99.), '丢线'),
            (dict(measurement_valid=True, measurement_stale=True), '丢线'),
            (dict(measurement_valid=True, lost_frames=2), '丢线'),
            (dict(measurement_valid=True, tracking_mode='LEFT_ONLY'), '左单线'),
            (dict(measurement_valid=True, tracking_mode='RIGHT_ONLY'), '右单线'),
            (dict(measurement_valid=True, single_line=True), '单线'),
            ({}, '未记录'),
        ]
        for measurement, expected in cases:
            with self.subTest(measurement=measurement):
                log = self.data.load_log(self.write_log([self.row(measurement=measurement)]))
                self.assertEqual(log.frames[0].line_state, expected)
                self.assertIsNone(log.frames[0].heading_deg)

    def test_raw_demand_is_independent_of_heading_filtered_demand_and_output(self):
        values = [9.327595389097295, 0., -12.3, None, float('nan')]
        rows = [self.row(i, measurement=dict(measurement_valid=True,
            steering_demand_deg=value, steering_heading_deg=7.6,
            steering_filtered_demand_deg=13.55)) for i, value in enumerate(values)]
        rows.append(self.row(8, measurement=dict(measurement_valid=False, steering_demand_deg=99.)))
        rows.append(self.row(9, measurement=dict(measurement_valid=True)))
        frames = self.data.load_log(self.write_log(rows)).frames
        self.assertEqual([f.demand_deg for f in frames], [values[0], 0., -12.3, None, None, None, None])

    def test_rowhdg_is_independent_raw_value_even_for_invalid_scan_or_lost_lane(self):
        rows = [self.row(1, measurement=dict(measurement_valid=True, steering_heading_deg=-32.,
                    rowhdg_deg=0., rowhdg_valid=False)),
                self.row(2, measurement=dict(measurement_valid=False,
                    rowhdg_deg=12.3456789, rowhdg_valid=False)),
                self.row(3, measurement=dict(measurement_valid=True, steering_heading_deg=20.)),
                self.row(4, measurement=dict(measurement_valid=True,
                    rowhdg_deg=-8.25, rowhdg_valid=True))]
        frames = self.data.load_log(self.write_log(rows)).frames
        self.assertEqual([f.rowhdg_deg for f in frames], [0., 12.3456789, None, -8.25])
        self.assertEqual([f.rowhdg_valid for f in frames], [False, False, None, True])
        self.assertEqual(frames[0].heading_deg, -32.)
        self.assertIsNone(frames[1].heading_deg)

    def test_legacy_valid_heading_and_single_line_fallback_and_nonfinite_values(self):
        rows = [self.row(1, measurement=dict(measurement_valid=True,
            heading_control_valid=True, heading_control_deg=0., steering_filter_heading_deg=-2.)),
            self.row(2, measurement=dict(measurement_valid=True, tracking_mode='RIGHT_ONLY',
                heading_control_valid=False, heading_control_deg=88.,
                single_edge_valid=True, single_edge_heading_deg=-7.)),
            self.row(3, measurement=dict(measurement_valid=True, steering_heading_deg=float('nan'),
                steering_filtered_heading_deg=float('inf')))]
        log = self.data.load_log(self.write_log(rows))
        self.assertEqual([f.heading_deg for f in log.frames], [0., -7., None])
        self.assertEqual([f.filtered_heading_deg for f in log.frames], [-2., None, None])

    def test_execution_values_require_written_enabled_feedback(self):
        rows = [self.row(i, robot_cmd_status=status, robot_cmd_wz=.17)
                for i, status in enumerate(('written', 'unknown', 'not_written', 'motors_off'), 1)]
        log = self.data.load_log(self.write_log(rows))
        self.assertEqual([f.robot_wz for f in log.frames], [.17, None, None, None])
        self.assertEqual([f.robot_status for f in log.frames], ['已采用', '未知/过期', '未发送', '电机未使能'])

    def test_discover_separate_runs_in_session_and_reject_missing_folder(self):
        a = self.write_log([self.row()])
        b = self.write_log([self.row()], 'vision/run_b/line_frames.jsonl')
        (self.folder / 'video').mkdir()
        (self.folder / 'video/frames.jsonl').write_text('{}')
        self.assertEqual(self.data.discover_logs(self.folder), (a, b))
        self.assertEqual(self.data.discover_logs(a), (a,))
        self.assertEqual(self.data.discover_logs(a.parent), (a,))
        with self.assertRaises(FileNotFoundError):
            self.data.discover_logs(self.folder / 'absent')

    def test_bad_rows_are_reported_and_empty_log_is_an_error(self):
        path = self.write_log([self.row(1), self.row(2)])
        with path.open('a', encoding='utf-8') as stream:
            stream.write('\n{truncated\n[]\n{"frame": 3}\n')
        log = self.data.load_log(path)
        self.assertEqual(len(log.frames), 2)
        self.assertEqual(log.skipped_rows, 3)
        path.write_text('')
        with self.assertRaises(ValueError):
            self.data.load_log(path)

    def test_unrecognized_status_and_numeric_timestamp_do_not_abort_other_frames(self):
        rows = [self.row(1, robot_cmd_status=[]), self.row(2)]
        rows[1]['host_time_ns'] = '1790000000200000000.0'
        log = self.data.load_log(self.write_log(rows))
        self.assertEqual(len(log.frames), 2)
        self.assertEqual(log.frames[0].robot_status, '未记录')
        self.assertEqual(log.frames[1].host_time_ns, 1790000000200000000)

    def test_monotonic_time_ignores_wall_clock_jumps_and_keeps_zero(self):
        rows = [self.row(1), self.row(2)]
        rows[0]['process_monotonic_s'] = 0.
        rows[1]['host_time_ns'] -= 10_000_000_000
        log = self.data.load_log(self.write_log(rows))
        self.assertEqual([f.elapsed_s for f in log.frames], [0., .2])

    def test_twenty_frame_window_and_five_frame_moves_clamp_to_edges(self):
        frames = self.data.load_log(self.write_log([self.row(i) for i in range(31)])).frames
        cursor = self.data.FrameCursor(frames)
        self.assertEqual(cursor.window(), (0, 20))
        cursor.step(5)
        self.assertEqual(cursor.index, 5)
        cursor.select(16)
        self.assertEqual(cursor.window(), (6, 26))
        cursor.step(100)
        self.assertEqual((cursor.index, cursor.window()), (30, (11, 31)))
        cursor.step(-100)
        self.assertEqual(cursor.index, 0)
        short = self.data.FrameCursor(frames[:3])
        self.assertEqual(short.window(), (0, 3))

    def test_time_selection_uses_nearest_recorded_frame_with_irregular_intervals(self):
        rows = [self.row(i) for i in range(3)]
        for row, stamp in zip(rows, (5., 5.1, 6.)):
            row['process_monotonic_s'] = stamp
        cursor = self.data.FrameCursor(self.data.load_log(self.write_log(rows)).frames)
        cursor.seek_time(.11)
        self.assertEqual(cursor.index, 1)
        cursor.seek_time(.8)
        self.assertEqual(cursor.index, 2)
        cursor.seek_time(-1)
        self.assertEqual(cursor.index, 0)


if __name__ == '__main__':
    unittest.main()
