"""Rejected scans survive to scalar telemetry without changing lane acceptance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from line_detector_v1_warp import LineDetector
from line_telemetry import LineTelemetry


class LaneRejectionDiagnosticsTests(unittest.TestCase):
    def detector(self):
        d = LineDetector(z_calib=(1., 0.), lane_width_cm=35.)
        d.startup_force_simple_bottom = False
        d.bottom_lock_enable = False
        d.robust_enable = False
        # These gate tests explicitly use a 140px lane and 50px hint tolerance.
        d.lane_width_tol_px = 50.
        return d

    def scan(self, d, rows=8, blocked=None):
        gray = np.zeros((400, 320), np.uint8)
        for y in range(350, 350+rows*2, 2):
            gray[y, 87:94] = gray[y, 227:234] = 255
        diag = {}
        with patch.object(d, '_detect_row_blocker', return_value=blocked or (False, False)):
            result = d._scan_band_midline(gray, np.zeros((400,320,3), np.uint8), 25, False,
                160., 140., .875, 1., 8, 2, diagnostics=diag)
        return result, diag

    def band(self, d, **changes):
        diag = {}
        ys = list(range(350, 366, 2))
        result = dict(band_name='low', ys_list=ys, centers_list=[160.]*8,
            modes_list=[2]*8, widths_list=[140.]*8, scan_rows=8,
            center_px=160., center_cm=0., dist_cm=25., lane_width_px=140.,
            weight=.65, conf=.9, pair_ratio=1., left_seen=True, right_seen=True,
            _scan_diagnostics=diag)
        result.update(changes)
        return result, diag

    def process(self, d, bands):
        with patch.object(d, '_detect_two_band_lanes', return_value=bands):
            return d.process(np.full((720,1280,3), 255, np.uint8), dt=.1)[-1]

    def test_empty_scan_is_distinct_from_insufficient_candidate_rows(self):
        d = self.detector()
        result, empty = self.scan(d, 0)
        self.assertIsNone(result)
        self.assertEqual(empty['rows_scanned'], 8)
        self.assertEqual(empty['run_candidate_rows'], 0)
        self.assertEqual(empty['candidate_rows'], 0)
        self.assertEqual(empty['paired_rows'], 0)
        self.assertEqual(empty['reject_reason'], 'no_candidates')
        result, sparse = self.scan(d, 2)
        self.assertIsNone(result)
        self.assertEqual(sparse['candidate_rows'], 2)
        self.assertEqual(sparse['paired_rows'], 2)
        self.assertEqual(sparse['reject_reason'], 'insufficient_rows')

    def test_blocked_scan_reports_obstacle_rows_without_saying_no_lines(self):
        d = self.detector()
        for blockers, field in [((True, False), 'red_block_rows'), ((False, True), 'black_block_rows')]:
            result, diag = self.scan(d, blocked=blockers)
            self.assertIsNone(result)
            self.assertEqual(diag[field], 8)
            self.assertEqual(diag['reject_reason'], 'rows_blocked')

    def test_row_pairing_rejects_distinguish_edge_width_distance_and_hint(self):
        d = self.detector()
        diag = d._new_scan_diagnostics('low')
        gray = np.zeros((400,320),np.uint8)
        gray[350,10:12]=255
        self.assertEqual(d._collect_track_runs_on_row(gray,350,0,319,25,False,diagnostics=diag),[])
        self.assertEqual(diag['raw_runs'],1)
        self.assertEqual(diag['line_width_rejected_runs'],1)
        self.assertIsNone(d._choose_pair_center_from_runs([(10,13),(150,156)],160,140,0,319,diagnostics=diag))
        self.assertGreater(diag['pair_thin_edge_skips'],0)
        self.assertIsNone(d._choose_pair_center_from_runs([(10,16),(25,31)],160,140,0,319,diagnostics=diag))
        self.assertEqual(diag['pair_width_range_rejects'],1)
        self.assertIsNone(d._choose_pair_center_from_runs([(10,16),(250,256)],160,140,0,319,diagnostics=diag))
        self.assertEqual(diag['pair_width_hint_rejects'],1)
        self.assertIsNone(d._centroid_pair_center(np.full((400,320),255,np.uint8),350,160,140,0,319,diagnostics=diag))
        self.assertEqual(diag['centroid_low_contrast_rows'],1)

    def test_pair_ratio_records_both_denominators(self):
        d = self.detector()
        result, diag = self.band(d, scan_rows=20)
        self.assertIsNone(d._qualified_band(result))
        self.assertEqual(diag['paired_rows'], 8)
        self.assertEqual(diag['pair_fraction'], 1.)
        self.assertEqual(diag['pair_support_ratio'], .4)
        self.assertEqual(diag['pair_ratio_min'], .55)
        self.assertEqual(diag['reject_reason'], 'pair_support_ratio')
        result, diag = self.band(d, modes_list=[2]*4+[1]*4)
        self.assertIsNone(d._qualified_band(result))
        self.assertEqual(diag['pair_fraction'], .5)
        self.assertEqual(diag['reject_reason'], 'pair_fraction')

    def test_geometry_gates_record_values_limits_and_independent_failures(self):
        d = self.detector()
        cases = [
            (dict(widths_list=[70.,210.]*4), 'width_cv', 'width_cv_pass'),
            (dict(centers_list=[120.,200.]*4), 'fit_residual', 'fit_residual_pass'),
            (dict(widths_list=[400.]*8, lane_width_px=400.), 'width_hint', 'width_hint_pass'),
        ]
        for changes, reason, flag in cases:
            with self.subTest(reason=reason):
                result, diag = self.band(d, **changes)
                self.assertIsNone(d._qualified_band(result))
                self.assertEqual(diag['reject_stage'], 'geometry')
                self.assertEqual(diag['reject_reason'], reason)
                self.assertFalse(diag[flag])
                self.assertEqual(diag['width_cv_max'], d.lock_width_cv_max)
                self.assertEqual(diag['fit_rmse_max_px'], d.observation_rmse_max_px)
                self.assertIsNotNone(diag['fit_rmse_px'])

    def test_single_width_missing_and_stale_history_are_separate(self):
        d = self.detector()
        result, diag = self.band(d, modes_list=[1]*8, left_seen=False, pair_ratio=0.)
        self.assertIsNone(d._qualified_band(result))
        self.assertEqual(diag['reject_reason'], 'single_width_history_missing')
        d._state.update(paired_width_frames=2, last_paired_tracking_time=0.)
        d._observation_clock_s=2.
        result, diag = self.band(d, modes_list=[1]*8, left_seen=False, pair_ratio=0.)
        self.assertIsNone(d._qualified_band(result))
        self.assertEqual(diag['reject_reason'], 'single_width_history_stale')
        self.assertEqual(diag['single_width_age_s'], 2.)

    def test_observation_quality_gate_keeps_failed_value(self):
        d = self.detector()
        result, diag = self.band(d, conf=.19)
        self.assertIsNone(d._qualified_band(result))
        self.assertEqual(diag['reject_reason'], 'measurement_quality')
        self.assertLess(diag['observation_quality'], diag['measurement_quality_min'])

    def test_confidence_and_total_weight_rejections_survive_to_final_debug(self):
        d = self.detector()
        band, _ = self.band(d, conf=.01)
        debug = self.process(d, [band])
        self.assertEqual(debug['scan_low_reject_stage'], 'confidence')
        self.assertEqual(debug['scan_low_reject_reason'], 'confidence')
        self.assertEqual(debug['scan_low_confidence'], .01)
        self.assertGreater(debug['scan_low_confidence_min'], .01)
        d = self.detector()
        d.min_weight = 10.
        band, _ = self.band(d)
        debug = self.process(d, [band])
        self.assertFalse(debug['measurement_valid'])
        self.assertEqual(debug['scan_low_reject_reason'], 'total_weight')
        self.assertGreater(debug['lane_min_weight_active'], debug['lane_total_weight'])
        self.assertEqual(debug['lane_reject_reason'], 'total_weight')

    def test_bottom_lock_reports_no_lines_pair_support_and_disabled(self):
        d = self.detector()
        d.bottom_lock_enable = True
        gray = np.zeros((400,320),np.uint8)
        bgr = np.zeros((400,320,3),np.uint8)
        d._detect_bottom_center_lock(gray,bgr,25,False)
        diag = d._scan_diagnostics['bottom_lock']
        self.assertEqual(diag['reject_reason'], 'no_candidates')
        self.assertEqual(diag['rows_scanned'], 10)
        y = int(d.bottom_lock_start_ratio*d.bird_h)
        gray[y:y+6,87:94]=gray[y:y+6,227:234]=255
        d._detect_bottom_center_lock(gray,bgr,25,False)
        self.assertEqual(d._scan_diagnostics['bottom_lock']['reject_reason'], 'pair_support_ratio')
        d.bottom_lock_enable = False
        d._detect_bottom_center_lock(gray,bgr,25,False)
        self.assertEqual(d._scan_diagnostics['bottom_lock']['reject_reason'], 'disabled')

    def test_per_frame_reset_and_jsonl_keep_only_current_scalar_diagnostics(self):
        d = self.detector()
        band, _ = self.band(d, widths_list=[70.,210.]*4)
        first = self.process(d,[band])
        second = d.process(np.full((720,1280,3),255,np.uint8),dt=.1)[-1]
        self.assertEqual(first['scan_low_reject_reason'], 'width_cv')
        self.assertEqual(second['scan_low_reject_reason'], 'no_candidates')
        self.assertIsNone(second['scan_low_width_cv'])
        self.assertEqual(second['scan_low_candidate_rows'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = LineTelemetry(tmp)
            telemetry.write(first, frame=1)
            telemetry.write(second, frame=2)
            telemetry.close()
            rows = [json.loads(line) for line in (Path(tmp)/'line_frames.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['measurement']['scan_low_reject_reason'], 'width_cv')
        self.assertIsNone(rows[1]['measurement']['scan_low_width_cv'])
        self.assertNotIn('binary', rows[0]['measurement'])


if __name__ == '__main__':
    unittest.main()
