"""Lane memory survives card-stop diagnostics while obstacle observations stay live."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision" / "jetson"))
from line_detector_v1_warp import LineDetector


class TrackingSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.detector = LineDetector(z_calib=(1.0, 0.0), lane_width_cm=35.0)
        self.detector._state.update(
            smoothed_err=0.42, last_lane_center_x=174.0,
            last_lane_width_px=139.0, startup_frames=75,
            near_err_history=[4.0, 5.0, 6.0], shake_active_frames=3,
        )
        self.detector.lateral_scale = 1.11
        self.detector._rebuild_err_scale()

    def test_snapshot_is_independent_of_mutable_tracking_history(self):
        snapshot = self.detector.snapshot_tracking_state()
        self.detector._state["near_err_history"].append(99.0)
        self.detector._state["smoothed_err"] = -0.9
        self.detector.lateral_scale = 0.85
        self.assertEqual(snapshot["tracking"]["near_err_history"], [4.0, 5.0, 6.0])
        self.assertEqual(snapshot["tracking"]["smoothed_err"], 0.42)
        self.assertEqual(snapshot["lateral_scale"], 1.11)
        snapshot["tracking"]["near_err_history"].append(7.0)
        self.assertEqual(self.detector._state["near_err_history"], [4.0, 5.0, 6.0, 99.0])

    def test_restored_history_does_not_alias_the_reusable_snapshot(self):
        snapshot = self.detector.snapshot_tracking_state()
        self.detector.restore_tracking_state(snapshot)
        self.detector._state["near_err_history"].append(99.0)
        self.detector.restore_tracking_state(snapshot)
        self.assertEqual(self.detector._state["near_err_history"], [4.0, 5.0, 6.0])
        self.assertIsNot(self.detector._state["near_err_history"],
                         snapshot["tracking"]["near_err_history"])

    def test_restore_preserves_current_camera_pitch_and_rebuilds_error_scale(self):
        snapshot = self.detector.snapshot_tracking_state()
        self.detector.set_camera_pitch_deg(53.0)
        current_matrix = self.detector.M.copy()
        current_inverse = self.detector.M_inv.copy()
        current_lut = self.detector._lut_cm_per_px.copy()
        current_z_lut = self.detector._lut_z_cm.copy()
        self.detector.lateral_scale = 0.87
        self.detector._rebuild_err_scale()
        old_error_scale = self.detector.err_scale_cm
        self.detector.restore_tracking_state(snapshot)
        self.assertAlmostEqual(self.detector.cam_pitch, math.radians(53.0))
        np.testing.assert_array_equal(self.detector.M, current_matrix)
        np.testing.assert_array_equal(self.detector.M_inv, current_inverse)
        np.testing.assert_array_equal(self.detector._lut_cm_per_px, current_lut)
        np.testing.assert_array_equal(self.detector._lut_z_cm, current_z_lut)
        self.assertEqual(self.detector.lateral_scale, 1.11)
        self.assertNotAlmostEqual(self.detector.err_scale_cm, old_error_scale)
        self.assertAlmostEqual(self.detector.err_scale_cm,
                               0.5 * self.detector.bird_w *
                               self.detector.cm_per_px_at(self.detector.NEAR_BAND_ROW))

    def test_obstacle_observations_stay_live_and_startup_is_restored_without_reset(self):
        self.detector._state["start_line_z"] = 50.0
        snapshot = self.detector.snapshot_tracking_state()
        state_object = self.detector._state
        self.assertNotIn("start_line_z", snapshot["tracking"])
        self.assertFalse(any(key.startswith("red_") for key in snapshot["tracking"]))
        self.detector._state.update(startup_frames=90, red_bar_count=4,
                                    red_bar_miss=2, red_bar_cx=20.0,
                                    red_bar_cy=30.0, red_bar_z_cm=15.0,
                                    start_line_z=12.0)
        self.detector.restore_tracking_state(snapshot)
        self.assertIs(self.detector._state, state_object)
        self.assertEqual(self.detector._state["startup_frames"], 75)
        self.assertEqual(self.detector._state["red_bar_count"], 4)
        self.assertEqual(self.detector._state["red_bar_miss"], 2)
        self.assertEqual(self.detector._state["red_bar_cx"], 20.0)
        self.assertEqual(self.detector._state["red_bar_cy"], 30.0)
        self.assertEqual(self.detector._state["red_bar_z_cm"], 15.0)
        self.assertEqual(self.detector._state["start_line_z"], 12.0)

    def test_repeated_diagnostic_frames_leave_walking_seed_and_confirm_red_bar(self):
        snapshot = self.detector.snapshot_tracking_state()
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        with patch.object(self.detector, "_detect_red_bar", return_value=(640.0, 500.0, 30.0)):
            for _ in range(self.detector.red_bar_confirm_frames):
                _, _, _, _, debug = self.detector.process(frame)
                self.detector.restore_tracking_state(snapshot)
                self.assertEqual(self.detector.snapshot_tracking_state(), snapshot)
            self.assertTrue(debug["red_bar_detected"])
            self.assertGreaterEqual(self.detector._state["red_bar_count"],
                                    self.detector.red_bar_confirm_frames)
            _, _, _, _, debug = self.detector.process(frame)
        # The first resumed frame advances from the pre-stop seed, not startup zero.
        self.assertEqual(debug["startup_frames"], 76)
        self.assertEqual(debug["lost_frames"], snapshot["tracking"]["lost_frames"] + 1)


if __name__ == "__main__":
    unittest.main()
