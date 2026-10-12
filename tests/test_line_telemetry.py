from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from line_telemetry import LineTelemetry

class LineTelemetryTests(unittest.TestCase):
    def test_keeps_scalar_geometry_command_and_clock_without_images(self):
        with tempfile.TemporaryDirectory() as folder:
            writer=LineTelemetry(folder)
            writer.write({'near_error_cm':np.float32(3.5),'measurement_valid':np.bool_(False),
                          'measurement_age_s':float('inf'),'bird':np.zeros((2,2))},
                         frame=2,process_monotonic_s=12.3,vx=0.,wz=0.)
            writer.close()
            row=json.loads(writer.path.read_text())
            self.assertEqual(row['measurement'], {'near_error_cm':3.5,'measurement_valid':False,'measurement_age_s':None})
            self.assertEqual(row['wz'],0.)
            self.assertEqual(row['process_monotonic_s'],12.3)
            with self.assertRaises(FileExistsError):LineTelemetry(folder)

    def test_bad_context_is_reported_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            writer=LineTelemetry(folder)
            try:
                with self.assertRaises(TypeError): writer.write({}, frame=[1,2])
            finally: writer.close()

    def test_write_cost_is_attached_to_the_following_row(self):
        with tempfile.TemporaryDirectory() as folder:
            writer=LineTelemetry(folder)
            writer.write({}, frame=1)
            writer.write({}, frame=2)
            writer.close()
            rows=[json.loads(line) for line in writer.path.read_text().splitlines()]
            self.assertIsNone(rows[0]['telemetry_prev_write_ms'])
            self.assertTrue(math.isfinite(rows[1]['telemetry_prev_write_ms']))
            self.assertGreaterEqual(rows[1]['telemetry_prev_write_ms'],0.0)

    def test_four_counters_have_a_readable_sidecar_with_correct_units(self):
        with tempfile.TemporaryDirectory() as folder:
            writer=LineTelemetry(folder)
            writer.write({}, frame=217, host_time_ns=1_700_000_000_000_000_000,
                         steering_paired_count=167, steering_single_count=20,
                         card_stop_box_hit_count=3, card_stop_cue_hit_count=18,
                         card_stop_box_trigger_count=0,
                         card_stop_cue_trigger_count=3)
            writer.close()
            summary=writer.counts_path.read_text(encoding='utf-8')
            self.assertIn('双边路径: 167',summary)
            self.assertIn('单边路径: 20',summary)
            self.assertIn('找到图卡框: 3',summary)
            self.assertIn('无框但 cue 命中: 18',summary)
            self.assertIn('单边占已计巡线帧: 10.7%',summary)
            self.assertIn('框占两种命中: 14.3%',summary)
            self.assertIn('不是停车次数',summary)
            self.assertIn('依据框宽: 0',summary)
            self.assertIn('依据 cue 高度: 3',summary)
