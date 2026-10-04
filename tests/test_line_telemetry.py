from pathlib import Path
import json
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
