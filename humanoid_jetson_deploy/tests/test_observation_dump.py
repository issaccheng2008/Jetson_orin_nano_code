from __future__ import annotations

import contextlib
import csv
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from one_foot_policy import OneFootPolicy
from policy_runner import HumanoidPolicy, ObservationDump, observation_columns, open_observation_dump


class ObservationDumpSchemaTests(unittest.TestCase):
    def read_rows(self, path):
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))

    def test_shared_requested_path_keeps_both_schemas_in_separate_files(self):
        for order in ((49, 46), (46, 49)):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "obs.csv"
                dumps = {}
                try:
                    for width in order:
                        dump = dumps[width] = ObservationDump(path, width)
                        dump.append(np.arange(width), np.arange(12) + 100,
                                    np.zeros(3) if width == 49 else None)
                        dump.file.flush()
                    self.assertEqual(dumps[49].path, str(path))
                    self.assertEqual(dumps[46].path, str(path.with_name("obs_onefoot46.csv")))
                    for width, dump in dumps.items():
                        rows = self.read_rows(dump.path)
                        self.assertEqual(len(rows), 2)
                        self.assertEqual(len(rows[0]), 3 + width + 12)
                        self.assertEqual(len(rows[1]), len(rows[0]))
                        row = dict(zip(rows[0], rows[1]))
                        self.assertEqual(row[observation_columns(width)[-1]], str(width - 1))
                        self.assertEqual(row["action_r_leg_pitch_joint"], "100")
                    self.assertEqual(self.read_rows(dumps[49].path)[1][2], "1")
                    self.assertEqual(self.read_rows(dumps[46].path)[1][2], "")
                    self.assertEqual(self.read_rows(dumps[46].path)[0][12], "lift_command")
                finally:
                    for dump in dumps.values():
                        dump.file.close()

    def test_reopening_appends_without_another_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "obs.csv"
            for _ in range(2):
                dump = ObservationDump(path, 46)
                dump.append(np.zeros(46), np.zeros(12), None)
                dump.file.close()
            rows = self.read_rows(Path(folder) / "obs_onefoot46.csv")
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(len(row) == 61 for row in rows))

    def test_schema_mismatch_does_not_modify_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "obs.csv"
            path.write_text("unrelated,header\n1,2\n", encoding="utf-8")
            previous = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                ObservationDump(path)
            self.assertEqual(path.read_bytes(), previous)

    def test_wrong_width_is_rejected_before_writing_a_row(self):
        with tempfile.TemporaryDirectory() as folder:
            dump = ObservationDump(Path(folder) / "obs.csv")
            try:
                with self.assertRaisesRegex(ValueError, "expected obs"):
                    dump.append(np.zeros(46), np.zeros(12), None)
                self.assertEqual(dump.step, 0)
                self.assertEqual(len(self.read_rows(dump.path)), 1)
            finally:
                dump.file.close()

    def test_actual_policy_class_selects_schema_and_reports_actual_path(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                os.environ, {"POLICY_OBS_CSV": str(Path(folder) / "obs.csv")}):
            policies = []
            output = io.StringIO()
            try:
                for policy_class, width in ((HumanoidPolicy, 49), (OneFootPolicy, 46)):
                    with patch("policy_runner.ort.InferenceSession") as factory, contextlib.redirect_stdout(output):
                        session = factory.return_value
                        session.get_inputs.return_value = [
                            SimpleNamespace(name="obs", type="tensor(float)", shape=[1, width])]
                        session.get_outputs.return_value = [SimpleNamespace(name="actions")]
                        session.run.return_value = [np.zeros((1, 12), dtype=np.float32)]
                        policy = policy_class("test.onnx")
                        policies.append(policy)
                    values = dict(accel_m_s2=np.array([0, 0, 9.81]), gyro_rad_s=np.zeros(3),
                                  projected_gravity=np.array([0, 0, -1]),
                                  joint_position_policy=np.zeros(12), joint_velocity_policy=np.zeros(12))
                    values.update({"velocity_command": np.zeros(3)} if width == 49 else {"lift_command": 1.})
                    policy.step(**values)
                    dump = policy.observation_dump
                    dump.file.flush()
                    rows = self.read_rows(dump.path)
                    self.assertEqual(len(rows[0]), len(rows[1]))
                    self.assertIn(dump.path, output.getvalue())
                    self.assertIn(f"schema={dump.mode}", output.getvalue())
            finally:
                for policy in policies:
                    policy.observation_dump.file.close()

    def test_dump_stays_off_without_environment_path(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(open_observation_dump(46))
            self.assertIsNone(open_observation_dump())


if __name__ == "__main__":
    unittest.main()
