from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import config
from fixed_joint_policy import FixedJointPolicy


class FixedJointPolicyTests(unittest.TestCase):
    def write_policy(self, directory, **changes):
        payload = {
            "format": "joint_frames_v1",
            "hz": 50,
            "joint_names": list(config.JOINT_NAMES),
            "frames": [config.Q_DEFAULT.tolist(), (config.Q_DEFAULT + 0.01).tolist()],
        }
        payload.update(changes)
        path = Path(directory) / "policy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_reads_one_frame_per_step_and_then_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = FixedJointPolicy(self.write_policy(directory))
            np.testing.assert_allclose(policy.next_target(), config.Q_DEFAULT)
            np.testing.assert_allclose(policy.next_target(), config.Q_DEFAULT + 0.01)
            self.assertIsNone(policy.next_target())

    def test_rejects_wrong_joint_order(self):
        with tempfile.TemporaryDirectory() as directory:
            names = list(config.JOINT_NAMES)
            names.reverse()
            with self.assertRaisesRegex(ValueError, "joint_names"):
                FixedJointPolicy(self.write_policy(directory, joint_names=names))

    def test_rejects_nonfinite_and_out_of_limit_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            for frames in ([config.Q_DEFAULT.tolist()[:-1]],
                           [[float("nan")] * 12],
                           [[5.0] * 12]):
                with self.subTest(frames=frames), self.assertRaises(ValueError):
                    FixedJointPolicy(self.write_policy(directory, frames=frames))

    def test_rejects_wrong_frequency_and_empty_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            for changes in ({"hz": 100}, {"frames": []}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    FixedJointPolicy(self.write_policy(directory, **changes))


if __name__ == "__main__":
    unittest.main()
