from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
from policy_runner import HumanoidPolicy


class PolicyInterfaceTests(unittest.TestCase):
    def make_policy(self, width=47, batch=1):
        with patch("policy_runner.ort.InferenceSession") as factory:
            session = factory.return_value
            session.get_inputs.return_value = [
                SimpleNamespace(name="obs", type="tensor(float)", shape=[batch, width])
            ]
            session.get_outputs.return_value = [SimpleNamespace(name="actions")]
            session.run.return_value = [np.arange(12, dtype=np.float32).reshape(1, 12)]
            return HumanoidPolicy("test.onnx"), session

    def test_exact_training_layout_and_action_history(self):
        policy, session = self.make_policy()
        values = dict(
            accel_m_s2=np.array([1, 2, 9.81]),
            gyro_rad_s=np.array([4, 5, 6]),
            projected_gravity=np.array([0, 0, -1]),
            velocity_command=np.array([0.4, 0, 0]),
            joint_position_policy=config.Q_DEFAULT + np.arange(12) / 100,
            joint_velocity_policy=np.arange(12) + 20,
        )
        expected = np.concatenate([
            [0.1, 0.2, 0.981, 4, 5, 6, 0, 0, -1, 0.4, 0],
            np.arange(12) / 100, np.arange(12) + 20, np.zeros(12),
        ]).astype(np.float32)
        target, action, obs, _ = policy.step(**values)
        np.testing.assert_allclose(obs, expected, atol=1e-7)
        np.testing.assert_allclose(target, config.Q_DEFAULT + 0.25 * np.arange(12))
        np.testing.assert_array_equal(session.run.call_args.args[1]["obs"], obs[None, :])
        next_obs = policy.build_observation(**values)
        np.testing.assert_array_equal(next_obs[35:47], action)
        # Command block is [vx, wz]; q_rel starts immediately after it.
        np.testing.assert_allclose(next_obs[9:11], [0.4, 0])
        policy.reset()
        np.testing.assert_array_equal(policy.build_observation(**values)[35:], np.zeros(12))

    def test_rejects_legacy_models_before_inference(self):
        # 46 is the one-foot policy's width, 48 and 49 are the older walking
        # layouts that still observed a step-distance command.
        for width in (46, 48, 49):
            with self.subTest(width=width), self.assertRaisesRegex(RuntimeError, "legacy"):
                self.make_policy(width)

    def test_accepts_dynamic_batch(self):
        self.make_policy(batch="batch")

    def test_rejects_nonfinite_observations(self):
        policy, _ = self.make_policy()
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            policy.build_observation(
                np.array([np.nan, 0, 9.81]), np.zeros(3), np.array([0, 0, -1]),
                np.array([0.4, 0, 0]), config.Q_DEFAULT, np.zeros(12),
            )


if __name__ == "__main__":
    unittest.main()
