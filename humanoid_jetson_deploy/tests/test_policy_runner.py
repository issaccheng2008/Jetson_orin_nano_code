from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
from policy_runner import HumanoidPolicy


class PolicyInterfaceTests(unittest.TestCase):
    def make_policy(self, width=49, batch=1):
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
            [0.1, 0.2, 0.981, 4, 5, 6, 0, 0, -1, 0.4, 0, 0.08, 0],
            np.arange(12) / 100, np.arange(12) + 20, np.zeros(12),
        ]).astype(np.float32)
        target, action, obs, _ = policy.step(**values)
        np.testing.assert_allclose(obs, expected, atol=1e-7)
        np.testing.assert_allclose(target, config.Q_DEFAULT + 0.25 * np.arange(12))
        np.testing.assert_array_equal(session.run.call_args.args[1]["obs"], obs[None, :])
        next_obs = policy.build_observation(**values)
        np.testing.assert_array_equal(next_obs[37:49], action)
        np.testing.assert_allclose(next_obs[9:13], [0.4, 0, 0.08, 0])
        policy.reset()
        np.testing.assert_array_equal(policy.build_observation(**values)[37:], np.zeros(12))

    def test_step_distance_tracks_the_commanded_speed(self):
        """The touchdown target scales with speed instead of being a fixed switch.

        It used to be 0.08 for any nonzero vx, so the first connector tick after a
        card stop asked for 0.08 m steps at 0.02 m/s - a pairing the policy had no
        reason to have seen in training.
        """
        policy, _ = self.make_policy()
        values = dict(
            accel_m_s2=np.array([0, 0, 9.81]),
            gyro_rad_s=np.zeros(3),
            projected_gravity=np.array([0, 0, -1]),
            joint_position_policy=config.Q_DEFAULT,
            joint_velocity_policy=np.zeros(12),
        )
        for vx, expected in ((0.4, 0.08), (0.2, 0.04), (0.02, 0.004), (0.0, 0.0)):
            with self.subTest(vx=vx):
                obs = policy.build_observation(
                    velocity_command=np.array([vx, 0, 0]), **values)
                self.assertAlmostEqual(float(obs[11]), expected, places=6)

    def test_rejects_legacy_models_before_inference(self):
        for width in (47, 48):
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
