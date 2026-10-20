from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
from policy_runner import HumanoidPolicy


class PolicyInterfaceTests(unittest.TestCase):
    def make_policy(self, width=49, batch=1, step_distance_m=None):
        with patch("policy_runner.ort.InferenceSession") as factory:
            session = factory.return_value
            session.get_inputs.return_value = [
                SimpleNamespace(name="obs", type="tensor(float)", shape=[batch, width])
            ]
            session.get_outputs.return_value = [SimpleNamespace(name="actions")]
            session.run.return_value = [np.arange(12, dtype=np.float32).reshape(1, 12)]
            return HumanoidPolicy("test.onnx", step_distance_m), session

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
            [0.1, 0.2, 0.981, 4, 5, 6, 0, 0, -1, 0.4, 0,
             config.STEP_LENGTH_CM / 100.0, 0],
            np.arange(12) / 100, np.arange(12) + 20, np.zeros(12),
        ]).astype(np.float32)
        target, action, obs, _ = policy.step(**values)
        np.testing.assert_allclose(obs, expected, atol=1e-7)
        np.testing.assert_allclose(target, config.Q_DEFAULT + 0.25 * np.arange(12))
        np.testing.assert_array_equal(session.run.call_args.args[1]["obs"], obs[None, :])
        next_obs = policy.build_observation(**values)
        np.testing.assert_array_equal(next_obs[37:49], action)
        np.testing.assert_allclose(
            next_obs[9:13], [0.4, 0, config.STEP_LENGTH_CM / 100.0, 0], rtol=1e-6)
        policy.reset()
        np.testing.assert_array_equal(policy.build_observation(**values)[37:], np.zeros(12))

    def test_the_step_does_not_move_when_the_speed_moves(self):
        """步长和速度是两个独立的数：步长就是个常数，不随 vx 变。

        以前步长是"每 m/s 给多大步距"的比例（--max-vx / --max-step-cm），
        实际步距 = vx × 比例。结果是 2026-10-03 把速度 0.3 改成 0.2 的时候，
        所有没显式写那两个参数的跑法步幅都短了 25%，而启动横幅上完全看不出来。"""
        policy, _ = self.make_policy()
        values = dict(
            accel_m_s2=np.array([0, 0, 9.81]),
            gyro_rad_s=np.zeros(3),
            projected_gravity=np.array([0, 0, -1]),
            joint_position_policy=config.Q_DEFAULT,
            joint_velocity_policy=np.zeros(12),
        )
        for vx in (0.05, 0.2, 0.4):
            with self.subTest(vx=vx):
                obs = policy.build_observation(
                    velocity_command=np.array([vx, 0, 0]), **values)
                self.assertAlmostEqual(float(obs[11]), config.STEP_LENGTH_CM / 100.0,
                                       places=6)          # 不随 vx 变
                self.assertAlmostEqual(float(obs[9]), vx, places=6)   # vx 照旧进观测
        # 唯一的例外：停下来时步长必须是 0，否则策略会迈原地步（真漂移）。
        stopped = policy.build_observation(
            velocity_command=np.array([0.0, 0, 0]), **values)
        self.assertEqual(float(stopped[11]), 0.0)

    def test_a_given_step_length_overrides_the_default(self):
        """给定值就盖过默认。"""
        for cm in (3.0, 5.0, 8.0):
            with self.subTest(cm=cm):
                policy, _ = self.make_policy(step_distance_m=cm / 100.0)
                obs = policy.build_observation(
                    accel_m_s2=np.array([0, 0, 9.81]), gyro_rad_s=np.zeros(3),
                    projected_gravity=np.array([0, 0, -1]),
                    velocity_command=np.array([0.2, 0, 0]),
                    joint_position_policy=config.Q_DEFAULT,
                    joint_velocity_policy=np.zeros(12))
                self.assertAlmostEqual(float(obs[11]), cm / 100.0, places=6)

    def test_the_two_numbers_are_pinned(self):
        """速度和步长各一个数，就这两个 —— 钉住，免得再被"顺手改速度"带偏。"""
        self.assertAlmostEqual(config.MAX_COMMAND_VX, 0.2, places=6)
        self.assertAlmostEqual(config.STEP_LENGTH_CM, 5.0, places=6)

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
