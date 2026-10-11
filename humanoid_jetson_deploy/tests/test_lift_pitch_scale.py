"""All-joint card lift scaling before safety, with automatic restoration."""

import unittest
from unittest.mock import patch

import numpy as np

import config
import main
from command_source import CommandSnapshot
import one_foot_policy
from shape_actions import ShapeActionController
from target_safety import TargetSafety
import test_shape_main


class LiftPitchScaleTests(unittest.TestCase):
    def test_scales_all_twelve_targets_once_without_mutating_model_output(self):
        target = np.linspace(-0.4, 0.4, 12, dtype=np.float32)
        original = target.copy()
        expected = target.copy()
        expected *= 0.9
        with patch.object(config, "SHAPE_LIFT_PITCH_SCALE", 0.9, create=True):
            for _ in range(2):
                np.testing.assert_allclose(one_foot_policy.apply_lift_target_scale(target, True), expected)
        np.testing.assert_array_equal(target, original)

    def test_configuration_is_adjustable_and_inactive_target_is_restored(self):
        target = config.Q_DEFAULT.copy()
        for scale in (0.8, 1.0):
            with self.subTest(scale=scale), patch.object(
                    config, "SHAPE_LIFT_PITCH_SCALE", scale, create=True):
                expected = target.copy()
                expected *= scale
                np.testing.assert_allclose(one_foot_policy.apply_lift_target_scale(target, True), expected)
                np.testing.assert_array_equal(one_foot_policy.apply_lift_target_scale(target, False), target)

    def test_invalid_active_scale_is_rejected(self):
        for scale in (-0.1, float("nan"), float("inf")):
            with self.subTest(scale=scale), patch.object(
                    config, "SHAPE_LIFT_PITCH_SCALE", scale, create=True):
                with self.assertRaises(ValueError):
                    one_foot_policy.apply_lift_target_scale(config.Q_DEFAULT, True)

    def test_mcu_cards_do_not_use_host_lift_scaling(self):
        harness = test_shape_main.ShapeMainTests()
        for card in (3, 4):
            with self.subTest(card=card):
                controller = ShapeActionController()
                controller.accept(765, card, 0.0)
                advance = controller.advance
                times = iter((0.0, 0.5, 3.0, 3.31))
                targets = []
                apply_safety = TargetSafety.apply_with_trace

                def record_target(safety, target, *args):
                    targets.append(target.copy())
                    return apply_safety(safety, target, *args)

                snapshots = [CommandSnapshot(np.zeros(3, dtype=np.float32),
                                             -1, 765, card)] * 4
                with (
                    patch.object(config, "SHAPE_LIFT_PITCH_SCALE", 0.8, create=True),
                    patch.object(main, "ShapeActionController", return_value=controller),
                    patch.object(controller, "advance", side_effect=lambda *args, **kwargs:
                                 advance(next(times), stopped=True)),
                    patch.object(TargetSafety, "apply_with_trace", autospec=True,
                                 side_effect=record_target),
                ):
                    walk, _link = harness.run_ticks(snapshots)
                expected = config.Q_DEFAULT.copy()
                self.assertEqual(len(targets), 4)
                for target in targets[:2]:
                    np.testing.assert_allclose(target, expected)
                for target in targets[2:]:
                    np.testing.assert_array_equal(target, config.Q_DEFAULT)
                self.assertEqual(walk.step.call_count, 0)


if __name__ == "__main__":
    unittest.main()
