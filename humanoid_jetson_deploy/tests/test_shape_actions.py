from __future__ import annotations

import unittest

from shape_actions import ShapeActionController


class ShapeActionControllerTests(unittest.TestCase):
    def test_square_lifts_left_leg_with_right_support_for_over_three_seconds(self):
        task = ShapeActionController(lift_seconds=3.2)
        self.assertTrue(task.accept(101, 3, 0.0))
        self.assertFalse(task.accept(101, 3, 0.1))
        self.assertEqual(task.advance(0.1, stopped=False).policy, "walking")
        self.assertEqual(task.advance(0.2, stopped=True).support_foot, "right")
        self.assertEqual(task.advance(0.8, stopped=True).lift_command, 1.0)
        self.assertEqual(task.advance(3.9, stopped=True).lift_command, 1.0)
        self.assertEqual(task.advance(4.1, stopped=True).lift_command, 0.0)
        self.assertTrue(task.advance(5.1, stopped=True).busy is False)

    def test_diamond_lifts_right_leg_with_left_support(self):
        task = ShapeActionController()
        self.assertTrue(task.accept(102, 4, 0.0))
        self.assertEqual(task.advance(0.1, stopped=True).support_foot, "left")

    def test_upper_body_cards_request_stm32_and_wait_for_completion(self):
        for card in (1, 2, 5, 6):
            with self.subTest(card=card):
                task = ShapeActionController()
                self.assertTrue(task.accept(card, card, 0.0))
                self.assertFalse(task.advance(0.1, stopped=False).send_upper)
                self.assertTrue(task.advance(0.2, stopped=True).send_upper)
                accepted = task.advance(0.3, stopped=True, upper_status=1)
                self.assertTrue(accepted.busy)
                self.assertTrue(accepted.send_upper)  # keep polling until completion is confirmed
                self.assertFalse(task.advance(3.6, stopped=True, upper_status=2).busy)

    def test_busy_keeps_waiting_instead_of_failing(self):
        """BUSY 不是"被拒"。STM32 一次只跑一个动作，撤重摆（8）还在跑的时候形状
        请求本来就是 BUSY —— 当成失败会直接 RuntimeError 把整条链路打掉，
        而等待是正常的：几帧之后它就会收下。"""
        task = ShapeActionController()
        self.assertTrue(task.accept(765, 1, 0.0))
        self.assertTrue(task.advance(0.2, stopped=True).send_upper)
        busy = task.advance(0.3, stopped=True, upper_status=3)
        self.assertTrue(busy.busy)
        self.assertTrue(busy.send_upper)            # 继续请求，直到它收下
        self.assertTrue(task.advance(0.4, stopped=True, upper_status=1).send_upper)
        self.assertFalse(task.advance(0.5, stopped=True, upper_status=2).busy)

    def test_no_card_and_busy_events_do_not_start_an_action(self):
        task = ShapeActionController()
        self.assertFalse(task.accept(0, -1, 0.0))
        self.assertFalse(task.advance(0.1, stopped=True).busy)
        self.assertTrue(task.accept(20, 3, 0.2))
        self.assertFalse(task.accept(21, 4, 0.3))


if __name__ == "__main__":
    unittest.main()
