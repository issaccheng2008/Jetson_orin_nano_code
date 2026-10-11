from __future__ import annotations

import unittest

from shape_actions import ShapeActionController


class ShapeActionControllerTests(unittest.TestCase):
    def test_all_six_cards_use_mcu_handshake_and_never_onefoot(self):
        for card in range(1, 7):
            with self.subTest(card=card):
                task = ShapeActionController()
                self.assertTrue(task.accept(100 + card, card, 0.0))
                self.assertFalse(task.advance(0.1, stopped=False).send_upper)
                request = task.advance(0.2, stopped=True)
                self.assertTrue(request.send_upper)
                self.assertEqual(request.policy, "walking")
                self.assertTrue(task.advance(0.3, True, upper_status=1).send_upper)
                self.assertTrue(task.advance(0.4, True, upper_status=3).send_upper)
                self.assertTrue(task.advance(5.9, True).busy)
                self.assertFalse(task.advance(6.0, True, upper_status=2).busy)
                self.assertFalse(task.accept(100 + card, card, 7.0))

    def test_leg_requests_wait_for_untilt(self):
        for card in (3, 4):
            task = ShapeActionController()
            task.accept(103, card, 0.0)
            self.assertFalse(task.advance(0.1, True, ready_to_lift=False).send_upper)
            self.assertTrue(task.advance(0.65, True, ready_to_lift=True).send_upper)

    def test_leg_actions_have_eight_second_timeout_and_reject_terminal_errors(self):
        task = ShapeActionController()
        self.assertGreaterEqual(task.action_timeout, 8.0)
        task.accept(103, 3, 0.0)
        task.advance(0.2, True)
        self.assertTrue(task.advance(7.9, True).send_upper)
        with self.assertRaises(TimeoutError):
            task.advance(8.21, True)
        for status in (4, 5):
            task = ShapeActionController()
            task.accept(104, 4, 0.0)
            task.advance(0.1, True)
            with self.assertRaises(RuntimeError):
                task.advance(0.2, True, upper_status=status)

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
                self.assertTrue(task.advance(0.4, stopped=True, upper_status=2).busy)
                self.assertFalse(task.advance(3.2, stopped=True, upper_status=2).busy)

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
        self.assertTrue(task.advance(0.5, stopped=True, upper_status=2).busy)
        self.assertFalse(task.advance(3.2, stopped=True, upper_status=2).busy)

    def test_no_card_and_busy_events_do_not_start_an_action(self):
        task = ShapeActionController()
        self.assertFalse(task.accept(0, -1, 0.0))
        self.assertFalse(task.advance(0.1, stopped=True).busy)
        self.assertTrue(task.accept(20, 3, 0.2))
        self.assertFalse(task.accept(21, 4, 0.3))


if __name__ == "__main__":
    unittest.main()
