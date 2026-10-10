from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
import main
from command_source import CommandSnapshot
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID


def robot_state(joint_position=None, joint_velocity=None):
    return SimpleNamespace(
        status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
        accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        joint_position=(config.Q_DEFAULT.copy() if joint_position is None
                        else np.asarray(joint_position, dtype=np.float32)),
        joint_velocity=(np.zeros(12, dtype=np.float32) if joint_velocity is None
                        else np.asarray(joint_velocity, dtype=np.float32)),
        sequence=1,
    )


class ShapeMainTests(unittest.TestCase):
    def run_one_tick(self, card):
        with patch("sys.argv", ["main.py", "--model", "walk.onnx", "--no-plot",
                                 "--command-source", "vision", "--one-foot-model", "foot.onnx"]):
            args = main.parse_args()
        state = robot_state()
        with (
            patch.object(main, "parse_args", return_value=args),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as walk_cls,
            patch.object(main, "OneFootPolicy") as foot_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "PositionCsvLogger"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            walk = walk_cls.return_value
            foot = foot_cls.return_value
            walk.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            foot.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(46), 0.0)
            source_cls.return_value.get_snapshot.return_value = CommandSnapshot(
                np.zeros(3, dtype=np.float32), -1, 765, card)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, RuntimeError("end test")]
            link.get_action_status.return_value = 0
            self.assertEqual(main.main(), 1)
            return walk, foot, link

    def run_ticks(self, snapshots, states=None, action_statuses=None):
        """像 run_one_tick，但喂多帧 —— 撤重摆那一帧和形状请求是同一帧，
        顺序只有在那一帧里才看得见。states 逐帧给，测站定和保持要用。"""
        with patch("sys.argv", ["main.py", "--model", "walk.onnx", "--no-plot",
                                 "--command-source", "vision", "--one-foot-model", "foot.onnx"]):
            args = main.parse_args()
        if states is None:
            states = [robot_state()] * len(snapshots)
        with (
            patch.object(main, "parse_args", return_value=args),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as walk_cls,
            patch.object(main, "OneFootPolicy") as foot_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "PositionCsvLogger"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            walk_cls.return_value.step.return_value = (
                config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            foot_cls.return_value.step.return_value = (
                config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(46), 0.0)
            source_cls.return_value.get_snapshot.side_effect = snapshots
            link = link_cls.return_value
            link.wait_for_state.return_value = states[0]
            link.get_latest_state.side_effect = (
                list(states) + [RuntimeError("end test")])
            link.get_action_status.return_value = 0
            if action_statuses is not None:
                link.get_action_status.side_effect = action_statuses
            self.assertEqual(main.main(), 1)
            return walk_cls.return_value, link

    def test_re_pose_done_is_broadcast_to_vision(self):
        from protocol import ACTION_DONE

        window = CommandSnapshot(np.zeros(3, dtype=np.float32), card_tilt=True)
        with patch.object(main, "AttitudeBroadcaster") as broadcaster_cls:
            self.run_ticks([window] * 6, action_statuses=[0, ACTION_DONE])

        packets = broadcaster_cls.return_value.publish.call_args_list
        self.assertGreaterEqual(len(packets), 2)
        self.assertEqual({k:v for k,v in packets[0].kwargs.items() if k != 'executed_command'},
                         {"card_tilt_event_id": 1, "card_tilt_done": False})
        self.assertEqual({k:v for k,v in packets[1].kwargs.items() if k != 'executed_command'},
                         {"card_tilt_event_id": 1, "card_tilt_done": True})
        self.assertIn('executed_command', packets[0].kwargs)

    def test_the_untilt_is_requested_before_the_shape_action(self):
        """8 必须排在 1-6 前面。STM32 一次只跑一个动作，后到的直接 BUSY 丢掉 ——
        而 8 是唯一没人轮询状态的那条，被丢掉是静默的，机身就一直保持重摆后的
        前倾姿态把整套动作做完。视觉的线上顺序是：窗口开着时只有 card_tilt=True、
        没有 event；识别到的那一帧才同时有 card_tilt=False 和 event。"""
        from protocol import ACTION_CARD_RESTORE, ACTION_CARD_TILT
        window = CommandSnapshot(np.zeros(3, dtype=np.float32), -1, 0, -1,
                                 card_tilt=True)
        identify = CommandSnapshot(np.zeros(3, dtype=np.float32), 1, 765, 1,
                                   card_tilt=False)
        _walk, link = self.run_ticks([window, identify])
        sends = [call.args for call in link.send_action.call_args_list]
        self.assertEqual(sends, [(1, ACTION_CARD_TILT),
                                 (2, ACTION_CARD_RESTORE),
                                 (765, 1)])

    def test_the_re_pose_waits_for_the_robot_to_settle(self):
        """7 是固件抓快照、也是起前倾斜坡的时刻。停车触发帧上机器人还在刹车，
        抓下来的是个走路中间的姿势 —— 2026-10-02 实机就是这样：机身 -2.85°（水平）、
        一条腿直一条腿弯 27°。+20° 于是不是"把后仰扳回来"，而是在水平上再往前
        推 20°，模型解除冻结时看到一个从没命令过的姿态，甩出 87° 的髋。

        有没有等站定，靠"窗口关了机器人还在动"来分：等的话 7 压根没发出去，
        也就没有 8 要撤。"""
        from protocol import ACTION_CARD_TILT
        window = CommandSnapshot(np.zeros(3, dtype=np.float32), -1, 0, -1,
                                 card_tilt=True)
        gone = CommandSnapshot(np.zeros(3, dtype=np.float32), -1, 0, -1,
                               card_tilt=False)
        moving = robot_state(joint_velocity=np.full(12, 2.0))   # 腿还在摆
        still = robot_state()

        _walk, link = self.run_ticks([window, gone], states=[moving, moving])
        self.assertEqual(link.send_action.call_args_list, [])

        _walk, link = self.run_ticks([window, window], states=[moving, still])
        self.assertEqual([call.args for call in link.send_action.call_args_list],
                         [(1, ACTION_CARD_TILT)])

    def test_the_policy_keeps_the_frozen_state_through_the_untilt_ramp(self):
        """发完 8，固件在同一瞬间解冻，可机身还要 0.58s 才从 20° 前倾爬回站姿。
        这期间模型必须继续看旧快照，否则它看到的是那个从没命令过的倾斜。"""
        from protocol import ACTION_CARD_RESTORE, ACTION_CARD_TILT
        window = CommandSnapshot(np.zeros(3, dtype=np.float32), -1, 0, -1,
                                 card_tilt=True)
        gone = CommandSnapshot(np.zeros(3, dtype=np.float32), -1, 0, -1,
                               card_tilt=False)
        frozen = robot_state()
        # 解冻之后固件开始报真值：位置一下子变了 0.35 rad
        live = robot_state(joint_position=config.Q_DEFAULT.copy() + 0.35)
        walk, link = self.run_ticks([window, gone, gone],
                                    states=[frozen, frozen, live])
        self.assertEqual([call.args for call in link.send_action.call_args_list],
                         [(1, ACTION_CARD_TILT), (2, ACTION_CARD_RESTORE)])
        fed = [call.kwargs["joint_position_policy"]
               for call in walk.step.call_args_list]
        self.assertEqual(len(fed), 3)
        # 第 2、3 帧喂给模型的必须一模一样 —— 第 3 帧的 state 已经变了
        np.testing.assert_array_equal(fed[1], fed[2])
        self.assertTrue(np.allclose(fed[2], config.Q_DEFAULT))
        self.assertFalse(np.allclose(fed[2], live.joint_position))

    def test_arm_card_sends_only_upper_body_request(self):
        walk, foot, link = self.run_one_tick(1)
        link.send_action.assert_called_once_with(765, 1)
        walk.step.assert_called_once()
        foot.step.assert_not_called()

    def test_square_uses_right_support_one_foot_model_without_mcu_action(self):
        walk, foot, link = self.run_one_tick(3)
        link.send_action.assert_not_called()
        foot.select_support_foot.assert_called_once_with("right")
        self.assertEqual(foot.step.call_args.kwargs["lift_command"], 1.0)
        walk.step.assert_not_called()

    def test_diamond_uses_left_support(self):
        _walk, foot, link = self.run_one_tick(4)
        link.send_action.assert_not_called()
        foot.select_support_foot.assert_called_once_with("left")


if __name__ == "__main__":
    unittest.main()
