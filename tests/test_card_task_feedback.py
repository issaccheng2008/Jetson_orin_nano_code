"""Real UDP feedback must identify the completed action and expire stale data."""
import json
from pathlib import Path
import select
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
sys.path.insert(0, str(ROOT / 'humanoid_jetson_deploy'))
from attitude_input import AttitudeInput
from attitude_broadcast import AttitudeBroadcaster
from run_policy_vision import ordinary_card_feedback


class CardTaskFeedbackTests(unittest.TestCase):
    def test_real_policy_packet_round_trip_and_expiry(self):
        receiver = AttitudeInput(0, 45.)
        sender = AttitudeBroadcaster(port=receiver.socket.getsockname()[1])
        try:
            task = dict(restore_id=2, restore_done=True, action_id=42,
                        action_done=True, monotonic_s=10.)
            sender.publish([0, 0, -1], 1., card_tilt_event_id=0, card_task=task)
            select.select([receiver.socket], [], [], .5)
            receiver.poll(10.1)
            feedback = ordinary_card_feedback(receiver, 10.1)
            self.assertEqual(feedback['action_id'], 42)
            self.assertTrue(feedback['action_done'])
            self.assertTrue(feedback['restore_done'])
            self.assertNotIn('action_id', ordinary_card_feedback(receiver, 10.6))
        finally:
            sender.close()
            receiver.close()

    def test_bad_or_out_of_order_completion_cannot_replace_current_event(self):
        receiver = AttitudeInput(0, 45.)
        try:
            current = dict(restore_id=2, restore_done=True, action_id=42,
                           action_done=False, monotonic_s=10.)
            receiver._observe_card_task(current, 10.1)
            for task in (dict(current, monotonic_s=9.9, action_done=True),
                         dict(current, action_id=True), dict(current, action_done=1),
                         dict(current, monotonic_s=float('nan')),
                         dict(current, monotonic_s=20.)):
                receiver._observe_card_task(task, 10.1)
            self.assertEqual(receiver.card_task(10.1), current)
        finally:
            receiver.close()

    def test_pose_only_packets_do_not_refresh_action_completion(self):
        receiver = AttitudeInput(0, 45.)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver._observe_card_task(dict(restore_id=2, restore_done=True,
                action_id=42, action_done=True, monotonic_s=10.), 10.)
            sender.sendto(json.dumps(dict(g=[0,0,-1], card_tilt_event_id=0,
                card_tilt_done=False)).encode(), receiver.socket.getsockname())
            select.select([receiver.socket], [], [], .5)
            receiver.poll(11.)
            self.assertEqual(receiver.card_task(11.), {})
        finally:
            sender.close()
            receiver.close()
