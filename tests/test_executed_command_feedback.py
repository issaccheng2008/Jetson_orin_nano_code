import json
from pathlib import Path
import socket
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'humanoid_jetson_deploy'))
from attitude_input import AttitudeInput
from attitude_broadcast import AttitudeBroadcaster


def executed(now=10., wz=.5):
    return dict(velocity=[.2, 0., wz], monotonic_s=now, host_unix_s=100.,
                step=5, policy_mode='walking49', send_result='written', enabled=True)


class ExecutedFeedbackTests(unittest.TestCase):
    def test_real_udp_transmits_actual_command_without_pose_smoothing(self):
        receiver = AttitudeInput(0, 45.)
        sender = AttitudeBroadcaster(port=receiver.socket.getsockname()[1])
        try:
            sender.publish([0, 0, -1], 1., executed_command=executed())
            receiver.poll(10.1)
            self.assertEqual(receiver.executed_command(10.1)['velocity'][2], .5)
            sender.publish([0, 0, -1], 1.1, executed_command=executed(10.2, -.5))
            receiver.poll(10.21)
            self.assertEqual(receiver.executed_command(10.21)['velocity'][2], -.5)
            self.assertIsNone(receiver.executed_command(10.8))
        finally:
            sender.close()
            receiver.close()

    def test_legacy_pose_packets_do_not_invent_or_refresh_execution(self):
        receiver = AttitudeInput(0, 45.)
        sender = AttitudeBroadcaster(port=receiver.socket.getsockname()[1])
        try:
            sender.publish([0, 0, -1], 1.)
            receiver.poll(10.)
            self.assertIsNone(receiver.executed_command(10.))
            sender.publish([0, 0, -1], 1., executed_command=executed())
            receiver.poll(10.)
            sender.publish([0, 0, -1], 2.)
            receiver.poll(11.)
            self.assertIsNone(receiver.executed_command(11.))
        finally:
            sender.close()
            receiver.close()

    def test_delayed_or_invalid_feedback_is_not_displayed_as_current(self):
        receiver = AttitudeInput(0, 45.)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            address = ('127.0.0.1', receiver.socket.getsockname()[1])
            for data in (executed(1.), executed(100.), executed(10., float('nan'))):
                sender.sendto(json.dumps({'g':[0,0,-1], 'executed_command':data}).encode(), address)
                receiver.poll(10.)
                self.assertIsNone(receiver.executed_command(10.))
        finally:
            sender.close()
            receiver.close()

    def test_nonfinite_feedback_does_not_break_valid_attitude(self):
        receiver = AttitudeInput(0, 45.)
        sender = AttitudeBroadcaster(port=receiver.socket.getsockname()[1])
        try:
            sender.publish([0, 0, -1], 1., executed_command=executed(10., float('nan')))
            self.assertTrue(receiver.poll(10.))
            self.assertAlmostEqual(receiver.value, 45.)
            self.assertIsNone(receiver.executed_command(10.))
        finally:
            sender.close()
            receiver.close()
