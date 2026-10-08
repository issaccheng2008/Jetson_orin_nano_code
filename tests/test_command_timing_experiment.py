"""Experimental defaults at visual output and the actual walking observation."""
from pathlib import Path
import json
import socket
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'new_vision/jetson'), str(ROOT / 'humanoid_jetson_deploy'),
               str(ROOT / 'humanoid_jetson_deploy/tests')]
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from walking_command_hold import WalkingCommandHold
import test_walking_command_hold as model_tests
import config
from connector import CommandSmoother, process_vision_output
from command_source import UdpCommandSource
from policy_bridge import ConnectorClient
from policy_runner import HumanoidPolicy
from segment_steering import SegmentSteeringController
from test_segment_steering import detection as segment_reading, bend

EXPERIMENT = 'no_hold'


def reading(heading):
    return dict(fused_err_cm=0., angle_err_deg=0., lost_frames=0,
                base_err_cm=0., near_error_cm=0., near_z_cm=0.,
                measurement_valid=True, heading_control_valid=True,
                heading_control_deg=heading)


class CommandTimingExperimentTests(unittest.TestCase):
    def test_fresh_low_rate_segments_confirm_without_restoring_command_hold(self):
        for dt in (.251, .3):
            with self.subTest(dt=dt):
                c = SegmentSteeringController(SteeringController())
                debug = segment_reading(bend, measurement_age_s=0., measurement_max_age_s=.25)
                for expected in (1, 2, 3):
                    out = c.command(debug, 1., dt)
                    self.assertEqual(c.diagnostics['segment_confirm_frames'], expected)
                self.assertTrue(c.diagnostics['segment_control_active'])
                self.assertGreater(out[1], 0.)
                self.assertEqual(c.turn_left, 0.)
                self.assertEqual(c.min_hold_s, 0.)

    def controller(self, yaw_sign=1):
        return HeadingSteeringController(SteeringController(yaw_sign=yaw_sign))

    def test_visual_default_releases_both_turn_directions_before_half_second(self):
        for sign in (-1, 1):
            for direction in (-1, 1):
                with self.subTest(sign=sign, direction=direction):
                    c = self.controller(sign)
                    first = c.command(reading(direction*40.), 1., .05)
                    outputs = [c.command(reading(0.), 1., .05) for _ in range(3)]
                    self.assertGreater(first[1]*sign*direction, 0.)
                    self.assertEqual(outputs[-1], (.2, 0.))
                    self.assertTrue(any(abs(out[1]) < abs(first[1]) for out in outputs))

    def test_visual_increase_and_reversal_follow_selected_experiment(self):
        for initial, target in ((10., 40.), (40., -40.), (-10., -40.)):
            with self.subTest(initial=initial, target=target):
                c = self.controller()
                first = c.command(reading(initial), 1., .05)
                output = first
                for _ in range(3):
                    output = c.command(reading(target), 1., .05)
                if EXPERIMENT == 'no_hold':
                    self.assertNotEqual(output, first)
                    self.assertEqual(c.turn_left, 0.)
                else:
                    self.assertEqual(output, first)
                    self.assertEqual(c.diagnostics['steering_reason'], 'minimum_hold')

    def test_filter_and_prediction_history_remain_available(self):
        c = self.controller()
        for _ in range(10):
            c.command(reading(20.), 1., .05)
        self.assertTrue(c.diagnostics['steering_prediction_valid'])
        self.assertGreaterEqual(c._samples[-1][0]-c._samples[0][0], .3)
        # A single opposite observation is still rejected by the existing median.
        self.assertGreater(c.command(reading(-40.), 1., .05)[1], 0.)

    def test_model_default_releases_only_yaw_with_linear_speed_unchanged(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                hold = WalkingCommandHold()
                hold.apply([.2, 0., direction*.5], 10.)
                np.testing.assert_allclose(hold.apply([.2, 0., direction*.37], 10.1),
                                           [.2, 0., direction*.37])
                np.testing.assert_allclose(hold.apply([.2, 0., 0.], 10.2), [.2, 0., 0.])

    def test_model_increase_reversal_and_velocity_change_policy(self):
        for first, request in (([.2, 0., .3], [.2, 0., .5]),
                               ([.2, 0., .5], [.2, 0., -.3]),
                               ([.2, 0., .5], [.3, 0., .3]),
                               ([.2, 0., .5], [.2, .1, 0.])):
            with self.subTest(first=first, request=request):
                hold = WalkingCommandHold()
                hold.apply(first, 10.)
                expected = request if EXPERIMENT == 'no_hold' else first
                np.testing.assert_allclose(hold.apply(request, 10.1), expected)
                # Repeated requests do not indefinitely renew the old block.
                hold.apply(request, 10.49)
                np.testing.assert_allclose(hold.apply(request, 10.5), request)

    def test_real_main_observation_uses_experimental_default(self):
        requests = [[.2, 0., .5], [.2, 0., .37], [.2, 0., 0.],
                    [.2, 0., -.5], [.2, 0., -.5], [.2, 0., -.5]]
        harness = model_tests.WalkingCommandHoldTests()
        seen, rows = harness.run_main([0., .1, .2, .3, .69, .700001], requests,
                                     command_hold=WalkingCommandHold())
        expected = requests if EXPERIMENT == 'no_hold' else requests[:3] + [requests[2]]*2 + [requests[-1]]
        for (_, observation), request, row in zip(seen, expected, rows):
            np.testing.assert_allclose(observation[9:11], [request[0], request[2]])
            self.assertAlmostEqual(float(row['cmd_wz']), request[2], places=6)

    def test_stop_invalid_clock_and_loss_still_preempt(self):
        hold = WalkingCommandHold()
        hold.apply([.2, 0., .5], 10.)
        np.testing.assert_array_equal(hold.apply([0., 0., 0.], 10.01), [0., 0., 0.])
        self.assertIsNone(hold.applied)
        c = self.controller()
        c.command(reading(40.), 1., .05)
        self.assertEqual(c.command(reading(40.), 1., float('nan')), (0., 0.))
        c.command(reading(40.), 1., .05)
        lost = reading(40.)
        lost.update(measurement_stale=True, measurement_valid=False, lost_frames=1)
        self.assertEqual(c.command(lost, 0., .05), (.2, 0.))
        c.drop_held_command()
        self.assertEqual(c.command(lost, 0., .05), (0., 0.))

    def test_actual_udp_connector_and_model_observation_release_segment_turn(self):
        # Camera geometry is known; both UDP boundaries and the command handling
        # through the real observation builder are actual production objects.
        source = UdpCommandSource(port=0)
        c = SegmentSteeringController(SteeringController())
        hold = WalkingCommandHold()
        smoother = CommandSmoother(max_vx_accel=1., max_wz_accel=2.)
        actor = HumanoidPolicy.__new__(HumanoidPolicy)
        actor.last_action = np.zeros(12, np.float32)
        actor.step_distance_m = .05
        tick = 0
        samples = []
        with (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver,
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as forwarding,
        ):
            receiver.bind(('127.0.0.1', 0))
            receiver.settimeout(1.)
            client = ConnectorClient(port=receiver.getsockname()[1])

            def apply(debug):
                nonlocal tick
                tick += 1
                command = c.command(debug, 1., .05)
                client.publish(*command, command_mode='held')
                packet = process_vision_output(json.loads(receiver.recv(4096)))
                forwarded = smoother.update(packet, .02)
                self.assertEqual((forwarded['vx'], forwarded['wz']), command)
                received_before = source.last_update
                # Windows monotonic() may give two receives the same tick.
                # Send on a fresh host tick before using time as the receive ack.
                while time.monotonic() <= received_before:
                    time.sleep(.001)
                forwarding.sendto(json.dumps(forwarded).encode(),
                                  ('127.0.0.1', source.sock.getsockname()[1]))
                deadline = time.monotonic()+1.
                while source.last_update == received_before and time.monotonic() < deadline:
                    time.sleep(.001)
                self.assertNotEqual(source.last_update, received_before)
                applied = hold.apply(source.get(), 10.+tick*.05)
                obs = actor.build_observation(np.array([0., 0., 9.81]), np.zeros(3),
                    np.array([0., 0., -1.]), applied, config.Q_DEFAULT, np.zeros(12))
                np.testing.assert_allclose(obs[9:11], applied[[0, 2]])
                samples.append((tick*.05, command[1], float(obs[10])))
                return applied

            try:
                for _ in range(20):
                    if apply(segment_reading(bend))[2] > 0:
                        break
                else:
                    self.fail('measured bend never reached model input')
                turn_started = samples[-1][0]
                for _ in range(3):
                    result = apply(segment_reading())
                self.assertEqual(float(result[2]), 0.)
                self.assertLess(samples[-1][0]-turn_started, .5)
                self.assertTrue(any(sample[2] > 0 for sample in samples))
            finally:
                client.close()
                source.close()


if __name__ == '__main__':
    unittest.main()
