"""Startup commands bypass collection, but tolerate normal slow camera frames."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
from connector import CommandWindowScheduler, process_vision_output
from policy_bridge import ConnectorClient


class StartupCommandWatchdogTests(unittest.TestCase):
    def startup(self, vx=.2, wz=.1):
        return process_vision_output(dict(vx=vx, wz=wz, command_mode='held',
                                          startup_sequence_active=True))

    def test_recorded_slow_frame_gaps_do_not_insert_zero_commands(self):
        scheduler = CommandWindowScheduler()
        # Publication times: frame timestamp + read_to_publish_ms, run 143339.
        publications = [505.276, 505.536, 505.815, 506.007, 506.215,
                        506.454, 506.680, 506.899, 507.100]
        frame = 0
        for tick in range(113):
            now = publications[0] + tick * .02
            while frame < len(publications) and publications[frame] <= now:
                scheduler.receive(self.startup(), publications[frame])
                frame += 1
            target, fresh, _ = scheduler.output(now)
            self.assertTrue(fresh, msg=f'unexpected watchdog stop at {now}')
            self.assertEqual((target['vx'], target['wz']), (.2, .1))
            self.assertNotIn('startup_sequence_active', target)

    def test_startup_watchdog_is_bounded_and_ordinary_timeout_is_unchanged(self):
        scheduler = CommandWindowScheduler(window_vision_timeout_s=.8)
        scheduler.receive(self.startup(), 10.)
        self.assertEqual(scheduler.output(10.)[0]['vx'], .2)
        self.assertTrue(scheduler.output(10.79)[1])
        target, fresh, _ = scheduler.output(10.81)
        self.assertFalse(fresh)
        self.assertEqual((target['vx'], target['wz']), (0., 0.))
        scheduler.receive(process_vision_output(dict(vx=.2, wz=.1)), 11.)
        self.assertFalse(scheduler.output(11.251)[1])

    def test_next_step_stop_posture_and_card_action_apply_immediately(self):
        for packet in [dict(vx=.2, wz=-.2, startup_sequence_active=True),
                       dict(vx=0., wz=0.),
                       dict(vx=0., wz=0., card_tilt=True),
                       dict(vx=0., wz=0., event_id=77, event_action=1)]:
            with self.subTest(packet=packet):
                scheduler = CommandWindowScheduler()
                scheduler.receive(self.startup(), 10.)
                scheduler.output(10.)
                scheduler.receive(process_vision_output(packet), 10.1)
                target, fresh, _ = scheduler.output(10.1)
                self.assertTrue(fresh)
                self.assertEqual((target['vx'], target['wz']), (packet['vx'], packet['wz']))
                if 'event_id' in packet:
                    self.assertEqual(target['event_id'], 77)

    def test_startup_tag_requires_a_boolean(self):
        for invalid in [None, 1, 'true', [], {}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                process_vision_output(dict(vx=.2, wz=.1, startup_sequence_active=invalid))

    def test_client_serializes_startup_tag_only_when_active(self):
        client = ConnectorClient.__new__(ConnectorClient)
        client.socket = Mock()
        client.address = ('127.0.0.1', 5006)
        for active in [True, False]:
            client.publish(.2, .1, startup_sequence_active=active)
            packet = json.loads(client.socket.sendto.call_args.args[0])
            self.assertEqual(packet.get('startup_sequence_active', False), active)
            normalized = process_vision_output(packet)
            self.assertEqual(normalized.get('startup_sequence_active', False), active)
