"""One startup indication using the existing UDP socket and serial packet."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
sys.path.insert(0, str(ROOT / 'humanoid_jetson_deploy'))

import protocol
import startup_button_link
import main as policy_main
from policy_bridge import ConnectorClient
from serial_link import SerialLink


class StartupReadyTests(unittest.TestCase):
    def test_policy_handoff_can_report_ready_after_first_connector_packet(self):
        notify = getattr(policy_main, 'send_host_ready_if_connected', None)
        self.assertTrue(callable(notify))
        link = Mock(spec=SerialLink)
        source = SimpleNamespace(last_update=0.)
        self.assertFalse(notify(link, source))
        link.send_startup_control.assert_not_called()
        source.last_update = 1.
        link.send_startup_control.side_effect = OSError('USB gone')
        self.assertFalse(notify(link, source))  # LED failure cannot stop the policy
        link.send_startup_control.side_effect = None
        self.assertTrue(notify(link, source))
        link.send_startup_control.assert_called_with(4)
        link.send_command.assert_not_called()
        link.send_action.assert_not_called()

    def test_ready_bit_preserves_frame_size_and_other_flags(self):
        self.assertEqual(getattr(protocol, 'STARTUP_HOST_READY', None), 4)
        for flags in range(8):
            with self.subTest(flags=flags):
                packet = protocol.StartupControlPacket(19, flags)
                frame = protocol.pack_startup_control(packet)
                self.assertEqual(len(frame), 11)
                self.assertEqual(list(protocol.FrameDecoder().feed(frame)), [packet])
        with self.assertRaises(ValueError):
            protocol.pack_startup_control(protocol.StartupControlPacket(0, 8))

    def test_ready_only_adds_a_bit_to_existing_button_heartbeat(self):
        link = Mock(spec=SerialLink)
        link.reader_alive.return_value = True
        link.get_latest_state.return_value = SimpleNamespace(status_flags=0)
        with patch.object(startup_button_link, 'SerialLink', return_value=link):
            button = startup_button_link.StartupButtonLink('/dev/ttyACM0')
            button.host_ready = True
            button.poll(False)
            button.poll(True)
            self.assertEqual(link.send_startup_control.call_args.args, (7,))
            link.send_command.assert_not_called()
            link.send_action.assert_not_called()
            link.reader_alive.return_value = False
            button.close()
            link.send_startup_control.assert_called_with(4)  # disarm without overwriting READY

    def test_missing_connector_does_not_claim_ready_or_block(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.bind(('127.0.0.1', 0))
            server.settimeout(.5)
            client = ConnectorClient(*server.getsockname())
            try:
                self.assertTrue(callable(getattr(client, 'poll_startup_ready', None)))
                started = time.monotonic()
                self.assertFalse(client.poll_startup_ready())
                self.assertLess(time.monotonic() - started, .2)
                data, address = server.recvfrom(512)
                self.assertEqual(json.loads(data), {'startup_probe': True})
                # A normal command or unrelated reply cannot claim readiness.
                server.sendto(b'{"vx":0,"wz":0}', address)
                self.assertFalse(client.poll_startup_ready())
                server.settimeout(.05)
                with self.assertRaises(socket.timeout):
                    server.recvfrom(512)  # probe retries are rate limited
            finally:
                client.close()

    def test_running_connector_acknowledges_probe_without_motion(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as policy, \
                socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reserved:
            policy.bind(('127.0.0.1', 0))
            policy.settimeout(5.)  # Windows process startup may take over half a second.
            reserved.bind(('127.0.0.1', 0))
            port = reserved.getsockname()[1]
            reserved.close()
            process = subprocess.Popen([
                sys.executable, '-B', str(ROOT / 'connector.py'),
                '--vision-port', str(port), '--policy-port', str(policy.getsockname()[1]),
                '--log-every', '100000',
            ], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            client = ConnectorClient('127.0.0.1', port)
            try:
                self.assertTrue(callable(getattr(client, 'poll_startup_ready', None)))
                # The normal policy stream confirms the receiver is listening.
                message = json.loads(policy.recvfrom(4096)[0])
                self.assertEqual((message['vx'], message['wz']), (0., 0.))
                deadline = time.monotonic() + 2.
                while not client.poll_startup_ready() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(client.poll_startup_ready())
                for _ in range(3):
                    message = json.loads(policy.recvfrom(4096)[0])
                    self.assertEqual((message['vx'], message['wz']), (0., 0.))
                    self.assertNotIn('startup_probe', message)
                    self.assertNotIn('connector_ready', message)
                # Confirmation is a one-time latch, not a new monitoring loop.
                with patch.object(client, 'socket', Mock()) as sock:
                    self.assertTrue(client.poll_startup_ready())
                    sock.sendto.assert_not_called()
            finally:
                client.close()
                process.terminate()
                process.communicate(timeout=3.)


if __name__ == '__main__':
    unittest.main()
