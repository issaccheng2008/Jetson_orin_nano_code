"""Managed startup tests use fake child processes; no serial port or model runs."""
import http.client
import json
import socket
import subprocess
from pathlib import Path
import sys
import tempfile
import time
import unittest

from remote_control.policy_process import PolicyProcess
from remote_control.server import RemoteService


class Child:
    def __init__(self):
        self.returncode = None
        self.signals = []
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.model = Path(self.folder.name) / "fake.onnx"
        self.model.touch()
        self.now = 10.
        self.children = []
        self.commands = []
        def spawn(command, **kwargs):
            child = Child()
            self.children.append(child)
            self.commands.append(command)
            return child
        self.manager = PolicyProcess(sys.executable, self.model, "fake-serial", 5507,
                                     startup_timeout=1., shutdown_timeout=.5,
                                     popen=spawn, clock=lambda: self.now)

    def tearDown(self):
        self.manager.close()
        self.folder.cleanup()

    def test_boot_never_starts_a_policy_and_start_waits_for_ready_marker(self):
        self.assertEqual(self.manager.status()["policy_state"], "stopped")
        self.assertEqual(self.children, [])
        self.manager.start()
        self.assertFalse(self.manager.status()["policy_ready"])
        command = self.commands[0]
        self.assertIn("--enable-motors", command)
        self.assertEqual(command[command.index("--command-source") + 1], "remote")
        self.assertEqual(command[command.index("--udp-command-port") + 1], "5507")
        Path(command[command.index("--startup-ready-file") + 1]).write_text("ready\n")
        self.manager.tick()
        self.assertTrue(self.manager.status()["policy_ready"])

    def test_timeout_interrupts_then_kills_unresponsive_child(self):
        self.manager.start()
        self.now += 1.1
        self.manager.tick()
        self.assertFalse(self.manager.status()["policy_ready"])
        self.assertTrue(self.children[0].signals)
        self.now += .6
        self.manager.tick()
        self.assertTrue(self.children[0].killed)
        self.manager.tick()
        self.assertEqual(len(self.children), 1)  # A failure never restarts itself.

    def test_late_marker_after_cancel_does_not_enable_or_restart(self):
        self.manager.start()
        old_marker = Path(self.commands[0][self.commands[0].index("--startup-ready-file") + 1])
        self.manager.cancel()
        old_marker.write_text("ready\n")
        self.manager.tick()
        self.assertFalse(self.manager.status()["policy_ready"])
        self.children[0].returncode = 0
        self.manager.tick()
        self.manager.start()
        self.assertFalse(self.manager.status()["policy_ready"])
        new_marker = self.commands[1][self.commands[1].index("--startup-ready-file") + 1]
        self.assertNotEqual(str(old_marker), new_marker)

    def test_process_failure_requires_manual_start_and_closed_manager_cannot_restart(self):
        self.manager.start()
        self.children[0].returncode = 1
        self.manager.tick()
        self.assertEqual(self.manager.status()["policy_state"], "failed")
        self.assertEqual(len(self.children), 1)
        self.manager.start()
        self.assertEqual(len(self.children), 2)
        self.manager.close()
        with self.assertRaises(ValueError):
            self.manager.start()

    def test_http_lease_suppresses_motion_until_ready_and_stops_child(self):
        service = RemoteService("127.0.0.1", 0, "policy-test", 15507,
                                policy_process=self.manager)
        service.start()
        def request(path, body):
            conn = http.client.HTTPConnection("127.0.0.1", service.address[1], timeout=2)
            conn.request("POST", path, json.dumps(body),
                         {"Content-Type": "application/json", "Authorization": "Bearer policy-test"})
            response = conn.getresponse()
            data = json.loads(response.read())
            conn.close()
            self.assertEqual(response.status, 200, data)
            return data
        try:
            self.assertEqual(self.children, [])
            result = request("/api/arm", {})
            self.assertFalse(result["policy_ready"])
            session = result["session"]
            result = request("/api/command", {"session": session, "seq": 1, "vx": .4, "wz": .5})
            self.assertEqual((result["target_vx"], result["target_wz"]), (0, 0))
            Path(self.commands[0][self.commands[0].index("--startup-ready-file") + 1]).write_text("ready\n")
            self.manager.tick()
            result = request("/api/command", {"session": session, "seq": 2, "vx": .2, "wz": .3})
            self.assertEqual(result["target_vx"], .2)
            request("/api/stop", {"session": session})
            deadline = time.monotonic() + .5
            while not self.children[0].signals and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(self.children[0].signals)
            self.assertEqual(len(self.children), 1)
        finally:
            service.close()

    def test_real_stub_child_handshake_udp_and_shutdown_without_hardware(self):
        stub = Path(self.folder.name) / "stub_policy.py"
        command_log = self.model.with_suffix(".commands")
        stub.write_text('''import json, socket, sys, time
from pathlib import Path
args = sys.argv
port = int(args[args.index('--udp-command-port') + 1])
marker = Path(args[args.index('--startup-ready-file') + 1])
log = Path(args[args.index('--model') + 1]).with_suffix('.commands')
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1', port))
        time.sleep(.15)
        marker.write_text('ready\\n')
        while True:
            message = json.loads(receiver.recv(2048))
            with log.open('a') as output:
                output.write(json.dumps(message) + '\\n')
except KeyboardInterrupt:
    pass
''', encoding="utf-8")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as finder:
            finder.bind(("127.0.0.1", 0))
            port = finder.getsockname()[1]
        children = []
        def stub_launcher(command, **kwargs):
            command = list(command)
            command[2] = str(stub)  # Never execute production main.py or a real model.
            child = subprocess.Popen(command, **kwargs)
            children.append(child)
            return child
        manager = PolicyProcess(sys.executable, self.model, "unused-test-serial", port,
                                startup_timeout=3., shutdown_timeout=1., popen=stub_launcher)
        service = RemoteService("127.0.0.1", 0, "stub-only", port, policy_process=manager)
        service.start()
        try:
            self.assertEqual(children, [])
            session = service.arm()["session"]
            seq = 0
            deadline = time.monotonic() + 3
            while not service.status()["policy_ready"] and time.monotonic() < deadline:
                seq += 1
                service.command(session, seq, 0., 0.)
                self.assertEqual(service.state.snapshot()["target_vx"], 0)
                time.sleep(.02)
            self.assertTrue(service.status()["policy_ready"])
            seq += 1
            service.command(session, seq, .2, -.3)
            deadline = time.monotonic() + .25
            messages = []
            while time.monotonic() < deadline:
                if command_log.exists():
                    messages = [json.loads(line) for line in command_log.read_text().splitlines()]
                    if any(message["vx"] > 0 and message["wz"] < 0 for message in messages):
                        break
                time.sleep(.01)
            self.assertTrue(any(message["vx"] > 0 and message["wz"] < 0 for message in messages))
            service.stop(session)
            children[0].wait(timeout=2)
            manager.tick()
            self.assertEqual(manager.status()["policy_state"], "stopped")
            self.assertFalse(service.status()["active"])
            self.assertEqual(len(children), 1)
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
