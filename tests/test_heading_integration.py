"""Real B entrypoint and wire contract; camera/detector only are synthetic."""
import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
import run_policy_vision
from policy_bridge import ConnectorClient


class HeadingIntegrationTests(unittest.TestCase):
    def test_heading_rejects_amplitude_that_connector_would_clip(self):
        with patch("sys.argv", ["run_policy_vision.py", "--max-wz", "1", "--wz-step", ".6"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_policy_vision.parse_args()

    def test_heading_requires_usable_corridor_and_recovery_caps(self):
        for extra in (("--heading-corridor-cm", "0"), ("--heading-corridor-cm", "nan"),
                      ("--max-wz-right", ".2"), ("--wz-step", ".3")):
            with self.subTest(extra=extra), patch("sys.argv", ["run_policy_vision.py", *extra]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_policy_vision.parse_args()

    def test_client_optional_mode_real_udp(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(1)
            client = ConnectorClient(port=receiver.getsockname()[1])
            try:
                client.publish(.2, .17, command_mode="held")
                packet = json.loads(receiver.recv(4096))
                self.assertEqual(packet["command_mode"], "held")
                self.assertEqual(packet["wz"], .17)
                client.publish(.2, .17)
                self.assertNotIn("command_mode", json.loads(receiver.recv(4096)))
                for bad in (True, 1, [], "bad"):
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        client.publish(.2, .1, command_mode=bad)
            finally:
                client.close()

    def test_default_entrypoint_held_levels_and_auditable_log(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        clock, count, sent = [0.0], [0], []

        def read():
            count[0] += 1
            clock[0] += .05
            if count[0] > 42:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame

        camera.read.side_effect = read
        detector = Mock()
        def process(_frame, **_kw):
            angle = (20., 40., -20., 0.)[min(3, (count[0]-1)//11)]
            return 0, 0, .9, None, dict(
                fused_err_cm=0., base_err_cm=0., near_error_cm=0., near_z_cm=0.,
                angle_err_deg=0., lost_frames=0, measurement_valid=True,
                heading_control_valid=True, heading_control_deg=angle)
        detector.process.side_effect = process
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--no-shape-detect",
                               "--attitude-port", "0", "--line-log-dir", tmp]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            client.return_value.publish.side_effect = lambda vx,wz,*a,**kw: sent.append((clock[0],vx,wz,kw))
            self.assertEqual(run_policy_vision.main(), 0)
            rows = [json.loads(line) for p in Path(tmp).rglob("line_frames.jsonl")
                    for line in p.read_text().splitlines()]
        self.assertEqual(len(rows), 42)
        self.assertTrue(all(r["mode"] == "heading" for r in rows))
        self.assertTrue(all("steering_heading_deg" in r["measurement"] for r in rows))
        walking = [s for s in sent if s[1] != 0]
        self.assertTrue(all(s[3].get("command_mode") == "held" for s in walking))
        levels = {round(s[2],6) for s in walking}
        self.assertGreaterEqual(len(levels), 3, levels)
        self.assertTrue(levels <= {0., .37, .43, .5, -.3, -.5}, levels)
        self.assertTrue(any(wz < 0 for wz in levels), levels)
        start, previous = walking[0][0], walking[0][1:3]
        for now,vx,wz,_ in walking[1:]:
            if (vx,wz) != previous:
                # No minimum delay; sensor filtering and levels still apply.
                self.assertGreater(now-start, 0.)
                start,previous = now,(vx,wz)
        self.assertEqual(sent[-1][1:3], (0.,0.))


if __name__ == "__main__":
    unittest.main()
