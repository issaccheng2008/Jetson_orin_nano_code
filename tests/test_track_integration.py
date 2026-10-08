"""Entry point selects the opt-in observer and retains the UDP hold contract."""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
import run_policy_vision
from tests.test_track_steering import observed


class TrackIntegrationTests(unittest.TestCase):
    def test_track_mode_is_explicit_and_publishes_held_discrete_levels(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        frame = np.zeros((720, 1280, 3), np.uint8)
        clock, reads = [0.], [0]

        def read():
            clock[0] += .1
            reads[0] += 1
            if reads[0] > 16:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame

        camera.read.side_effect = read
        detector = Mock()
        detector.process.side_effect = lambda *a, **k: (
            0, 0, .9, None, deepcopy(observed()))
        with (
            patch("sys.argv", ["run_policy_vision.py", "--wz-mode", "track",
                               "--headless", "--no-shape-detect", "--attitude-port", "0"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        self.assertTrue(detector.track_geometry_enable)
        published = client.return_value.publish.call_args_list
        walking = [call for call in published if call.args[0] > 0]
        self.assertTrue(walking)
        self.assertTrue(all(call.kwargs.get("command_mode") == "held" for call in walking))
        self.assertTrue(all(call.args[1] in (0., .37, .43, .5, -.3, -.5)
                            for call in walking))
        self.assertTrue(any(call.args[1] > 0 for call in walking))


if __name__ == "__main__":
    unittest.main()
