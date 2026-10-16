"""Startup frames must reach the recorder without running lane detection."""
import ast
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
from command_video import CommandVideo


def submit_from_main(recorder, frame, debug, source):
    """Execute the real submission block without camera, serial, or motor setup."""
    tree = ast.parse((ROOT / 'new_vision/jetson/run_policy_vision.py').read_text(encoding='utf-8'))
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and isinstance(node.test, (ast.Compare, ast.BoolOp))
                  and ast.unparse(node.test).startswith('command_video is not None')
                  and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                          and n.func.attr == 'submit' for n in ast.walk(node)))
    env = dict(command_video=recorder, startup_card_only=debug.get('startup_card_only', False),
               frame=frame, debug=debug, args=SimpleNamespace(video_source=source),
               attitude=None, frames=1, command_host_time_ns=1, vx=0., wz=0.,
               confidence=0., time=time)
    import command_video
    env.update(vars(command_video))
    exec(compile(ast.Module(body=[branch], type_ignores=[]), '<main video submission>', 'exec'), env)


class StartupVideoTests(unittest.TestCase):
    def test_startup_main_submits_camera_without_lane_geometry(self):
        for source in ('bird_pair', 'binary', 'camera'):
            with self.subTest(source=source):
                recorder = Mock(error=None)
                image = np.full((180, 320, 3), 90, np.uint8)
                submit_from_main(recorder, image, dict(startup_card_only=True,
                    tracking_mode='STARTUP_CARD_ONLY', measurement_valid=False), source)
                recorder.submit.assert_called_once()
                self.assertIs(recorder.submit.call_args.args[0], image)
                self.assertTrue(recorder.submit.call_args.kwargs['camera_fallback'])
                self.assertIsNone(recorder.error)

    def test_startup_then_lane_same_avi_decodes_with_fixed_dimensions(self):
        for source in ('bird_pair', 'binary', 'camera'):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                size = (640, 400) if source == 'bird_pair' else (320, 400)
                camera = np.full((180, 320, 3), (25, 100, 220), np.uint8)
                original = camera.copy()
                colour = np.full((400, 320, 3), (180, 70, 20), np.uint8)
                mask = np.zeros((400, 320), np.uint8)
                mask[:, 100:120] = 255
                normal = ((colour, mask) if source == 'bird_pair' else
                          mask if source == 'binary' else camera)
                recorder = CommandVideo(tmp, frame_source=source, image_size=size, queue_size=8)
                try:
                    recorder.submit(camera, frame_id=1, host_time_ns=1, monotonic_s=1.,
                        vx=0., wz=0., lost=True, camera_fallback=True,
                        lane_debug=dict(startup_card_only=True))
                    recorder.submit(camera, frame_id=2, host_time_ns=2, monotonic_s=1.2,
                        vx=0., wz=0., lost=True, camera_fallback=True,
                        lane_debug=dict(startup_card_only=True))
                    recorder.submit(normal, frame_id=3, host_time_ns=3, monotonic_s=1.4,
                        vx=.2, wz=.1, lost=False,
                        lane_debug=dict(measurement_valid=True, heading_control_valid=True,
                                        heading_control_deg=12.))
                finally:
                    recorder.close()
                self.assertIsNone(recorder.error)
                rows = [json.loads(l) for l in (Path(tmp) / 'frames.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 5)
                self.assertEqual([r['line_state'] for r in rows], ['STARTUP'] * 4 + ['DUAL'])
                self.assertTrue(all(r['heading_deg'] is None and not r['line_lost'] for r in rows[:4]))
                self.assertTrue(all(r['frame_source'] == 'startup_camera' for r in rows[:4]))
                self.assertEqual(rows[-1]['frame_source'], source)
                cap = cv2.VideoCapture(str(Path(tmp) / 'camera_commands.avi'))
                decoded = []
                try:
                    while True:
                        ok, image = cap.read()
                        if not ok:
                            break
                        decoded.append(image)
                finally:
                    cap.release()
                self.assertEqual(len(decoded), len(rows))
                self.assertEqual(len({im.shape for im in decoded}), 1)
                expected_h = 400 + (110 if source != 'camera' else 0)
                self.assertEqual(decoded[0].shape, (expected_h, size[0], 3))
                # The startup view is the actual camera, with no fake binary lane.
                np.testing.assert_allclose(decoded[0][200, size[0] // 2], camera[90, 160], atol=5)
                np.testing.assert_array_equal(camera, original)
                if source == 'bird_pair':
                    restored = cv2.cvtColor(decoded[-1][:400, 320:], cv2.COLOR_BGR2GRAY) >= 128
                    self.assertGreater(np.mean(restored == (mask > 0)), .999)

    def test_startup_hud_is_explicit_and_has_no_lane_heading(self):
        from command_video import draw_command
        with patch('command_video.cv2.putText', wraps=cv2.putText) as text:
            draw_command(np.zeros((510, 640, 3), np.uint8), 0., 0., False, .5, 0.,
                         line_state='STARTUP')
        self.assertTrue(any('STARTUP CAMERA' in c.args[1] for c in text.call_args_list))


if __name__ == '__main__':
    unittest.main()
