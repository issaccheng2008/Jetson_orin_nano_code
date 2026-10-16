"""Real vision entrypoint publication; substitute only camera and mapped frame output."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
import run_policy_vision
from connector import CommandWindowScheduler, process_vision_output


class WindowPublicationTests(unittest.TestCase):
    def run_frames(self, side, stop_frame=None, recorder=None):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        clock, count, sent = [0.], [0], []
        scheduler = CommandWindowScheduler()
        next_tick = [.02]

        def publish(vx, wz, qr=-1, **kw):
            now = clock[0]+10.
            while next_tick[0]+10. < now-1e-8:
                scheduler.output(next_tick[0]+10.)
                next_tick[0] += .02
            packet = dict(vx=vx, wz=wz, qr=qr, **kw)
            if 'steering_window' in packet:
                packet['steering_window'] = dict(packet['steering_window'])
                packet['steering_window']['origin_s'] += 10.
            scheduler.receive(process_vision_output(packet), now)
            applied = scheduler.output(now)[0]
            sent.append((applied['vx'], applied['wz']))
        def read():
            count[0] += 1
            clock[0] = count[0]*.05
            if count[0] > 31:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame
        camera.read.side_effect = read
        detector = Mock()
        def detect(image, **kw):
            valid = count[0] <= 7 or count[0] >= 22
            debug = dict(fused_err_cm=0., base_err_cm=0., near_error_cm=0.,
                         near_z_cm=0., angle_err_deg=0., measurement_valid=valid,
                         lost_frames=0 if valid else 1,
                         heading_control_valid=valid, heading_control_deg=0.)
            return 0., 0., 1. if valid else 0., image, debug
        detector.process.side_effect = detect
        def mapped(controller, debug, confidence, dt):
            i = count[0]
            wz = .1 if i == 1 or i >= 22 else .25 if i <= 6 else -.2 if i == 7 or i <= 20 else .3
            controller.diagnostics = dict(steering_reason='new_block' if confidence else 'loss_history_turn',
                                          steering_applied_wz=wz)
            return (0., 0.) if i == stop_frame else (.2, wz)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch('sys.argv', ['run_policy_vision.py', '--no-camera-async', '--headless', '--no-shape-detect',
                              '--record-video' if recorder is not None else '--no-record-video',
                              '--video-source', 'camera', '--attitude-port', '0', '--recording-root', tmp,
                              '--steering-command-median', side]),
            patch.object(run_policy_vision.signal, 'signal'),
            patch('utils.open_camera', return_value=camera),
            patch('line_detector_v1_warp.LineDetector', return_value=detector),
            patch('camera_controls.apply_camera_controls', return_value=dict(status='unchanged', settings=[])),
            patch('heading_steering.HeadingSteeringController.command', mapped),
            patch.object(run_policy_vision, 'ConnectorClient') as client,
            patch('command_video.CommandVideo', return_value=recorder),
            patch.object(run_policy_vision.time, 'monotonic', lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            client.return_value.publish.side_effect = publish
            self.assertEqual(run_policy_vision.main(), 0)
            rows = [json.loads(line) for path in Path(tmp).rglob('line_frames.jsonl')
                    for line in path.read_text().splitlines()]
        return sent, rows

    def test_median_of_mapped_frames_excludes_lost_tail_and_all_lost_uses_recovery(self):
        sent, rows = self.run_frames('lower')
        self.assertEqual(sent[:10], [(0., 0.)]*10)
        self.assertEqual(sent[10:20], [(.2, .25)]*10)
        self.assertEqual(sent[20:30], [(.2, -.2)]*10)
        self.assertEqual(sent[30], (.2, .1))
        self.assertEqual(sent[-1], (0., 0.))
        completed = rows[10]['measurement']
        self.assertEqual(completed['steering_window_reason'], 'connector_collection')
        self.assertEqual(completed['steering_frame_wz'], -.2)
        self.assertEqual(completed['steering_applied_wz'], -.2)  # Raw candidate in vision telemetry.

    def test_zero_command_bypasses_window_and_discards_pre_stop_samples(self):
        sent, rows = self.run_frames('upper', stop_frame=3)
        self.assertEqual(sent[2], (0., 0.))
        self.assertEqual(sent[3:13], [(0., 0.)]*10)
        self.assertEqual(sent[13], (.2, .25))
        self.assertEqual(rows[2]['vx'], 0.)

    def test_video_receives_current_geometry_without_changing_published_commands(self):
        without_video, _ = self.run_frames('lower')
        recorder = Mock(error=None)
        with_video, _ = self.run_frames('lower', recorder=recorder)
        self.assertEqual(with_video, without_video)
        self.assertEqual(recorder.submit.call_count, 31)
        samples = [call.kwargs for call in recorder.submit.call_args_list]
        for i, sample in enumerate(samples, 1):
            valid = i <= 7 or i >= 22
            self.assertEqual(sample['lane_debug']['heading_control_valid'], valid)
            self.assertEqual(sample['lane_debug']['heading_control_deg'], 0.)
            self.assertEqual(sample['lost'], not valid)
        recorder.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
