import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from command_video import CommandVideo, draw_command, line_lost


class CommandVideoTests(unittest.TestCase):
    def test_bird_pair_preserves_colour_mask_and_existing_footer(self):
        colour = np.full((400,320,3), (30,100,220), np.uint8)
        mask = np.zeros((400,320), np.uint8)
        mask[:,100:115] = 255
        source_colour, source_mask = colour.copy(), mask.copy()
        writer = Mock()
        writer.isOpened.return_value = True
        feedback = dict(velocity=[.2,0.,-.5], enabled=True, send_result='written',
                        policy_mode='walking49', monotonic_s=1., step=1)
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer) as factory:
            recorder = CommandVideo(Path(tmp), frame_source='bird_pair')
            recorder.submit((colour,mask), frame_id=1, host_time_ns=1, monotonic_s=1.,
                            vx=.2, wz=.3, lost=False, executed=feedback)
            recorder.close()
            manifest = json.loads((Path(tmp)/'manifest.json').read_text())
        self.assertIsNone(recorder.error)
        self.assertEqual(factory.call_args.args[3], (640,510))
        encoded = writer.write.call_args.args[0]
        np.testing.assert_array_equal(encoded[:400,:320], source_colour)
        for channel in range(3):
            np.testing.assert_array_equal(encoded[:400,320:,channel], source_mask)
        expected_footer = np.zeros((510,640,3),np.uint8)
        draw_command(expected_footer,.2,.3,False,.5,0.,executed=feedback)
        np.testing.assert_array_equal(encoded[400:], expected_footer[400:])
        np.testing.assert_array_equal(colour,source_colour)
        np.testing.assert_array_equal(mask,source_mask)
        self.assertEqual(manifest['frame_source'],'bird_pair')

    def test_bird_pair_resize_keeps_binary_values(self):
        colour = np.full((400,320,3), (30,100,220), np.uint8)
        mask = np.zeros((400,320),np.uint8)
        mask[:,100:115]=255
        writer = Mock()
        writer.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer):
            recorder=CommandVideo(Path(tmp),width=320,frame_source='bird_pair')
            recorder.submit((colour,mask),frame_id=1,host_time_ns=1,monotonic_s=1.,vx=.2,wz=0.,lost=False)
            recorder.close()
        self.assertIsNone(recorder.error)
        encoded=writer.write.call_args.args[0]
        self.assertEqual(encoded.shape,(310,320,3))
        self.assertEqual(set(np.unique(encoded[:200,160:])),{0,255})

    def test_real_bird_pair_decodes_left_colour_and_right_detector_mask(self):
        colour = np.full((400,320,3), (30,100,220), np.uint8)
        mask = np.zeros((400,320), np.uint8)
        mask[:,100:115]=255
        with tempfile.TemporaryDirectory() as tmp:
            recorder=CommandVideo(Path(tmp),frame_source='bird_pair')
            recorder.submit((colour,mask),frame_id=1,host_time_ns=1,monotonic_s=1.,vx=.2,wz=.3,lost=False)
            recorder.close()
            self.assertIsNone(recorder.error)
            cap=cv2.VideoCapture(str(Path(tmp)/'camera_commands.avi'))
            try:
                ok, decoded=cap.read()
                self.assertTrue(ok)
                self.assertEqual(decoded.shape,(510,640,3))
                self.assertLess(np.mean(np.abs(decoded[:400,:316].astype(float)-colour[:,:316])),3.)
                restored=(cv2.cvtColor(decoded[:400,320:],cv2.COLOR_BGR2GRAY)>=128).astype(np.uint8)*255
                self.assertGreater(np.mean(restored==mask),.999)
            finally:
                cap.release()

    def test_binary_video_keeps_entire_detector_mask_and_puts_arrows_below_it(self):
        mask = np.zeros((400, 320), np.uint8)
        mask[40:360, 96:112] = 255
        original = mask.copy()
        writer = Mock()
        writer.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer) as factory:
            recorder = CommandVideo(Path(tmp), width=960, frame_source='binary')
            recorder.submit(mask, frame_id=1, host_time_ns=1, monotonic_s=1.,
                            vx=.2, wz=.3, lost=False)
            recorder.close()
            manifest = json.loads((Path(tmp)/'manifest.json').read_text())
        self.assertIsNone(recorder.error)
        self.assertEqual(factory.call_args.args[3], (320, 510))
        encoded = writer.write.call_args.args[0]
        self.assertEqual(encoded.shape, (510, 320, 3))
        for channel in range(3):
            np.testing.assert_array_equal(encoded[:400, :, channel], original)
        np.testing.assert_array_equal(mask, original)
        self.assertEqual(manifest['frame_source'], 'binary')
        self.assertEqual(manifest['detector_debug_key'], 'binary')

    def test_binary_downsampling_keeps_white_black_candidates(self):
        mask = np.zeros((400, 320), np.uint8)
        mask[:, 96:112] = 255
        writer = Mock()
        writer.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer):
            recorder = CommandVideo(Path(tmp), width=160, frame_source='binary')
            recorder.submit(mask, frame_id=1, host_time_ns=1, monotonic_s=1.,
                            vx=.2, wz=0., lost=False)
            recorder.close()
        encoded = writer.write.call_args.args[0]
        self.assertEqual(encoded.shape, (310, 160, 3))
        self.assertEqual(set(np.unique(encoded[:200])), {0, 255})

    def test_real_binary_video_decodes_to_same_lane_mask(self):
        mask = np.zeros((400, 320), np.uint8)
        mask[40:360, 96:112] = 255
        with tempfile.TemporaryDirectory() as tmp:
            recorder = CommandVideo(Path(tmp), frame_source='binary')
            recorder.submit(mask, frame_id=1, host_time_ns=1, monotonic_s=1.,
                            vx=.2, wz=.3, lost=False)
            recorder.close()
            self.assertIsNone(recorder.error)
            cap = cv2.VideoCapture(str(Path(tmp)/'camera_commands.avi'))
            try:
                ok, decoded = cap.read()
                self.assertTrue(ok)
                self.assertEqual(decoded.shape[:2], (510, 320))
                restored = (cv2.cvtColor(decoded[:400], cv2.COLOR_BGR2GRAY) >= 128).astype(np.uint8)*255
                self.assertGreater(np.mean(restored == mask), .999)
            finally:
                cap.release()

    def test_resampled_frames_expire_actual_feedback_during_camera_gap(self):
        feedback = dict(velocity=[.2,0.,-.5], enabled=True, send_result='written',
                        policy_mode='walking49', monotonic_s=10., step=1)
        writer = Mock()
        writer.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer):
            recorder = CommandVideo(Path(tmp), width=320, queue_size=16)
            image = np.zeros((180,320,3), np.uint8)
            recorder.submit(image, frame_id=1, host_time_ns=100, monotonic_s=10.1,
                            vx=.2, wz=.5, lost=False, executed=feedback)
            recorder.submit(image, frame_id=2, host_time_ns=200, monotonic_s=11.1,
                            vx=.2, wz=.5, lost=False, executed=None)
            recorder.close()
            rows = [json.loads(x) for x in (Path(tmp)/'frames.jsonl').read_text().splitlines()]
        self.assertIsNotNone(rows[0]['executed_command'])
        for row in rows:
            if 10.1+row['video_time_s'] > 10.5+1e-7:
                self.assertIsNone(row['executed_command'], row)

    def test_frame_index_keeps_both_commands_and_owns_feedback_snapshot(self):
        feedback = dict(velocity=[.2, 0., -.5], enabled=True, send_result='written',
                        policy_mode='walking49', monotonic_s=10., step=1)
        writer = Mock()
        writer.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer):
            recorder = CommandVideo(Path(tmp), width=320)
            recorder.submit(np.zeros((180,320,3), np.uint8), frame_id=1, host_time_ns=100,
                monotonic_s=10.1, vx=.2, wz=.5, lost=False, executed=feedback)
            feedback['velocity'][2] = 9.
            recorder.close()
            row = json.loads((Path(tmp)/'frames.jsonl').read_text().splitlines()[0])
        self.assertEqual(row['wz'], .5)
        self.assertEqual(row['executed_command']['velocity'][2], -.5)

    def test_video_has_separate_vision_and_actual_held_arrows(self):
        image = np.zeros((360, 640, 3), np.uint8)
        feedback = dict(velocity=[.2, 0., -.5], enabled=True, send_result='written',
                        policy_mode='walking49', monotonic_s=10., step=1)
        with patch('command_video.cv2.arrowedLine', wraps=cv2.arrowedLine) as arrow:
            draw_command(image, .2, .5, False, .5, 0., executed=feedback)
            self.assertEqual(arrow.call_count, 2)
            first, second = arrow.call_args_list
            self.assertLess(first.args[2][0], first.args[1][0])
            self.assertGreater(second.args[2][0], second.args[1][0])
        with patch('command_video.cv2.arrowedLine') as arrow:
            draw_command(image, .2, .5, True, .5, 0., executed=feedback)
            self.assertTrue(all(call.args[3] == (0, 0, 255) for call in arrow.call_args_list))

    def test_missing_actual_feedback_is_unknown_and_never_copies_vision(self):
        image = np.zeros((360, 640, 3), np.uint8)
        with patch('command_video.cv2.arrowedLine') as arrow, patch('command_video.cv2.putText') as text:
            draw_command(image, .2, .5, False, .5, 0., executed=None)
        self.assertEqual(arrow.call_count, 1)
        self.assertTrue(any('UNKNOWN' in call.args[1] for call in text.call_args_list))

    def test_copy_failure_cannot_escape_into_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = CommandVideo(Path(tmp))
            image = Mock()
            image.copy.side_effect = MemoryError('copy unavailable')
            try:
                self.assertFalse(recorder.submit(image, frame_id=1, host_time_ns=1,
                    monotonic_s=1., vx=.2, wz=0., lost=False))
                self.assertIn('copy unavailable', recorder.error)
            finally:
                recorder.close()

    def test_shutdown_truncates_backlog_but_releases_and_saves_manifest(self):
        writer = Mock()
        writer.isOpened.return_value = True
        writer.write.side_effect = lambda _frame: time.sleep(.01)
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer):
            recorder = CommandVideo(Path(tmp), queue_size=16)
            frame = np.zeros((180, 320, 3), np.uint8)
            for i, timestamp in enumerate((0., 60.)):
                recorder.submit(frame, frame_id=i, host_time_ns=i, monotonic_s=timestamp,
                    vx=.2, wz=.5, lost=False)
            recorder.close(drain_timeout_s=.03)
            self.assertFalse(recorder._thread.is_alive())
            writer.release.assert_called_once()
            manifest = json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertTrue(manifest['truncated'])
            self.assertLess(manifest['written_frames'], 601)
            rows = (Path(tmp)/'frames.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), manifest['written_frames'])

    def test_arrow_matches_published_sign_and_loss_colour_without_changing_source(self):
        source = np.zeros((180, 320, 3), dtype=np.uint8)
        with patch('command_video.cv2.arrowedLine', wraps=cv2.arrowedLine) as arrow:
            draw_command(source.copy(), .2, .5, False, .5, 0.)
            _, start, end, colour = arrow.call_args.args[:4]
            self.assertLess(end[0], start[0])
            self.assertEqual(colour, (255, 0, 0))
            draw_command(source.copy(), .2, -.5, True, .5, 0.)
            _, start, end, colour = arrow.call_args.args[:4]
            self.assertGreater(end[0], start[0])
            self.assertEqual(colour, (0, 0, 255))
            draw_command(source.copy(), .2, 0., False, .5, 0.)
            self.assertEqual(arrow.call_args.args[1][0], arrow.call_args.args[2][0])
        self.assertFalse(source.any())

    def test_stopped_frame_has_no_forward_arrow(self):
        with patch('command_video.cv2.arrowedLine') as arrow:
            draw_command(np.zeros((180, 320, 3), np.uint8), 0., 0., True, .5, 0.)
        arrow.assert_not_called()

    def test_loss_covers_invalid_stale_and_history_turn_but_not_valid_single_edge(self):
        self.assertFalse(line_lost({'measurement_valid': True, 'single_line': True}, .4))
        for debug in ({'measurement_valid': False}, {'measurement_stale': True},
                      {'lost_frames': 1}, {'steering_reason': 'loss_history_turn'},
                      {'steering_reason': 'brief_loss_hold'}):
            self.assertTrue(line_lost(debug, .9), debug)

    def test_timestamp_sampling_preserves_elapsed_time_and_sent_commands(self):
        writer = Mock()
        writer.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer):
            recorder = CommandVideo(Path(tmp), fps=10, width=320, queue_size=16)
            frame = np.zeros((180, 320, 3), np.uint8)
            recorder.submit(frame, frame_id=1, host_time_ns=100, monotonic_s=5., vx=.2, wz=.3, lost=False)
            recorder.submit(frame, frame_id=2, host_time_ns=200, monotonic_s=5.31, vx=.2, wz=-.5, lost=True)
            recorder.close()
            rows = [json.loads(x) for x in (Path(tmp)/'frames.jsonl').read_text().splitlines()]
            manifest = json.loads((Path(tmp)/'manifest.json').read_text())
        self.assertEqual([r['video_time_s'] for r in rows], [0., .1, .2, .3, .4])
        self.assertEqual([r['source_frame'] for r in rows], [1, 1, 1, 1, 2])
        self.assertEqual(rows[-1]['wz'], -.5)
        self.assertTrue(rows[-1]['line_lost'])
        self.assertEqual(manifest['written_frames'], 5)
        writer.release.assert_called_once()

    def test_full_queue_never_waits_for_encoder(self):
        entered, release = threading.Event(), threading.Event()
        writer = Mock()
        writer.isOpened.return_value = True
        def slow_write(_frame):
            entered.set()
            release.wait(3)
        writer.write.side_effect = slow_write
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer):
            recorder = CommandVideo(Path(tmp), fps=10, width=320, queue_size=1)
            frame = np.zeros((180, 320, 3), np.uint8)
            def submit(i):
                return recorder.submit(frame, frame_id=i, host_time_ns=i, monotonic_s=float(i), vx=.2, wz=0., lost=False)
            try:
                self.assertTrue(submit(1))
                self.assertTrue(entered.wait(2))
                self.assertTrue(submit(2))
                self.assertFalse(submit(3))
                self.assertGreater(recorder.dropped_samples, 0)
            finally:
                release.set()
                recorder.close()

    def test_encoder_failure_is_reported_without_raising_into_control(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.cv2.VideoWriter', return_value=writer):
            recorder = CommandVideo(Path(tmp), fps=10, width=320)
            recorder.submit(np.zeros((180, 320, 3), np.uint8), frame_id=1, host_time_ns=1,
                            monotonic_s=1., vx=.2, wz=0., lost=False)
            recorder.close()
            manifest = json.loads((Path(tmp)/'manifest.json').read_text())
        self.assertIn('encoder', manifest['error'])
        self.assertFalse(recorder.submit(np.zeros((2, 2, 3), np.uint8), frame_id=2,
            host_time_ns=2, monotonic_s=2., vx=.2, wz=0., lost=False))

    def test_real_video_decodes_and_matches_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = CommandVideo(Path(tmp), fps=10, width=320, queue_size=16)
            frame = np.full((180, 320, 3), 100, np.uint8)
            for i in range(3):
                recorder.submit(frame, frame_id=i, host_time_ns=i, monotonic_s=1.+i*.1,
                                vx=.2, wz=.5, lost=bool(i))
            recorder.close()
            self.assertIsNone(recorder.error)
            rows = (Path(tmp)/'frames.jsonl').read_text().splitlines()
            cap = cv2.VideoCapture(str(Path(tmp)/'camera_commands.avi'))
            try:
                decoded = 0
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    decoded += 1
                    self.assertEqual(image.shape[:2], (180, 320))
                self.assertEqual(decoded, len(rows))
                self.assertGreaterEqual(decoded, 3)
            finally:
                cap.release()
