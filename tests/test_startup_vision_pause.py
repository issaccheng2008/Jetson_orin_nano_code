"""First-card timings and actual entry bypass, without USB, serial or motors."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from startup_sequence import StartupSequence, parse_sequence
from startup_vision_pause import StartupVisionPause
import run_policy_vision
from test_policy_vision import detection


class StartupVisionPauseTests(unittest.TestCase):
    def test_walk_action_and_resume_keep_single_event_and_50hz_ticks(self):
        for action, hold in ((1, 3.), (3, 6.)):
            with self.subTest(action=action):
                sequence = StartupSequence(parse_sequence('', .5, .2, .5))
                factory = Mock(return_value=123)
                runner = StartupVisionPause(sequence, action, hold, .2, factory)
                commands = []
                for i in range(1000):
                    now = 10. + i*.02
                    cmd = runner.command(now)
                    runner.published(now)
                    commands.append((now, cmd))
                    if cmd.phase == 'RESUME':
                        break
                walks = [(t,c) for t,c in commands if c.phase == 'WALK']
                acts = [(t,c) for t,c in commands if c.phase == 'ACTION']
                self.assertEqual(len(walks), 25)
                self.assertTrue(all((c.vx,c.wz)==(.2,0.) for _,c in walks))
                self.assertTrue(all((c.vx,c.wz)==(0.,0.) for _,c in acts))
                self.assertEqual({c.event_id for _,c in acts}, {123})
                self.assertEqual(sum(c.qr == action for _,c in commands), 1)
                self.assertAlmostEqual(commands[-1][0]-acts[0][0], hold, places=5)
                self.assertEqual((commands[-1][1].vx,commands[-1][1].wz), (.2,0.))
                factory.assert_called_once()
                self.assertAlmostEqual(runner.diagnostics()['startup_pause_publish_hz'], 50.)
                self.assertAlmostEqual(runner.max_publish_gap_s, .02)

    def test_delayed_tick_does_not_skip_or_shorten_requested_sequence_steps(self):
        steps = parse_sequence('[{"duration_s":0.5,"vx":0.2,"wz":0},'
                               '{"duration_s":0.5,"vx":0.2,"wz":0.3}]', .5,.2,.5)
        runner = StartupVisionPause(StartupSequence(steps), 2, 3., .2, lambda: 42)
        self.assertEqual(runner.command(10.).wz, 0.)
        self.assertEqual(runner.command(11.).wz, .3)
        self.assertEqual(runner.command(11.49).phase, 'WALK')
        self.assertEqual(runner.command(11.5).phase, 'ACTION')
        self.assertEqual(runner.command(14.49).phase, 'ACTION')
        self.assertEqual(runner.command(14.5).phase, 'RESUME')

    def test_entry_skips_all_image_calls_and_resumes_fresh_capture(self):
        clock = [0.]
        reads = [0]
        frame = np.zeros((720,1280,3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280,720]
        capture = Mock()
        capture.close.return_value = True
        detector = Mock()
        detector.process.return_value = (0,0,.9,None,detection())
        shape = Mock()
        shape.action_map = {'circle':1}
        shape.update.return_value = (None, dict(presence=True,card_found=True,
            presence_cy_frac=.9,shape='circle'))
        sent = []
        output = io.StringIO()

        def read():
            reads[0] += 1
            clock[0] += .1
            if reads[0] > 8:
                run_policy_vision.signal.signal.call_args.args[1](None,None)
                return False,None,{}
            # The capture worker may still hold one pre-resume exposure.
            return True,frame,{'camera_capture_started_s':(-1000. if reads[0] == 3 else clock[0]),
                              'camera_capture_returned_s':clock[0]}

        capture.read.side_effect = read

        def publish(vx,wz,qr=-1,**kw):
            sent.append(dict(time=clock[0],vx=vx,wz=wz,qr=qr,kw=kw,
                reads=reads[0],line_calls=detector.process.call_count,
                shape_calls=shape.update.call_count))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch('sys.argv', ['vision','--headless','--start-gate','both',
                '--startup-vision-pause','--startup-first-walk-s','.5',
                '--start-gate-shape-votes','2','--shape-every','1','--qr-every','1',
                '--attitude-port','0','--card-hold-ms','500','--line-log-dir',directory]),
            patch.object(run_policy_vision.signal,'signal'),
            patch.object(run_policy_vision,'ConnectorClient') as client,
            patch('utils.open_camera',return_value=camera),
            patch('latest_camera.LatestCamera') as camera_worker,
            patch('line_detector_v1_warp.LineDetector',return_value=detector),
            patch('shape_detector.ShapeDetector',return_value=shape),
            patch('qr_reader.QrReader') as qr,
            patch('command_video.CommandVideo') as video,
            patch.object(run_policy_vision.time,'monotonic',lambda: clock[0]),
            patch.object(run_policy_vision.time,'sleep',lambda duration: clock.__setitem__(0,clock[0]+duration)),
            contextlib.redirect_stdout(output),
        ):
            camera_worker.return_value.start.return_value = capture
            qr.return_value.decode.return_value = SimpleNamespace(
                payload='1',strategy='raw',edge_px=83.,cost_ms=1.)
            qr.return_value.scans = 1
            qr.return_value.geom_rejects = 0
            qr.return_value.last_cost_ms = 1.
            client.return_value.publish.side_effect = publish
            self.assertEqual(run_policy_vision.main(),0)
            video.assert_not_called()
            client.return_value.close.assert_called_once()
            rows = [json.loads(s) for s in next(Path(directory).rglob('line_frames.jsonl')).read_text().splitlines()]
        first_walk = next(i for i,r in enumerate(sent) if r['vx'] > 0)
        first_event = next(i for i,r in enumerate(sent) if r['kw'].get('event_id'))
        resume = next(i for i in range(first_event+1,len(sent)) if sent[i]['vx'] > 0)
        interval = sent[first_walk:resume+1]
        self.assertEqual(len({r['reads'] for r in interval}),1)
        self.assertEqual(len({r['line_calls'] for r in interval}),1)
        self.assertEqual(len({r['shape_calls'] for r in interval}),1)
        self.assertLess(max(b['time']-a['time'] for a,b in zip(interval,interval[1:])), .021)
        self.assertAlmostEqual(sent[first_event]['time']-sent[first_walk]['time'],.5,delta=.021)
        self.assertAlmostEqual(sent[resume]['time']-sent[first_event]['time'],3.,delta=.021)
        self.assertGreater(sent[resume+1]['reads'],sent[resume]['reads'])
        self.assertEqual(sent[resume+1]['line_calls'],sent[resume]['line_calls'])
        self.assertGreater(sent[resume+2]['line_calls'],sent[resume+1]['line_calls'])
        self.assertEqual(len({r['kw']['event_id'] for r in sent if r['kw'].get('event_id')}),1)
        paused = [r for r in rows if r.get('startup_vision_paused')]
        self.assertTrue(paused)
        self.assertTrue(all(not r['observation_processed'] for r in paused))
        self.assertTrue(any(r.get('observation_processed') for r in rows[rows.index(paused[-1])+1:]))
        self.assertIn('phase=RESUME',output.getvalue())


if __name__ == '__main__':
    unittest.main()
