import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from startup_sequence import parse_sequence, StartupSequence


class StartupSequenceTests(unittest.TestCase):
    def test_default_and_explicit_sequence(self):
        default = parse_sequence('', .3, .2, .5)
        self.assertEqual([(s.duration_s,s.vx,s.wz) for s in default], [(.3,.2,0.)])
        steps = parse_sequence('[{"duration_s":0.5,"vx":0.2,"wz":0},'
                               '{"duration_s":0.5,"vx":0.2,"wz":0.5}]', .3, .2, .5)
        self.assertEqual([(s.duration_s,s.vx,s.wz) for s in steps], [(.5,.2,0.),(.5,.2,.5)])

    def test_each_step_gets_its_duration_even_after_slow_frame_and_reset(self):
        runner = StartupSequence(parse_sequence('[{"duration_s":0.5,"vx":0.2,"wz":0},'
            '{"duration_s":0.5,"vx":0.2,"wz":-0.5},{"duration_s":0.2,"vx":0,"wz":0}]',.5,.2,.5))
        self.assertEqual(runner.command(10.), (.2,0.))
        self.assertEqual(runner.command(10.49), (.2,0.))
        self.assertEqual(runner.command(11.), (.2,-.5))
        self.assertEqual(runner.command(11.49), (.2,-.5))
        self.assertEqual(runner.command(11.5), (0.,0.))
        self.assertIsNone(runner.command(11.7))
        self.assertIsNone(runner.command(12.))
        runner.reset()
        self.assertEqual(runner.command(20.), (.2,0.))

    def test_bad_sequences_rejected_before_hardware(self):
        for raw in ('[]', '{}', 'bad', '[{}]', '[{"duration_s":0,"vx":0.2,"wz":0}]',
            '[{"duration_s":-1,"vx":0.2,"wz":0}]', '[{"duration_s":1,"vx":2,"wz":0}]',
            '[{"duration_s":1,"vx":0.2,"wz":0.6}]', '[{"duration_s":1,"vx":0.2,"wz":NaN}]',
            '[{"duration_s":true,"vx":0.2,"wz":0}]', '[{"duration_s":1,"vx":0.2,"wz":0,"typo":1}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_sequence(raw,.5,.2,.5)
