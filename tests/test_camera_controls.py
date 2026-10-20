import argparse
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from camera_controls import add_arguments, validate_args, apply_camera_controls

CONTROLS = '''User Controls
 brightness 0x00980900 (int) : min=-64 max=64 step=1 default=0 value=0
 contrast 0x00980901 (int) : min=0 max=95 step=1 default=2 value=2
 saturation 0x00980902 (int) : min=0 max=100 step=1 default=75 value=75
 sharpness 0x0098091b (int) : min=1 max=7 step=1 default=2 value=2
 white_balance_automatic 0x0098090c (bool) : default=1 value=1
 white_balance_temperature 0x0098091a (int) : min=2800 max=6500 step=1 default=4600 value=4600 flags=inactive
 power_line_frequency 0x00980918 (menu) : min=0 max=2 default=1 value=1 (50 Hz)
 0: Disabled
 1: 50 Hz
 2: 60 Hz
Camera Controls
 auto_exposure 0x009a0901 (menu) : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
 1: Manual Mode
 3: Aperture Priority Mode
 exposure_time_absolute 0x009a0902 (int) : min=3 max=2047 step=1 default=166 value=166 flags=inactive
'''


def arguments(*argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(argv)
    validate_args(args)
    return args


class FakeV4l:
    def __init__(self, text=CONTROLS, fail=None, wrong_readback=False):
        self.text, self.fail, self.wrong_readback = text, fail, wrong_readback
        self.calls = []
        self.values = {}

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        flag = command[-1]
        if flag == '--list-ctrls-menus':
            return subprocess.CompletedProcess(command,0,self.text,'')
        if flag.startswith('--set-ctrl='):
            key, value = flag.split('=',1)[1].split('=')
            if key == self.fail:
                return subprocess.CompletedProcess(command,1,'','setting rejected')
            self.values[key] = int(value)
            return subprocess.CompletedProcess(command,0,'','')
        key = flag.split('=',1)[1]
        value = self.values[key]+(1 if self.wrong_readback else 0)
        return subprocess.CompletedProcess(command,0,f'{key}: {value}\n','')


class CameraControlsTests(unittest.TestCase):
    def test_keep_defaults_never_access_device(self):
        with patch('camera_controls.subprocess.run') as run:
            report = apply_camera_controls('/dev/video0', arguments())
        run.assert_not_called()
        self.assertEqual(report['status'], 'unchanged')

    def test_manual_exposure_units_order_readback_and_other_settings(self):
        fake = FakeV4l()
        args = arguments('--camera-exposure-mode','manual','--camera-exposure-ms','5',
            '--camera-brightness','-2','--camera-white-balance-mode','manual',
            '--camera-white-balance-k','4600','--camera-power-line-hz','50')
        with patch('camera_controls.sys.platform','linux'), patch('camera_controls.subprocess.run',side_effect=fake):
            report = apply_camera_controls('/dev/video2',args)
        writes = [c[-1] for c in fake.calls if c[-1].startswith('--set-ctrl=')]
        self.assertLess(writes.index('--set-ctrl=auto_exposure=1'), writes.index('--set-ctrl=exposure_time_absolute=50'))
        self.assertLess(writes.index('--set-ctrl=white_balance_automatic=0'), writes.index('--set-ctrl=white_balance_temperature=4600'))
        self.assertIn('--set-ctrl=brightness=-2',writes)
        self.assertIn('--set-ctrl=power_line_frequency=1',writes)
        self.assertTrue(all(c[2]=='/dev/video2' for c in fake.calls))
        self.assertEqual(report['status'],'applied')
        exposure = next(s for s in report['settings'] if s['control']=='exposure_time_absolute')
        self.assertEqual(exposure['actual'],50)
        self.assertEqual(exposure['actual_ms'],5.)

    def test_automatic_mode_and_older_driver_names(self):
        fake = FakeV4l(CONTROLS.replace('auto_exposure','exposure_auto').replace('exposure_time_absolute','exposure_absolute'))
        with patch('camera_controls.sys.platform','linux'), patch('camera_controls.subprocess.run',side_effect=fake):
            report = apply_camera_controls('/dev/video0',arguments('--camera-exposure-mode','auto'))
        self.assertEqual(fake.values,{'exposure_auto':3})
        self.assertEqual(report['status'],'applied')

    def test_failed_auto_disable_never_sets_manual_exposure(self):
        fake = FakeV4l(fail='auto_exposure')
        with patch('camera_controls.sys.platform','linux'), patch('camera_controls.subprocess.run',side_effect=fake):
            report = apply_camera_controls('/dev/video0',arguments('--camera-exposure-mode','manual','--camera-exposure-ms','5'))
        self.assertNotIn('exposure_time_absolute',fake.values)
        self.assertEqual(report['status'],'partial')

    def test_unsupported_out_of_range_and_readback_mismatch_are_not_success(self):
        for args, text, wrong in ((arguments('--camera-sharpness','8'),CONTROLS,False),
                (arguments('--camera-contrast','3'),'',False),
                (arguments('--camera-brightness','1'),CONTROLS,True)):
            fake=FakeV4l(text,wrong_readback=wrong)
            with patch('camera_controls.sys.platform','linux'), patch('camera_controls.subprocess.run',side_effect=fake):
                report=apply_camera_controls('/dev/video0',args)
            self.assertEqual(report['status'],'partial')
            self.assertTrue(any(s.get('error') for s in report['settings']))

    def test_missing_utility_and_timeout_leave_control_running_with_error(self):
        for error in (FileNotFoundError('v4l2-ctl missing'),subprocess.TimeoutExpired('v4l2-ctl',3)):
            with patch('camera_controls.sys.platform','linux'), patch('camera_controls.subprocess.run',side_effect=error):
                report=apply_camera_controls('/dev/video0',arguments('--camera-exposure-mode','manual','--camera-exposure-ms','5'))
            self.assertEqual(report['status'],'unavailable')
            self.assertTrue(report['error'])

    def test_invalid_modes_and_units_fail_without_hardware(self):
        for argv in (('--camera-exposure-mode','manual'), ('--camera-exposure-ms','5'),
                ('--camera-exposure-mode','manual','--camera-exposure-ms','nan'),
                ('--camera-exposure-mode','manual','--camera-exposure-ms','0'),
                ('--camera-exposure-mode','manual','--camera-exposure-ms','.333'),
                ('--camera-exposure-mode','manual','--camera-exposure-ms','1e308'),
                ('--camera-white-balance-mode','manual'), ('--camera-white-balance-k','4600')):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                arguments(*argv)
