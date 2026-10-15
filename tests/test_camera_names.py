from pathlib import Path, PurePosixPath
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'new_vision/jetson'))
from camera_devices import resolve_devices, MAIN_LINK, CARD_LINK, MAIN_PORT_LINKS


class CameraRoleTests(unittest.TestCase):
    def resolve(self, *, ports=(0,), main_node='/dev/video2', card_node='/dev/video0',
                extras=(), disabled=False, card_present=True):
        other = '/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0'
        mapping = {MAIN_LINK: main_node, CARD_LINK: card_node}
        for port in ports:
            mapping[MAIN_PORT_LINKS[port]] = '/dev/video2' if port == ports[0] else '/dev/video4'
        if card_present:
            mapping[other] = '/dev/video0'
        for i, node in enumerate(extras):
            mapping[f'/dev/v4l/by-path/platform-3610000.usb-usb-0:3.{i+1}:1.0-video-index0'] = node
        class LinuxPath(PurePosixPath):
            def exists(self):
                return str(self) in mapping

            def glob(self, pattern):
                return [LinuxPath(p) for p in mapping if p.startswith('/dev/v4l/by-path/')]

        with patch('camera_devices.Path', LinuxPath), \
             patch('camera_devices.os.path.realpath', side_effect=lambda p: mapping.get(str(p), str(p))):
            return resolve_devices(0, -1 if disabled else 1, MAIN_LINK, None if disabled else CARD_LINK)

    def test_either_main_port_survives_video_number_order(self):
        for port in (0, 1):
            with self.subTest(port=port):
                self.assertEqual(self.resolve(ports=(port,)), (MAIN_LINK, CARD_LINK))

    def test_wrong_alias_cannot_silently_swap_roles(self):
        with self.assertRaisesRegex(ValueError, 'USB 2.3/2.4'):
            self.resolve(main_node='/dev/video4')
        with self.assertRaisesRegex(ValueError, 'other capture'):
            self.resolve(card_node='/dev/video4')

    def test_both_main_ports_occupied_rejected_even_if_secondary_disabled(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled):
                with self.assertRaisesRegex(ValueError, 'Multiple cameras'):
                    self.resolve(ports=(0, 1), disabled=disabled)

    def test_third_camera_makes_other_role_ambiguous(self):
        with self.assertRaisesRegex(ValueError, 'exactly one secondary'):
            self.resolve(extras=('/dev/video6',))

    def test_missing_main_cannot_promote_secondary(self):
        with self.assertRaisesRegex(ValueError, 'Main camera missing'):
            self.resolve(ports=())

    def test_missing_secondary_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one secondary'):
            self.resolve(card_present=False)

    def test_explicit_secondary_disabled(self):
        self.assertEqual(self.resolve(disabled=True, card_present=False), (MAIN_LINK, None))

    def test_stm32_on_unused_main_port_does_not_count_as_camera(self):
        # A ttyACM/ttyUSB device creates no V4L by-path link and is ignored.
        self.assertEqual(self.resolve(ports=(1,)), (MAIN_LINK, CARD_LINK))
        rules = (ROOT/'config/udev/99-robot-cameras.rules').read_text()
        for line in rules.splitlines():
            if line and not line.startswith('#'):
                self.assertIn('SUBSYSTEM=="video4linux"', line)
                self.assertIn('ENV{ID_V4L_CAPABILITIES}=="*:capture:*"', line)


if __name__ == '__main__':
    unittest.main()
