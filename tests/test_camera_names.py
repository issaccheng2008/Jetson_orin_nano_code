from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'new_vision/jetson'))
from camera_devices import resolve_devices, MAIN_LINK, CARD_LINK, MAIN_PORT_LINK


class CameraRoleTests(unittest.TestCase):
    def resolve(self, *, main_node='/dev/video2', card_node='/dev/video0', extras=(), disabled=False):
        other = '/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0'
        paths = [Path(MAIN_PORT_LINK),Path(other)]
        mapping = {MAIN_LINK:main_node, CARD_LINK:card_node,
                   MAIN_PORT_LINK:'/dev/video2',other:'/dev/video0'}
        for i,node in enumerate(extras):
            name=f'/dev/v4l/by-path/platform-3610000.usb-usb-0:2.{i+3}:1.0-video-index0'
            paths.append(Path(name));mapping[name]=node
        with patch('camera_devices.Path.exists',return_value=True), \
             patch('camera_devices.Path.glob',return_value=paths), \
             patch('camera_devices.os.path.realpath',side_effect=lambda p:mapping.get(str(p),str(p))):
            return resolve_devices(0,-1 if disabled else 1,MAIN_LINK,None if disabled else CARD_LINK)

    def test_named_roles_survive_video_number_order(self):
        # USB2.1 enumerated as video2; numeric0 is explicitly overridden.
        self.assertEqual(self.resolve(),(MAIN_LINK,CARD_LINK))

    def test_wrong_alias_cannot_silently_swap_roles(self):
        with self.assertRaisesRegex(ValueError,'USB 2.1'):
            self.resolve(main_node='/dev/video4')
        with self.assertRaisesRegex(ValueError,'other capture'):
            self.resolve(card_node='/dev/video4')

    def test_third_camera_makes_other_role_ambiguous(self):
        with self.assertRaisesRegex(ValueError,'exactly one secondary'):
            self.resolve(extras=('/dev/video4',))

    def test_missing_main_port_rejected(self):
        from camera_devices import verify_camera_roles
        with patch('camera_devices.Path.exists',return_value=False):
            with self.assertRaisesRegex(ValueError,'Main camera missing'):
                verify_camera_roles(MAIN_LINK,CARD_LINK)

    def test_explicit_secondary_disabled(self):
        self.assertEqual(self.resolve(disabled=True),(MAIN_LINK,None))


if __name__=='__main__':unittest.main()
