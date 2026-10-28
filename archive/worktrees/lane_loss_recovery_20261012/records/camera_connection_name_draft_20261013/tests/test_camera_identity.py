import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'new_vision/jetson'))
import camera_identity as identity


class CameraNameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.root/'camera_roles.json'
        # Same hardware name and serial, different connection IDs.
        self.main = dict(node='/dev/video2', name='2M: 2M', device_id='usb:001:005',
                         vendor='1bcf', product='2281', serial='01.00.00', initialized='100')
        self.card = dict(self.main, node='/dev/video0', device_id='usb:001:010', initialized='200')

    def register(self, main='usb:001:005', card='usb:001:010'):
        with patch.object(identity, 'list_cameras', return_value=[self.main,self.card]), \
             patch.object(identity, 'boot_id', return_value='boot-a'):
            return identity.register_roles(main,card,registry=self.registry,device_dir=self.root)

    def verify(self, *, boot='boot-a', device=None):
        with patch.object(identity, 'boot_id', return_value=boot), \
             patch.object(identity, 'camera_identity', return_value=device or self.main):
            identity.verify_named_camera(self.root/'robot_main_camera', registry=self.registry)

    def test_same_hardware_names_get_distinct_explicit_roles(self):
        self.register()
        self.assertEqual((self.root/'robot_main_camera').readlink(),Path('/dev/video2'))
        self.assertEqual((self.root/'robot_card_camera').readlink(),Path('/dev/video0'))
        self.verify()
        saved=json.loads(self.registry.read_text())
        self.assertEqual(saved['cameras']['robot_main_camera'],self.main)

    def test_duplicate_missing_or_hardware_name_selection_rejected(self):
        for main,card in [('usb:001:005','usb:001:005'),('missing','usb:001:010'),
                          ('2M: 2M','usb:001:010')]:
            with self.assertRaises(ValueError):self.register(main,card)
        self.assertFalse(self.registry.exists())

    def test_reboot_invalidates_names(self):
        self.register()
        with self.assertRaisesRegex(ValueError,'expired after reboot'):self.verify(boot='boot-b')

    def test_replug_and_role_swap_invalidates_names(self):
        self.register()
        for changed in [dict(self.main,device_id='usb:001:011'),
                        dict(self.main,initialized='300'),self.card]:
            with self.assertRaisesRegex(ValueError,'identity changed'):self.verify(device=changed)

    def test_old_udev_name_without_registration_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'re-register'):self.verify()

    def test_non_symlink_file_cannot_be_overwritten(self):
        (self.root/'robot_main_camera').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'non-symlink'):self.register()
        self.assertEqual((self.root/'robot_main_camera').read_text(),'preserve')

    def test_no_port_rules_remain_active(self):
        text=(ROOT/'config/udev/99-robot-cameras.rules').read_text()
        self.assertFalse(any(line.strip() and not line.lstrip().startswith('#')
                             for line in text.splitlines()))


if __name__=='__main__':unittest.main()
