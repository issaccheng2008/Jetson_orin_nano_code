"""Window boundaries are driven by control ticks, never by a slow camera."""
import unittest

import connector


class ExecutedYawWindowTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(connector, 'CommandWindowScheduler'),
                        'connector needs an independent command-window clock')
        self.scheduler = connector.CommandWindowScheduler()

    def frame(self, now, wz, *, vx=.2, valid=True, side='lower', origin=None, **metadata):
        packet = dict(vx=vx, wz=wz, qr=-1, command_mode='held',
                      steering_window=dict(seconds=.5, median=side, valid=valid), **metadata)
        if origin is not None:
            packet['steering_window']['origin_s'] = origin
        self.scheduler.receive(connector.process_vision_output(packet), now)

    def output(self, now):
        return self.scheduler.output(now)[0]

    def test_user_example_closes_without_a_boundary_frame(self):
        self.frame(10., .12)  # Starts [10, 10.5); first interval stays stopped.
        self.frame(10.12, .12)
        self.frame(10.3, .3)
        self.frame(10.4, .3)
        self.assertEqual(self.output(10.499)['vx'], 0.)
        self.assertEqual(self.output(10.5)['wz'], .12)  # lower of four
        self.frame(10.52, -.2)
        self.assertEqual(self.output(10.999)['wz'], .12)
        self.assertEqual(self.output(11.)['wz'], -.2)

    def test_exact_user_three_samples_and_late_next_sample(self):
        self.frame(10.12, .12, origin=10.)
        for time, wz in [(10.3, .3), (10.4, .3)]:
            self.frame(time, wz)
        self.assertEqual(self.output(10.5)['wz'], .3)
        self.frame(10.52, -.2)
        self.assertEqual(self.output(10.98)['wz'], .3)
        self.assertEqual(self.output(11.)['wz'], -.2)

    def test_first_frame_finishing_after_boundary_is_not_backdated(self):
        self.frame(10.52, .3, origin=10.)
        self.assertEqual(self.output(10.52)['vx'], 0.)
        self.assertEqual(self.output(11.)['wz'], .3)

    def test_upper_lower_signed_values_and_boundary_sample(self):
        for side, expected in [('lower', -.2), ('upper', .3)]:
            self.scheduler = connector.CommandWindowScheduler()
            self.frame(10., -.2, side=side)
            self.frame(10.3, .3, side=side)
            self.frame(10.5, .5, side=side)
            self.assertEqual(self.output(10.5)['wz'], expected)
            self.assertEqual(self.output(11.)['wz'], .5)

    def test_repeated_control_ticks_do_not_count_as_visual_samples(self):
        self.frame(10., -.2)
        for i in range(24):
            self.output(10.+i*.02)
        self.frame(10.48, .3, side='lower')
        self.assertEqual(self.output(10.5)['wz'], -.2)

    def test_empty_window_keeps_motion_and_loss_window_uses_compensation(self):
        self.scheduler = connector.CommandWindowScheduler(window_vision_timeout_s=1.5)
        self.frame(10., .3)
        self.assertEqual(self.output(10.5)['wz'], .3)
        self.assertEqual(self.output(11.)['wz'], .3)
        self.frame(11.01, -.1, valid=False)
        self.assertEqual(self.output(11.49)['wz'], .3)
        self.assertEqual(self.output(11.5)['wz'], -.1)

    def test_valid_samples_win_over_loss_compensation(self):
        self.frame(10., -.2)
        self.frame(10.1, .1)
        self.frame(10.2, .3)
        self.frame(10.4, .5, valid=False)
        self.assertEqual(self.output(10.5)['wz'], .1)

    def test_delayed_ticks_never_shorten_an_applied_block(self):
        self.frame(10., .3)
        self.assertEqual(self.output(10.73)['wz'], .3)
        self.frame(10.8, -.2)
        self.assertEqual(self.output(11.02)['wz'], .3)
        self.assertEqual(self.output(11.229)['wz'], .3)
        self.assertEqual(self.output(11.23)['wz'], -.2)

    def test_card_stop_preempts_and_discards_all_pending_samples(self):
        self.frame(10., .3)
        self.output(10.5)
        self.frame(10.6, -.2)
        self.frame(10.61, 0., vx=0., card_tilt=True, event_id=7, event_action=3)
        out = self.output(10.61)
        self.assertEqual((out['vx'], out['wz']), (0., 0.))
        self.assertTrue(out['card_tilt'])
        self.assertEqual((out['event_id'], out['event_action']), (7, 3))
        self.frame(10.7, .1)
        self.assertEqual(self.output(11.19)['vx'], 0.)
        self.assertEqual(self.output(11.2)['wz'], .1)

    def test_window_watchdog_tolerates_slow_frame_then_stops_and_resets(self):
        self.frame(10., .3)
        self.output(10.5)
        self.frame(10.55, .1)
        self.assertEqual(self.output(10.9)['vx'], .2)  # > .25 since frame
        out, fresh, age = self.scheduler.output(11.56)
        self.assertFalse(fresh)
        self.assertAlmostEqual(age, 1.01)
        self.assertEqual((out['vx'], out['wz']), (0., 0.))
        self.frame(11.6, -.2)
        self.assertEqual(self.output(11.6)['vx'], 0.)
        self.assertEqual(self.output(12.1)['wz'], -.2)

    def test_watchdog_restart_does_not_reuse_an_old_window_origin(self):
        self.frame(10., .3, origin=10.)
        self.output(10.5)
        self.assertEqual(self.output(11.1)['vx'], 0.)
        self.frame(11.4, -.2, origin=10.)
        self.assertEqual(self.output(11.5)['vx'], 0.)
        self.frame(11.6, .1, origin=10.)
        self.assertEqual(self.output(11.9)['wz'], -.2)

    def test_legacy_packets_keep_original_timeout_and_bypass_windows(self):
        self.frame(10., .3)
        packet = connector.process_vision_output(dict(vx=.2, wz=-.1))
        self.scheduler.receive(packet, 10.1)
        self.assertEqual(self.output(10.1)['wz'], -.1)
        self.assertEqual(self.output(10.36)['vx'], 0.)

    def test_window_settings_and_model_hold_survive_only_where_needed(self):
        self.frame(10., .3, side='upper')
        out = self.output(10.5)
        self.assertEqual(out['command_min_hold_s'], .5)
        self.assertNotIn('steering_window', out)
        forwarded = connector.CommandSmoother(1., 2.).update(out, .02)
        self.assertEqual(forwarded['command_min_hold_s'], .5)
        self.assertEqual((forwarded['vx'], forwarded['wz']), (.2, .3))


class WindowProtocolTests(unittest.TestCase):
    def test_invalid_window_metadata_is_rejected(self):
        for window in [None, {}, dict(seconds=0, median='lower', valid=True),
                       dict(seconds=float('nan'), median='lower', valid=True),
                       dict(seconds=.5, median='mean', valid=True),
                       dict(seconds=.5, median='upper', valid='yes')]:
            with self.subTest(window=window), self.assertRaises(ValueError):
                connector.process_vision_output(dict(vx=.2, wz=.3, steering_window=window))


if __name__ == '__main__':
    unittest.main()
