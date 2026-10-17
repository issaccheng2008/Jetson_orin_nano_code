"""Configured tables consume current geometry, without demand-history corrections."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from steering_filter import FilterConfig
from test_heading_steering import detection

TABLE = [[-90, -45, -.1], [-45, -15, 0], [-15, 15, .1],
         [15, 46, .22], [46, 55, .3], [55, 90, .5]]


class RawDemandTableTests(unittest.TestCase):
    def make(self, **options):
        return HeadingSteeringController(SteeringController(), angle_wz_table=TABLE,
            filter_config=FilterConfig(algorithm='robust'), **options)

    def test_frame_839_geometry_maps_to_point_one_after_larger_demands(self):
        c = self.make()
        for heading in (35., 30., 25.):
            c.command(detection(heading, near=-7.7, z=21.), 1., .2)
        output = c.command(detection(7.600468663464752,
            near=-4.355807511770191, z=21.09691353383458), 1., .21936)
        self.assertAlmostEqual(c.diagnostics['steering_demand_deg'], 9.327595389097295)
        self.assertEqual(output, (.2, .1))
        self.assertEqual(c.diagnostics['steering_table_input_deg'], c.diagnostics['steering_demand_deg'])
        for removed in ('steering_position_correction_deg', 'steering_filtered_demand_deg',
                        'steering_combined_demand_deg', 'steering_filter_demand_deg'):
            self.assertNotIn(removed, c.diagnostics)
        self.assertNotEqual(c.diagnostics['steering_filtered_heading_deg'],
                            c.diagnostics['steering_heading_deg'])

    def test_each_frame_uses_exact_table_boundary_without_history_or_prediction(self):
        for yaw in (-1, 1):
            c = HeadingSteeringController(SteeringController(yaw_sign=yaw),
                angle_wz_table=TABLE, filter_config=FilterConfig(algorithm='robust'))
            for angle, expected in ((70., .5), (50., .3), (20., .22),
                    (15., .22), (14.9, .1), (-15., .1), (-15.1, 0.),
                    (-45., 0.), (-45.1, -.1), (0., .1)):
                with self.subTest(yaw=yaw, angle=angle):
                    output = c.command(detection(angle), 1., .2)
                    self.assertEqual(output, (.2, expected*yaw))

    def test_confirmed_position_does_not_override_raw_table_demand(self):
        c = self.make(position_recovery_cm=8.)
        # Lane position and heading cancel at the forward target: raw demand = 0.
        heading = math.degrees(math.atan2(12., 25.))
        for _ in range(4):
            self.assertEqual(c.command(detection(heading, near=12., z=25.), 1., .2), (.2, .1))
            self.assertAlmostEqual(c.diagnostics['steering_demand_deg'], 0.)

    def test_configured_hold_and_geometry_loss_still_apply(self):
        c = self.make(min_hold_s=.5)
        self.assertEqual(c.command(detection(60.), 1., .01), (.2, .5))
        self.assertEqual(c.command(detection(0.), 1., .2), (.2, .5))
        self.assertEqual(c.command(detection(0.), 1., .301), (.2, .1))
        c.command(detection(measurement_valid=False), 0., .3)
        self.assertIsNone(c.diagnostics.get('steering_demand_deg'))
        c.drop_held_command()
        self.assertEqual(c.command(detection(measurement_valid=False), 0., .1), (0., 0.))


if __name__ == '__main__':
    unittest.main()
