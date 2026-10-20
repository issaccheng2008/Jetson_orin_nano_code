"""F/C target-angle blending is confined to active heading angle tables."""
import contextlib
import io
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from steering_filter import FilterConfig, GeometryFilter, configured_controller
from segment_steering import SegmentSteeringController
from test_heading_steering import detection

TABLE = [[-90,-10,-.5],[-10,-3,-.3],[-3,3,0],[3,10,.3],[10,90,.5]]

class PositionBlendTests(unittest.TestCase):
    def controller(self, **options):
        defaults = dict(angle_wz_table=TABLE, filter_config=FilterConfig(algorithm='none'))
        defaults.update(options)
        return HeadingSteeringController(SteeringController(), **defaults)

    def test_weights_endpoints_arithmetic_and_discrete_output(self):
        for w, expected in ((0,0.), (.3,-.3), (1,-.5)):
            c = self.controller(steering_position_blend_weight=w)
            heading = math.degrees(math.atan2(20, 50))
            output = c.command(detection(heading, near=20), 1, .1)
            d = c.diagnostics
            self.assertAlmostEqual(d['steering_demand_deg'], 0.)
            self.assertAlmostEqual(d['steering_filtered_demand_deg'], 0.)
            self.assertEqual(d['steering_position_correction_deg'], -12.)
            self.assertAlmostEqual(d['steering_combined_demand_deg'], -12.)
            self.assertEqual(d['steering_position_blend_weight'], w)
            self.assertAlmostEqual(d['steering_blended_demand_deg'], -12*w)
            self.assertEqual(d['steering_table_input_deg'], d['steering_blended_demand_deg'])
            self.assertEqual(output[1], expected)
            self.assertFalse(d['steering_braked'])

    def test_default_weight_and_mixed_sign_are_not_heading_blend(self):
        c = self.controller()
        heading = math.degrees(math.atan(math.tan(math.radians(2))+20/50))
        c.command(detection(heading, near=20), 1, .1)
        d = c.diagnostics
        self.assertAlmostEqual(d['steering_filtered_demand_deg'], 2.)
        self.assertAlmostEqual(d['steering_combined_demand_deg'], -10.)
        self.assertAlmostEqual(d['steering_blended_demand_deg'], -1.6)
        self.assertGreater(d['steering_heading_deg'], 20.)

    def test_position_deadband_cap_and_settings(self):
        for near, expected in ((-100,12.), (-2,0.), (0,0.), (2,0.), (100,-12.),
                               (6,-math.degrees(math.atan2(4,50)))):
            c = self.controller()
            c.command(detection(0,near=near), 1, .1)
            self.assertAlmostEqual(c.diagnostics['steering_position_correction_deg'], expected)
        c = self.controller(position_gain=2, position_dead_cm=1, position_lookahead_cm=40, position_max_deg=5)
        c.command(detection(0,near=4), 1, .1)
        self.assertEqual(c.diagnostics['steering_position_correction_deg'], -5.)

    def test_demand_step_is_filtered_independently_of_constant_heading(self):
        c = self.controller(filter_config=FilterConfig(algorithm='ema'), steering_position_blend_weight=0)
        c.command(detection(0), 1, .1)
        c.command(detection(0, near=-20), 1, .1)
        d = c.diagnostics
        self.assertEqual(d['steering_filtered_heading_deg'], 0.)
        self.assertGreater(d['steering_filtered_demand_deg'], 0.)
        self.assertLess(d['steering_filtered_demand_deg'], d['steering_demand_deg'])
        self.assertEqual(d['steering_table_input_deg'], d['steering_filtered_demand_deg'])

    def test_filter_none_retains_causal_three_sample_median(self):
        c = self.controller(steering_position_blend_weight=0)
        for angle in (0,0,40):
            c.command(detection(angle),1,.1)
        self.assertEqual(c.diagnostics['steering_filtered_demand_deg'],0.)
        self.assertEqual(c.diagnostics['steering_demand_deg'],40.)

    def test_no_position_direction_has_no_extra_correction(self):
        c = self.controller()
        c.command(detection(15,near=40, position_valid=False,
            heading_control_source='visible_line_direction'),1,.1)
        self.assertEqual(c.diagnostics['steering_position_correction_deg'],0.)
        self.assertEqual(c.diagnostics['steering_combined_demand_deg'],15.)
        self.assertEqual(c.diagnostics['steering_table_input_deg'],15.)

    def test_source_transitions_preserve_robust_demand_and_reset_other_filters(self):
        # Choose headings independently so measured near position and raw target
        # bearing differ. A lane-heading blend cannot satisfy these assertions.
        def measured(raw_demand, near):
            heading = math.degrees(math.atan(math.tan(math.radians(raw_demand))+near/50))
            return detection(heading, near=near)

        first = measured(0,20)
        direction = detection(30,near=99,position_valid=False,
                              heading_control_source='visible_line_direction')
        restored = measured(-25,20)
        edge_heading = math.degrees(math.atan(math.tan(math.radians(40))-20/50))
        edge = detection(heading_control_valid=False, single_edge_valid=True,
                         single_edge_heading_deg=edge_heading,
                         single_edge_near_cm=-20, single_edge_z_cm=0)
        frames = (first,direction,restored,edge,restored)
        sources = ('ground_x_z','visible_line_direction','ground_x_z','single_edge','ground_x_z')
        for algorithm in ('robust','ema','one-euro'):
            with self.subTest(algorithm=algorithm):
                c = self.controller(filter_config=FilterConfig(algorithm=algorithm))
                previous_f = None
                for index, (frame, source) in enumerate(zip(frames,sources)):
                    output = c.command(frame,1,.1)
                    d = c.diagnostics
                    f = d['steering_filtered_demand_deg']
                    self.assertEqual(d['steering_heading_source'],source)
                    if algorithm == 'robust' and previous_f is not None:
                        self.assertLessEqual(abs(f-previous_f),45*.1+1e-9)
                        self.assertNotAlmostEqual(f,d['steering_demand_deg'])
                    else:
                        self.assertAlmostEqual(f,d['steering_demand_deg'])
                    if index == 0:
                        self.assertEqual(d['steering_position_correction_deg'],-12.)
                    if source == 'visible_line_direction':
                        self.assertEqual(d['steering_position_correction_deg'],0.)
                        if algorithm == 'robust':
                            self.assertGreater(d['steering_filter_near_cm'],2.)
                            self.assertAlmostEqual(f,previous_f)
                    p = d['steering_position_correction_deg']
                    blend = f+.3*p
                    self.assertAlmostEqual(d['steering_combined_demand_deg'],f+p)
                    self.assertAlmostEqual(d['steering_table_input_deg'],blend)
                    self.assertAlmostEqual(d['steering_blended_demand_deg'],blend)
                    expected_gear = (-.5 if blend < -10 else -.3 if blend < -3
                                     else 0. if blend < 3 else .3 if blend < 10 else .5)
                    self.assertEqual(output,(.2,expected_gear))
                    self.assertEqual(d['steering_table_wz'],expected_gear)
                    previous_f = f

    def test_stop_and_expired_loss_clear_filter_history(self):
        for stop in ('reset','drop','loss','invalid_clock'):
            c=self.controller(filter_config=FilterConfig(algorithm='robust'))
            c.command(detection(0),1,.1)
            if stop=='loss':
                c.command(detection(measurement_valid=False, measurement_stale=True),0,.5)
            elif stop=='invalid_clock': c.command(detection(),1,0)
            elif stop=='drop': c.drop_held_command()
            else: c.reset()
            c.command(detection(40),1,.1)
            self.assertAlmostEqual(c.diagnostics['steering_filtered_demand_deg'],40.)

    def test_legacy_non_table_and_segments_are_isolated(self):
        for cls, options in ((HeadingSteeringController,dict(filter_config=None)),
            (HeadingSteeringController,dict(angle_wz_table=None)), (SegmentSteeringController,{})):
            defaults=dict(angle_wz_table=TABLE,filter_config=FilterConfig(algorithm='ema'))
            defaults.update(options)
            c=cls(SteeringController(),**defaults)
            c.command(detection(0),1,.1)
            c.command(detection(40,near=20),1,.1)
            self.assertNotIn('steering_blended_demand_deg',c.diagnostics)
            if c.angle_wz_table is not None:
                self.assertEqual(c.diagnostics['steering_table_input_deg'],c.diagnostics['steering_demand_deg'])
            if cls is SegmentSteeringController:
                self.assertFalse(c.shadow._position_blend_active)

    def test_optional_geometry_demand_filter_preserves_default_api(self):
        cfg=FilterConfig(algorithm='ema')
        baseline=GeometryFilter(cfg)
        trial=GeometryFilter(cfg,filter_demand=True)
        for f in (baseline,trial): f.apply((0,25,0,0,'ground_x_z'),0)
        step=(0,25,0,30,'ground_x_z')
        self.assertEqual(baseline.apply(step,.1)[3],30)
        out=trial.apply(step,.1)
        self.assertGreater(out[3],0)
        self.assertLess(out[3],30)
        self.assertFalse(hasattr(baseline,'demand'))
        trial.reset()
        self.assertEqual(trial.apply(step,.2),step)

    def test_invalid_constructor_values(self):
        for name, values in (('steering_position_blend_weight',(-.1,1.1,math.nan,math.inf)),
            ('position_gain',(-1,math.nan)),('position_dead_cm',(-1,math.inf)),
            ('position_lookahead_cm',(0,-1,math.nan)),('position_max_deg',(-1,math.inf))):
            for value in values:
                with self.subTest(name=name,value=value), self.assertRaises(ValueError):
                    self.controller(**{name:value})

class BlendCliTests(unittest.TestCase):
    def parse(self,*flags):
        import run_policy_vision
        with patch.dict(os.environ,{},clear=True),patch('sys.argv',['vision',*flags]):
            return run_policy_vision.parse_args()

    def test_defaults_and_explicit_values(self):
        args=self.parse()
        self.assertEqual(args.steering_position_blend_weight,.3)
        self.assertEqual((args.position_gain,args.position_dead_cm,args.position_lookahead_cm,args.position_max_deg),(1,2,50,12))
        args=self.parse('--steering-position-blend-weight','1','--position-gain','2')
        self.assertEqual(args.steering_position_blend_weight,1)
        self.assertEqual(args.position_gain,2)

    def test_invalid_cli_values_fail_before_camera(self):
        for value in ('nan','inf','-0.1','1.01'):
            with self.subTest(value=value),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:
                self.parse('--steering-position-blend-weight',value)
            self.assertEqual(caught.exception.code,2)

    def test_environment_weight_uses_parser_validation_and_cli_override(self):
        import run_policy_vision
        with patch.dict(os.environ, {'STEERING_POSITION_BLEND_WEIGHT':'bad'}):
            with patch('sys.argv',['vision']), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    run_policy_vision.parse_args()
                self.assertEqual(caught.exception.code,2)
            with patch('sys.argv',['vision','--steering-position-blend-weight','.8']):
                self.assertEqual(run_policy_vision.parse_args().steering_position_blend_weight,.8)
        with patch.dict(os.environ, {'STEERING_POSITION_BLEND_WEIGHT':'.7'}), patch('sys.argv',['vision']):
            self.assertEqual(run_policy_vision.parse_args().steering_position_blend_weight,.7)

    def test_shadow_actual_stays_baseline_and_trial_logs_blend(self):
        args=self.parse('--steering-filter-mode','shadow','--steering-filter-algorithm','none')
        c=configured_controller(HeadingSteeringController,SteeringController(),dict(angle_wz_table=TABLE),args)
        # Match main entrypoint options, which must pass the editable weight.
        c.command(detection(0,near=-6,z=25),1,.1)
        self.assertEqual(c.diagnostics['steering_applied_wz'],.3)
        self.assertEqual(c.diagnostics['steering_filter_shadow_steering_position_blend_weight'],.3)
        self.assertNotIn('steering_blended_demand_deg',c.diagnostics)

if __name__=='__main__': unittest.main()
