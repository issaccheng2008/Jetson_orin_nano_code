"""Position proxy veto survives rule order, windows and observation loss."""
import math
import unittest
from test_heading_steering import detection
from heading_steering import HeadingSteeringController
from segment_steering import SegmentSteeringController
from policy_bridge import SteeringController
from steering_recovery import RecoveryConfig
from steering_command_window import SteeringCommandWindow

class GuardTests(unittest.TestCase):
    def test_report_counterexample_and_mirror_all_sources(self):
        for cls in (HeadingSteeringController, SegmentSteeringController):
            for mode,source in [('RIGHT_ONLY','right_boundary'),('LEFT_ONLY','left_boundary'),('BOTH','ground_x_z')]:
                for sign in (-1,1):
                    c=cls(SteeringController(),recovery_config=RecoveryConfig(fallback_wz=.5))
                    d=detection(35*sign,near=10*sign,z=25,tracking_mode=mode,heading_control_source=source)
                    c.command(d,1,.1);pair=c.command(d,1,.1)
                    self.assertLess(pair[1]*sign,0)
                    self.assertEqual(c.diagnostics['steering_control_intent'],'boundary_guard')
                    self.assertFalse(c.diagnostics['foot_clearance_valid'])

    def test_window_immediate_veto_and_clean_release(self):
        w=SteeringCommandWindow(.5)
        w.update(.2,.5,0,valid=True);w.update(.2,.5,.1,valid=True)
        self.assertEqual(w.update(.2,-.3,.2,valid=True,intent='boundary_guard'),(.2,-.3))
        self.assertEqual(w.samples,[])
        self.assertEqual(w.update(.2,0,.3,valid=True),(.2,0))
        self.assertEqual(w.update(.2,0,.8,valid=True),(.2,0))
        self.assertEqual(w.update(.2,.5,.9,valid=True,intent='controlled_stop'),(0.,0.))

    def test_guard_survives_loss_and_direction_only_until_position_release(self):
        c=HeadingSteeringController(SteeringController(),recovery_config=RecoveryConfig())
        d=detection(35,near=10,z=25)
        c.command(d,1,.1);c.command(d,1,.1)
        for _ in range(20):
            self.assertEqual(c.command(detection(measurement_valid=False),0,.1),(0.,0.))
        self.assertEqual(c.command(detection(30,position_valid=False,heading_control_source='visible_line_direction'),1,.1),(0.,0.))
        c.command(detection(0,near=0,z=25),1,.1)
        self.assertNotEqual(c._guard_side,0)
        c.command(detection(0,near=0,z=25),1,.1)
        self.assertEqual(c._guard_side,0)
        c.reset();self.assertEqual(c.hold,(0.,0.))

    def test_published_history_records_window_output(self):
        c=HeadingSteeringController(SteeringController(),recovery_config=RecoveryConfig())
        c.published_feedback=True
        c.command(detection(35),1,.1)
        c.observe_published(.2,-.3)
        self.assertEqual(c._turn_history.previous[1],(.2,-.3))
        c.command(detection(35),1,.1)
        self.assertEqual(c._turn_history.previous[1],(.2,-.3))
        c.observe_published(0.,0.,external=True)
        self.assertEqual(c._command,(0.,0.));self.assertEqual(list(c._turn_history.samples),[])

    def test_curvature_near_is_not_body_position(self):
        radius=77.6
        points=[(z,5-(radius-math.sqrt(radius**2-z**2))) for z in (0,25,30)]
        self.assertEqual(points[0][1],5)
        self.assertLess(abs(points[1][1]),2)
        self.assertLess(points[2][1],0)
        c=HeadingSteeringController(SteeringController())
        c.command(detection(math.degrees(math.asin(25/radius)),near=points[1][1],z=25),1,.1)
        self.assertFalse(c.diagnostics['body_cross_track_valid'])
        self.assertFalse(c.diagnostics['foot_clearance_valid'])
