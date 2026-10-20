"""Independent mechanism checks on exactly the source used in offline replay."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
SNAP=HERE/"source_final"
sys.path.insert(0,str(SNAP))

def load(name):
    spec=importlib.util.spec_from_file_location("verified_"+name,SNAP/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

inner_cls=load("policy_bridge").SteeringController
controller_cls=load("heading_steering").HeadingSteeringController

def debug(near=0,angle=0):
    return dict(fused_err_cm=near,base_err_cm=near,near_error_cm=near,near_z_cm=25,
        angle_err_deg=angle,lost_frames=0,measurement_valid=True,measurement_stale=False,
        heading_control_deg=angle,heading_control_valid=True)

def bearing(value):
    return debug(-50*math.tan(math.radians(value)))

class FrozenMechanismChecks(unittest.TestCase):
    def controller(self,**kwargs):
        return controller_cls(inner_cls(vx=.5,**kwargs),allow_right=True)

    def test_ground_sign_and_output_yaw_mapping(self):
        for yaw_sign in (-1,1):
            for near,angle in ((-5,0),(5,0),(0,20),(0,-20)):
                with self.subTest(yaw=yaw_sign,near=near,angle=angle):
                    c=controller_cls(inner_cls(vx=.5,yaw_sign=yaw_sign),
                        right_tolerance_deg=0,left_tolerance_deg=0,allow_right=True)
                    output=c.command(debug(near,angle),1,.05)
                    expected=-math.degrees(math.atan((near-25*math.tan(math.radians(angle)))/50))
                    self.assertAlmostEqual(c.diagnostics["steering_demand_deg"],expected,places=12)
                    self.assertGreater(output[1]*yaw_sign*expected,0)

    def test_normal_changes_hold_half_second_at_different_fps(self):
        for fps in (10,20,50):
            with self.subTest(fps=fps):
                c=self.controller();previous=None;started=0;changes=0
                for frame in range(fps*5):
                    output=c.command(bearing(50 if (frame//3)%2 else -50),1,1/fps)
                    if previous is not None and previous!=output:
                        self.assertGreaterEqual((frame-started)/fps,.5-1e-9)
                        changes+=1
                        started=frame
                    previous=output
                self.assertGreater(changes,1)

    def test_brief_loss_keeps_exact_pair_but_loss_deadline_clears_it(self):
        c=self.controller()
        moving=c.command(bearing(50),1,.05)
        missing=bearing(50);missing["heading_control_valid"]=False
        self.assertEqual(c.command(missing,1,.05),moving)
        self.assertEqual(c.command(missing,1,.21),(0.,0.))
        self.assertEqual(c.inner.hold,(0.,0.))
        restarted=c.command(bearing(-50),1,.05)
        self.assertLess(restarted[1],0)
        self.assertEqual(c.diagnostics["steering_reason"],"new_block")

    def test_stale_geometry_and_invalid_clock_interrupt_immediately(self):
        for stale in (True,False):
            with self.subTest(stale=stale):
                c=self.controller();c.command(bearing(50),1,.05)
                reading=bearing(50);reading["measurement_stale"]=stale
                output=c.command(reading,1,.01 if stale else float("nan"))
                self.assertEqual(output,(0.,0.))
                self.assertEqual(c.inner.hold,(0.,0.))

    def test_braking_cannot_increase_existing_amplitude_or_launch_new_prediction(self):
        for current in (.1,0.):
            with self.subTest(current=current):
                c=self.controller();c._clock=.5;c._started=0
                c._command=(.5,current);c.inner.hold=c._command
                c._samples=[(.1,22.4,0),(.2,22,0),(.3,21.6,0),(.4,21.2,0),(.5,20.8,0)]
                output=c.command(bearing(20.4),1,.1)
                if current:
                    self.assertTrue(c.diagnostics["steering_braked"])
                    self.assertLessEqual(abs(output[1]),current)
                else:
                    self.assertFalse(c.diagnostics["steering_braked"])
                    self.assertAlmostEqual(output[1],.22)

    def test_reverse_yaw_uses_physical_negative_model_command_cap(self):
        for yaw_sign in (-1,1):
            for demand in (-50,50):
                with self.subTest(yaw=yaw_sign,demand=demand):
                    c=self.controller(yaw_sign=yaw_sign,max_wz_right=.2)
                    output=c.command(bearing(demand),1,.05)[1]
                    self.assertAlmostEqual(abs(output),.2 if output<0 else .5)

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FrozenMechanismChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    manifest=json.loads((SNAP/"manifest.json").read_text())
    payload=dict(tests_run=result.testsRun,successful=result.wasSuccessful(),
        failures=len(result.failures),errors=len(result.errors),source_sha256=manifest["sha256"])
    (HERE/"replay_final/mechanism_checks.json").write_text(json.dumps(payload,indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
