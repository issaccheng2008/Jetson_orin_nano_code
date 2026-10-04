"""Targeted independent checks of final command-level source and CSV restoration."""
import importlib.util
import json
import math
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
SNAP=HERE/"source_final"

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

C=load(SNAP/"heading_steering.py","verified_level_controller").HeadingSteeringController
I=load(SNAP/"policy_bridge.py","verified_level_inner").SteeringController
restore=load(HERE/"replay_cached_levels.py","cache_restore").scalar

def debug(near,heading=0):
    return dict(fused_err_cm=near,base_err_cm=near,near_error_cm=near,near_z_cm=25,
        angle_err_deg=heading,lost_frames=0,measurement_valid=True,measurement_stale=False,
        heading_control_deg=heading,heading_control_valid=True)

def controller():return C(I(vx=.5))

def seed(c,current,demands):
    c._clock=.5;c._started=0;c._command=(.5,current);c.inner.hold=c._command
    c._samples=[((i+1)/10,d,0.,-50*math.tan(math.radians(d))) for i,d in enumerate(demands)]

class LevelChecks(unittest.TestCase):
    def test_cached_scalar_types_are_restored(self):
        self.assertIs(restore("False"),False);self.assertIs(restore("True"),True)
        self.assertIsNone(restore(""));self.assertEqual(restore("ground_x_z"),"ground_x_z")
        self.assertIsInstance(restore("0"),int);self.assertAlmostEqual(restore("0.0499"),.0499)

    def test_weak_boundary_braking_is_release_after_predicted_reentry(self):
        c=controller();seed(c,-.4,[-17,-16,-15,-14,-13])
        output=c.command(debug(10),1,.1)
        self.assertEqual(c.diagnostics["steering_decision"],"left_corridor")
        self.assertEqual(output,(.5,0.))
        self.assertTrue(c.diagnostics["steering_braked"])
        self.assertLessEqual(abs(c.diagnostics["steering_predicted_demand_deg"]),math.degrees(math.atan2(8,50)))

    def test_outside_forecast_restores_strong_boundary_not_weak_point_one(self):
        c=controller();seed(c,-.1,[-26,-25,-24,-23,-22])
        output=c.command(debug(20),1,.1)
        self.assertEqual(output,(.5,-.4))
        self.assertFalse(c.diagnostics["steering_braked"])

    def test_tiny_inward_heading_cannot_exempt_outside_corridor(self):
        for near,heading in ((-20,-.001),(-8.1,-.001),(8.1,.001),(20,.001)):
            with self.subTest(near=near,heading=heading):
                c=controller();output=c.command(debug(near,heading),1,.05)
                self.assertGreaterEqual(abs(output[1]),.4)
        for near,heading in ((-20,-30),(20,30)):
            with self.subTest(near=near,heading=heading):
                c=controller();self.assertEqual(c.command(debug(near,heading),1,.05),(.5,0.))

    def test_heading_crossing_zero_alone_does_not_stop_outside_target(self):
        c=controller();c._clock=.5;c._started=0;c._command=(.5,.4);c.inner.hold=c._command
        c._samples=[]
        for i,angle in enumerate((4,3,2,1,0)):
            demand=-math.degrees(math.atan2(-20-25*math.tan(math.radians(angle)),50))
            c._samples.append(((i+1)/10,demand,angle,-20.))
        self.assertEqual(c.command(debug(-20,-1),1,.1),(.5,.4))

    def test_normal_levels_and_half_second_hold_across_fps(self):
        for fps in (10,20,50):
            with self.subTest(fps=fps):
                c=controller();previous=None;started=0;changes=0
                for frame in range(fps*5):
                    output=c.command(debug(30 if (frame//3)%2 else -30),1,1/fps)
                    self.assertIn(output[1],(-.5,-.4,-.1,0.,.4,.5))
                    if previous is not None and output!=previous:
                        self.assertGreaterEqual((frame-started)/fps,.5-1e-9)
                        changes+=1;started=frame
                    previous=output
                self.assertGreater(changes,1)

if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LevelChecks))
    output=dict(tests_run=result.testsRun,successful=result.wasSuccessful(),
        failures=len(result.failures),errors=len(result.errors),
        source_manifest=json.loads((SNAP/"manifest.json").read_text()))
    (HERE/"replay_final/mechanism_checks.json").write_text(json.dumps(output,indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
