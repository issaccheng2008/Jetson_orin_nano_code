import argparse
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

JETSON = Path(__file__).resolve().parents[1] / "new_vision" / "jetson"
sys.path.insert(0, str(JETSON))

from fit_photometric_mapping import _build_mapping, _cell_rejection, fit_session
from photometric_capture import (CalibrationError, CameraReader, CaptureStabilityConfig,
                                 V4L2Controller, capture_group, create_session_directory,
                                 assess_group_drift, wait_for_stability)
from photometric_lut import (aggregate_group_cells, apply_mapping, extract_board_cells,
                             estimate_translation, fit_monotonic_lut, read_png,
                             save_mapping, write_png)


V4L2_CONTROLS = """User Controls
 white_balance_automatic 0x0098090c (bool) : min=0 max=1 step=1 default=1 value=1
 white_balance_temperature 0x0098091a (int) : min=2800 max=6500 step=100 default=4600 value=4600
 gain 0x00980913 (int) : min=0 max=100 step=1 default=12 value=12
Camera Controls
 auto_exposure 0x009a0901 (menu) : min=0 max=2 step=1 default=2 value=2
 0: Manual Mode
 1: Aperture Priority Mode
 2: Fully Auto Mode
 exposure_time_absolute 0x009a0902 (int) : min=1 max=2000 step=1 default=100 value=100
"""


class FakeV4L2:
    def __init__(self, wrong_control=None):
        self.values = {"white_balance_automatic": 1,
                       "white_balance_temperature": 4600,
                       "gain": 12, "auto_exposure": 2,
                       "exposure_time_absolute": 100}
        self.calls = []
        self.wrong_control = wrong_control

    def __call__(self, command, **kwargs):
        flag = command[-1]
        self.calls.append(flag)
        if flag == "--list-ctrls-menus":
            return subprocess.CompletedProcess(command, 0, V4L2_CONTROLS, "")
        if flag.startswith("--set-ctrl="):
            name, raw = flag[len("--set-ctrl="):].split("=", 1)
            value = int(raw)
            self.values[name] = value + 1 if name == self.wrong_control else value
            return subprocess.CompletedProcess(command, 0, "", "")
        name = flag[len("--get-ctrl="):]
        return subprocess.CompletedProcess(command, 0, f"{name}: {self.values[name]}\n", "")


class CaptureControllerStub:
    initial = {"gain": 10}

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def apply_group(self, mode, exposure, wb):
        self.calls += 1
        if self.fail:
            raise CalibrationError("mock control readback mismatch")
        return {"mode": mode, "verified": True,
                "operations": [{"control": "auto_exposure"}]}

    def _find(self, aliases):
        return None

    def read_all(self):
        return {"gain": 10}


class FreshFakeReader:
    def __init__(self, frame):
        self.frame = frame
        self.sequence = 0

    def read_after(self, sequence, timeout=1.0):
        self.sequence = max(self.sequence + 1, sequence + 1)
        return self.sequence, self.frame.copy(), time.monotonic()


class PacedCapture:
    def __init__(self):
        self.counter = 0
        self.released = False

    def read(self):
        time.sleep(.002)
        self.counter += 1
        return True, np.full((8,10,3),self.counter%255,np.uint8)

    def release(self):
        self.released = True


class PhotometricCalibrationToolsTests(unittest.TestCase):
    def test_png_round_trip_preserves_gray_bgr_and_bgra(self):
        rng = np.random.default_rng(91)
        images = [rng.integers(0,256,(13,17),dtype=np.uint8),
                  rng.integers(0,256,(13,17,3),dtype=np.uint8),
                  rng.integers(0,256,(13,17,4),dtype=np.uint8)]
        with tempfile.TemporaryDirectory() as tmp:
            for i, image in enumerate(images):
                path = Path(tmp)/f"{i}.png"
                write_png(path,image)
                np.testing.assert_array_equal(read_png(path),image)

    def test_lut_is_monotonic_handles_duplicate_inputs_and_rejects_outlier(self):
        x = np.asarray([10,30,50,70,90,90,110,130,150,170,190,210,230],float)
        y = 20 + .8*x
        y[5] += 1.0  # repeated gray level from the second physical row
        y[7] = 245.0 # isolated bad patch
        lut, details = fit_monotonic_lut(x,y,max_slope=2.0)
        self.assertEqual(lut.dtype,np.uint8)
        self.assertEqual(lut.shape,(256,))
        self.assertTrue(np.all(np.diff(lut.astype(np.int16)) >= 0))
        self.assertLessEqual(int(np.max(np.diff(lut.astype(np.int16)))),2)
        self.assertTrue(details["outliers"])
        self.assertEqual(details["unique_input_groups"],12)
        self.assertFalse(details["conflicts"])

    def test_near_duplicate_input_conflict_is_reported(self):
        _, details = fit_monotonic_lut([80,81,120,160,200],
                                       [50,130,150,180,220], conflict_limit=16)
        self.assertTrue(details["conflicts"])
        self.assertGreater(details["conflicts"][0]["spread"],16)

    def test_apply_mapping_preserves_shape_dtype_and_alpha(self):
        lut_bgr = np.tile(np.arange(256,dtype=np.uint8),(3,1))
        lut_bgr[0] = np.clip(np.arange(256)+2,0,255).astype(np.uint8)
        lut_gray = np.arange(256,dtype=np.uint8)[::-1].copy()
        mapping = {"lut_bgr":lut_bgr,"lut_gray":lut_gray}
        gray = np.arange(24,dtype=np.uint8).reshape(4,6)
        mapped_gray = apply_mapping(gray,mapping)
        self.assertEqual(mapped_gray.shape,gray.shape)
        self.assertEqual(mapped_gray.dtype,np.uint8)
        np.testing.assert_array_equal(mapped_gray,lut_gray[gray])
        bgr = np.stack((gray,gray+1,gray+2),axis=2)
        mapped_bgr = apply_mapping(bgr,mapping)
        self.assertEqual(mapped_bgr.shape,bgr.shape)
        self.assertEqual(mapped_bgr.dtype,np.uint8)
        np.testing.assert_array_equal(mapped_bgr[:,:,0],lut_bgr[0,bgr[:,:,0]])
        bgra = np.concatenate((bgr,np.full((4,6,1),77,np.uint8)),axis=2)
        mapped_bgra = apply_mapping(bgra,mapping)
        self.assertEqual(mapped_bgra.shape,bgra.shape)
        np.testing.assert_array_equal(mapped_bgra[:,:,3],bgra[:,:,3])

    def test_sample_central_half_of_both_reversed_gray_rows(self):
        n, cell, margin = 16, 20, 10
        image = np.full((2*cell+2*margin,n*cell+2*margin,3),100,np.uint8)
        for row in range(2):
            for col in range(n):
                level = col if row == 0 else n-1-col
                value = 20 + 10*level
                y0, x0 = margin+row*cell, margin+col*cell
                image[y0:y0+cell,x0:x0+cell] = value
        corners = [[margin,margin],[margin+n*cell-1,margin],
                   [margin+n*cell-1,margin+2*cell-1],[margin,margin+2*cell-1]]
        cells, warped, _ = extract_board_cells(image,corners,n,cell,.5)
        self.assertEqual(warped.shape,(2*cell,n*cell,3))
        self.assertEqual(len(cells),2*n)
        upper = {c["gray_level_index"]:c["gray_median"] for c in cells if c["row"]==0}
        lower = {c["gray_level_index"]:c["gray_median"] for c in cells if c["row"]==1}
        for level in range(n):
            self.assertAlmostEqual(upper[level],20+10*level,delta=1)
            self.assertAlmostEqual(lower[level],20+10*level,delta=1)

    def test_saturated_and_unstable_cells_are_excluded(self):
        auto, fixed = [], []
        for i, value in enumerate([30,55,80,105,130,155,180,205]):
            base = {"row":i//4,"col":i%4,"gray_level_index":i,
                    "bgr_median":[value]*3,"gray_median":float(value),
                    "bgr_group_spread_p90_p10":[1.,1.,1.],
                    "gray_group_spread_p90_p10":1.,"saturated_fraction":0.0}
            base["bgr_patch_std_median"]=[0.,0.,0.]
            base["gray_patch_std_median"]=0.
            f, a = dict(base), dict(base)
            a["bgr_median"]=[min(250,value+10)]*3
            a["gray_median"]=float(min(250,value+10))
            if i == 0:
                f["saturated_fraction"] = .8
            if i == 1:
                a["gray_group_spread_p90_p10"] = 12.
            fixed.append(f); auto.append(a)
        _,_,paired,rejected,_,_,_ = _build_mapping(
            None,auto,fixed,8,.4,8.,4.,16.)
        self.assertEqual(len(rejected),2)
        self.assertEqual(sum(row["used_for_fit"] for row in paired),6)
        reasons = {reason for row in rejected for reason in row["reasons"]}
        self.assertIn("fixed_saturated",reasons)
        self.assertIn("auto_unstable",reasons)

    def test_v4l2_modes_are_selected_from_menu_and_fixed_values_read_back(self):
        fake = FakeV4L2()
        controller = V4L2Controller("/dev/video2",runner=fake,platform="linux")
        controller.query_initial_state()
        report = controller.apply_group("fixed",5.0)
        self.assertTrue(report["verified"])
        writes = [s for s in fake.calls if s.startswith("--set-ctrl=")]
        self.assertEqual(writes[:3],["--set-ctrl=auto_exposure=0",
                                     "--set-ctrl=white_balance_automatic=1",
                                     "--set-ctrl=exposure_time_absolute=50"])
        auto = controller.apply_group("auto")
        self.assertTrue(auto["verified"])
        self.assertIn("--set-ctrl=auto_exposure=2",fake.calls)
        self.assertIn("--set-ctrl=white_balance_automatic=1",fake.calls)
        fake.values["gain"] = 13  # emulate an auto-control side effect
        self.assertEqual(controller.restore_initial()["status"],"restored")
        self.assertEqual(fake.values["gain"],12)
        self.assertIn("--set-ctrl=gain=12",fake.calls)

    def test_driver_readback_mismatch_prevents_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeV4L2(wrong_control="auto_exposure")
            controller = V4L2Controller("/dev/video0",runner=fake,platform="linux")
            controller.query_initial_state()
            reader = FreshFakeReader(np.full((32,40,3),100,np.uint8))
            result = capture_group("auto",reader,controller,Path(tmp)/"auto",
                                   CaptureStabilityConfig())
            self.assertEqual(result["status"],"control_failed")
            self.assertEqual(list((Path(tmp)/"auto").glob("*.png")),[])

    def test_auto_control_drift_disqualifies_the_capture_group(self):
        cfg = CaptureStabilityConfig()
        samples = []
        for i in range(10):
            samples.append({"roi_metrics":{"brightness_gray_median":100.,
                                            "r_over_g":1.,"b_over_g":1.},
                            "controls":{"exposure_time_absolute":100+i*10,
                                        "white_balance_temperature":4500+i*30}})
        result = assess_group_drift(samples,cfg,"auto", "exposure_time_absolute",
                                    "white_balance_temperature")
        self.assertFalse(result["qualified"])
        self.assertIn("auto_exposure_parameter_drift",result["reasons"])
        self.assertIn("auto_white_balance_parameter_drift",result["reasons"])

    def test_control_failure_never_writes_calibration_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = FreshFakeReader(np.full((32,40,3),100,np.uint8))
            controller = CaptureControllerStub(fail=True)
            result = capture_group("fixed",reader,controller,Path(tmp)/"fixed",
                                   CaptureStabilityConfig())
            self.assertEqual(result["status"],"control_failed")
            self.assertEqual(list((Path(tmp)/"fixed").glob("*.png")),[])

    def test_stability_timeout_is_recorded_but_advisory(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = FreshFakeReader(np.full((32,40,3),100,np.uint8))
            controller = CaptureControllerStub()
            config = CaptureStabilityConfig()
            with patch("photometric_capture.wait_for_stability",
                       return_value={"stable":False,"reason":"stability_timeout"}):
                result = capture_group("auto",reader,controller,Path(tmp)/"auto",config)
            self.assertEqual(result["status"],"complete")
            self.assertIn("stability_timeout",result["warnings"])
            self.assertFalse(result["quality_qualified"])
            self.assertEqual(len(result["frames"]),config.count)
            self.assertEqual(len(list((Path(tmp)/"auto").glob("*.png"))),config.count)

    def test_non_timeout_stability_failure_aborts_before_any_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = FreshFakeReader(np.full((32,40,3),100,np.uint8))
            controller = CaptureControllerStub()
            config = CaptureStabilityConfig()
            with patch("photometric_capture.wait_for_stability",
                       return_value={"stable":False,"reason":"camera_stalled"}):
                result = capture_group("auto",reader,controller,Path(tmp)/"auto",config)
            self.assertEqual(result["status"],"stability_failed")
            self.assertEqual(list((Path(tmp)/"auto").glob("*.png")),[])

    def test_new_capture_session_never_reuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = create_session_directory(tmp,"same-session")
            (first/"sentinel").write_text("keep")
            second = create_session_directory(tmp,"same-session")
            self.assertNotEqual(first,second)
            self.assertEqual((first/"sentinel").read_text(),"keep")
            self.assertTrue((second/"auto").is_dir())

    def test_board_translation_is_detected_instead_of_silently_reusing_corners(self):
        reference = np.full((160,340,3),100,np.uint8)
        for y in range(20,140):
            for x in range(20,320):
                reference[y,x] = 30+((x//20)%2)*150+((y//20)%2)*50
        moved = np.zeros_like(reference)
        moved[:,5:] = reference[:,:-5]
        corners = [[20,20],[319,20],[319,139],[20,139]]
        same = estimate_translation(reference,reference,corners)
        shift = estimate_translation(reference,moved,corners)
        self.assertFalse(same["requires_reselection"])
        self.assertTrue(shift["requires_reselection"])
        self.assertEqual(abs(shift["dx_px"]),5)

    def test_capture_saves_exactly_ten_fresh_pngs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"auto"
            reader = FreshFakeReader(np.full((32,40,3),120,np.uint8))
            controller = CaptureControllerStub()
            config = CaptureStabilityConfig(stability_window_seconds=.1,
                                            stability_timeout_seconds=5,
                                            sample_interval_seconds=.001)
            with patch("photometric_capture.wait_for_stability",
                       return_value={"stable":True,"last_frame_sequence":0}):
                result = capture_group("auto",reader,controller,directory,config)
            self.assertEqual(result["status"],"complete")
            self.assertEqual([p.name for p in sorted(directory.glob("*.png"))],
                             [f"{i:03d}.png" for i in range(10)])
            self.assertEqual(read_png(directory/"000.png")[0,0].tolist(),[120,120,120])
            before = (directory/"000.png").read_bytes()
            with self.assertRaises(CalibrationError):
                capture_group("auto",reader,controller,directory,config)
            self.assertEqual((directory/"000.png").read_bytes(),before)

    def test_background_camera_keeps_advancing_while_terminal_is_idle(self):
        cap = PacedCapture()
        reader = CameraReader(cap).start()
        try:
            seq1, frame1, _ = reader.read_after(-1,timeout=1)
            time.sleep(.03)  # stand in for time spent waiting at the terminal prompt
            seq2, frame2, _ = reader.read_after(seq1,timeout=1)
            self.assertGreater(seq2,seq1)
            self.assertNotEqual(int(frame1[0,0,0]),int(frame2[0,0,0]))
        finally:
            self.assertTrue(reader.close())
        self.assertTrue(cap.released)

    def test_stability_wait_requires_warmup_and_stable_roi_window(self):
        clock = [0.0]
        stable_frame = np.full((40,60,3),100,np.uint8)
        class VirtualReader:
            def __init__(self,unstable=False):
                self.sequence=0; self.unstable=unstable
            def read_after(self,sequence,timeout=1.0):
                self.sequence=max(self.sequence+1,sequence+1)
                clock[0]+=.16
                value=100 if not self.unstable or self.sequence%2 else 180
                return self.sequence,np.full_like(stable_frame,value),clock[0]
        config = CaptureStabilityConfig(warmup_seconds=3.,stability_window_seconds=.6,
                                        stability_timeout_seconds=4.,check_interval_seconds=.15)
        with patch("photometric_capture.time.monotonic",side_effect=lambda: clock.__setitem__(0,clock[0]+.01) or clock[0]):
            stable = wait_for_stability(VirtualReader(),config)
        self.assertTrue(stable["stable"])
        self.assertGreaterEqual(stable["elapsed_seconds"],3.)
        clock[0]=0.
        with patch("photometric_capture.time.monotonic",side_effect=lambda: clock.__setitem__(0,clock[0]+.01) or clock[0]):
            unstable = wait_for_stability(VirtualReader(unstable=True),config)
        self.assertFalse(unstable["stable"])
        self.assertEqual(unstable["reason"],"stability_timeout")

    def test_end_to_end_offline_fit_exports_luts_and_same_scene_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root/"session"
            (session/"auto").mkdir(parents=True)
            (session/"fixed").mkdir()
            height,width,margin,cell,n = 60,340,10,20,16
            corners = [[10,10],[329,10],[329,49],[10,49]]
            for frame_index in range(10):
                fixed = np.full((height,width,3),70,np.uint8)
                auto = np.full((height,width,3),95,np.uint8)
                for row in range(2):
                    for col in range(n):
                        level = col if row==0 else n-1-col
                        src = 20+10*level
                        dst = int(round(.75*src+30))
                        y0,x0 = margin+row*cell,margin+col*cell
                        fixed[y0:y0+cell,x0:x0+cell] = src
                        auto[y0:y0+cell,x0:x0+cell] = dst
                write_png(session/"fixed"/f"{frame_index:03d}.png",fixed)
                write_png(session/"auto"/f"{frame_index:03d}.png",auto)
            capture_meta = {"session_id":"offline","groups":{
                "auto":{"status":"complete","drift":{"qualified":True}},
                "fixed":{"status":"complete","drift":{"qualified":True}}}}
            (session/"metadata.json").write_text(json.dumps(capture_meta),encoding="utf-8")
            corner_file = root/"corners.json"
            corner_file.write_text(json.dumps({"auto_corners":corners,"fixed_corners":corners}),encoding="utf-8")
            args = argparse.Namespace(
                session=str(session),corners_json=str(corner_file),no_gui=True,
                cells_per_row=16,fit_images=8,held_out_images=2,cell_px=cell,
                center_fraction=.5,max_slope=4.,max_saturation_fraction=.4,
                max_patch_spread=8.,conflict_limit=16.,max_alignment_shift_px=20,
                comparison_index=8,output_dir=None,apply_to=[],validation_mode="auto")
            output,result = fit_session(args)
            self.assertTrue((output/"lut_bgr.npy").is_file())
            self.assertTrue((output/"lut_gray.npy").is_file())
            self.assertTrue((output/"lut_max.npy").is_file())
            self.assertTrue((output/"mapping.json").is_file())
            self.assertTrue((output/"comparison_008.png").is_file())
            self.assertTrue((output/"sampling_overlay.png").is_file())
            self.assertLess(result["quality"]["gray_mae_after"],
                            result["quality"]["gray_mae_before"])
            self.assertLess(result["quality"]["max_channel_mae_after"],
                            result["quality"]["max_channel_mae_before"])
            mapping = {"lut_bgr":np.load(output/"lut_bgr.npy"),
                       "lut_gray":np.load(output/"lut_gray.npy")}
            self.assertTrue(np.all(np.diff(mapping["lut_gray"].astype(np.int16))>=0))
            self.assertTrue(np.all(np.diff(np.load(output/"lut_max.npy").astype(np.int16))>=0))


if __name__ == "__main__":
    unittest.main()

