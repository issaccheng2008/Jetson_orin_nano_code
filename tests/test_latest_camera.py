import sys,unittest,queue,time,threading
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from latest_camera import LatestCamera

class Camera:
    def __init__(self):self.q=queue.Queue();self.released=threading.Event()
    def read(self):
        try:v=self.q.get(timeout=.05)
        except queue.Empty:return False,None
        if isinstance(v,Exception):raise v
        return v
    def release(self):self.released.set()

class LatestCameraTests(unittest.TestCase):
    def wait_sequence(self,r,n):
        with r.condition:
            self.assertTrue(r.condition.wait_for(lambda:r.sequence>=n,timeout=1.))
    def test_latest_only_owned_frame_and_no_duplicate(self):
        cap=Camera();r=LatestCamera(cap).start()
        try:
            im=np.full((3,4,3),7,np.uint8)
            for n in range(3):cap.q.put((True,im+n))
            self.wait_sequence(r,3)
            ok,frame,meta=r.read();self.assertTrue(ok);self.assertEqual(frame[0,0,0],9)
            self.assertEqual(meta['camera_frames_overwritten'],2)
            self.assertEqual(meta['camera_frames_skipped'],2)
            self.assertFalse(r.read(timeout=.005)[0])
        finally:self.assertTrue(r.close())
        self.assertTrue(cap.released.is_set())
    def test_failure_discards_previous_frame(self):
        cap=Camera();r=LatestCamera(cap).start()
        try:
            cap.q.put((True,np.ones((2,2,3),np.uint8)));self.wait_sequence(r,1)
            cap.q.put((False,None))
            with r.condition:self.assertTrue(r.condition.wait_for(lambda:r.failed,timeout=1.))
            self.assertFalse(r.read()[0])
        finally:r.close()
    def test_exception_releases_camera(self):
        cap=Camera();r=LatestCamera(cap).start();cap.q.put(RuntimeError('disconnected'))
        self.assertTrue(cap.released.wait(1.));self.assertFalse(r.read()[0]);self.assertIn('disconnected',r.error);r.close()
    def test_stale_slot_rejected(self):
        cap=Camera();r=LatestCamera(cap).start()
        try:
            cap.q.put((True,np.ones((2,2,3),np.uint8)));self.wait_sequence(r,1)
            with r.condition:r.latest[1]['camera_capture_returned_s']-=1.
            self.assertFalse(r.read(timeout=.1)[0])
        finally:r.close()

    def test_vision_entrypoint_uses_async_capture_and_logs_metadata(self):
        import contextlib,io,json,tempfile
        from unittest.mock import patch,Mock
        import run_policy_vision
        class PacedCamera:
            def __init__(self):self.released=False
            def isOpened(self):return True
            def get(self,prop):
                import cv2
                return 1280 if prop==cv2.CAP_PROP_FRAME_WIDTH else 720
            def read(self):
                time.sleep(.005)
                return True,np.full((720,1280,3),180,np.uint8)
            def release(self):self.released=True
        cap=PacedCamera()
        with tempfile.TemporaryDirectory() as tmp, \
             patch('sys.argv',['run_policy_vision.py','--headless','--no-shape-detect',
                   '--no-record-video','--attitude-port','0','--max-seconds','.25',
                   '--recording-root',tmp]), \
             patch('utils.open_camera',return_value=cap), \
             patch('camera_controls.apply_camera_controls',return_value={}), \
             patch.object(run_policy_vision,'ConnectorClient'), \
             patch.object(run_policy_vision.signal,'signal'), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_policy_vision.main(),0)
            rows=[json.loads(line) for p in Path(tmp).rglob('line_frames.jsonl') for line in p.read_text().splitlines()]
            self.assertTrue(rows)
            seq=[r['camera_sequence'] for r in rows]
            self.assertEqual(seq,sorted(set(seq)))
            self.assertTrue(all(r['camera_async'] for r in rows))
            self.assertTrue(all(r['camera_frame_age_ms']>=0 for r in rows))
        self.assertTrue(cap.released)
