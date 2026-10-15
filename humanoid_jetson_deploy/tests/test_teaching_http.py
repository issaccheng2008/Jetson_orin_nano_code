"""HTTP command boundary tests, including cross-origin command rejection."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teaching import TeachingController, TeachingStore
from teaching_console import make_handler, SimulatedLink


class TeachingHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.controller=TeachingController(SimulatedLink(),TeachingStore(self.temp.name))
        self.server=ThreadingHTTPServer(('127.0.0.1',0), make_handler(self.controller,'test-token',True))
        self.worker=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.worker.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=1)
        self.temp.cleanup()

    def post(self,action,headers):
        request=Request(self.base+'/api/command',json.dumps(dict(action=action)).encode(),headers=headers)
        return urlopen(request,timeout=3)

    def test_static_page_and_monitor_endpoint_work_without_motor_commands(self):
        with urlopen(self.base+'/',timeout=3) as r:
            self.assertIn('动作示教台',r.read().decode())
            self.assertEqual(r.headers['X-Frame-Options'],'DENY')
        with urlopen(self.base+'/api/state',timeout=3) as r:
            self.assertEqual(json.load(r)['mode'],'idle')

    def test_missing_token_and_cross_origin_rejected(self):
        for headers in ({}, {'X-Teaching-Token':'test-token','Origin':'http://evil.example'}):
            with self.assertRaises(HTTPError) as caught:
                self.post('estop',headers)
            self.assertEqual(caught.exception.code,403)
            self.assertEqual(self.controller.mode,'idle')

    def test_token_allows_local_commands_and_bad_action_returns_error(self):
        with self.post('estop',{'X-Teaching-Token':'test-token'}) as r:
            self.assertEqual(json.load(r)['mode'],'fault')
        with self.assertRaises(HTTPError) as caught:
            self.post('unknown',{'X-Teaching-Token':'test-token'})
        self.assertEqual(caught.exception.code,400)

    def test_unknown_assets_and_download_traversal_rejected(self):
        for path in ('/../config.py','/api/download?name=../config.py'):
            with self.assertRaises(HTTPError) as caught:
                urlopen(self.base+path,timeout=3)
            self.assertIn(caught.exception.code,(400,404))


if __name__=='__main__':
    unittest.main()
