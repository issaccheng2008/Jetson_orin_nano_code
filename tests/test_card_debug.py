"""Status and measured accuracy must not confuse weak cues with classification."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from card_debug import Statistics, scalar_debug, state, dashboard


class CardDebugTests(unittest.TestCase):
    def test_states_distinguish_unknown_from_absent(self):
        self.assertEqual(state({})[0],'NO CARD')
        self.assertEqual(state({'presence':True})[0],'PRESENCE CUE ONLY')
        self.assertEqual(state({'card_found':True})[0],'BOX FOUND / UNKNOWN')
        self.assertEqual(state({'card_found':True,'shape':'circle'})[0],'RECOGNIZED')

    def test_accuracy_includes_missed_and_unknown_frames(self):
        stats=Statistics()
        for dbg in ({'card_found':True,'shape':'circle'}, {},
                    {'presence':True}, {'card_found':True,'shape':'square'}):
            stats.observe(dbg,'circle')
        result=stats.report()
        self.assertEqual(result['frame_accuracy'],.25)
        self.assertEqual((result['verified_boxes'],result['classified']),(2,2))
        self.assertEqual(stats.confusion[('circle','none')],1)
        self.assertEqual(stats.confusion[('circle','unknown')],1)

    def test_unlabelled_does_not_invent_accuracy(self):
        stats=Statistics()
        stats.observe({},'unlabelled')
        self.assertIsNone(stats.report()['frame_accuracy'])

    def test_blank_and_detected_dashboard(self):
        stats=Statistics()
        for dbg in ({}, {'card_found':True,'shape':'triangle',
                         'quad':np.array([[10,10],[50,10],[50,50],[10,50]]),
                         'presence_box':(10,10,40,40),'appearance_gray':np.zeros((200,200),np.uint8)}):
            view=dashboard(np.zeros((720,1280,3),np.uint8),dbg,stats,'unlabelled',10.,1,False,'selective/legacy')
            self.assertEqual(view.shape,(1000,1440,3))

    def test_log_retains_geometry_not_images(self):
        result=scalar_debug({'gray':np.zeros((540,960),np.uint8),
                             'quad':np.zeros((4,2),np.float32),'score':np.float32(.5)})
        self.assertNotIn('gray',result)
        self.assertEqual(len(result['quad']),4)
        self.assertEqual(result['score'],.5)


if __name__=='__main__':
    unittest.main()
