"""Teaching regressions with a fake 50 Hz STM32 and no powered hardware."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import (COMMAND_ENABLE, COMMAND_ESTOP, COMMAND_TEACHING,
                      STATE_ENCODERS_VALID, STATE_FAULT, STATE_TEACHING_READY, StatePacket,
                      FrameDecoder, pack_command, CommandPacket, CanDiagnostics,
                      STATE_MOTORS_ENABLED)
from teaching import TeachingController, TeachingStore, Trajectory, NAMES


class FakeLink:
    state_queue_drops = 0

    def __init__(self):
        self.frames, self.sent = [], []
        self.last_command_sequence = 0

    def drain_states(self):
        frames, self.frames = self.frames, []
        return frames

    def send_command(self, *args, **kwargs):
        self.sent.append(args)
        self.last_command_sequence = (self.last_command_sequence + 1)&0xffff
        return True

    def close(self):
        pass


class TeachingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = 100.0
        self.link = FakeLink()
        self.store = TeachingStore(self.temp.name)
        self.control = TeachingController(self.link, self.store, enable_motors=True, clock=lambda:self.now)
        self.addCleanup(self.control._stop_recording)
        self.seq = 0
        self.timestamp_base = 0

    def frame(self, value=0., flags=STATE_ENCODERS_VALID | STATE_TEACHING_READY | STATE_MOTORS_ENABLED,
              *, timestamp=None, diagnostics=True, ages=None, applied_mode=None):
        self.seq += 1
        mode = 0
        if self.link.sent:
            last = self.link.sent[-1]
            mode = 0 if not last[-1]&COMMAND_ENABLE or last[-1]&COMMAND_ESTOP else 1 if last[2]==last[3]==0 else 2
        diag = CanDiagnostics(tuple(ages or [0]*12), (self.seq,)*12,
                              applied_command_sequence=self.link.last_command_sequence,
                              applied_gain_mode=mode if applied_mode is None else applied_mode)
        state = StatePacket(self.seq & 0xffff,
                            timestamp if timestamp is not None else (self.timestamp_base+self.seq*20000)&0xffffffff,
                            np.full(12, value), np.zeros(12), np.zeros(3), np.zeros(3),
                            np.array([1,0,0,0]), flags, len(self.link.sent), self.seq,
                            diag if diagnostics else None)
        self.link.frames.append((state, self.now))
        return state

    def tick(self, value=0., step=.02, **kwargs):
        self.now += step
        self.frame(value, **kwargs)
        self.control.tick()

    def free(self):
        self.tick()
        self.control.execute("release")
        for _ in range(6):
            self.tick()

    def test_monitor_never_sends_even_with_enable_argument(self):
        self.tick()
        self.assertEqual(self.link.sent, [])

    def test_explicit_emergency_stop_also_works_before_first_release(self):
        self.tick()
        self.control.enable_motors=False
        self.control.execute("estop")
        self.tick()
        self.assertEqual(self.link.sent[-1][-1], COMMAND_ESTOP | COMMAND_TEACHING)
        self.assertFalse(self.link.sent[-1][-1] & COMMAND_ENABLE)

    def test_zero_gains_echo_latest_feedback_without_clamping(self):
        self.free()
        self.tick(1.0)
        _, q, kp, kd, flags = self.link.sent[-1]
        np.testing.assert_allclose(q, 1.)
        self.assertEqual((kp,kd,flags), (0,0,COMMAND_ENABLE | COMMAND_TEACHING))

    def test_hold_captures_feedback_not_last_command(self):
        self.free()
        self.tick(.1)
        self.control.execute("hold")
        self.tick(.11)
        _, q, kp, kd, flags = self.link.sent[-1]
        np.testing.assert_allclose(q,.1)
        self.assertEqual((kp,kd), (1,1))
        self.tick(.12)
        np.testing.assert_allclose(self.link.sent[-1][1], .1)

    def test_no_enable_and_old_firmware_refused(self):
        self.tick()
        self.control.enable_motors=False
        with self.assertRaisesRegex(ValueError,"enable-motors"):
            self.control.execute("release")
        self.control.enable_motors=True
        self.tick(flags=STATE_ENCODERS_VALID)
        with self.assertRaisesRegex(ValueError,"固件"):
            self.control.execute("release")
        self.assertEqual(self.link.sent,[])

    def test_record_all_received_frames_once_and_ignore_prestart(self):
        self.free()
        self.frame(.01)  # queued before record boundary
        self.now += .001
        self.control.execute("start-recording")
        self.now += .02
        a=self.frame(.02)
        self.now += .02
        b=self.frame(.03)
        self.link.frames.append((b,self.now))  # repeated state is not a new sample
        self.control.tick()
        self.control.execute("stop-recording")
        name=self.control.record_name
        lines=[json.loads(s) for s in self.store.path(name).read_text().splitlines()]
        self.assertEqual(len(lines),3)
        self.assertEqual(lines[0]["joint_names"], NAMES)
        self.assertEqual([s["sequence"] for s in lines[1:]], [a.sequence,b.sequence])
        times, poses=self.store.recording(name)
        np.testing.assert_allclose(times,[0,.02])
        np.testing.assert_allclose(poses[-1],.03)

    def test_timestamp_wrap_and_gap_accounting(self):
        self.timestamp_base = 0xffffd8f0 - 8*20000
        self.free()
        self.control.execute("start-recording")
        self.tick(timestamp=0xffffd8f0)
        self.tick(timestamp=(0xffffd8f0+40000)&0xffffffff, step=.04)
        self.control.execute("stop-recording")
        self.assertEqual(self.control.record_missing,1)
        self.assertAlmostEqual(self.store.recording(self.control.record_name)[0][-1],.04)

    def test_checkpoint_persists_immediately_and_duration_edit_survives_restart(self):
        self.free()
        self.tick(.05)
        self.control.execute("checkpoint",dict(label="抬脚"))
        name=self.control.checkpoint_name
        np.testing.assert_allclose(self.store.checkpoints(name)["checkpoints"][0]["position_rad"],.05)
        self.tick(.1)
        self.control.execute("checkpoint")
        points=self.store.checkpoints(name)["checkpoints"]
        points[0]["duration_s"]=3.25
        self.control.execute("save-checkpoints",dict(name=name,checkpoints=points))
        other=TeachingStore(self.temp.name)
        self.assertEqual(other.checkpoints(name)["checkpoints"][0]["duration_s"],3.25)

    def test_linear_shared_time_for_all_joints(self):
        end=np.arange(12)*.01
        trajectory=Trajectory([0,2,5],[np.zeros(12),end,end*2])
        np.testing.assert_allclose(trajectory.at(1),end*.5)
        np.testing.assert_allclose(trajectory.at(3.5),end*1.5)
        np.testing.assert_allclose(trajectory.at(100),end*2)

    def test_playback_approach_complete_and_stop_holds_actual(self):
        self.free()
        points=[dict(label="A",position_rad=[.02]*12,duration_s=.2),
                dict(label="B",position_rad=[.06]*12,duration_s=2)]
        self.store.save_checkpoints("test.json",points)
        self.control.execute("play-checkpoints",dict(name="test.json",approach_s=.2))
        start=self.now
        for _ in range(21):
            self.tick(self.control.target[0])
        self.assertEqual(self.control.mode,"hold")
        np.testing.assert_allclose(self.control.target,.06)
        self.tick(.05)
        self.control.execute("stop-playback")
        self.tick(.05)
        np.testing.assert_allclose(self.control.target,.05)

    def test_log_playback_speed_and_completion(self):
        self.free()
        self.control.execute("start-recording")
        for i in range(6):
            self.tick(.002*i)
        self.control.execute("stop-recording")
        self.control.execute("play-log",dict(name=self.control.record_name,approach_s=.2,speed=.5))
        self.assertAlmostEqual(self.control.trajectory.times[-1],.4)
        for _ in range(21):
            self.tick(self.control.target[0])
        self.assertEqual(self.control.mode,"hold")
        np.testing.assert_allclose(self.control.target,.01)

    def test_refuse_unsafe_trajectory_speed_limits_and_nan(self):
        with self.assertRaisesRegex(ValueError,"速度"):
            Trajectory([0,.02],[np.zeros(12),np.full(12,.1)]).validate(3)
        with self.assertRaisesRegex(ValueError,"限位"):
            Trajectory([0,10],[np.zeros(12),np.full(12,10)]).validate(3)
        with self.assertRaises(ValueError):
            Trajectory([0,1],[np.zeros(12),np.full(12,np.nan)])
        for duration in (0, -.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.store.save_checkpoints('bad.json',[dict(position_rad=[0]*12,duration_s=duration)])

    def test_latched_fault_stale_feedback_no_automatic_resume(self):
        self.free()
        self.now+=.12
        with self.assertRaises(ValueError) as caught:
            self.control.tick()
        self.control.fail(caught.exception)
        self.assertEqual(self.control.mode,"fault")
        self.assertEqual(self.link.sent[-1][-1],COMMAND_ESTOP | COMMAND_TEACHING)
        self.tick()
        with self.assertRaisesRegex(ValueError,"锁存"):
            self.control.execute("release")
        self.control.execute("reset")
        self.assertEqual(self.control.mode,"disabled")
        self.control.execute("release")
        self.tick()
        self.assertEqual(self.control.mode,"free")

    def test_frozen_timestamp_is_not_fresh(self):
        self.free()
        stamp=self.control.state.timestamp_us
        for _ in range(4):
            self.tick(timestamp=stamp)
        with self.assertRaisesRegex(ValueError,"停止更新"):
            for _ in range(3):
                self.tick(timestamp=stamp)

    def test_fault_encoder_and_loop_overrun_refused(self):
        self.free()
        with self.assertRaisesRegex(ValueError,"故障"):
            self.tick(flags=STATE_FAULT | STATE_TEACHING_READY)
        self.control.fail("test")
        self.tick()
        self.control.execute("reset")
        self.control.execute("release")
        with self.assertRaisesRegex(ValueError,"循环延迟"):
            self.tick(step=.09)

    def test_record_queue_overflow_faults_and_closes_log(self):
        self.free()
        self.control.execute("start-recording")
        self.link.state_queue_drops=1
        with self.assertRaises(ValueError) as caught:
            self.tick()
        self.control.fail(caught.exception)
        self.assertIsNone(self.control.record_file)

    def test_disable_estop_never_restore_pd_until_explicit_release(self):
        self.free()
        self.control.execute("estop")
        self.tick(.02)
        self.assertEqual(self.link.sent[-1][2:4], (0,0))
        self.assertTrue(self.link.sent[-1][-1] & COMMAND_ESTOP)
        self.control.execute("disable")
        with self.assertRaisesRegex(ValueError,"锁存"):
            self.control.execute("release")
        self.control.execute("reset")
        self.tick(.02)
        self.assertFalse(self.link.sent[-1][-1] & COMMAND_ENABLE)

    def test_reboot_latches_fault_but_new_telemetry_can_be_reset(self):
        self.free()
        self.seq = 0
        with self.assertRaisesRegex(ValueError,"重启") as caught:
            self.tick()
        self.control.fail(caught.exception)
        self.tick()
        self.control.execute("reset")
        self.assertEqual(self.control.mode,"disabled")
        self.control.execute("release")
        self.tick()
        self.assertEqual(self.control.mode,"free")

    def test_file_traversal_and_wrong_joint_order_refused(self):
        for name in ('../x.json','/tmp/x.json','.hidden.json','a/b.json','a\\b.json'):
            with self.assertRaises(ValueError):
                self.store.path(name)
        self.store.save_checkpoints('test.json',[])
        path=self.store.path('test.json')
        data=json.loads(path.read_text())
        data['joint_names']=NAMES[::-1]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"坐标系"):
            self.store.checkpoints('test.json')

    def test_protocol_roundtrip_zero_and_restored_gains(self):
        for scale in (0,1):
            import config
            kp,kd=config.command_gains(scale,scale)
            packet=CommandPacket(10,100,np.full(12,.1),kp,kd,COMMAND_ENABLE|COMMAND_TEACHING)
            result=list(FrameDecoder().feed(pack_command(packet)))[0]
            np.testing.assert_allclose(result.kp,kp)
            np.testing.assert_allclose(result.kd,kd)
            self.assertEqual(result.command_flags,COMMAND_ENABLE|COMMAND_TEACHING)


    def test_usb_fresh_left_can_stale_faults_and_stops_recording(self):
        self.free()
        self.control.execute('start-recording')
        self.tick()
        with self.assertRaisesRegex(ValueError,'l_leg_pitch_joint') as caught:
            self.tick(ages=[200]*6+[0]*6)
        self.control.fail(caught.exception)
        self.assertEqual(self.control.mode,'fault')
        self.assertIsNone(self.control.record_file)
        self.assertTrue(self.link.sent[-1][-1]&COMMAND_ESTOP)

    def test_stationary_angles_with_fresh_can_are_valid(self):
        self.free()
        for _ in range(50):
            self.tick(0.)
        self.assertEqual(self.control.mode,'free')
        self.assertEqual(self.control.snapshot()['gain_confirmation'],'submitted')

    def test_release_must_be_confirmed_by_stm32_not_just_http_success(self):
        self.free()
        self.tick()
        self.control.execute('hold')
        for _ in range(6):
            self.tick(applied_mode=2)
        self.control.execute('release')
        self.assertEqual(self.control.snapshot()['gain_confirmation'],'pending')
        with self.assertRaisesRegex(ValueError,'未确认') as caught:
            for _ in range(25):
                self.tick(applied_mode=2)
        self.control.fail(caught.exception)
        self.assertEqual(self.control.mode,'fault')

    def test_legacy_firmware_cannot_silently_lock_stale_pose(self):
        self.tick(diagnostics=False)
        self.control.execute('release')
        for _ in range(6):
            self.tick(diagnostics=False)
        self.assertEqual(self.control.snapshot()['gain_confirmation'],'unavailable')
        with self.assertRaisesRegex(ValueError,'CAN'):
            self.control.execute('hold')

    def test_reconnect_does_not_restore_previous_hold_or_send_enable(self):
        self.free()
        self.control.execute('hold')
        self.control.fail('USB disconnected')
        replacement=FakeLink()
        self.control.reconnect_factory=lambda:replacement
        self.control.execute('reconnect')
        self.assertEqual(self.control.mode,'idle')
        self.assertFalse(self.control.session_started)
        self.assertIsNone(self.control.state)
        self.control.tick()
        self.assertEqual(replacement.sent,[])


if __name__=='__main__':
    unittest.main()
