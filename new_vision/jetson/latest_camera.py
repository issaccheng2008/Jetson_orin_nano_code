"""One capture owner, one replaceable frame slot: never queue stale observations."""
import threading
import time


class LatestCamera:
    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.sequence = self.consumed = self.dropped = 0
        self.failed = False
        self.error = ''
        self.previous_return = None
        self.period_count = 0
        self.period_mean = self.period_m2 = 0.
        self.first_return = None
        self.thread = threading.Thread(target=self._run, name='camera-capture', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                ok, frame = self.capture.read()
                returned = time.monotonic()
                with self.condition:
                    if not ok or frame is None:
                        self.failed = True
                        self.latest = None
                        self.condition.notify_all()
                    else:
                        self.sequence += 1
                        if self.first_return is None:
                            self.first_return = returned
                        if self.previous_return is not None:
                            period_ms = (returned-self.previous_return)*1000
                            self.period_count += 1
                            delta = period_ms-self.period_mean
                            self.period_mean += delta/self.period_count
                            self.period_m2 += delta*(period_ms-self.period_mean)
                        if self.latest is not None:
                            self.dropped += 1
                        meta = dict(camera_sequence=self.sequence,
                                    camera_capture_started_s=started,
                                    camera_capture_returned_s=returned,
                                    camera_capture_read_ms=(returned-started)*1000,
                                    camera_capture_period_ms=(None if self.previous_return is None
                                        else (returned-self.previous_return)*1000),
                                    camera_frames_overwritten=self.dropped)
                        meta.update(camera_capture_period_count=self.period_count,
                                    camera_capture_period_mean_ms=self.period_mean,
                                    camera_capture_period_std_ms=(self.period_m2/max(1,self.period_count))**.5,
                                    camera_capture_elapsed_s=returned-self.first_return)
                        self.previous_return = returned
                        self.latest = (frame.copy(), meta)
                        self.failed = False
                        self.condition.notify_all()
                if not ok or frame is None:
                    self.stop.wait(.02)
        except Exception as exc:
            with self.condition:
                self.error = f'{type(exc).__name__}: {exc}'
                self.failed = True
                self.latest = None
                self.condition.notify_all()
        finally:
            self.capture.release()  # Never race release against a read in another thread.

    def read(self, timeout=.25):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.latest is None and not self.failed and not self.stop.is_set():
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    return False, None, {}
                self.condition.wait(remaining)
            if self.latest is None:
                return False, None, {}
            frame, meta = self.latest
            self.latest = None
            now = time.monotonic()
            # Timestamp is app read completion, not the sensor exposure time.
            meta['camera_frame_age_ms'] = (now-meta['camera_capture_returned_s'])*1000
            meta['camera_frames_skipped'] = max(0,meta['camera_sequence']-self.consumed-1)
            self.consumed = meta['camera_sequence']
            if now-meta['camera_capture_returned_s'] > timeout:
                return False, None, meta
            return True, frame, meta

    def close(self, timeout=1.):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout)
        return not self.thread.is_alive()
