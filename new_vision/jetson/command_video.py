"""Bounded, asynchronous camera recording of commands published by vision.

No motor/connector feedback is inferred here. The AVI is a timestamp-resampled
view of processed camera frames; its JSONL index identifies duplicated frames.
"""
import json
import math
from pathlib import Path
import queue
import threading
import time

import cv2


def line_lost(debug, confidence):
    reason = str(debug.get('steering_reason', ''))
    return (not bool(debug.get('measurement_valid', confidence > 0))
            or bool(debug.get('measurement_stale', False))
            or float(debug.get('lost_frames', 0) or 0) > 0
            or reason.startswith('loss_')
            or reason in ('brief_loss_hold', 'geometry_lost_yaw_zero', 'invalid_clock'))


def draw_command(image, vx, wz, lost, max_wz, elapsed_s):
    """Draw a direction symbol on the encoder's private image, never on control input."""
    h, w = image.shape[:2]
    colour = (0, 0, 255) if lost else (255, 0, 0)  # BGR: red / blue
    panel = image[max(0, h-110):h]
    panel[:] = (panel * .35).astype(image.dtype)
    scale = max(.35, min(.65, w/960))
    label = 'LINE LOST' if lost else 'TRACK'
    cv2.putText(image, f'{label}  VISION SENT vx={vx:+.2f} wz={wz:+.2f}',
                (8, h-85), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    cv2.putText(image, f't={elapsed_s:.2f}s', (8, h-60),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    start = (w//2, h-18 if vx >= 0 else h-72)
    if vx == 0 and wz == 0:
        cv2.putText(image, 'STOP', (w//2-24, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2, cv2.LINE_AA)
        cv2.line(image, (w//2-8, h-65), (w//2+8, h-49), colour, 3)
        cv2.line(image, (w//2+8, h-65), (w//2-8, h-49), colour, 3)
    else:
        # Positive published yaw means LEFT; amplitude scales the symbol's angle,
        # not an estimated trajectory or physical steering angle.
        angle = max(-1., min(1., wz/max(max_wz, 1e-6))) * math.pi/3
        end = (start[0]-round(52*math.sin(angle)),
               start[1]-round(52*math.cos(angle))*(1 if vx >= 0 else -1))
        cv2.arrowedLine(image, start, end, colour, 4, cv2.LINE_AA, tipLength=.3)
    return image


class CommandVideo:
    def __init__(self, directory, fps=10., width=960, max_wz=.5, queue_size=2):
        if not math.isfinite(fps) or not 1 <= fps <= 30:
            raise ValueError('video fps must be finite and in [1, 30]')
        if not 64 <= width <= 1920 or queue_size < 1:
            raise ValueError('video width must be in [64, 1920]; queue must be positive')
        self.directory = Path(directory)
        self.fps, self.width, self.max_wz = fps, width, max_wz
        self.dropped_samples = 0
        self.written_frames = 0
        self.error = None
        self.truncated = False
        self._shutdown_deadline = None
        self._queue = queue.Queue(maxsize=queue_size)
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name='command-video', daemon=True)
        self._thread.start()

    def submit(self, image, *, frame_id, host_time_ns, monotonic_s, vx, wz, lost):
        if self.error is not None or self._closing.is_set():
            return False
        if self._queue.full():
            self.dropped_samples += 1
            return False
        # Copy before enqueue: a display window/detector may modify the camera buffer.
        try:
            sample = (image.copy(), dict(source_frame=frame_id,
                command_host_time_ns=host_time_ns, command_monotonic_s=monotonic_s,
                vx=vx, wz=wz, line_lost=bool(lost)))
            self._queue.put_nowait(sample)
        except queue.Full:
            self.dropped_samples += 1
            return False
        except Exception as exc:
            self.error = str(exc)
            print(f'[video] recording disabled: {exc}', flush=True)
            return False
        return True

    def close(self, drain_timeout_s=2.):
        if self._shutdown_deadline is None:
            self._shutdown_deadline = time.monotonic()+max(0., drain_timeout_s)
        self._closing.set()
        # Stop resampling after the drain budget, then wait for the current codec
        # call and release/flush. Never leave a daemon writing on normal exit.
        self._thread.join()

    def _run(self):
        writer = index = None
        origin = last_time = None
        previous = None
        next_tick = 0
        dimensions = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            index = (self.directory/'frames.jsonl').open('x', encoding='utf-8', buffering=65536)
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    raw, metadata = self._queue.get(timeout=.05)
                except queue.Empty:
                    continue
                now = metadata['command_monotonic_s']
                if not math.isfinite(now) or (last_time is not None and now < last_time):
                    raise ValueError('video command clock regressed or is non-finite')
                if writer is None:
                    origin = now
                    h, w = raw.shape[:2]
                    out_w = min(self.width, w)//2*2
                    out_h = max(2, round(h*out_w/w)//2*2)
                    dimensions = (out_w, out_h)
                    writer = cv2.VideoWriter(str(self.directory/'camera_commands.avi'),
                        cv2.VideoWriter_fourcc(*'MJPG'), self.fps, dimensions)
                    if not writer.isOpened():
                        raise RuntimeError('MJPG video encoder could not open')
                # Put each observation at its first output tick, at most 1/fps late.
                # Repeat the previous image between observations instead of speeding
                # up the movie when camera processing is slow or samples are dropped.
                tick = math.ceil((now-origin)*self.fps-1e-7)
                while previous is not None and next_tick < tick:
                    self._write(writer, index, previous, next_tick)
                    next_tick += 1
                image = cv2.resize(raw, dimensions, interpolation=cv2.INTER_AREA)
                elapsed = now-origin
                draw_command(image, metadata['vx'], metadata['wz'], metadata['line_lost'],
                             self.max_wz, elapsed)
                previous = (image, metadata)
                if next_tick <= tick:
                    self._write(writer, index, previous, tick)
                    next_tick = tick+1
                last_time = now
        except Exception as exc:
            # Recording is auxiliary: codec/filesystem failures never stop motors or
            # change the published command. Expose the failure in both log and manifest.
            self.error = str(exc)
            print(f'[video] recording disabled: {exc}', flush=True)
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    self.error = self.error or str(exc)
            if index is not None:
                try:
                    index.close()
                except OSError as exc:
                    self.error = self.error or str(exc)
            try:
                manifest = dict(schema='command_video_v1', codec='MJPG', video='camera_commands.avi',
                    frame_index='frames.jsonl', fps=self.fps, dimensions=dimensions,
                    command_source='vision publish -> connector; before connector bias/model hold',
                    colour='BGR blue for tracking, red for lost line; STOP has no direction arrow',
                    clock='video_time_s + origin_monotonic_s; ceil to next tick; repeated source frames allowed',
                    origin_monotonic_s=origin, written_frames=self.written_frames,
                    dropped_samples=self.dropped_samples, truncated=self.truncated, error=self.error)
                (self.directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            except OSError as exc:
                self.error = self.error or str(exc)
                print(f'[video] cannot save manifest: {exc}', flush=True)

    def _write(self, writer, index, sample, tick):
        if (self._closing.is_set() and self._shutdown_deadline is not None
                and time.monotonic() >= self._shutdown_deadline):
            self.truncated = True
            raise RuntimeError('shutdown drain deadline reached; video tail truncated')
        image, metadata = sample
        writer.write(image)
        index.write(json.dumps(dict(video_frame=self.written_frames,
            video_time_s=tick/self.fps, **metadata), allow_nan=False)+'\n')
        self.written_frames += 1
