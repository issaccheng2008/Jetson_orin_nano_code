"""Bounded, asynchronous recording of commands published by vision.

No motor/connector feedback is inferred here. The AVI is a timestamp-resampled
view of camera frames, detector lane masks or paired birdseye views; its JSONL index identifies duplicated frames.
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


def draw_command(image, vx, wz, lost, max_wz, elapsed_s, executed=None):
    """Draw a direction symbol on the encoder's private image, never on control input."""
    h, w = image.shape[:2]
    colour = (0, 0, 255) if lost else (255, 0, 0)  # BGR: red / blue
    panel = image[max(0, h-110):h]
    panel[:] = (panel * .35).astype(image.dtype)
    scale = max(.35, min(.65, w/960))
    label = 'LINE LOST' if lost else 'TRACK'
    cv2.putText(image, f'{label} t={elapsed_s:.2f}s', (8, h-92),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    columns = [(w//4, 'VISION', vx, wz)]
    status = None
    if executed is None:
        status = 'UNKNOWN'
    elif executed.get('send_result') != 'written':
        status = 'UNKNOWN (not sent)'
    elif not executed.get('enabled', False):
        status = 'MOTORS OFF'
    else:
        velocity = executed['velocity']
        columns.append((3*w//4, 'ROBOT CMD', velocity[0], velocity[2]))
        cv2.putText(image, f'{executed.get("policy_mode", "")} hold={executed.get("hold_remaining_s", 0.):.2f}s',
                    (w//2+4, h-92), cv2.FONT_HERSHEY_SIMPLEX, max(.3, scale*.8),
                    (255, 255, 255), 1, cv2.LINE_AA)
    if status:
        cv2.putText(image, f'ROBOT CMD {status}', (w//2+4, h-72),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.3, scale*.8), (180, 180, 180), 1, cv2.LINE_AA)
    for centre, name, column_vx, column_wz in columns:
        cv2.putText(image, f'{name} vx={column_vx:+.2f} wz={column_wz:+.2f}',
                    (max(4, centre-w//4+4), h-72), cv2.FONT_HERSHEY_SIMPLEX,
                    max(.3, scale*.8), colour, 1, cv2.LINE_AA)
        _draw_arrow(image, centre, column_vx, column_wz, colour, max_wz)
    return image


def _draw_arrow(image, centre, vx, wz, colour, max_wz):
    h = image.shape[0]
    start = (centre, h-12 if vx >= 0 else h-62)
    if vx == 0 and wz == 0:
        cv2.putText(image, 'STOP', (centre-24, h-22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2, cv2.LINE_AA)
        cv2.line(image, (centre-8, h-59), (centre+8, h-43), colour, 3)
        cv2.line(image, (centre+8, h-59), (centre-8, h-43), colour, 3)
    else:
        # Positive published yaw means LEFT; amplitude scales the symbol's angle,
        # not an estimated trajectory or physical steering angle.
        angle = max(-1., min(1., wz/max(max_wz, 1e-6))) * math.pi/3
        end = (start[0]-round(52*math.sin(angle)),
               start[1]-round(52*math.cos(angle))*(1 if vx >= 0 else -1))
        cv2.arrowedLine(image, start, end, colour, 4, cv2.LINE_AA, tipLength=.3)


class CommandVideo:
    def __init__(self, directory, fps=10., width=960, max_wz=.5, queue_size=2, frame_source='camera'):
        if not math.isfinite(fps) or not 1 <= fps <= 30:
            raise ValueError('video fps must be finite and in [1, 30]')
        if not 64 <= width <= 1920 or queue_size < 1:
            raise ValueError('video width must be in [64, 1920]; queue must be positive')
        if frame_source not in ('camera', 'binary', 'bird_pair'):
            raise ValueError('video source must be camera, binary or bird_pair')
        self.frame_source = frame_source
        self.directory = Path(directory)
        self.fps, self.width, self.max_wz = fps, width, max_wz
        self.dropped_samples = 0
        self.written_frames = 0
        self.error = None
        self.truncated = False
        self._shutdown_deadline = None
        self._origin_monotonic_s = None
        self._queue = queue.Queue(maxsize=queue_size)
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name='command-video', daemon=True)
        self._thread.start()

    def submit(self, image, *, frame_id, host_time_ns, monotonic_s, vx, wz, lost, executed=None):
        if self.error is not None or self._closing.is_set():
            return False
        if self._queue.full():
            self.dropped_samples += 1
            return False
        # Copy before enqueue: a display window/detector may modify the camera buffer.
        try:
            private_image = (tuple(part.copy() for part in image)
                             if self.frame_source == 'bird_pair' else image.copy())
            sample = (private_image, dict(source_frame=frame_id,
                command_host_time_ns=host_time_ns, command_monotonic_s=monotonic_s,
                vx=vx, wz=wz, line_lost=bool(lost), executed_command=(
                    None if executed is None else dict(executed, velocity=list(executed['velocity'])))))
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
        dimensions = image_dimensions = None
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
                    self._origin_monotonic_s = origin
                    h, w = (raw[0].shape[:2] if self.frame_source == 'bird_pair' else raw.shape[:2])
                    if self.frame_source == 'bird_pair':
                        w *= 2
                    out_w = min(self.width, w)//2*2
                    out_h = max(2, round(h*out_w/w)//2*2)
                    image_dimensions = (out_w, out_h)
                    dimensions = (out_w, out_h + (110 if self.frame_source in ('binary', 'bird_pair') else 0))
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
                if self.frame_source == 'bird_pair':
                    colour, mask = raw
                    if (colour.ndim != 3 or colour.shape[2] != 3 or colour.dtype != 'uint8'
                            or mask.ndim != 2 or mask.dtype != 'uint8'
                            or colour.shape[:2] != mask.shape):
                        raise ValueError('bird_pair needs matching BGR birdseye and uint8 lane mask')
                    # Resize each half independently so binary edges stay binary.
                    left_w = image_dimensions[0] // 2
                    right_w = image_dimensions[0] - left_w
                    height = image_dimensions[1]
                    left = cv2.resize(colour, (left_w, height), interpolation=cv2.INTER_AREA)
                    right = cv2.resize(mask, (right_w, height), interpolation=cv2.INTER_NEAREST)
                    raw = cv2.hconcat((left, cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)))
                if self.frame_source == 'binary':
                    # This is the detector's final candidate mask, not a new
                    # threshold of the camera image. Preserve it above the HUD.
                    if raw.ndim != 2 or raw.dtype != 'uint8':
                        raise ValueError('binary video needs the detector uint8 lane mask')
                    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                image = (raw if self.frame_source == 'bird_pair' else
                         cv2.resize(raw, image_dimensions,
                             interpolation=cv2.INTER_NEAREST if self.frame_source == 'binary' else cv2.INTER_AREA))
                if self.frame_source in ('binary', 'bird_pair'):
                    image = cv2.copyMakeBorder(image, 0, 110, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
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
                manifest = dict(schema='command_video_v2', codec='MJPG', video='camera_commands.avi',
                    frame_index='frames.jsonl', fps=self.fps, dimensions=dimensions,
                    frame_source=self.frame_source,
                    detector_debug_key=('bird_color,binary' if self.frame_source == 'bird_pair'
                                        else 'binary' if self.frame_source == 'binary' else None),
                    image_source=('left: original BGR birdseye; right: final lane candidate mask; HUD below'
                                  if self.frame_source == 'bird_pair' else
                                  'detector birdseye final lane candidate mask; white on black; HUD below'
                                  if self.frame_source == 'binary' else 'original camera BGR; HUD overlay'),
                    command_source='vision publish -> connector; before connector bias/model hold',
                    executed_command_source='policy feedback after model hold/takeovers and serial send; not measured body motion',
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
        output_time = self._origin_monotonic_s + tick/self.fps
        actual = metadata['executed_command']
        if actual is not None:
            age = output_time-float(actual['monotonic_s'])
            if not -.01 <= age <= .5+1e-7:
                actual = None
        drawn = image.copy()
        draw_command(drawn, metadata['vx'], metadata['wz'], metadata['line_lost'],
                     self.max_wz, tick/self.fps, executed=actual)
        writer.write(drawn)
        index.write(json.dumps(dict(video_frame=self.written_frames,
            video_time_s=tick/self.fps, **dict(metadata, executed_command=actual)), allow_nan=False)+'\n')
        self.written_frames += 1
