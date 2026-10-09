#!/usr/bin/env python3
"""Crop one recorded test to its start gate and plot vision/control/IMU offline."""
import argparse
import csv
import json
import math
from pathlib import Path


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except (ValueError, TypeError):
        return math.nan


def find_start(rows, mode='auto'):
    for row in rows:
        chosen = row.get('start_gate_mode') if mode == 'auto' else mode
        passed = (chosen == 'both' and row.get('qr_passed') is True and row.get('shape_passed') is True)
        button = (chosen == 'button' and row.get('button_passed') is True and row.get('start_released') is True)
        stamp = number(row.get('host_time_ns')) / 1e9
        if (passed or button) and math.isfinite(stamp):
            return stamp, 'both_valves' if passed else 'button_release'
    raise ValueError('Cannot determine start: required gate states missing or never passed. '
                     'Use --start-unix-s only with a verified start timestamp.')


def crop_data(visual, control, start):
    def crop(rows, field, scale):
        result = []
        for row in rows:
            timestamp = number(row.get(field)) / scale
            if math.isfinite(timestamp) and timestamp >= start:
                result.append(dict(row, t_s=timestamp-start))
        return sorted(result, key=lambda r: r['t_s'])
    return crop(visual, 'host_time_ns', 1e9), crop(control, 'host_unix_s', 1.)


def orientation_degrees(quaternion):
    w, x, y, z = [number(v) for v in quaternion]
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if not math.isfinite(norm) or norm == 0:
        return (math.nan,)*3
    w, x, y, z = (v/norm for v in (w, x, y, z))
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def display_orientation_degrees(quaternion):
    return tuple(v % 360 if math.isfinite(v) else v for v in orientation_degrees(quaternion))


def deviation_values(row):
    measurement = row.get('measurement', {})
    valid = row.get('body_track_deviation_valid', measurement.get('heading_control_valid', False))
    raw = (number(row.get('body_track_deviation_deg', measurement.get('heading_control_deg')))
           if valid is True else math.nan)
    filtered = number(measurement.get('steering_filter_heading_deg',
                      measurement.get('steering_filter_shadow_steering_filter_heading_deg')))
    return raw, filtered


def unique_file(root, pattern):
    paths = list(root.rglob(pattern))
    # Analysis outputs must never become inputs to a later invocation.
    paths = [p for p in paths if 'analysis' not in p.relative_to(root).parts]
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {pattern} in {root}, found {len(paths)}. Select one test folder.')
    return paths[0]


def export_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_control_columns(fields):
    required = {'host_unix_s', 'cmd_wz'}
    required.update('received_orientation_'+axis for axis in 'wxyz')
    required.update('received_'+group+'_'+axis for group in ('gyro', 'accel') for axis in 'xyz')
    missing = required - set(fields or [])
    if missing:
        raise ValueError('Control CSV missing required columns: ' + ', '.join(sorted(missing)))


def plot(visual, control, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vt = [r['t_s'] for r in visual]
    ct = [r['t_s'] for r in control]
    angles = [deviation_values(row) for row in visual]
    deviation = [a[0] for a in angles]
    filtered_deviation = [a[1] for a in angles]
    fig, left = plt.subplots(figsize=(14, 5), constrained_layout=True)
    right = left.twinx()
    left.plot(vt, deviation, color='tab:blue', label='Visual track deviation')
    left.plot(vt, filtered_deviation, color='tab:purple', label='Filtered steering heading',
              linewidth=1.6, linestyle='--')
    right.step(vt, [number(r.get('wz')) for r in visual], where='post',
               color='tab:orange', label='Visual WZ', linewidth=1.5)
    right.step(ct, [number(r.get('cmd_wz')) for r in control], where='post',
               color='tab:green', label='Executed model WZ', linestyle='--', linewidth=1.4)
    left.set(xlabel='Time after start (s)', ylabel='Track deviation (deg)')
    right.set_ylabel('Angular velocity command (rad/s)')
    left.grid(alpha=.25)
    handles, labels = left.get_legend_handles_labels()
    h, l = right.get_legend_handles_labels()
    left.legend(handles+h, labels+l, loc='upper center', bbox_to_anchor=(.5, 1.18), ncol=2)
    fig.savefig(directory/'steering.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    attitudes = [display_orientation_degrees([r.get('received_orientation_'+a) for a in 'wxyz']) for r in control]
    for i, label in enumerate(('Roll', 'Pitch', 'Yaw')):
        axes[0].plot(ct, [a[i] for a in attitudes], label=label, linewidth=1.)
    for axis in 'xyz':
        axes[1].plot(ct, [number(r.get('received_gyro_'+axis)) for r in control], label=axis.upper(), linewidth=1.)
        axes[2].plot(ct, [number(r.get('received_accel_'+axis)) for r in control], label=axis.upper(), linewidth=1.)
    for ax, label in zip(axes, ('Attitude (deg)', 'Angular velocity (rad/s)', 'Acceleration (m/s²)')):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.legend(loc='upper right')
    axes[0].set_ylim(0, 360)
    axes[0].set_title('Received IMU (attitude 0–360 deg; negative angles +360)')
    axes[2].set_xlabel('Time after start (s)')
    fig.savefig(directory/'imu_0_360.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('test_directory', type=Path)
    parser.add_argument('--gate-mode', choices=('auto', 'both', 'button'), default='auto')
    parser.add_argument('--start-unix-s', type=float, help='Verified manual start override for legacy recordings')
    args = parser.parse_args(argv)
    try:
        root = args.test_directory.expanduser().resolve()
        visual_path = unique_file(root, 'line_frames.jsonl')
        control_path = unique_file(root, 'control_trace_*.csv')
        visual = [json.loads(line) for line in visual_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        visual.sort(key=lambda r: number(r.get('host_time_ns')))
        with control_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            validate_control_columns(reader.fieldnames)
            control = list(reader)
        if args.start_unix_s is None:
            start, reason = find_start(visual, args.gate_mode)
        else:
            start, reason = args.start_unix_s, 'manual_override'
            if not math.isfinite(start):
                raise ValueError('Manual start must be finite')
        cropped_v, cropped_c = crop_data(visual, control, start)
        if not cropped_v or not cropped_c:
            raise ValueError('No visual or control data after start; verify the test pairing/start time.')
        output = root/'analysis'
        # Import the plotting dependency before creating outputs.
        import matplotlib
        output.mkdir(exist_ok=True)
        plot(cropped_v, cropped_c, output)
        with (output/'vision_after_start.jsonl').open('w', encoding='utf-8') as handle:
            for row in cropped_v:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        export_csv(output/'control_after_start.csv', cropped_c)
        export_csv(output/'vision_after_start.csv', [dict(t_s=r['t_s'], host_time_ns=r['host_time_ns'],
            deviation_deg=r.get('body_track_deviation_deg', r.get('measurement', {}).get('heading_control_deg')),
            deviation_valid=r.get('body_track_deviation_valid', r.get('measurement', {}).get('heading_control_valid', False)),
            filtered_deviation_deg=(deviation_values(r)[1] if math.isfinite(deviation_values(r)[1]) else None),
            steering_raw_heading_deg=r.get('measurement', {}).get('steering_heading_deg'),
            steering_heading_source=r.get('measurement', {}).get('steering_heading_source'),
            vx=r.get('vx'), wz=r.get('wz')) for r in cropped_v])
        export_csv(output/'imu_after_start.csv', [dict(t_s=r['t_s'], host_unix_s=r['host_unix_s'],
            **dict(zip(('roll_deg', 'pitch_deg', 'yaw_deg'), display_orientation_degrees(
                [r.get('received_orientation_'+a) for a in 'wxyz']))),
            **{key:r.get(key) for group in ('gyro','accel') for a in 'xyz'
               for key in ['received_'+group+'_'+a]}) for r in cropped_c])
        summary = dict(start_unix_s=start, start_reason=reason, visual_source=str(visual_path),
                       control_source=str(control_path), visual_rows_before=len(visual), control_rows_before=len(control),
                       visual_rows_after=len(cropped_v), control_rows_after=len(cropped_c),
                       first_control_time_s=cropped_c[0]['t_s'],
                       filtered_angle_samples=sum(math.isfinite(deviation_values(r)[1]) for r in cropped_v),
                       imu_attitude_display='All Euler angles modulo 360; accel/gyro unchanged.',
                       semantics='Both valves: first frame with both latched; button: first released frame. '
                                 'Executed WZ is model input, not measured body yaw rate. No resampling or filtering.')
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        if not summary['filtered_angle_samples']:
            print('No logged filtered heading; its curve is empty, never reconstructed from raw data.')
        print(f'Start: {start:.9f} ({reason}); vision {len(visual)} -> {len(cropped_v)}, '
              f'control {len(control)} -> {len(cropped_c)}; output: {output}')
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
