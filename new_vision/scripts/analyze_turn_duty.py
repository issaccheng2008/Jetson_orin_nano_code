#!/usr/bin/env python3
"""Measure commanded turn duty from frame telemetry, or count journal snapshots.

Examples:
  python new_vision/scripts/analyze_turn_duty.py line_frames.jsonl --json
  python new_vision/scripts/analyze_turn_duty.py journal.txt --max-gap 0.5

JSONL uses the actual LineTelemetry context: process_monotonic_s, vx, wz, and
measurement diagnostics. Each valid command is attributed to [this timestamp,
next timestamp), excluding the terminal sample. A gap above --max-gap is ENTIRELY
unknown; no partial hold is invented. Malformed rows break continuity. Run or
clock changes and non-increasing time are boundaries with unquantifiable duration.
Host timestamps are a fallback, and cannot be mixed with a monotonic clock.

This measures commands, not physical yaw. Journal output is subsampled, with
second-resolution timestamps, and only supports snapshot proportions.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re


CLASSES = ('turn', 'straight', 'stop', 'other')
CLOCKS = (('process_monotonic_s', 'monotonic', 1), ('monotonic_s', 'monotonic', 1),
          ('timestamp_s', 'wall', 1), ('timestamp', 'wall', 1),
          ('host_time_ns', 'wall', 1e-9))


def _number(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not a numeric command or timestamp')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('non-finite numeric value')
    return value


def _classify(vx, wz):
    if vx > 0:
        return 'turn' if wz != 0 else 'straight'
    return 'stop' if vx == 0 and wz == 0 else 'other'


def _percentage(turn, straight):
    return 100 * turn / (turn + straight) if turn + straight else None


def _active(value):
    if value is True or value == 1 or value in ('1', 'true', 'True'):
        return 'true'
    if value is False or value == 0 or value in ('0', 'false', 'False'):
        return 'false'
    return 'missing'


def _diagnostics(row):
    measurement = row.get('measurement', {})
    if not isinstance(measurement, dict):
        measurement = {}
    return {'reason': str(measurement.get('steering_reason', 'missing')),
            'active': _active(measurement.get('segment_control_active')),
            'gate': str(measurement.get('segment_gate_reason', 'missing'))}


def analyze_jsonl(lines, max_gap=.5):
    """Return time-weighted duty. Invalid records are reported, never silently joined."""
    max_gap = _number(max_gap)
    if max_gap <= 0:
        raise ValueError('max_gap must be positive')
    durations = dict.fromkeys((*CLASSES, 'unknown'), 0.)
    counts = dict.fromkeys(CLASSES, 0)
    issues, boundaries, unknown_causes = [], Counter(), Counter()
    reason_time, active_time, gate_time = Counter(), Counter(), Counter()
    reason_count, active_count, gate_count = Counter(), Counter(), Counter()
    previous, interrupted, samples = None, False, 0
    clock_fields = Counter()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get('schema') != 'line_frames_v1':
                raise ValueError('expected schema=line_frames_v1')
            field = next((key for key, _, _ in CLOCKS if key in row), None)
            if field is None:
                raise ValueError('missing frame timestamp')
            _, source, scale = next(clock for clock in CLOCKS if clock[0] == field)
            timestamp = _number(row[field]) * scale
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            issues.append({'line': line_number, 'message': str(error)})
            interrupted = True
            continue
        sample = dict(time=timestamp, source=source, run=row.get('run_id'),
                      kind='unknown', **_diagnostics(row))
        clock_fields[field] += 1
        samples += 1
        try:
            sample['kind'] = _classify(_number(row['vx']), _number(row['wz']))
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            issues.append({'line': line_number, 'message': 'invalid vx/wz: ' + str(error)})
        else:
            counts[sample['kind']] += 1
        reason_count[sample['reason']] += 1
        active_count[sample['active']] += 1
        gate_count[sample['gate']] += 1
        if previous is not None:
            gap = timestamp - previous['time']
            if previous['run'] != sample['run']:
                boundaries['run_change'] += 1
            elif previous['source'] != source:
                boundaries['clock_source_change'] += 1
            elif gap <= 0:
                boundaries['non_increasing_time'] += 1
            else:
                cause = ('malformed_record' if interrupted else
                         'long_gap' if gap > max_gap + 1e-9 else
                         'invalid_command' if previous['kind'] == 'unknown' else None)
                kind = 'unknown' if cause else previous['kind']
                durations[kind] += gap
                if cause:
                    unknown_causes[cause] += gap
                if kind in ('turn', 'straight'):
                    reason_time[previous['reason']] += gap
                    active_time[previous['active']] += gap
                    gate_time[previous['gate']] += gap
        previous, interrupted = sample, False
    return dict(format='line_frames_v1', samples=samples, counts=counts,
                duration_s=durations, moving_duration_s=durations['turn'] + durations['straight'],
                moving_turn_pct=_percentage(durations['turn'], durations['straight']),
                max_gap_s=max_gap, unknown_causes_duration_s=dict(unknown_causes),
                discontinuities=dict(boundaries), clock_fields=dict(clock_fields),
                reason_counts=dict(reason_count), segment_active_counts=dict(active_count),
                gate_counts=dict(gate_count), moving_reason_duration_s=dict(reason_time),
                moving_segment_active_duration_s=dict(active_time), moving_gate_duration_s=dict(gate_time),
                issues=issues,
                timing_caveat='Time-weighted command estimate between recorded frames only; '
                    'terminal duration and discontinuity duration are unquantified. '
                    'Long gaps are wholly unknown. Stop, other, and unknown are excluded from moving duty. '
                    'Commands are not measured body yaw.')


def _snapshot_summary(rows):
    counts = dict.fromkeys(CLASSES, 0)
    for row in rows:
        counts[row['kind']] += 1
    return dict(samples=len(rows), counts=counts,
                moving_samples=counts['turn'] + counts['straight'],
                moving_turn_pct=_percentage(counts['turn'], counts['straight']),
                reason_counts=dict(Counter(row.get('reason', 'missing') for row in rows)),
                segment_active_counts=dict(Counter(_active(row.get('active')) for row in rows)),
                gate_counts=dict(Counter(row.get('gate', 'missing') for row in rows)),
                support_counts=dict(Counter(row.get('pattern', 'missing') for row in rows)),
                anchored_counts=dict(Counter(row.get('anchored', 'missing') for row in rows)),
                moving_reason_counts=dict(Counter(row.get('reason', 'missing') for row in rows
                                                 if row['kind'] in ('turn', 'straight'))))


def analyze_journal(lines):
    """Keep vision (B) and model input (C) snapshot counts independent."""
    vision, policy, issues, timestamps = [], [], [], []
    current = None
    for line_number, line in enumerate(lines, 1):
        stamp = re.match(r'\w+\s+\d+\s+\d\d:\d\d:\d\d', line)
        if stamp:
            timestamps.append(stamp[0])
        is_vision = bool(re.search(r'\[vision\]\s+\d+Hz\s+vx=', line))
        is_policy = 'policy_target_velocity=[' in line
        if is_vision or is_policy:
            if is_vision:
                current = None
            payload = line.split('policy_target_velocity=[', 1)[1] if is_policy else line
            try:
                vx = _number(re.search(r'\bvx=([^\s,\]]+)', payload)[1])
                wz = _number(re.search(r'\bwz=([^\s,\]]+)', payload)[1])
            except (ValueError, TypeError, OverflowError) as error:
                issues.append({'line': line_number, 'message': 'invalid vx/wz: ' + str(error)})
                continue
            row = dict(line=line_number, kind=_classify(vx, wz))
            if is_vision:
                reason = re.search(r'\breason=(\S+)', line)
                row['reason'] = reason[1] if reason else 'missing'
                vision.append(row)
                current = row
            else:
                step = re.search(r'\bstep=\s*(\d+)', line)
                row['step'] = int(step[1]) if step else None
                policy.append(row)
        elif current is not None and ('[segment-control]' in line or '[lane-segments]' in line):
            for key in ('active', 'gate', 'pattern', 'anchored'):
                match = re.search(r'\b' + key + r'=(\S+)', line)
                if match:
                    current[key] = match[1]
    result = dict(format='journal', B_vision=_snapshot_summary(vision),
                  C_policy=_snapshot_summary(policy), issues=issues,
                  first_timestamp=timestamps[0] if timestamps else None,
                  last_timestamp=timestamps[-1] if timestamps else None,
                  timing_caveat='Journal snapshot proportions only: second-resolution timestamps '
                    'and sparse sampling cannot establish precise time duty. '
                    'B and C are asynchronously sampled; same-second differences do not prove packet loss. '
                    'Stops are excluded from moving duty; commands are not measured body yaw.')
    steps = [row['step'] for row in policy]
    if len(steps) >= 2 and all(step is not None for step in steps):
        gaps = [b - a for a, b in zip(steps, steps[1:])]
        result['assumption_diagnostics'] = dict(policy_hz_assumed=50,
            nominal_policy_step_span_s=(steps[-1] - steps[0]) / 50 if all(gap > 0 for gap in gaps) else None,
            policy_step_gap_counts=dict(Counter(gaps)),
            caveat='Nominal span assumes a single uninterrupted run at repository default 50Hz; '
                   'it is not a measured turn or straight duration.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=Path, help='line_frames.jsonl or journal text (UTF-8)')
    parser.add_argument('--format', choices=('auto', 'jsonl', 'journal'), default='auto')
    parser.add_argument('--max-gap', type=float, default=.5,
                        help='JSONL: entire longer interval is unknown (default: 0.5 seconds)')
    parser.add_argument('--json', action='store_true', help='emit JSON to stdout')
    args = parser.parse_args(argv)
    try:
        lines = args.file.read_text(encoding='utf-8-sig').splitlines()
        format_name = args.format
        if format_name == 'auto':
            first = next((line.lstrip() for line in lines if line.strip()), '')
            format_name = 'jsonl' if args.file.suffix.lower() == '.jsonl' or first.startswith('{') else 'journal'
        report = analyze_jsonl(lines, args.max_gap) if format_name == 'jsonl' else analyze_journal(lines)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        reports = [('frames', report)] if format_name == 'jsonl' else [
            ('B vision', report['B_vision']), ('C policy', report['C_policy'])]
        for label, item in reports:
            pct = item['moving_turn_pct']
            proportion = 'unavailable' if pct is None else f'{pct:.1f}%'
            print(f'{label}: samples={item["samples"]}, counts={item["counts"]}; moving turn={proportion}')
            if 'duration_s' in item:
                print('duration_s=' + json.dumps(item['duration_s']))
                print('moving_reason_duration_s=' + json.dumps(item['moving_reason_duration_s']))
                print('moving_segment_active_duration_s=' + json.dumps(item['moving_segment_active_duration_s']))
                print('moving_gate_duration_s=' + json.dumps(item['moving_gate_duration_s']))
            else:
                print('reason_counts=' + json.dumps(item['reason_counts']))
                print('segment_active_counts=' + json.dumps(item['segment_active_counts']))
                print('gate_counts=' + json.dumps(item['gate_counts']))
        print(report['timing_caveat'])
        if report['issues']:
            print('Issues: ' + json.dumps(report['issues']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
