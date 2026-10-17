"""Small startup-only validators for editable steering arrays; no camera imports."""
import json
import math

DEFAULT_SEGMENT_REGIONS_CM = ((20., 32.), (32., 44.), (44., 56.))


def _rows(value, columns, name):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, (list, tuple)):
            raise ValueError()
        rows = []
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != columns:
                raise ValueError()
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in row):
                raise ValueError()
            rows.append(tuple(float(v) for v in row))
        if not all(math.isfinite(v) for row in rows for v in row):
            raise ValueError()
        return tuple(rows)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a JSON array of finite numeric {columns}-value rows') from exc


def validate_angle_wz_table(value, cap=.5, right_cap=None, allow_right=True):
    if value is None or value == '':
        return None
    rows = _rows(value, 3, 'steering-angle-wz-table')
    cap = min(.5, cap)
    right_cap = min(.5, cap if right_cap is None else right_cap)
    if (not 2 <= len(rows) <= 32 or rows[0][0] != -90 or rows[-1][1] != 90
            or not any(rate > 0 for _, _, rate in rows)
            or (allow_right and not any(rate < 0 for _, _, rate in rows))
            or any(not -90 <= lo < hi <= 90 or not -right_cap <= rate <= cap for lo,hi,rate in rows)
            or any(a[1] != b[0] for a, b in zip(rows, rows[1:]))):
        raise ValueError('angle table must cover [-90,+90] without gaps/overlap, '
                         'include turn levels for enabled directions; '
                         f'limits are [-{right_cap:g},+{cap:g}] rad/s')
    return rows


def table_level(table, demand):
    """Use current raw target bearing; bins are [lo,hi), endpoints saturate."""
    index = next((i for i, (_, hi, _) in enumerate(table) if demand < hi-1e-9), len(table)-1)
    return table[index][2]


def validate_heading_regions(value):
    if value is None or value == '':
        return None
    rows = _rows(value, 2, 'heading-regions-cm')
    if (len(rows) != 2 or any(not 20 <= lo < hi <= 85 for lo,hi in rows)
            or rows[0][1] > rows[1][0]):
        raise ValueError('heading regions need exactly two ordered non-overlapping intervals within 20–85cm')
    return rows


def validate_segment_regions(value=DEFAULT_SEGMENT_REGIONS_CM):
    rows = _rows(value, 2, 'segment-regions-cm')
    if (not 2 <= len(rows) <= 5 or rows[0][0] < 20 or rows[-1][1] > 70
            or any(hi-lo < 6 for lo, hi in rows)
            or any(a[1] != b[0] for a, b in zip(rows, rows[1:]))):
        raise ValueError('segment regions need 2–5 contiguous ordered intervals, each >=6cm, within 20–70cm')
    return rows
