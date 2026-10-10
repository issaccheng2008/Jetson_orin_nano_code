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


def validate_angle_wz_table(value, cap=.5, hysteresis_deg=0.):
    if value is None or value == '':
        return None
    rows = _rows(value, 3, 'steering-angle-wz-table')
    if (not 2 <= len(rows) <= 32 or rows[0][0] != 0 or rows[-1][1] != 90
            or rows[0][2] != 0 or not any(rate > 0 for _, _, rate in rows)
            or any(not 0 <= lo < hi <= 90 or not 0 <= rate <= cap for lo, hi, rate in rows)
            or any(a[1] != b[0] for a, b in zip(rows, rows[1:]))
            or any(hi-lo <= 2*hysteresis_deg for lo, hi, _ in rows)):
        raise ValueError('angle table must cover [0,90] without gaps/overlap, start at 0 rad/s, '
                         f'contain a turn, use rates <= {cap:g} rad/s and bins wider than twice hysteresis')
    return rows


def table_level(table, demand, current=0., width=0.):
    """Use |final decision angle|; bins are [lo,hi), with saturated last bin.

    Keep an adjacent applied level within the configured boundary hysteresis.
    Never retain a command pointing in the opposite direction.
    """
    magnitude = abs(demand)
    index = next((i for i, (_, hi, _) in enumerate(table) if magnitude < hi-1e-9), len(table)-1)
    level = table[index][2]
    if width > 0 and current*demand >= 0:
        for neighbour, boundary in ((index-1, table[index][0]), (index+1, table[index][1])):
            if (0 <= neighbour < len(table) and abs(magnitude-boundary) < width
                    and table[neighbour][2] == abs(current)):
                level = abs(current)
                break
    return math.copysign(level, demand) if demand else 0.


def validate_segment_regions(value=DEFAULT_SEGMENT_REGIONS_CM):
    rows = _rows(value, 2, 'segment-regions-cm')
    if (not 2 <= len(rows) <= 5 or rows[0][0] < 20 or rows[-1][1] > 70
            or any(hi-lo < 6 for lo, hi in rows)
            or any(a[1] != b[0] for a, b in zip(rows, rows[1:]))):
        raise ValueError('segment regions need 2–5 contiguous ordered intervals, each >=6cm, within 20–70cm')
    return rows
