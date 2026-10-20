"""Optional startup-only V4L2 controls, in physical exposure units.

Kept separate from generic camera opening and all per-frame/control logic.
No controls are accessed when every option is left at keep/None.
"""
import math
import re
import subprocess
import sys


def add_arguments(parser):
    parser.add_argument('--camera-exposure-mode', choices=('keep','auto','manual'), default='keep')
    parser.add_argument('--camera-exposure-ms', type=float, default=None,
                        help='Manual exposure in milliseconds, positive multiples of 0.1; requires manual mode.')
    for name in ('brightness','contrast','saturation','sharpness'):
        parser.add_argument('--camera-'+name, type=int, default=None,
                            help='Optional hardware value; range checked against this camera at startup.')
    parser.add_argument('--camera-white-balance-mode', choices=('keep','auto','manual'), default='keep')
    parser.add_argument('--camera-white-balance-k', type=int, default=None,
                        help='Manual white-balance temperature in Kelvin; requires manual mode.')
    parser.add_argument('--camera-power-line-hz', type=int, choices=(0,50,60), default=None,
                        help='Optional anti-flicker frequency; 0 disables. Does not change frame rate.')


def validate_args(args):
    ms = args.camera_exposure_ms
    if args.camera_exposure_mode == 'manual':
        if (ms is None or not math.isfinite(ms) or not 0 < ms <= 214748364.7
                or not math.isclose(ms*10, round(ms*10), rel_tol=0, abs_tol=1e-7)):
            raise ValueError('manual camera exposure requires positive camera-exposure-ms in 0.1 ms steps')
    elif ms is not None:
        raise ValueError('camera-exposure-ms requires camera-exposure-mode=manual; clear it for auto/keep')
    kelvin = args.camera_white_balance_k
    if args.camera_white_balance_mode == 'manual':
        if kelvin is None or kelvin <= 0:
            raise ValueError('manual white balance requires positive camera-white-balance-k')
    elif kelvin is not None:
        raise ValueError('camera-white-balance-k requires camera-white-balance-mode=manual')


def _catalog(text):
    controls, current = {}, None
    for line in text.splitlines():
        match = re.match(r'^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\(([^)]+)\)\s*:\s*(.*)$', line)
        if match:
            name, kind, description = match.groups()
            current = dict(kind=kind, description=description, menu={})
            current.update({k:int(v) for k,v in re.findall(r'\b(min|max|step|value)=(-?\d+)',description)})
            controls[name] = current
        else:
            menu = re.match(r'^\s+(-?\d+):\s*(.*)$',line)
            if menu and current is not None:
                current['menu'][int(menu[1])] = menu[2]
    return controls


def _call(device, flag):
    result = subprocess.run(['v4l2-ctl','-d',str(device),flag], capture_output=True,
                            text=True, timeout=3, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or 'v4l2-ctl failed').strip())
    return result.stdout


def apply_camera_controls(device, args):
    validate_args(args)
    report = dict(device=str(device), status='unchanged', settings=[])
    requested = (args.camera_exposure_mode != 'keep' or args.camera_white_balance_mode != 'keep'
        or args.camera_power_line_hz is not None
        or any(getattr(args,'camera_'+name) is not None for name in ('brightness','contrast','saturation','sharpness')))
    if not requested:
        return report
    try:
        if not sys.platform.startswith('linux'):
            raise RuntimeError('camera controls require Linux V4L2')
        catalog = _catalog(_call(device,'--list-ctrls-menus'))
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report.update(status='unavailable',error=str(exc))
        print(f'[camera-controls] unavailable: {exc}; camera defaults/current state retained',flush=True)
        return report

    def prepare(aliases, value):
        name = next((n for n in aliases if n in catalog), aliases[0])
        setting = dict(control=name,requested=value,actual=None)
        info = catalog.get(name)
        if info is None:
            setting['error'] = 'control not supported'
        elif ('read-only' in info['description'] or 'read_only' in info['description']):
            setting['error'] = 'control is read-only'
        else:
            low, high = info.get('min',0 if info['kind']=='bool' else value), info.get('max',1 if info['kind']=='bool' else value)
            step = info.get('step',1)
            if not low <= value <= high or (step > 0 and (value-low)%step):
                setting['error'] = f'value outside driver range {low}..{high} step {step}'
            elif info['menu'] and value not in info['menu']:
                setting['error'] = 'menu choice not supported'
        return setting

    def record(setting):
        report['settings'].append(setting)
        detail = setting.get('error') or f"actual={setting['actual']}"
        if setting.get('actual_ms') is not None:
            detail += f" ({setting['actual_ms']:g} ms)"
        print(f"[camera-controls] {setting['control']} requested={setting['requested']}: {detail}",flush=True)
        return not bool(setting.get('error'))

    def apply(setting, exposure=False):
        if 'error' not in setting:
            try:
                name = setting['control']
                _call(device,f"--set-ctrl={name}={setting['requested']}")
                text = _call(device,f'--get-ctrl={name}')
                match = re.search(r'^'+re.escape(name)+r':\s*(-?\d+)',text,re.MULTILINE)
                if match is None:
                    raise RuntimeError('could not read back control')
                setting['actual'] = int(match[1])
                if exposure:
                    setting['actual_ms'] = setting['actual']/10.
                if setting['actual'] != setting['requested']:
                    setting['error'] = 'driver readback differs from requested value'
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                setting['error'] = str(exc)
        return record(setting)

    if args.camera_exposure_mode != 'keep':
        aliases = ('auto_exposure','exposure_auto')
        mode_name = next((n for n in aliases if n in catalog), aliases[0])
        menu = catalog.get(mode_name,{}).get('menu',{})
        mode = 1 if args.camera_exposure_mode == 'manual' else (3 if 3 in menu else 0)
        exposure = (prepare(('exposure_time_absolute','exposure_absolute'),round(args.camera_exposure_ms*10))
                    if args.camera_exposure_mode=='manual' else None)
        # Validate the dependent value before switching off auto exposure.
        if exposure is not None and 'error' in exposure:
            record(exposure)
        elif apply(prepare(aliases,mode)) and exposure is not None:
            apply(exposure,exposure=True)

    for name in ('brightness','contrast','saturation','sharpness'):
        value = getattr(args,'camera_'+name)
        if value is not None:
            apply(prepare((name,),value))

    if args.camera_white_balance_mode != 'keep':
        auto = ('white_balance_automatic','white_balance_temperature_auto','auto_white_balance')
        temperature = (prepare(('white_balance_temperature',),args.camera_white_balance_k)
                       if args.camera_white_balance_mode=='manual' else None)
        if temperature is not None and 'error' in temperature:
            record(temperature)
        elif apply(prepare(auto,0 if temperature is not None else 1)) and temperature is not None:
            apply(temperature)
    if args.camera_power_line_hz is not None:
        apply(prepare(('power_line_frequency',),{0:0,50:1,60:2}[args.camera_power_line_hz]))
    report['status'] = 'partial' if any(s.get('error') for s in report['settings']) else 'applied'
    return report
