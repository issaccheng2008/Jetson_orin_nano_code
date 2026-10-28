"""Small causal observation filters and level hysteresis; no camera or transport.

Angles are degrees, positions centimeters, time monotonic seconds. To add a
filter, implement update(value, dt) and register it in make_angle_filter().
Only accepted geometry enters this module; missing observations are not zeros.
"""
from dataclasses import dataclass
from copy import deepcopy
import math


@dataclass(frozen=True)
class FilterConfig:
    algorithm: str = 'one-euro'
    min_cutoff_hz: float = 1.5
    max_cutoff_hz: float = 4.0
    beta: float = .03
    derivative_cutoff_hz: float = 1.0
    position_tau_s: float = .1
    hysteresis_deg: float = 1.0
    enter_deg: float = 2.0
    exit_deg: float = 1.0
    robust_tau_s: float = .45
    robust_window_s: float = .6
    robust_slew_deg_s: float = 45.

    def __post_init__(self):
        values = (self.min_cutoff_hz, self.max_cutoff_hz, self.beta,
                  self.derivative_cutoff_hz, self.position_tau_s,
                  self.hysteresis_deg, self.enter_deg, self.exit_deg,
                  self.robust_tau_s, self.robust_window_s, self.robust_slew_deg_s)
        if (self.algorithm not in ('one-euro', 'ema', 'none', 'robust')
                or not all(math.isfinite(v) for v in values)
                or self.min_cutoff_hz <= 0 or self.max_cutoff_hz < self.min_cutoff_hz
                or self.beta < 0 or self.derivative_cutoff_hz <= 0
                or self.position_tau_s <= 0 or self.hysteresis_deg < 0
                or min(self.robust_tau_s, self.robust_window_s, self.robust_slew_deg_s) <= 0
                or not 0 <= self.exit_deg <= self.enter_deg < 90):
            raise ValueError('invalid steering filter settings: finite positive cutoffs/tau, '
                             'max >= min, beta/hysteresis >= 0, 0 <= exit <= enter < 90')


class LowPass:
    def __init__(self, tau_s):
        self.tau_s = tau_s
        self.value = None

    def update(self, value, dt):
        if self.value is None:
            self.value = value
        else:
            self.value += dt/(dt+self.tau_s)*(value-self.value)
        return self.value


class OneEuro:
    def __init__(self, config):
        self.config = config
        self.raw_previous = None
        self.signal = LowPass(1/(2*math.pi*config.min_cutoff_hz))
        self.derivative = LowPass(1/(2*math.pi*config.derivative_cutoff_hz))

    def update(self, value, dt):
        rate = 0. if self.raw_previous is None else (value-self.raw_previous)/dt
        speed = abs(self.derivative.update(rate, dt))
        self.raw_previous = value
        cutoff = min(self.config.max_cutoff_hz,
                     self.config.min_cutoff_hz+self.config.beta*speed)
        self.signal.tau_s = 1/(2*math.pi*cutoff)
        return self.signal.update(value, dt)


class Identity:
    def update(self, value, dt):
        return value


class RobustAngle:
    """Time-window median-of-three, fixed EMA, bounded output slew."""
    def __init__(self, config):
        self.config = config
        self.samples = []
        self.clock = 0.
        self.value = None

    def update(self, value, dt):
        from statistics import median
        self.clock += dt
        self.samples.append((self.clock, value))
        self.samples = [p for p in self.samples[-3:] if self.clock-p[0] <= self.config.robust_window_s]
        # With two samples, retaining the first rejects a one-frame spike.
        target = self.samples[0][1] if len(self.samples)==2 else median(p[1] for p in self.samples)
        if self.value is None:
            self.value = value
        else:
            elapsed = min(dt, .25)  # Missing frames do not open the filter on return.
            change = elapsed/(elapsed+self.config.robust_tau_s)*(target-self.value)
            limit = self.config.robust_slew_deg_s*elapsed
            self.value += max(-limit, min(limit, change))
        return self.value


def make_angle_filter(config):
    """The only factory to change when adding another angle-filter formula."""
    if config.algorithm == 'robust':
        return RobustAngle(config)
    if config.algorithm == 'one-euro':
        return OneEuro(config)
    if config.algorithm == 'ema':
        return LowPass(1/(2*math.pi*config.min_cutoff_hz))
    return Identity()


class GeometryFilter:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.source = self.last_time = None
        self.heading = make_angle_filter(self.config)
        self.demand = make_angle_filter(self.config)
        self.near = (Identity() if self.config.algorithm == 'none'
                     else LowPass(self.config.position_tau_s))

    def apply(self, geometry, now):
        near, z, heading, demand, source = geometry
        if not all(math.isfinite(v) for v in (near, z, heading, demand, now)):
            raise ValueError('filter geometry/time must be finite')
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('filter time must increase monotonically')
        if source != self.source and self.config.algorithm != 'robust':
            self.reset()
        dt = now-self.last_time if self.last_time is not None else 0.
        self.source, self.last_time = source, now
        return (self.near.update(near, dt), z, self.heading.update(heading, dt),
                self.demand.update(demand, dt), source)


def validate_ladder(levels, cap, full_scale, width):
    thresholds = [(a+b)*.5*full_scale/cap for a,b in zip(levels, levels[1:])]
    if any(b-a <= 2*width for a,b in zip(thresholds, thresholds[1:])):
        raise ValueError('steering hysteresis too wide for the configured levels; '
                         'reduce steering-hysteresis-deg or separate the levels')


def select_level(angle, current, *, levels, cap, full_scale, gate, width):
    """Choose a positive magnitude; zero/direction/priority live in the controller."""
    target = cap*min(1., max(0., angle-gate)/full_scale)
    nearest = min(levels, key=lambda level: (abs(level-target), level))
    if current not in levels or width == 0:
        return nearest
    index = levels.index(current)
    while index+1 < len(levels):
        threshold = gate+(levels[index]+levels[index+1])*.5*full_scale/cap
        if angle <= threshold+width:
            break
        index += 1
    while index > 0:
        threshold = gate+(levels[index-1]+levels[index])*.5*full_scale/cap
        if angle >= threshold-width:
            break
        index -= 1
    return levels[index]


class FilterComparison:
    """Shadow runs on independent state and never publishes a command."""
    def __init__(self, actual, trial):
        self.actual, self.trial = actual, trial
        self.diagnostics = {}

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def reset(self, clear_hold=False):
        self.actual.reset(clear_hold=clear_hold)
        self.trial.reset(clear_hold=clear_hold)
        self.diagnostics = {}

    def drop_held_command(self):
        self.actual.drop_held_command()
        self.trial.drop_held_command()
        self.diagnostics = {}

    def command(self, debug, confidence, dt):
        result = self.actual.command(debug, confidence, dt)
        trial = self.trial.command(debug, confidence, dt)
        self.diagnostics = dict(self.actual.diagnostics)
        self.diagnostics.update(steering_filter_mode='shadow',
                                steering_filter_shadow_vx=trial[0],
                                steering_filter_shadow_wz=trial[1])
        for key, value in self.trial.diagnostics.items():
            self.diagnostics['steering_filter_shadow_'+key] = value
        return result


def add_arguments(parser):
    parser.add_argument('--steering-filter-mode', choices=('legacy','shadow','active'), default='legacy',
                        help='heading/segments only: original, comparison-only, or applied filter/hysteresis')
    parser.add_argument('--steering-filter-algorithm', choices=('one-euro','ema','none','robust'), default='robust',
                        help='none disables new smoothing and retains the legacy median; hysteresis remains adjustable')
    for flag, default, help_text in (
        ('robust-tau-s',.45,'Robust fixed EMA time constant; larger smooths more'),
        ('robust-window-s',.6,'Robust median history maximum age'),
        ('robust-slew-deg-s',45.,'Robust filtered angle maximum change per second'),
        ('min-hz',1.5,'Minimum angle cutoff Hz; smaller smooths more'),
        ('max-hz',4.,'Maximum adaptive angle cutoff Hz'),
        ('beta',.03,'Adaptive strength using degrees and degrees/second'),
        ('derivative-hz',1.,'Cutoff Hz for adaptive speed estimate'),
        ('position-tau-s',.1,'Near-position smoothing time constant in seconds')):
        parser.add_argument('--steering-filter-'+flag,type=float,default=default,help=help_text)
    parser.add_argument('--steering-hysteresis-deg',type=float,default=1.,help='Same-direction level half-band; 0 disables')
    parser.add_argument('--steering-enter-deg',type=float,default=2.,help='Minimum ordinary turn-entry angle')
    parser.add_argument('--steering-exit-deg',type=float,default=1.,help='Ordinary turn-release angle; <= enter')


def config_from_args(args):
    return FilterConfig(algorithm=args.steering_filter_algorithm,
        min_cutoff_hz=args.steering_filter_min_hz, max_cutoff_hz=args.steering_filter_max_hz,
        beta=args.steering_filter_beta, derivative_cutoff_hz=args.steering_filter_derivative_hz,
        position_tau_s=args.steering_filter_position_tau_s,
        hysteresis_deg=args.steering_hysteresis_deg, enter_deg=args.steering_enter_deg,
        exit_deg=args.steering_exit_deg,
        robust_tau_s=args.steering_filter_robust_tau_s,
        robust_window_s=args.steering_filter_robust_window_s,
        robust_slew_deg_s=args.steering_filter_robust_slew_deg_s)


def configured_controller(controller_type, inner, options, args):
    from steering_recovery import config_from_args as recovery_from_args
    options = dict(options, recovery_config=recovery_from_args(args),
                   segment_fallback=args.steering_segment_fallback)
    config = config_from_args(args)
    mode = args.steering_filter_mode
    actual = controller_type(inner, **options, filter_config=config if mode == 'active' else None)
    if mode == 'shadow':
        actual = FilterComparison(actual, controller_type(deepcopy(inner), **options, filter_config=config))
    return actual
