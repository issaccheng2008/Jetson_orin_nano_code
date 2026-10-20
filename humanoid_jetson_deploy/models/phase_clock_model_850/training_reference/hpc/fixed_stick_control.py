"""Shared train/play command-mode settings and safe resume validation."""
from deployment.phase_clock import PhaseClockConfig


def add_control_args(parser):
    parser.add_argument('--command-mode', choices=('phase_clock', 'touchdown'), default=None,
                        help='Default: saved mode for new checkpoints; phase_clock for old warm starts')
    parser.add_argument('--walk-end-s', type=float, default=None)
    parser.add_argument('--lead-end-s', type=float, default=None)
    parser.add_argument('--sequence-end-s', type=float, default=None)


def saved_control(checkpoint):
    return (checkpoint.get('infos') or {}).get('fixed_stick_control')


def resolve_control(args, checkpoint):
    saved = saved_control(checkpoint) or {}
    mode = args.command_mode or saved.get('command_mode', 'phase_clock')
    values = dict(saved.get('phase_clock', {}))
    for name in ('walk_end_s', 'lead_end_s', 'sequence_end_s'):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    clock = PhaseClockConfig(**values)
    return {'command_mode': mode, 'phase_clock': clock.to_dict()}


def validate_control_resume(checkpoint, control, resume):
    if not resume:
        return
    saved = saved_control(checkpoint) or {'command_mode': 'touchdown'}
    if saved['command_mode'] != control['command_mode']:
        raise ValueError('Command mode changed: warm-start without --resume and reset optimizer/metrics')
    if control['command_mode'] == 'phase_clock' and saved.get('phase_clock') != control['phase_clock']:
        raise ValueError('Phase boundaries changed: warm-start without --resume')


def configure_control(cfg, control):
    clock = PhaseClockConfig(**control['phase_clock'])
    if abs(cfg.sim.dt * cfg.decimation - clock.control_dt) > 1e-8:
        raise ValueError('Simulation control period must match phase-clock control_dt')
    cfg.fixed_command_mode = control['command_mode']
    cfg.fixed_walk_end_s = clock.walk_end_s
    cfg.fixed_lead_end_s = clock.lead_end_s
    cfg.fixed_sequence_end_s = clock.sequence_end_s
    cfg.fixed_forward_velocity = clock.forward_velocity
    cfg.fixed_walk_step = clock.walk_step
    cfg.fixed_crossing_step = clock.crossing_step
    cfg.commands.base_velocity.ranges.lin_vel_x = (clock.forward_velocity, clock.forward_velocity)
