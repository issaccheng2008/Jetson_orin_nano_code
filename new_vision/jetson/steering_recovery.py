"""Keep walking through lane loss; reuse a recent turn, then search left."""
from collections import deque
from dataclasses import dataclass
import argparse
import math


@dataclass(frozen=True)
class RecoveryConfig:
    max_loss_s: float = .8
    history_s: float = .8

    def __post_init__(self):
        if not (math.isfinite(self.max_loss_s) and self.max_loss_s >= 0
                and math.isfinite(self.history_s) and self.history_s > 0):
            raise ValueError('loss max must be finite >=0 and history must be finite >0')


class TurnHistory:
    def __init__(self, config, straight_wz=0.):
        self.config = config
        self.straight_wz = straight_wz
        self.reset()

    def reset(self):
        self.samples = deque()
        self.recovery = None

    def observe(self, now, pair, dt):
        self.recovery = None
        self.samples.append((now, pair, dt))
        while self.samples and now-self.samples[0][0] > self.config.history_s:
            self.samples.popleft()

    def command(self, now, loss_s, current, *, walking_vx=0., left_wz=.3):
        # Called only during normal lane following. The caller owns QR/button,
        # card and manual stops and does not call recovery during those windows.
        speed = current[0] if current[0] != 0. else walking_vx
        if speed == 0.:
            return (0., 0.), 'loss_walking_disabled'
        fallback = (speed, left_wz)
        # This deadline bounds reuse of an old turn, not forward motion.
        if loss_s >= self.config.max_loss_s:
            return fallback, 'loss_default_left'
        if self.recovery is None:
            recent = [(t,p,d) for t,p,d in self.samples if now-t <= self.config.history_s]
            turns = [(t,p,d) for t,p,d in recent if p[0] != 0 and p[1] not in (0., self.straight_wz)]
            if not turns or now-turns[-1][0] > self.config.history_s*.5:
                return fallback, 'loss_default_left'
            # Recent time-weighted modal command, not frame count. Newer wins ties.
            weights = {}
            for t, pair, dt in turns:
                weights[pair] = weights.get(pair, 0.) + min(dt, self.config.history_s)*(
                    1-(now-t)/self.config.history_s)
            pair = max(weights, key=lambda p: (weights[p], next(t for t,p2,_ in reversed(turns) if p2==p)))
            # A real last turn has priority over older opposite-direction history.
            self.recovery = current if current[1] not in (0., self.straight_wz) else (current[0], pair[1])
        return (speed, self.recovery[1]), 'loss_history_turn'


def add_arguments(parser):
    parser.add_argument('--steering-loss-mode', choices=('history-turn','history-stop','legacy'),
                        default='history-turn',
                        help='history-turn keeps walking and searches left without a recent turn; '
                             'history-stop is a compatibility alias with the same behavior')
    parser.add_argument('--steering-loss-max-s', type=float, default=.8,
                        help='Maximum reuse of a recent turn during lane loss; then search left, never stop')
    parser.add_argument('--steering-loss-history-s', type=float, default=.8)
    parser.add_argument('--steering-segment-fallback', action=argparse.BooleanOptionalAction,
                        default=True, help='Allow quality-gated near segment when global direction is invalid')


def config_from_args(args):
    config = RecoveryConfig(args.steering_loss_max_s, args.steering_loss_history_s)
    return config if args.steering_loss_mode in ('history-turn', 'history-stop') else None
