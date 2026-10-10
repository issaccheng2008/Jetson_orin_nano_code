"""Select a discrete yaw from every valid vision frame in a fixed time window."""
import math


def effective_policy_hold(args):
    """The vision window owns timing; do not hold its result a second time."""
    return 0. if args.steering_command_window_s > 0 else args.command_min_hold_s


class SteeringCommandWindow:
    def __init__(self, window_s=.5, median_side='lower'):
        if not math.isfinite(window_s) or window_s < 0:
            raise ValueError('steering-command-window-s must be finite and nonnegative')
        if median_side not in ('lower', 'upper'):
            raise ValueError('steering-command-median must be lower or upper')
        self.window_s, self.median_side = window_s, median_side
        self.reset()

    def reset(self):
        self.deadline = self.previous_time = None
        self.samples = []
        self.ignored = 0
        self.held_wz = self.loss_wz = 0.
        self.diagnostics = dict(steering_window_reason='reset',
                               steering_window_valid_samples=0,
                               steering_window_ignored_samples=0)

    def update(self, vx, wz, now, *, valid):
        if not all(math.isfinite(v) for v in (vx, wz, now)):
            self.reset()
            return 0., 0.
        if vx == 0. and wz == 0.:
            self.reset()
            self.diagnostics['steering_window_reason'] = 'stop'
            return 0., 0.
        if self.window_s == 0.:
            self.diagnostics['steering_window_reason'] = 'disabled'
            return vx, wz
        # A backwards clock or an entirely missed window starts a fresh session.
        # Do not revive a command sampled before a camera/connector timeout.
        if (self.previous_time is not None and now < self.previous_time
                or self.deadline is not None and now >= self.deadline+self.window_s):
            self.reset()
        self.previous_time = now
        if not valid:
            self.loss_wz = wz
        if self.deadline is None:
            # First command after start/stop is immediate. It is also the first
            # sample of [now, now+window_s); subsequent updates use whole windows.
            self.deadline = now+self.window_s
            self.held_wz = wz
            self.diagnostics['steering_window_reason'] = 'initial'
        elif now+1e-9 >= self.deadline:
            # The boundary frame belongs to the NEXT window. Lost frames never
            # enter the median, even when the final frame before publication loses
            # the line. An empty window uses the controller's loss compensation.
            if self.samples:
                ordered = sorted(self.samples)
                index = (len(ordered)-1)//2 if self.median_side == 'lower' else len(ordered)//2
                self.held_wz = ordered[index]
                reason = 'median'
            else:
                self.held_wz = self.loss_wz
                reason = 'loss_compensation'
            self.diagnostics.update(steering_window_reason=reason,
                                   steering_window_valid_samples=len(self.samples),
                                   steering_window_ignored_samples=self.ignored)
            self.deadline += self.window_s
            self.samples = []
            self.ignored = 0
        if valid:
            self.samples.append(wz)
        else:
            self.ignored += 1
        self.diagnostics.update(steering_window_pending_samples=len(self.samples),
                                steering_window_pending_ignored=self.ignored,
                                steering_window_remaining_s=max(0., self.deadline-now),
                                steering_window_median_side=self.median_side)
        return vx, self.held_wz


def add_arguments(parser):
    parser.add_argument('--steering-command-window-s', type=float, default=.5,
                        help='Median of mapped valid frame yaw commands per window; 0 disables it')
    parser.add_argument('--steering-command-median', choices=('lower', 'upper'), default='lower',
                        help='Even sample count selects lower/upper middle; never averages levels')
