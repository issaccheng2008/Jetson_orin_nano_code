"""把连续的航向指令换成 {0, ±step_lo, ±step_hi} 的短脉冲。

连续 PID 的毛病是执行端跟不上：直道上一直在微调、弯道轨迹有凹有凸、
参数怎么调都差一口气。这里只改**发出去的 wz** —— 平时是 0，`|err|`
过了阈值才打一个很短的脉冲，然后回 0。直道几乎完美直行，弯道靠
"脉冲 + 滑行"走成折线。

`SteeringController` 一个字不改：本类包住它，只替换它发出的 wz。
`--wz-mode continuous`（默认）时根本不构造本类，走的是原来那条路。

两个约定，改之前先读：

* **触发看 `last_err_eff`**（加过 `--bias-*` / `--single-line-gain` 的那个），
  也就是日志里的 `eff=`。这样现场按日志调阈值不会调错；`--bias-straight-cm 0`
  保证直道上它等于原始 `fused_err_cm`。代价是 `--center-dead-cm`、
  `--steer-full-scale-cm` 和三项 PID 增益在离散模式下不再影响输出 ——
  阈值带本身就是死区。
* **`hold` 写回内层**。它是丢线淡出的唯一来源（`policy_bridge.py:196`），
  不写回的话丢线那 `--lost-hold-s` 秒回放的是连续的 PID 值，两条路不一致。
"""

from __future__ import annotations

import math

from policy_bridge import clamp


class DiscreteSteeringController:
    def __init__(self, inner, fire_cm=3.0, strong_cm=8.0,
                 step_lo=0.4, step_hi=0.5, pulse_s=0.15):
        values = (fire_cm, strong_cm, step_lo, step_hi, pulse_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("discrete steering settings must be finite")
        if not 0.0 < fire_cm < strong_cm:
            raise ValueError("need 0 < fire-cm < fire-strong-cm")
        if not 0.0 < step_lo <= step_hi <= inner.max_wz:
            raise ValueError("need 0 < step-lo <= step-hi <= max-wz")
        if pulse_s <= 0.0:
            raise ValueError("pulse-s must be positive")
        self.inner = inner
        self.fire_cm = fire_cm
        self.strong_cm = strong_cm
        self.step_lo = step_lo
        self.step_hi = step_hi
        self.pulse_s = pulse_s
        self.pulse_left = 0.0
        self.pulse_wz = 0.0

    # 只读转发：run_policy_vision 读的就是这几个（日志、停车交接打印）。
    @property
    def vx(self):
        return self.inner.vx

    @property
    def yaw_sign(self):
        return self.inner.yaw_sign

    @property
    def hold(self):
        return self.inner.hold

    @property
    def lost_s(self):
        return self.inner.lost_s

    @property
    def last_steer(self):
        return self.inner.last_steer

    @property
    def last_err_eff(self):
        return self.inner.last_err_eff

    @property
    def rejected_lateral(self):
        return self.inner.rejected_lateral

    def reset(self, clear_hold=False):
        self.inner.reset(clear_hold)
        self.pulse_left = 0.0
        self.pulse_wz = 0.0

    def drop_held_command(self):
        self.inner.drop_held_command()
        self.pulse_left = 0.0
        self.pulse_wz = 0.0

    def command(self, debug, confidence, dt):
        vx, wz_pid = self.inner.command(debug, confidence, dt)
        # 丢线/无效帧：内层已经在按 lost_hold_s 淡出它自己那条指令，原样透传。
        # lost_s 只在有效帧被清 0（policy_bridge.py:274），所以它 > 0 就是
        # 这一帧刚判过无效。
        if self.inner.lost_s > 0.0:
            return (vx, wz_pid)
        wz = self._pulse(self.inner.last_err_eff, dt)
        self.inner.hold = (vx, wz)
        return (vx, wz)

    def _pulse(self, err, dt):
        if self.pulse_left > 0.0:
            # 先扣再判：扣到 0 的那一帧就该回到阈值判断，否则每次脉冲都会多挂
            # 一帧（dt=0.05、pulse_s=0.2 时是 5 帧而不是 4 帧）。
            # round 到 ns 是必须的：0.05 累减四次会留下 1.7e-17 的浮点尘，
            # 不加这一下脉冲就永远多一帧。
            self.pulse_left = round(
                max(0.0, self.pulse_left - clamp(dt, 0.01, 0.2)), 9)
            if self.pulse_left > 0.0:
                return self.pulse_wz
        if abs(err) >= self.strong_cm:
            level = self.step_hi
        elif abs(err) >= self.fire_cm:
            level = self.step_lo
        else:
            self.pulse_wz = 0.0
            return 0.0
        self.pulse_wz = self.inner.yaw_sign * math.copysign(level, err)
        self.pulse_left = self.pulse_s
        return self.pulse_wz
