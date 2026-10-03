"""把连续的航向指令换成 {0, ±step_lo, ±step_hi} 的短脉冲。

连续 PID 的毛病是执行端跟不上：直道上一直在微调、弯道轨迹有凹有凸、
参数怎么调都差一口气。这里只改**发出去的 wz** —— 平时是 0，`|err|`
过了阈值才打一个很短的脉冲，然后回 0。直道几乎完美直行，弯道靠
"脉冲 + 滑行"走成折线。

`SteeringController` 一个字不改：本类包住它，只替换它发出的 wz。
`--wz-mode continuous`（默认）时根本不构造本类，走的是原来那条路。

两个约定，改之前先读：

* **触发看原始的 `fused_err_cm`**，也就是日志里的 `err=`。一开始用的是加过
  `--bias-*` 的 `last_err_eff`（日志里的 `eff=`），但默认 `--bias-cm` 是 3.0、
  而 `--wz-fire-cm` 也是 3.0 —— **bias 自己就把阈值顶穿了**，弯道上车正对着
  中心（`err=0`）也会一直打脉冲。偏置的来历是补连续 PID 的稳态内偏，而离散模式
  的阈值带本身就是那个机制，两个叠在一起只会双算。所以偏置在这里不参与。
* **`hold` 写回内层**。它是丢线淡出的唯一来源（`policy_bridge.py:196`），
  不写回的话丢线那 `--lost-hold-s` 秒回放的是连续的 PID 值，两条路不一致。

`--center-dead-cm`、`--steer-full-scale-cm`、`--preview-gain` 和三项 PID 增益
同样不参与 —— 它们都是往 `steer` 里加的，而 `steer` 在这里被整个丢掉。
"""

from __future__ import annotations

import math

from policy_bridge import clamp

# 单次转弯的上限和两次转弯之间的下限。用户定的形状要求：转弯是一下一下的
# （弯道上近似多边形），不是连续转。所以一发不能长过 1 秒，两发之间要空出
# 2 秒以上。以前打完只要 |err| 还在阈值上就立刻再打，间隔是 0 —— 那出来的是
# 连续转弯，不是多边形。
MAX_PULSE_S = 1.0
MIN_GAP_S = 2.0


class DiscreteSteeringController:
    def __init__(self, inner, fire_cm=5.0, strong_cm=8.0,
                 step_lo=0.4, step_hi=0.5, pulse_s=0.15,
                 min_gap_s=2.5, allow_right=False):
        values = (fire_cm, strong_cm, step_lo, step_hi, pulse_s, min_gap_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("discrete steering settings must be finite")
        if not 0.0 < fire_cm < strong_cm:
            raise ValueError("need 0 < fire-cm < fire-strong-cm")
        if not 0.0 < step_lo <= step_hi <= inner.max_wz:
            raise ValueError("need 0 < step-lo <= step-hi <= max-wz")
        if not 0.0 < pulse_s <= MAX_PULSE_S:
            raise ValueError(f"pulse-s must be in (0, {MAX_PULSE_S}]")
        if min_gap_s <= MIN_GAP_S:
            raise ValueError(f"min-gap-s must exceed {MIN_GAP_S}")
        self.inner = inner
        self.fire_cm = fire_cm
        self.strong_cm = strong_cm
        self.step_lo = step_lo
        self.step_hi = step_hi
        self.pulse_s = pulse_s
        self.min_gap_s = min_gap_s
        self.allow_right = bool(allow_right)
        self.pulse_left = 0.0
        self.gap_left = 0.0
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
        self.gap_left = 0.0
        self.pulse_wz = 0.0

    def drop_held_command(self):
        self.inner.drop_held_command()
        self.pulse_left = 0.0
        self.gap_left = 0.0
        self.pulse_wz = 0.0

    def command(self, debug, confidence, dt):
        vx, wz_pid = self.inner.command(debug, confidence, dt)
        # 丢线/无效帧：内层已经在按 lost_hold_s 淡出它自己那条指令，原样透传。
        # lost_s 只在有效帧被清 0（policy_bridge.py:274），所以它 > 0 就是
        # 这一帧刚判过无效。
        if self.inner.lost_s > 0.0:
            return (vx, wz_pid)
        try:
            err = float(debug["fused_err_cm"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return (vx, wz_pid)
        wz = self._pulse(err, dt)
        self.inner.hold = (vx, wz)
        return (vx, wz)

    def _pulse(self, err, dt):
        step = clamp(dt, 0.01, 0.2)
        if self.pulse_left > 0.0:
            # 先扣再判：扣到 0 的那一帧这一发就结束了，间隔从这一帧起算。
            # round 到 ns 是必须的：0.05 累减四次会留下 1.7e-17 的浮点尘，
            # 不加这一下脉冲就永远多一帧。
            self.pulse_left = round(max(0.0, self.pulse_left - step), 9)
            if self.pulse_left > 0.0:
                return self.pulse_wz
            self.pulse_wz = 0.0
            self.gap_left = self.min_gap_s
            return 0.0
        # 强制间隔：这一发打完之后，不管 err 还有多大都要空出 min_gap_s。
        # 没有这一段，|err| 停在阈值上时发出去的是一串连着的脉冲 —— 那是连续
        # 转弯，不是"转一下、滑一段"的多边形。
        if self.gap_left > 0.0:
            self.gap_left = round(max(0.0, self.gap_left - step), 9)
            self.pulse_wz = 0.0
            return 0.0
        # 单边：只有车身偏右（err > 0）才允许左转。偏左一律不转 —— 发一个负的
        # wz 去纠，在只有左弯的赛道上等于把自己往弯外推。
        # err <= 0 时是精确的 0，不是"很小"。
        if err <= 0.0 and not self.allow_right:
            self.pulse_wz = 0.0
            return 0.0
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
