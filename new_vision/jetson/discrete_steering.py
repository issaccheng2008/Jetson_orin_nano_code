"""把连续的航向指令换成"一段固定时长的转向 + 一段强制滑行"。

`SteeringController` 一个字不改：本类包住它，只替换它发出的 wz。
`--wz-mode continuous` 时根本不构造本类，走的是原来那条路。

**为什么不是脉冲。** 一开始写成 0.15s 的"脉冲"，在 3:8 的步频下那只有 2~3 步 ——
指令还没被机械执行完就变了，每次转出来的量都不可复现。这不是精确执行器：
关节在小角度和大角度上都不准，同一个指令每次的表现都不一样。所以一段转向要
**够长**，长到机械真的做出来，而且幅度要用**大且稳**的那一档（0.4/0.5 就是
这个原因 —— 小角度下它更不稳）。

**一串转向只有四个数**（曾经有六个：两档幅度 + 强阈值 + 时长的两套名字）：

    --wz-fire-cm   什么时候开
    --wz-turn-s    转多久（≤ 1.0）
    --wz-gap-s     转完空多久（> 2.0）
    --wz-step      转多猛（一个数，不分档）

开的时候只看 err 过没过阈值，开了之后就不再看 err —— 中间不改幅度、不提前收、
不换档。四个数之外没有别的判断。

两个约定，改之前先读：

* **触发看原始的 `fused_err_cm`**，也就是日志里的 `err=`。不用加过 `--bias-*`
  的 `last_err_eff`（日志里的 `eff=`）：偏置是给连续 PID 补稳态内偏的，而这里的
  阈值带本身就是那个机制，两个叠在一起只会双算。
* **`hold` 写回内层**。它是丢线淡出的唯一来源（`policy_bridge.py:196`），
  不写回的话丢线那 `--lost-hold-s` 秒回放的是连续的 PID 值，两条路不一致。

`--center-dead-cm`、`--steer-full-scale-cm`、`--preview-gain` 和三项 PID 增益
同样不参与 —— 它们都是往 `steer` 里加的，而 `steer` 在这里被整个丢掉。
"""

from __future__ import annotations

import math

from policy_bridge import clamp

# 一段转向的上限、和两段之间的下限。这是形状要求，不是调参建议：转和滑分开，
# 弯道上才是"转一下、滑一段"的多边形。
MAX_TURN_S = 1.0
MIN_GAP_S = 2.0


class DiscreteSteeringController:
    def __init__(self, inner, fire_cm=5.0, turn_s=1.0, gap_s=2.5,
                 step=0.5, allow_right=False):
        values = (fire_cm, turn_s, gap_s, step)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("discrete steering settings must be finite")
        if fire_cm <= 0.0:
            raise ValueError("fire-cm must be positive")
        if not 0.0 < turn_s <= MAX_TURN_S:
            raise ValueError(f"turn-s must be in (0, {MAX_TURN_S}]")
        if gap_s <= MIN_GAP_S:
            raise ValueError(f"gap-s must exceed {MIN_GAP_S}")
        if not 0.0 < step <= inner.max_wz:
            raise ValueError("need 0 < step <= max-wz")
        self.inner = inner
        self.fire_cm = fire_cm
        self.turn_s = turn_s
        self.gap_s = gap_s
        self.step = step
        self.allow_right = bool(allow_right)
        self.turn_left = 0.0
        self.gap_left = 0.0
        self.turn_wz = 0.0

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
        self.turn_left = 0.0
        self.gap_left = 0.0
        self.turn_wz = 0.0

    def drop_held_command(self):
        self.inner.drop_held_command()
        self.turn_left = 0.0
        self.gap_left = 0.0
        self.turn_wz = 0.0

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
        wz = self._steer(err, dt)
        self.inner.hold = (vx, wz)
        return (vx, wz)

    def _steer(self, err, dt):
        step = clamp(dt, 0.01, 0.2)
        if self.turn_left > 0.0:
            # 先扣再判：扣到 0 的那一帧这一段就结束了。
            # round 到 ns 是必须的：0.05 累减会留下 1.7e-17 的浮点尘。
            self.turn_left = round(max(0.0, self.turn_left - step), 9)
            if self.turn_left > 0.0:
                return self.turn_wz
            self.turn_wz = 0.0
            self.gap_left = self.gap_s
            return 0.0
        if self.gap_left > 0.0:
            self.gap_left = round(max(0.0, self.gap_left - step), 9)
            self.turn_wz = 0.0
            return 0.0
        # 单边：只有车身偏右（err > 0）才开。赛道按行进方向只有左弯，反向打
        # 就是把自己往弯外推。err <= 0 时是精确的 0。
        if err <= 0.0 and not self.allow_right:
            self.turn_wz = 0.0
            return 0.0
        if abs(err) < self.fire_cm:
            self.turn_wz = 0.0
            return 0.0
        self.turn_wz = self.inner.yaw_sign * math.copysign(self.step, err)
        self.turn_left = self.turn_s
        return self.turn_wz
