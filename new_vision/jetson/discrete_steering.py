"""把连续的航向指令换成"一段固定时长的转向 + 一段强制滑行"。

离散模式只共享测量校验和丢线处理，不运行连续 PID 的积分或微分。
`--wz-mode continuous` 时根本不构造本类，走的是原来那条路。

**为什么不是脉冲。** 一开始写成 0.15s 的"脉冲"，在 3:8 的步频下那只有 2~3 步 ——
指令还没被机械执行完就变了，每次转出来的量都不可复现。这不是精确执行器：
关节在小角度和大角度上都不准，同一个指令每次的表现都不一样。所以一段转向要
**够长**，长到机械真的做出来，而且幅度要用**大且稳**的那一档（0.4/0.5 就是
这个原因 —— 小角度下它更不稳）。

**一串转向只有五个数**（曾经有六个：两档幅度 + 强阈值 + 时长的两套名字）：

    --wz-fire-cm   什么时候开
    --wz-stop-cm   什么时候收手（不写 = --wz-fire-cm，即镜像；不设范围）
    --wz-turn-s    最多转多久（是**上限**，不是定长）
    --wz-gap-s     收手后空多久
    --wz-step      转多猛（一个数，不分档）

开的时候只看 err 过没过阈值；开了之后只看一件事：err 到了停手线的哪一边。
`stop_cm` 是**带符号**的：正数 = 另一侧的量（+4.5 触发 → 转到 −4.5 才收），
0 = 只翻过中心，负数 = 同侧的提前收（−2 = err 回到同侧 2cm 以内就收）。
**不写就是开火线的镜像**（= --wz-fire-cm），右转按触发时误差的符号镜像，输出边界再应用 yaw_sign。

2026-10-03 实车走了两步：先是定长 2.5s 谁也叫不停（出弯那一下把车带出直道）；
改成的"同侧 2cm 就收"又收得太早 —— 每转完都还在中心右边，直道上再攒新的右偏，
最后从**右**边出去。所以收手线改成开火线的对称位置。
收手照旧进 gap；幅度不改、档不换。

定长是 2026-10-03 实车改掉的：出弯进直道那一下打出一段 2.5s 的转向，err 已经回中
也没有任何东西能打断它 —— 0.5×2.5 = 72° 的左转指令全落在只有 3 秒长的直道上。
直道需要的转向是 0，而当时控制器能给出的最小值就是"一整段"。

两个约定，改之前先读：

* **触发看原始的 `fused_err_cm`**，也就是日志里的 `err=`。不用加过 `--bias-*`
  的 `last_err_eff`（日志里的 `eff=`）：偏置是给连续 PID 补稳态内偏的，而这里的
  阈值带本身就是那个机制，两个叠在一起只会双算。
* **`hold` 写回内层**。它是丢线淡出的唯一来源（`policy_bridge.py:196`），
  不写回的话丢线那 `--lost-hold-s` 秒回放的是连续的 PID 值，两条路不一致。

`--center-dead-cm`、`--steer-full-scale-cm`、`--preview-gain` 和三项 PID 增益
同样不参与。离散模式的 last_err_eff 是原始阈值输入，last_steer 是实际 wz。
"""

from __future__ import annotations

import math


class DiscreteSteeringController:
    def __init__(self, inner, fire_cm=5.0, stop_cm=None, turn_s=1.0, gap_s=2.5,
                 step=0.5, allow_right=False):
        # 不写 stop_cm 就是开火线的镜像：+fire 触发，转到 −fire 才收。
        stop_cm = fire_cm if stop_cm is None else float(stop_cm)
        values = (fire_cm, stop_cm, turn_s, gap_s, step)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("discrete steering settings must be finite")
        if fire_cm <= 0.0:
            raise ValueError("fire-cm must be positive")
        if turn_s <= 0.0:
            raise ValueError("turn-s must be positive")
        if gap_s < 0.0:
            raise ValueError("gap-s must not be negative; 0 = no forced coast")
        if not 0.0 < step <= inner.max_wz:
            raise ValueError("need 0 < step <= max-wz")
        self.inner = inner
        self.fire_cm = fire_cm
        self.stop_cm = stop_cm
        self.turn_s = turn_s
        self.gap_s = gap_s
        self.step = step
        self.allow_right = bool(allow_right)
        self.turn_left = 0.0
        self.gap_left = 0.0
        self.turn_wz = 0.0
        self.turn_error_sign = 0.0

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
        self.turn_error_sign = 0.0

    def drop_held_command(self):
        self.inner.drop_held_command()
        self.turn_left = 0.0
        self.gap_left = 0.0
        self.turn_wz = 0.0
        self.turn_error_sign = 0.0

    def _end_turn(self):
        self.turn_left = 0.0
        self.turn_wz = 0.0
        self.turn_error_sign = 0.0
        self.gap_left = self.gap_s

    def _advance_clock(self, elapsed):
        """Advance existing phases, carrying long-frame overshoot into the gap."""
        if self.turn_left > 0.0:
            remaining = self.turn_left
            self.turn_left = round(max(0.0, remaining - elapsed), 9)
            if self.turn_left == 0.0:
                self._end_turn()
                self.gap_left = round(max(0.0, self.gap_left -
                                          max(0.0, elapsed - remaining)), 9)
                # No yaw may be replayed after the turn deadline, even on loss.
                self.inner.hold = (self.inner.hold[0], 0.0)
        elif self.gap_left > 0.0:
            self.gap_left = round(max(0.0, self.gap_left - elapsed), 9)

    def command(self, debug, confidence, dt):
        measurement = self.inner.read_detection(debug, confidence, dt)
        try:
            elapsed = dt if math.isfinite(dt) and dt > 0.0 else 0.0
        except (TypeError, ValueError, OverflowError):
            elapsed = 0.0
        if measurement is None:
            if debug.get("measurement_stale", False) and self.turn_left > 0.0:
                # Expired geometry cannot resume an old pulse after reacquisition,
                # even when the configured loss hold exceeds the measurement TTL.
                self._end_turn()
            loss_stop_after = max(0.0, self.inner.lost_hold_s - self.inner.lost_s)
            if self.turn_left > 0.0 and elapsed >= loss_stop_after:
                # The loss deadline may have passed inside this long frame.
                # Credit time on both sides of it instead of starting a full
                # gap only when this call finally observes the stop.
                self._advance_clock(loss_stop_after)
                if self.turn_left > 0.0:
                    self._end_turn()
                self._advance_clock(elapsed - loss_stop_after)
            else:
                self._advance_clock(elapsed)
            fallback = self.inner.lost_command(elapsed, clamp_elapsed=False)
            if self.inner.lost_s >= self.inner.lost_hold_s and self.turn_left > 0.0:
                # Once loss has stopped motion, reacquisition must start a new
                # decision rather than continue the pre-loss turn.
                self._end_turn()
            self.inner.last_steer = fallback[1]
            return fallback
        err, _angle = measurement
        # In discrete mode diagnostics name the actual input and output; no PID
        # bias, integral or derivative is evaluated and silently thrown away.
        self.inner.last_err_eff = err
        self.inner.lost_s = 0.0
        wz = self._steer(err, elapsed)
        self.inner.last_steer = wz
        self.inner.hold = (self.inner.vx, wz)
        return self.inner.hold

    def _steer(self, err, elapsed):
        if self.turn_left > 0.0:
            # The error coordinate determines the stop line. yaw_sign describes
            # the model's output coordinate and may reverse the command sign.
            demand = self.turn_error_sign * err
            if demand <= -self.stop_cm:
                self._end_turn()
                return 0.0
            self._advance_clock(elapsed)
            return self.turn_wz if self.turn_left > 0.0 else 0.0
        if self.gap_left > 0.0:
            self._advance_clock(elapsed)
            return 0.0
        if err <= 0.0 and not self.allow_right:
            self.turn_wz = 0.0
            return 0.0
        if abs(err) < self.fire_cm:
            self.turn_wz = 0.0
            return 0.0
        self.turn_error_sign = math.copysign(1.0, err)
        self.turn_wz = self.inner.yaw_sign * self.turn_error_sign * self.step
        self.turn_left = self.turn_s
        return self.turn_wz
