# Steering recovery and robust filtering

User explicitly authorizes choosing the design and implementing without approval questions.

Priority 1: bounded, recent command-history turn on missing geometry; never substitute configured forward speed. Stop both vx/wz without turn evidence or after 0.8 seconds. Explicit stops clear history. Legacy behavior remains selectable.

Priority 2: add causal median + fixed EMA + angle slew limit. Defaults: 0.45-second EMA, median of recent three accepted samples within 0.6 seconds, 45 deg/s maximum output change. No adaptive opening on every noisy jump. Use filtered geometry for prediction; preserve raw two-frame offset safeguards. Preserve legacy formulas.

Priority 3: qualified segment near-direction fallback independent of failed global fit, requiring fresh paired near observations, existing segment width/span/point/residual/anchor gates. Do not relax five-point, 1-cm-span global heading invalidity.

Verify tests first, replay three recordings with separate filter/loss/fallback ablations and fixed-observation semantics, inspect plots, document new controls and old config migration, review and push policy49flitter. Hardware trajectory gains remain unverified.
