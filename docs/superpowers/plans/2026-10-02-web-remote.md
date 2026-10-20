# Web Remote Implementation Plan

> For agentic workers: use subagent-driven-development for the independent frontend task, with integration and final review in this session.

**Goal:** A polished Chinese dual-axis browser controller for the existing Jetson walking runtime.

**Architecture:** A standard-library Python HTTP service owns an expiring exclusive control lease and publishes bounded, smoothed commands over loopback UDP. The existing runtime gains a remote mode on port 5007. Static assets work offline and support touch, mouse, and keyboard.

**Tech Stack:** Python standard library, unittest, HTML/CSS/JavaScript, browser automation.

## 1. Control state and HTTP service

- [x] Add `tests/test_remote_control.py`: deterministic clock tests for forward-only limits, finite values, exclusive ownership, stale sequences, timeout revocation, immediate stop, and rate limits; real HTTP/UDP tests for authorization and command delivery.
- [x] Run `python -m unittest discover -s tests -p test_remote_control.py -v` and verify missing-feature failure.
- [x] Add `remote_control/control.py` with `ControlState.arm()`, `command(session, seq, vx, wz)`, `stop(session)`, `tick(dt)`, and `snapshot()`; serialize state access with an RLock and use monotonic time.
- [x] Add `remote_control/server.py` and `remote_control/__main__.py`. API: authenticated GET `/api/status`; POST `/api/arm` returns `{session, ...status}`; POST `/api/command` accepts `{session, seq, vx, wz}`; POST `/api/stop` accepts `{session}`. Access via `Authorization: Bearer <token>`; no CORS; enforce same Origin if supplied; strictly bound JSON bodies and socket timeouts.
- [x] State returns `active`, `target_vx`, `target_wz`, `vx`, `wz`, `max_vx`, `max_wz`, `yaw_sign`, `timeout_ms`, `command_age_ms`; no session id in public status. Errors return JSON `{error}` with 400, 401, 409, or 503.
- [x] Publish every 20 ms to `127.0.0.1:5007`. Watchdog 300 ms; abort stale sessions; send zeros on stop/shutdown. Re-run tests until green.

## 2. Browser controller

- [x] Add `remote_control/static/index.html`, `style.css`, `app.js`: warm gray instrument panel, dark thumb controls, restrained orange accents; no CDN.
- [x] Connect with an in-memory token; require explicit arm; never auto-arm after reconnect. Use a client generation counter to discard stale async results.
- [x] POST command at 20 Hz with at most one in flight. Every stop clears local input immediately and revokes the captured lease, independent of pending commands.
- [x] Pointer capture supports both thumbs with per-axis release. Keyboard W/up, A/left, D/right and space; ignore text fields. Blur, pagehide, visibilitychange and pointercancel stop control.
- [x] Show commanded values, published values, link RTT and explicit lack of hardware feedback; mobile landscape and portrait plus desktop; no fabricated telemetry.
- [x] Verify DOM interaction and screenshots at desktop, 390px portrait, and phone landscape widths.

## 3. Runtime integration and documentation

- [x] Add remote-mode tests to `humanoid_jetson_deploy/tests/test_remote_mode.py`; check port defaults and override, incompatible modes, and live commands after 5 seconds.
- [x] Extend `main.py` with remote option, default port 5007, shared UDP setup, and validation. Preserve fixed and vision behavior.
- [x] Add `docs/web_remote_zh.md` covering LAN/hotspot setup, two-terminal launch, token entry, keyboard/touch, limits, zero command versus hardware stop, and a hardware-free UDP demonstration.
- [x] Link guide from README. Run both repository unittest suites; any unrelated existing failures are reported separately.

## 4. Review and handoff

- [x] Inspect final diff and ownership/race failure paths. Request an independent spec and code quality review, resolve actionable findings, then verify changed behavior.
- [x] Record verified test counts and browser viewport results. Leave existing user edits intact; do not connect to hardware, enable motors, publish a site, or commit unrelated files.

UI references: https://teenage.engineering/products/tp-7 and https://shop.porsche.com/us/en-US/porsche-design-timepieces-custom-built-timepieces . Original layout and CSS; no third-party images or branding copied.


## Verification completed — 2026-10-20

- Repository Python suite: 44 tests passed.
- Humanoid runtime Python suite: 87 tests passed.
- Frontend state/race suite: 11 tests passed (`node --test tests/test_remote_ui.cjs`).
- Real Edge/Chromium browser: 21 checks passed at desktop 1440×1000, portrait 390×844, landscape 844×390; no JavaScript errors.
- Browser checks exercised real HTTP and two-finger input, independent release, keyboard, blur, server watchdog and recovery without automatic arming.
- Publisher shutdown and injected UDP failure revoke control; failure returns HTTP 503.
- Independent review found the Space-stop focus issue; fixed and regression-tested.
- Screenshots are in ignored `logs/remote-qa/`. The CDP harness sends the ended touch in partial touchEnd events.
- No hardware motion was tested. Existing user edits remain untouched. UI refinement deferred at the user’s request.
- Before pushing, the exact staged source was exported and tested independently: 44 repository tests, 86 runtime tests, and 11 frontend tests passed. The extra local joint-calibration test is outside this feature commit.
