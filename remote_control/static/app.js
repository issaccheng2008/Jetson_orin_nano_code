/* Offline, same-origin controller. A browser lease is never restored implicitly. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const ui = {
    panel: $("control-panel"), forward: $("forward-pad"), yaw: $("yaw-pad"),
    forwardThumb: $("forward-thumb"), yawThumb: $("yaw-thumb"),
    targetVx: $("target-vx"), targetWz: $("target-wz"),
    publishedVx: $("published-vx"), publishedWz: $("published-wz"),
    arm: $("arm-button"), armLabel: $("arm-label"), stop: $("stop-button"),
    form: $("connect-form"), token: $("token-input"), connect: $("connect-button"),
    disconnect: $("disconnect-button"), profile: $("speed-profile"),
    link: $("link-status"), linkText: $("link-text"), rtt: $("rtt-value"),
    controlState: $("control-state"), description: $("control-description"),
    notice: $("notice"), noticeText: $("notice-text"), publication: $("publication-state"),
    limitVx: $("limit-vx"), limitWz: $("limit-wz"), timeout: $("timeout-value"),
    reversed: $("yaw-reversed"), connectionDetails: $("connection-details"),
    tuningDetails: $("tuning-details"), keyboardDetails: $("keyboard-details"),
  };

  const state = {
    token: "", connected: false, connecting: false, rejected: false,
    arming: false, session: null, generation: 0, sequence: 0,
    serverActive: false, maxVx: 0.4, maxWz: 0.5, yawSign: 1, timeoutMs: 300,
    publishedVx: 0, publishedWz: 0, publicationKnown: false, rtt: null,
    pointerVx: 0, pointerWz: 0, profile: 1,
    policyManaged: false, policyReady: true,
  };
  const keys = new Set();
  const pointers = new Map();
  const motionCodes = new Set(["KeyW", "ArrowUp", "KeyA", "ArrowLeft", "KeyD", "ArrowRight"]);
  let commandRequest = null;
  let stopRequest = null;
  let pumpTimer = null;
  let pollInFlight = false;

  class ApiError extends Error {
    constructor(message, status = 0) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, body, options = {}) {
    const controller = options.controller || new AbortController();
    const timer = setTimeout(() => controller.abort(), 700);
    const started = performance.now();
    try {
      const response = await fetch(path, {
        method: body === undefined ? "GET" : "POST",
        headers: {
          Authorization: `Bearer ${options.token ?? state.token}`,
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store",
        credentials: "same-origin",
        keepalive: Boolean(options.keepalive),
        signal: controller.signal,
      });
      let data;
      try { data = await response.json(); }
      catch { throw new ApiError("服务响应无法读取", response.status); }
      if (!response.ok) throw new ApiError(data.error || "服务请求失败", response.status);
      return { data, rtt: Math.round(performance.now() - started) };
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(error.name === "AbortError" ? "请求超时" : "无法连接服务");
    } finally {
      clearTimeout(timer);
    }
  }

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const fixed = (value) => Math.abs(value) < 0.005 ? "0.00" : value.toFixed(2);

  function inputs() {
    if (!state.session || !state.policyReady) return { vx: 0, wz: 0 };
    const forward = keys.has("KeyW") || keys.has("ArrowUp") ? 1 : 0;
    const left = keys.has("KeyA") || keys.has("ArrowLeft") ? 1 : 0;
    const right = keys.has("KeyD") || keys.has("ArrowRight") ? 1 : 0;
    return {
      vx: Math.max(state.pointerVx, forward) * state.maxVx * state.profile,
      wz: (hasPointer("yaw") ? state.pointerWz : left - right) * state.maxWz * state.profile,
    };
  }

  function hasPointer(axis) {
    return [...pointers.values()].some((pointer) => pointer.axis === axis);
  }

  function setNotice(message, error = false) {
    ui.noticeText.textContent = message;
    ui.notice.classList.toggle("error", error);
  }

  function render() {
    const active = Boolean(state.session);
    const waiting = active && !state.policyReady;
    const input = inputs();
    const busy = state.connected && state.serverActive && !active;
    document.body.dataset.control = active && !waiting ? "active" : "idle";
    ui.link.classList.toggle("connected", state.connected);
    ui.link.classList.toggle("failed", Boolean(state.token) && !state.connected && !state.connecting);
    ui.linkText.textContent = state.connecting ? "连接中" : state.connected ? "服务已连接" : state.token ? "连接中断" : "未连接";
    ui.rtt.replaceChildren(document.createTextNode(`${state.connected && state.rtt !== null ? state.rtt : "—"} `));
    const milliseconds = document.createElement("small");
    milliseconds.textContent = "ms";
    ui.rtt.append(milliseconds);
    ui.controlState.textContent = waiting ? "上位机启动中" : active ? "控制中" : state.arming ? "启用中" : busy ? "已被占用" : state.connected ? "待启用" : "待连接";
    ui.description.textContent = waiting ? "正在等待上位机和 STM32 就绪。就绪前保持零指令，也可随时停止。" : active ? "保持页面在前台。松手归零，停止后需重新启用。" : state.arming ? "正在申请控制权，请保持页面在前台。" : busy ? "另一会话正在控制，请等待其释放控制权。" : state.connected ? "确认环境安全后，点击启用控制。" : "输入服务口令连接，再手动启用控制。";
    ui.arm.disabled = !state.connected || state.connecting || state.arming || active || busy;
    ui.armLabel.textContent = waiting ? "上位机启动中…" : active ? "控制已启用" : state.arming ? "正在启用…" : "启用控制";
    ui.profile.disabled = active || state.arming;
    ui.connect.disabled = state.connecting;
    ui.disconnect.hidden = !state.token;
    ui.targetVx.textContent = fixed(input.vx);
    ui.targetWz.textContent = fixed(input.wz * state.yawSign);
    ui.publishedVx.textContent = state.publicationKnown ? fixed(state.publishedVx) : "—";
    ui.publishedWz.textContent = state.publicationKnown ? fixed(state.publishedWz) : "—";
    ui.publication.textContent = state.publicationKnown ? "服务端发布指令 · 非实际速度" : state.token ? "等待服务确认" : "等待服务连接";
    ui.limitVx.textContent = fixed(state.maxVx * state.profile);
    ui.limitWz.textContent = fixed(state.maxWz * state.profile);
    ui.timeout.textContent = String(Math.round(state.timeoutMs));
    ui.reversed.hidden = state.yawSign !== -1;
    for (const pad of [ui.forward, ui.yaw]) {
      pad.setAttribute("aria-disabled", String(!active || waiting));
      pad.dataset.held = String(hasPointer(pad === ui.forward ? "forward" : "yaw"));
    }
    ui.forward.setAttribute("aria-valuemax", String(state.maxVx * state.profile));
    ui.forward.setAttribute("aria-valuenow", fixed(input.vx));
    ui.yaw.setAttribute("aria-valuemin", String(-state.maxWz * state.profile));
    ui.yaw.setAttribute("aria-valuemax", String(state.maxWz * state.profile));
    ui.yaw.setAttribute("aria-valuenow", fixed(input.wz * state.yawSign));
    const forwardNormalized = state.maxVx * state.profile > 0 ? input.vx / (state.maxVx * state.profile) : 0;
    const yawNormalized = state.maxWz * state.profile > 0 ? input.wz / (state.maxWz * state.profile) : 0;
    ui.forwardThumb.style.setProperty("--ty", `${-forwardNormalized * travel(ui.forward)}px`);
    ui.yawThumb.style.setProperty("--tx", `${-yawNormalized * travel(ui.yaw)}px`);
  }

  function applyStatus(data, rtt) {
    const wasReady = state.policyReady;
    state.policyManaged = data.policy_managed === true;
    state.policyReady = !state.policyManaged || data.policy_ready === true;
    // A held key/thumb during startup must not become motion once hardware is ready.
    if (!state.policyReady || !wasReady) clearInputs();
    if (state.policyReady && !wasReady && state.session) {
      setNotice("上位机已就绪。请重新推动摇杆或重新按方向键开始移动。");
    }
    state.serverActive = data.active === true;
    if (finite(data.max_vx) && data.max_vx >= 0) state.maxVx = data.max_vx;
    if (finite(data.max_wz) && data.max_wz >= 0) state.maxWz = data.max_wz;
    if (data.yaw_sign === 1 || data.yaw_sign === -1) state.yawSign = data.yaw_sign;
    if (finite(data.timeout_ms)) state.timeoutMs = data.timeout_ms;
    state.publicationKnown = finite(data.vx) && finite(data.wz);
    if (state.publicationKnown) {
      state.publishedVx = data.vx;
      state.publishedWz = data.wz;
    }
    state.rtt = rtt;
  }

  function clearInputs() {
    // Forget pointers before releasing capture: the resulting lost-capture event
    // must never cancel the other thumb or a later control generation.
    const captured = [...pointers.entries()];
    pointers.clear();
    keys.clear();
    state.pointerVx = 0;
    state.pointerWz = 0;
    for (const [id, pointer] of captured) {
      try { if (pointer.pad.hasPointerCapture(id)) pointer.pad.releasePointerCapture(id); }
      catch { /* A browser may already have canceled this pointer. */ }
    }
  }

  function stopPump() {
    clearInterval(pumpTimer);
    pumpTimer = null;
    if (commandRequest) commandRequest.controller.abort();
    commandRequest = null;
  }

  function revokeLease(session, token) {
    return api("/api/stop", { session }, { token, keepalive: true });
  }

  function stopControl(message, { connectionLost = false, error = false } = {}) {
    // Capture both credentials before changing any local state. Stop is sent
    // independently of an in-flight command and invalidates every old response.
    const oldSession = state.session;
    const oldToken = state.token;
    const generation = ++state.generation;
    state.session = null;
    state.arming = false;
    state.connecting = false;
    state.sequence = 0;
    state.publicationKnown = false;
    if (connectionLost) state.connected = false;
    stopPump();
    clearInputs();
    setNotice(message, error);
    render();
    if (!oldSession || !oldToken) return;
    // This request deliberately does not share the command's AbortController.
    const request = { generation };
    stopRequest = request;
    void revokeLease(oldSession, oldToken).then(({ data, rtt }) => {
      if (generation !== state.generation || oldToken !== state.token) return;
      applyStatus(data, rtt);
      render();
    }).catch((failure) => {
      if (generation !== state.generation || oldToken !== state.token) return;
      if (failure.status === 409) {
        // An expired lease is already revoked. Let the next status poll report
        // whether another controller has acquired a new lease in the meantime.
        setNotice("本次控制权已结束，请重新启用。", error);
      } else {
        state.connected = false;
        state.rejected = failure.status === 401;
        setNotice("停止请求未获确认；服务端超时保护将归零指令。恢复后请手动启用。", true);
      }
      render();
    }).finally(() => {
      if (stopRequest === request) stopRequest = null;
    });
  }

  async function connect(event) {
    event.preventDefault();
    const token = ui.token.value.trim();
    if (!token || state.connecting) return;
    stopControl("正在连接遥控服务…");
    state.token = token;
    state.connected = false;
    state.connecting = true;
    state.rejected = false;
    state.serverActive = false;
    const generation = ++state.generation;
    render();
    try {
      const { data, rtt } = await api("/api/status", undefined, { token });
      if (generation !== state.generation) return;
      state.connected = true;
      state.connecting = false;
      applyStatus(data, rtt);
      ui.token.value = "";
      ui.token.placeholder = "已在当前页面连接";
      setNotice(data.active ? "服务已连接，当前控制权被占用。" : "连接成功。启用前，请确认机器人已按本地流程准备就绪。");
      if (landscape.matches) ui.connectionDetails.open = false;
    } catch (failure) {
      if (generation !== state.generation) return;
      state.connecting = false;
      state.connected = false;
      state.rejected = failure.status === 401;
      setNotice(failure.status === 401 ? "口令无效，请核对 Jetson 启动终端中的口令。" : "连接失败，请检查服务地址、Wi-Fi 与 Jetson 服务。", true);
    }
    render();
  }

  async function arm() {
    if (!state.connected || state.session || state.arming || state.connecting || state.serverActive) return;
    if (document.visibilityState === "hidden") return;
    if (keys.size) {
      setNotice("请先松开所有方向键，再启用控制。", true);
      return;
    }
    clearInputs();
    state.arming = true;
    const generation = ++state.generation;
    const token = state.token;
    render();
    try {
      const { data, rtt } = await api("/api/arm", {}, { token });
      if (generation !== state.generation || document.visibilityState === "hidden") {
        // Blur/stop can happen while /arm is in flight. A late lease must never
        // enable the UI; best-effort revoke it, with watchdog expiry as fallback.
        if (typeof data.session === "string") void revokeLease(data.session, token).catch(() => {});
        return;
      }
      if (typeof data.session !== "string" || !data.session || data.active !== true) throw new ApiError("控制会话无效");
      // A key pressed while the arm request was pending must not become a
      // queued movement command. Motion requires a fresh press after enabling.
      clearInputs();
      state.session = data.session;
      state.sequence = 0;
      state.arming = false;
      applyStatus(data, rtt);
      setNotice(state.policyReady ? "控制已启用。上推前进、左右转向；按空格或停止按钮结束控制。" : "上位机正在启动，就绪前摇杆保持零指令。请保持页面在前台。");
      ui.panel.focus({ preventScroll: true });
      render();
      // The first heartbeat is immediate; waiting for a status poll would let
      // the short server lease expire before the first command.
      void sendCommand();
      pumpTimer = setInterval(() => void sendCommand(), 50);
    } catch (failure) {
      if (generation !== state.generation) return;
      state.arming = false;
      if (failure.status === 409) {
        state.serverActive = true;
        setNotice("控制权已被占用，请等待另一会话停止后再试。", true);
      } else {
        state.connected = false;
        state.rejected = failure.status === 401;
        setNotice("未能启用控制。连接恢复后，需要手动再次启用。", true);
      }
      render();
    }
  }

  async function sendCommand() {
    if (!state.session || !state.connected || commandRequest) return;
    const request = { controller: new AbortController(), generation: state.generation };
    commandRequest = request;
    const input = inputs();
    const body = { session: state.session, seq: ++state.sequence, vx: input.vx, wz: input.wz };
    try {
      const { data, rtt } = await api("/api/command", body, { controller: request.controller });
      if (request.generation !== state.generation) return;
      if (!data.active) {
        stopControl("控制权已失效，请重新启用。", { error: true });
        return;
      }
      applyStatus(data, rtt);
      render();
    } catch (failure) {
      if (request.generation !== state.generation) return;
      state.rejected = failure.status === 401;
      stopControl(failure.status === 409 ? "控制权已失效，输入已清空。请重新启用。" : "指令发送中断，已退出控制。连接恢复后请重新启用。", { connectionLost: failure.status !== 409, error: true });
    } finally {
      if (commandRequest === request) commandRequest = null;
    }
  }

  async function pollStatus() {
    if (!state.token || state.connecting || state.rejected || pollInFlight || stopRequest || state.arming || document.visibilityState === "hidden") return;
    // Command responses carry the same status while this browser owns a lease.
    // Avoid a concurrent older status snapshot overwriting newer command data.
    if (state.session) return;
    pollInFlight = true;
    const generation = state.generation;
    const token = state.token;
    try {
      const { data, rtt } = await api("/api/status", undefined, { token });
      if (generation !== state.generation) return;
      const recovered = !state.connected;
      state.connected = true;
      applyStatus(data, rtt);
      if (recovered) setNotice(data.active ? "连接已恢复，控制权暂被占用。" : "连接已恢复，当前未启用控制。请手动启用。");
      render();
    } catch (failure) {
      if (generation !== state.generation) return;
      state.connected = false;
      state.publicationKnown = false;
      state.rejected = failure.status === 401;
      setNotice(failure.status === 401 ? "访问口令已失效，请重新连接。" : "无法连接服务，请检查 Wi-Fi 与 Jetson 服务。", true);
      render();
    } finally {
      pollInFlight = false;
    }
  }

  function travel(pad) { return pad.clientWidth * 0.265; }

  function updatePointer(event, pointer) {
    const bounds = pointer.pad.getBoundingClientRect();
    const range = Math.max(1, travel(pointer.pad));
    const raw = pointer.axis === "forward" ? (bounds.top + bounds.height / 2 - event.clientY) / range : (bounds.left + bounds.width / 2 - event.clientX) / range;
    const bounded = pointer.axis === "forward" ? clamp(raw, 0, 1) : clamp(raw, -1, 1);
    const value = Math.abs(bounded) < 0.035 ? 0 : bounded;
    if (pointer.axis === "forward") state.pointerVx = value;
    else state.pointerWz = value;
    render();
  }

  function releasePointer(event) {
    const pointer = pointers.get(event.pointerId);
    if (!pointer) return;
    pointers.delete(event.pointerId);
    if (pointer.axis === "forward") state.pointerVx = 0;
    else state.pointerWz = 0;
    try { if (pointer.pad.hasPointerCapture(event.pointerId)) pointer.pad.releasePointerCapture(event.pointerId); }
    catch { /* Capture can already be released when a pointer leaves the page. */ }
    render();
  }

  function installPointer(pad, axis) {
    pad.addEventListener("pointerdown", (event) => {
      if (!state.session || !state.policyReady || (event.pointerType === "mouse" && event.button !== 0) || hasPointer(axis)) return;
      event.preventDefault();
      const pointer = { pad, axis };
      pointers.set(event.pointerId, pointer);
      try { pad.setPointerCapture(event.pointerId); }
      catch {
        stopControl("浏览器未能保持触控，已停止控制。", { error: true });
        return;
      }
      updatePointer(event, pointer);
    });
    pad.addEventListener("pointermove", (event) => {
      const pointer = pointers.get(event.pointerId);
      if (!pointer || pointer.pad !== pad || !state.session) return;
      event.preventDefault();
      updatePointer(event, pointer);
    });
    pad.addEventListener("pointerup", releasePointer);
    // Normal pointerup also dispatches lostpointercapture. Release is idempotent
    // and axis-local, so releasing one thumb never zeros the other thumb.
    pad.addEventListener("lostpointercapture", releasePointer);
  }

  function isTypingTarget(target) {
    return target instanceof Element && Boolean(target.closest("input, textarea, select, button, [contenteditable]:not([contenteditable='false'])"));
  }

  window.addEventListener("keydown", (event) => {
    // Once enabled (or requesting a lease), Space is always the stop key,
    // including when focus has moved into a field or a button.
    if (event.code === "Space" && (state.session || state.arming)) {
      event.preventDefault();
      stopControl("已停止。再次移动前，请重新启用控制。");
      return;
    }
    if (isTypingTarget(event.target) || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.code === "Space") {
      event.preventDefault();
      if (!event.repeat) stopControl("已停止。再次移动前，请重新启用控制。");
      return;
    }
    if (!motionCodes.has(event.code)) return;
    event.preventDefault();
    if (state.session && !state.policyReady) return;
    // Repeated events cannot revive a key held through stop, blur, or arming.
    if (event.repeat) return;
    keys.add(event.code);
    render();
  });
  window.addEventListener("keyup", (event) => {
    if (keys.delete(event.code)) {
      if (!isTypingTarget(event.target)) event.preventDefault();
      render();
    }
  });

  function interrupt(reason) {
    if (state.session || state.arming) stopControl(reason);
    else clearInputs();
  }
  window.addEventListener("blur", () => interrupt("页面已失焦，控制已停止。返回后请手动启用。"));
  window.addEventListener("pagehide", () => {
    // Also invalidate a connecting request; nothing may revive this document.
    stopControl("页面已退出，控制已停止。");
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") interrupt("页面已切至后台，控制已停止。返回后请手动启用。");
    else { render(); void pollStatus(); }
  });
  window.addEventListener("pointercancel", () => interrupt("触控已被中断，控制已停止。请重新启用。"));
  window.addEventListener("offline", () => stopControl("网络已断开，控制已停止。恢复后请手动启用。", { connectionLost: true, error: true }));
  window.addEventListener("online", () => void pollStatus());
  window.addEventListener("resize", render);

  ui.form.addEventListener("submit", (event) => void connect(event));
  ui.arm.addEventListener("click", () => void arm());
  ui.stop.addEventListener("click", () => stopControl("已停止。再次移动前，请重新启用控制。"));
  ui.disconnect.addEventListener("click", () => {
    stopControl("已断开连接，当前页面的口令已清除。");
    // Invalidate the stop response, but keep its captured credentials in flight.
    ++state.generation;
    state.token = "";
    state.connected = false;
    state.serverActive = false;
    state.rejected = false;
    state.rtt = null;
    ui.token.value = "";
    ui.token.placeholder = "输入访问口令";
    render();
  });
  ui.profile.addEventListener("change", () => {
    if (state.session || state.arming) return;
    const value = Number(ui.profile.value);
    state.profile = [0.25, 0.5, 1].includes(value) ? value : 1;
    render();
  });
  installPointer(ui.forward, "forward");
  installPointer(ui.yaw, "yaw");

  const landscape = window.matchMedia("(max-height: 520px) and (min-width: 600px)");
  function adaptDetails() {
    ui.keyboardDetails.open = !landscape.matches;
    ui.tuningDetails.open = !landscape.matches;
    ui.connectionDetails.open = !landscape.matches || !state.connected;
    render();
  }
  landscape.addEventListener("change", adaptDetails);
  adaptDetails();
  setInterval(() => void pollStatus(), 800);
})();
