/* Run with: node --test tests/test_remote_ui.cjs
 * Contract and safety-race tests. Browser layout/pointer tests remain separate.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

class Surface {
  constructor(id = "") {
    this.id = id;
    this.handlers = new Map();
    this.attributes = new Map();
    this.captured = new Set();
    this.dataset = {};
    this.textContent = "";
    this.value = "";
    this.clientWidth = 200;
    this.disabled = false;
    this.open = true;
    this.classList = { toggle() {} };
    this.style = { setProperty() {} };
  }
  addEventListener(type, fn) {
    if (!this.handlers.has(type)) this.handlers.set(type, []);
    this.handlers.get(type).push(fn);
  }
  dispatch(type, props = {}) {
    const event = { target: this, preventDefault() {}, ...props };
    for (const fn of this.handlers.get(type) || []) fn(event);
  }
  setAttribute(name, value) { this.attributes.set(name, value); }
  getAttribute(name) { return this.attributes.get(name); }
  replaceChildren() {}
  append() {}
  focus() {}
  closest() { return this.editable ? this : null; }
  getBoundingClientRect() { return { top: 0, left: 0, width: 200, height: 200 }; }
  setPointerCapture(id) { this.captured.add(id); }
  hasPointerCapture(id) { return this.captured.has(id); }
  releasePointerCapture(id) {
    this.captured.delete(id);
    this.dispatch("lostpointercapture", { pointerId: id });
  }
}

const snapshot = (active = false, extra = {}) => ({
  active, target_vx: 0, target_wz: 0, vx: 0, wz: 0,
  max_vx: 0.4, max_wz: 0.5, yaw_sign: 1, timeout_ms: 300, command_age_ms: 0, ...extra,
});
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const flush = async () => { for (let i = 0; i < 15; i++) await Promise.resolve(); };

function harness() {
  const elements = new Map();
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, new Surface(id));
    return elements.get(id);
  };
  const window = new Surface("window");
  const document = new Surface("document");
  document.visibilityState = "visible";
  document.body = new Surface("body");
  document.getElementById = get;
  document.createTextNode = () => ({});
  document.createElement = () => new Surface();
  const media = new Surface("media");
  media.matches = false;
  window.matchMedia = () => media;
  let timerId = 0;
  const intervals = new Map();
  const calls = [];
  const routes = new Map();
  const request = (url, options) => {
    const call = { url, ...options, parsed: options.body ? JSON.parse(options.body) : undefined };
    calls.push(call);
    const route = routes.get(url);
    if (route) return route(call);
    if (url === "/api/status") return Promise.resolve(response(snapshot()));
    if (url === "/api/arm") return Promise.resolve(response({ session: "lease-one", ...snapshot(true) }));
    if (url === "/api/command") return Promise.resolve(response(snapshot(true)));
    if (url === "/api/stop") return Promise.resolve(response(snapshot()));
    throw Error(`Unknown request ${url}`);
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../remote_control/static/app.js"), "utf8"), {
    window, document, Element: Surface, AbortController, fetch: request,
    performance: { now: () => 42 },
    setTimeout: () => ++timerId, clearTimeout() {},
    setInterval: (fn, period) => { const id = ++timerId; intervals.set(id, { fn, period }); return id; },
    clearInterval: (id) => intervals.delete(id), console,
  });
  return {
    get, window, document, calls, routes,
    last(url) { return calls.filter((call) => call.url === url).at(-1); },
    count(url) { return calls.filter((call) => call.url === url).length; },
    async tick(period) { for (const timer of [...intervals.values()]) if (timer.period === period) timer.fn(); await flush(); },
    async connect() { get("token-input").value = "test-token"; get("connect-form").dispatch("submit"); await flush(); },
    async arm() { get("arm-button").dispatch("click"); await flush(); },
    key(type, code, extra = {}) { window.dispatch(type, { target: document.body, code, repeat: false, ...extra }); },
  };
}

test("starts disabled; authenticated connect never arms; arm sends immediate zero heartbeat", async () => {
  const h = harness();
  assert.equal(h.get("arm-button").disabled, true);
  await h.connect();
  assert.equal(h.get("arm-button").disabled, false);
  assert.equal(h.count("/api/arm"), 0);
  assert.equal(h.last("/api/status").headers.Authorization, "Bearer test-token");
  assert.equal(h.get("token-input").value, "");
  await h.arm();
  assert.equal(h.get("control-state").textContent, "控制中");
  assert.deepEqual(h.last("/api/command").parsed, { session: "lease-one", seq: 1, vx: 0, wz: 0 });
});

test("two pointers operate independently and releasing one preserves the other axis", async () => {
  const h = harness();
  await h.connect(); await h.arm();
  const forward = h.get("forward-pad"), yaw = h.get("yaw-pad");
  forward.dispatch("pointerdown", { pointerId: 1, pointerType: "touch", clientX: 100, clientY: 40 });
  yaw.dispatch("pointerdown", { pointerId: 2, pointerType: "touch", clientX: 40, clientY: 100 });
  await h.tick(50);
  assert.equal(h.last("/api/command").parsed.vx, 0.4);
  assert.equal(h.last("/api/command").parsed.wz, 0.5);
  forward.dispatch("pointerup", { pointerId: 1 });
  await h.tick(50);
  assert.equal(h.last("/api/command").parsed.vx, 0);
  assert.equal(h.last("/api/command").parsed.wz, 0.5);
  assert.equal(h.get("control-state").textContent, "控制中");
  assert.equal(h.count("/api/stop"), 0);
  yaw.dispatch("pointerup", { pointerId: 2 });
  await h.tick(50);
  assert.equal(h.last("/api/command").parsed.wz, 0);
});

test("forward cannot reverse; right is negative; profile scales both limits", async () => {
  const h = harness(); await h.connect();
  h.get("speed-profile").value = "0.25";
  h.get("speed-profile").dispatch("change");
  await h.arm();
  h.get("forward-pad").dispatch("pointerdown", { pointerId: 1, pointerType: "touch", clientX: 100, clientY: 190 });
  h.get("yaw-pad").dispatch("pointerdown", { pointerId: 2, pointerType: "touch", clientX: 190, clientY: 100 });
  await h.tick(50);
  assert.equal(h.last("/api/command").parsed.vx, 0);
  assert.equal(h.last("/api/command").parsed.wz, -0.125);
  h.get("forward-pad").dispatch("pointermove", { pointerId: 1, clientX: 100, clientY: 10 });
  await h.tick(50);
  assert.equal(h.last("/api/command").parsed.vx, 0.1);
});

test("stop bypasses a pending command and its late response cannot re-enable motion", async () => {
  const h = harness(); await h.connect(); await h.arm();
  let release;
  h.routes.set("/api/command", () => new Promise((resolve) => { release = resolve; }));
  h.key("keydown", "KeyW"); await h.tick(50);
  const pending = h.last("/api/command");
  const commandCount = h.count("/api/command");
  await h.tick(50);
  assert.equal(h.count("/api/command"), commandCount, "only one command may be in flight");
  h.get("stop-button").dispatch("click");
  assert.deepEqual(h.last("/api/stop").parsed, { session: "lease-one" });
  assert.equal(pending.signal.aborted, true);
  release(response(snapshot(true, { vx: 0.4 })));
  await flush(); await h.tick(50);
  assert.notEqual(h.get("control-state").textContent, "控制中");
  assert.equal(h.get("target-vx").textContent, "0.00");
  assert.equal(h.count("/api/command"), commandCount);
});

test("late arm response after blur is revoked without a single movement command", async () => {
  const h = harness(); await h.connect();
  let release;
  h.routes.set("/api/arm", () => new Promise((resolve) => { release = resolve; }));
  await h.arm();
  h.window.dispatch("blur");
  release(response({ session: "late-lease", ...snapshot(true) }));
  await flush();
  assert.equal(h.count("/api/command"), 0);
  assert.deepEqual(h.last("/api/stop").parsed, { session: "late-lease" });
  assert.notEqual(h.get("control-state").textContent, "控制中");
});

test("stale idle poll cannot disarm a newly acquired session", async () => {
  const h = harness(); await h.connect();
  let release;
  h.routes.set("/api/status", () => new Promise((resolve) => { release = resolve; }));
  await h.tick(800);
  await h.arm();
  release(response(snapshot())); await flush();
  assert.equal(h.get("control-state").textContent, "控制中");
  h.key("keydown", "KeyW"); await h.tick(50);
  assert.equal(h.last("/api/command").parsed.vx, 0.4);
});

test("held keys block arm; a key pressed during arm requires a new press", async () => {
  const h = harness(); await h.connect();
  h.key("keydown", "KeyW"); await h.arm();
  assert.equal(h.count("/api/arm"), 0);
  h.key("keyup", "KeyW");
  let release;
  h.routes.set("/api/arm", () => new Promise((resolve) => { release = resolve; }));
  await h.arm(); h.key("keydown", "KeyW");
  release(response({ session: "lease-two", ...snapshot(true) })); await flush();
  assert.equal(h.last("/api/command").parsed.vx, 0);
  h.key("keydown", "KeyW", { repeat: true }); await h.tick(50);
  assert.equal(h.last("/api/command").parsed.vx, 0);
  h.key("keyup", "KeyW"); h.key("keydown", "KeyW"); await h.tick(50);
  assert.equal(h.last("/api/command").parsed.vx, 0.4);
});

test("input fields ignore movement keys; keyup, pointercancel and space clear motion", async () => {
  const h = harness(); await h.connect(); await h.arm();
  h.get("token-input").editable = true;
  h.key("keydown", "KeyW", { target: h.get("token-input") }); await h.tick(50);
  assert.equal(h.last("/api/command").parsed.vx, 0);
  h.key("keydown", "KeyA"); await h.tick(50);
  assert.equal(h.last("/api/command").parsed.wz, 0.5);
  h.key("keyup", "KeyA"); await h.tick(50);
  assert.equal(h.last("/api/command").parsed.wz, 0);
  h.window.dispatch("pointercancel"); await flush();
  assert.notEqual(h.get("control-state").textContent, "控制中");
  await h.arm(); h.key("keydown", "Space"); await flush();
  assert.notEqual(h.get("control-state").textContent, "控制中");
});

test("command failure exits control; connection recovery never re-arms", async () => {
  const h = harness(); await h.connect(); await h.arm();
  h.routes.set("/api/command", async () => { throw Error("network failure"); });
  h.key("keydown", "KeyW"); await h.tick(50); await flush();
  assert.notEqual(h.get("control-state").textContent, "控制中");
  assert.equal(h.get("target-vx").textContent, "0.00");
  await h.tick(800);
  assert.equal(h.get("control-state").textContent, "待启用");
  assert.equal(h.count("/api/arm"), 1);
});

test("Space stops active or pending control from fields and buttons, but idle Space is normal", async () => {
  const h = harness(); await h.connect();
  const field = h.get("token-input"), button = h.get("arm-button");
  field.editable = true;
  button.editable = true;
  let prevented = false;
  h.key("keydown", "Space", { target: field, preventDefault() { prevented = true; } });
  assert.equal(prevented, false, "idle input retains its normal space behavior");
  assert.equal(h.count("/api/stop"), 0);
  await h.arm();
  h.key("keydown", "Space", { target: field, preventDefault() { prevented = true; } });
  await flush();
  assert.equal(prevented, true);
  assert.equal(h.count("/api/stop"), 1);
  assert.notEqual(h.get("control-state").textContent, "控制中");
  await h.arm();
  h.key("keydown", "Space", { target: button }); await flush();
  assert.equal(h.count("/api/stop"), 2);
  assert.notEqual(h.get("control-state").textContent, "控制中");
  let release;
  h.routes.set("/api/arm", () => new Promise((resolve) => { release = resolve; }));
  await h.arm();
  const commandCount = h.count("/api/command");
  h.key("keydown", "Space", { target: field });
  release(response({ session: "space-canceled-arm", ...snapshot(true) })); await flush();
  assert.equal(h.count("/api/command"), commandCount);
  assert.deepEqual(h.last("/api/stop").parsed, { session: "space-canceled-arm" });
});

test("yaw sign affects the display but is applied by the server, only once", async () => {
  const h = harness(); await h.connect();
  h.routes.set("/api/arm", async () => response({ session: "lease-flip", ...snapshot(true, { yaw_sign: -1 }) }));
  h.routes.set("/api/command", async () => response(snapshot(true, { yaw_sign: -1 })));
  await h.arm(); h.key("keydown", "KeyA"); await h.tick(50);
  assert.equal(h.last("/api/command").parsed.wz, 0.5);
  assert.equal(h.get("target-wz").textContent, "-0.50");
});
