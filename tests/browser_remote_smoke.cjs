/* Optional real-browser smoke test against a LOCAL, hardware-free service.
 * ROBOT_REMOTE_TEST_TOKEN=... node tests/browser_remote_smoke.cjs
 * Requires playwright and a Chromium installation; set ROBOT_BROWSER_CHANNEL=msedge if needed.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const base = process.env.ROBOT_REMOTE_TEST_URL || 'http://127.0.0.1:8088';
const token = process.env.ROBOT_REMOTE_TEST_TOKEN;
assert.equal(new URL(base).hostname, '127.0.0.1', 'Only a local simulation may be tested');
assert.ok(token, 'Set ROBOT_REMOTE_TEST_TOKEN for the local simulation');
const screenshotDir = process.env.ROBOT_REMOTE_SCREENSHOTS || 'logs/remote-qa';
fs.mkdirSync(screenshotDir, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.ROBOT_BROWSER_CHANNEL ? { channel: process.env.ROBOT_BROWSER_CHANNEL } : {}) });
  let checks = 0;
  const errors = [];
  async function status() {
    const result = await fetch(`${base}/api/status`, { headers: { Authorization: `Bearer ${token}` } });
    assert.equal(result.status, 200);
    return result.json();
  }
  async function until(predicate, label) {
    const deadline = Date.now() + 3500;
    let last;
    while (Date.now() < deadline) {
      last = await status();
      if (predicate(last)) { checks++; console.log('PASS', label); return last; }
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    assert.fail(`${label}: ${JSON.stringify(last)}`);
  }
  async function connect(page) {
    await page.locator('#token-input').fill(token);
    await page.locator('#connect-button').click();
    await page.waitForFunction(() => !document.getElementById('arm-button').disabled);
  }
  async function arm(page) {
    await page.locator('#arm-button').click();
    await page.waitForFunction(() => document.getElementById('control-state').textContent === '控制中');
  }
  try {
    const desktop = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await desktop.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.waitForLoadState('networkidle');
    await connect(page);
    await until(s => !s.active, 'connection alone never arms');
    await arm(page);
    await page.keyboard.down('w');
    await page.keyboard.down('a');
    await until(s => s.target_vx === .4 && s.target_wz === .5, 'keyboard forward plus left turn');
    await page.keyboard.up('w');
    await until(s => s.target_vx === 0 && s.target_wz === .5, 'forward release keeps yaw');
    await page.keyboard.up('a');
    await until(s => s.vx === 0 && s.wz === 0 && s.active, 'normal release ramps to zero');
    await page.keyboard.down('w');
    await until(s => s.vx > 0, 'movement reaches real service');
    await page.keyboard.press('Space');
    await until(s => !s.active && s.vx === 0 && s.wz === 0, 'space revokes lease');
    await page.keyboard.up('w');
    await page.waitForFunction(() => !document.getElementById('arm-button').disabled);
    await arm(page);
    await page.keyboard.down('w');
    await until(s => s.target_vx > 0, 'rearm works');
    await page.evaluate(() => window.dispatchEvent(new Event('blur')));
    await until(s => !s.active && s.vx === 0, 'blur immediately zeros and revokes');
    await page.keyboard.up('w');
    await page.waitForFunction(() => !document.getElementById('arm-button').disabled);
    await arm(page);
    await page.keyboard.down('w');
    await until(s => s.vx > 0, 'pre-disconnect movement');
    await desktop.setOffline(true);
    await until(s => !s.active && s.vx === 0, 'network loss expires on server');
    await desktop.setOffline(false);
    await page.keyboard.up('w');
    await page.waitForFunction(() => !document.getElementById('arm-button').disabled);
    await until(s => !s.active, 'network recovery does not auto-arm');
    await page.screenshot({ path: path.join(screenshotDir, 'desktop.png'), fullPage: true });
    await desktop.close();

    for (const [name, width, height] of [['mobile', 390, 844], ['landscape', 844, 390]]) {
      const context = await browser.newContext({ viewport: { width, height }, isMobile: true, hasTouch: true });
      const p = await context.newPage();
      p.on('pageerror', error => errors.push(error.message));
      await p.goto(base);
      await p.waitForLoadState('networkidle');
      await connect(p);
      await arm(p);
      await p.locator('#control-panel').scrollIntoViewIfNeeded();
      const forward = await p.locator('#forward-pad').boundingBox();
      const yaw = await p.locator('#yaw-pad').boundingBox();
      const a = { id: 1, x: forward.x + forward.width / 2, y: forward.y + forward.height / 2 };
      const b = { id: 2, x: yaw.x + yaw.width / 2, y: yaw.y + yaw.height / 2 };
      const cdp = await context.newCDPSession(p);
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [a, b] });
      a.y -= forward.width * .22;
      b.x += yaw.width * .22;
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [a, b] });
      await until(s => s.target_vx > .2 && s.target_wz < -.2, `${name}: real two-finger input`);
      // CDP ends the listed touch here; end the forward finger, preserving yaw.
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [a] });
      await until(s => s.target_vx === 0 && s.target_wz < -.2, `${name}: releasing one finger preserves the other`);
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      await until(s => s.target_vx === 0 && s.target_wz === 0, `${name}: both fingers released`);
      await p.locator('#stop-button').click();
      await until(s => !s.active, `${name}: stop button`);
      const layout = await p.evaluate(() => ({ width: innerWidth,
        content: document.documentElement.scrollWidth,
        padWidth: document.getElementById('forward-pad').clientWidth,
        stopBottom: document.getElementById('stop-button').getBoundingClientRect().bottom }));
      assert.equal(layout.width, width);
      assert.ok(layout.content <= width, `${name}: horizontal overflow`);
      assert.ok(layout.padWidth >= 120, `${name}: touch target too small`);
      if (name === 'landscape') assert.ok(layout.stopBottom <= height, 'landscape stop must stay on screen');
      checks++;
      await p.screenshot({ path: path.join(screenshotDir, `${name}.png`), fullPage: true });
      await context.close();
    }
    assert.deepEqual(errors, [], 'browser JavaScript errors');
    console.log(`PASS ${checks} browser checks; no JavaScript errors; screenshots in ${screenshotDir}`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
