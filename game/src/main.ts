import './style.css';
import { World, DT, ACTION_SECONDS, GRID_SIZE, ACTIONS, rng, ruleAction, type Controller, type Action, type Input } from './simulation.ts';
import { GameView, GameAudio } from './view.ts';
import { decide, type Decision, type ModelMode } from './decision.ts';
import { GameplayRecording } from './recording.ts';

const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id)! as T;
const canvas = element<HTMLCanvasElement>('world');
let world = new World(42);
const view = new GameView(canvas, world);
const audio = new GameAudio();
type Screen = 'menu' | 'playing' | 'paused' | 'result' | 'error';
let screen: Screen = 'menu', controller: Controller = 'manual', generation = 0;
let pending: AbortController | null = null, thinking = false, action: Action | null = null, actionTicks = 0;
let accumulator = 0, previousFrame = performance.now(), hudAt = 0, eventCursor = 0, messageUntil = 0;
let runId = '', startedAt = 0, endedAt = 0, mouseTurn = 0, firePressed = false;
let decisions: (Decision & { time: number; state: string })[] = [];
let frames: { ms: number; waiting: boolean; drawCalls: number; triangles: number }[] = [];
let recording: GameplayRecording | null = null;
const recordingStatus = document.createElement('p');
recordingStatus.setAttribute('role', 'status'); recordingStatus.id = 'recording-status';
element('result-stats').after(recordingStatus);
const keys = new Set<string>();
let pilotRandom = rng(42 ^ 0x1baf);
const missionSeed = () => {
  const input = element<HTMLInputElement>('mission-seed'), seed = Number(input.value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) { input.value = '42'; return 42; }
  return seed;
};
const names: Record<string, string> = { manual: 'Manual control', rule: 'Rule baseline', random: 'Random baseline',
  laya: 'Laya · local', decoder: 'ModernBERT Decoder · local', jev: 'Jev · TypeSafe API' };

function setScreen(next: Screen) {
  screen = next;
  for (const id of ['menu', 'pause', 'result', 'error']) element(id).hidden = id === 'pause' ? next !== 'paused' : next !== id;
  document.body.classList.toggle('menu-open', next === 'menu');
  element('touch-controls').hidden = next !== 'playing';
  element('thinking').style.display = thinking && next === 'playing' ? 'flex' : 'none';
  if (next !== 'playing') {
    keys.clear(); mouseTurn = 0; firePressed = false; accumulator = 0;
    if (document.pointerLockElement) document.exitPointerLock();
  }
}

function start(mode: Controller = 'manual', seed = 42) {
  if (recording && (screen === 'result' || screen === 'error')) return;
  discardRecording();
  generation++; pending?.abort(); pending = null;
  world = new World(seed); view.setWorld(world); audio.cursor = 0;
  controller = mode; action = null; actionTicks = 0; thinking = false;
  pilotRandom = rng(seed ^ 0x1baf);
  decisions = []; frames = []; eventCursor = 0; accumulator = 0; keys.clear(); mouseTurn = 0; firePressed = false;
  runId = `browser-${mode}-${seed}-${Date.now()}`; startedAt = performance.now(); endedAt = 0;
  recordingStatus.textContent = '';
  element('result-stats').after(recordingStatus);
  if (element<HTMLInputElement>('record-gameplay').checked) {
    try { recording = new GameplayRecording(); }
    catch (error) { element('error-message').textContent = String(error); setScreen('error'); return; }
  }
  element('telemetry').hidden = mode === 'manual';
  element('telemetry-model').textContent = names[mode];
  element('confidence-label').textContent = mode === 'jev' ? 'Jev confidence' : 'Entropy confidence';
  element('pilot').textContent = names[mode].toUpperCase();
  element('last-action').textContent = element('last-latency').textContent = element('last-confidence').textContent = '—';
  element('decision-count').textContent = '0';
  element('action-probabilities').replaceChildren();
  element('observed-state').textContent = 'Waiting for the first decision.';
  element('hint').innerHTML = mode === 'manual' ? 'W A S D <span>move</span> · MOUSE / ARROWS <span>aim</span> · CLICK / SPACE <span>fire</span> · ESC <span>pause</span>' : 'LIVE AI PILOT <span>2-second tactical actions</span> · TAB <span>telemetry</span> · ESC <span>pause</span>';
  setScreen('playing'); audio.start(); announce('CONTAINMENT BREACHED · CLEAR ALL HOSTILES'); updateHUD();
}

function discardRecording() {
  const capture = recording;
  recording = null;
  if (capture) void capture.stop().catch(error => console.warn('Recording stopped:', error));
}

function showMenu() {
  if (recording && (screen === 'result' || screen === 'error')) return;
  discardRecording();
  generation++; pending?.abort(); pending = null; thinking = false;
  world = new World(42); view.setWorld(world); element('telemetry').hidden = true;
  element('message').style.opacity = '0'; setScreen('menu');
}

function pause() { if (screen === 'playing') setScreen('paused'); }
function resume() { if (screen === 'paused') { accumulator = 0; setScreen('playing'); } }
function announce(text: string) { element('message').textContent = text; messageUntil = performance.now() + 2100; }

async function nextAction() {
  if (thinking || screen !== 'playing' || controller === 'manual') return;
  const observed = world.observe(), token = generation;
  if (controller === 'rule' || controller === 'random') {
    action = controller === 'rule' ? ruleAction(observed) : ACTIONS[Math.floor(pilotRandom() * ACTIONS.length)];
    actionTicks = Math.round(ACTION_SECONDS / DT);
    decisions.push({ action, source: controller, mode: controller, elapsed_ms: 0, client_ms: 0, laya_ms: 0,
      escalated: false, time: world.time, state: observed.text });
    updateTelemetry(); return;
  }
  thinking = true; element('thinking').style.display = 'flex'; pending = new AbortController();
  try {
    const result = await decide(observed.text, controller as ModelMode, runId, '', pending.signal);
    if (generation !== token) return;
    decisions.push({ ...result, time: world.time, state: observed.text });
    action = result.action; actionTicks = Math.round(ACTION_SECONDS / DT); updateTelemetry();
  } catch (e) {
    if (generation !== token) return;
    element('error-message').textContent = e instanceof Error ? e.message : String(e);
    endedAt = performance.now(); thinking = false;
    setScreen('error');
    element('error-message').after(recordingStatus);
    saveRecording();
  } finally {
    if (generation === token) { thinking = false; pending = null; accumulator = 0; element('thinking').style.display = 'none'; }
  }
}

function updateTelemetry() {
  const d = decisions.at(-1); if (!d) return;
  element('last-action').textContent = `${d.action.toUpperCase()} / ${d.source}`;
  element('last-latency').textContent = `${Math.round(d.client_ms).toLocaleString()} ms`;
  const confidence = d.jev_confidence ?? d.entropy_confidence;
  element('last-confidence').textContent = confidence === undefined ? '—' : `${(confidence * 100).toFixed(1)}%`;
  element('decision-count').textContent = String(decisions.length);
  element('observed-state').textContent = d.state;
  const probabilities = element('action-probabilities');
  probabilities.replaceChildren();
  if (d.probabilities) for (const name of ACTIONS) {
    const row = document.createElement('div'); row.className = 'probability-row';
    const label = document.createElement('span'); label.textContent = name;
    const bar = document.createElement('meter'); bar.min = 0; bar.max = 1; bar.value = d.probabilities[name];
    bar.setAttribute('aria-label', `${name} probability`);
    const value = document.createElement('span'); value.textContent = `${(d.probabilities[name] * 100).toFixed(1)}%`;
    row.append(label, bar, value); probabilities.append(row);
  }
}

function finish() {
  endedAt = performance.now();
  const won = world.status === 'won';
  element('result-kicker').textContent = `${names[controller].toUpperCase()} / SEED ${world.seed}`;
  element('result-title').textContent = won ? 'EXTRACTED' : world.status === 'dead' ? 'SIGNAL LOST' : 'TIME EXPIRED';
  element('result-copy').textContent = won ? 'Core secured. Every hostile signal silenced. You made it out.' : 'The mission is over. Run the station again with another pilot.';
  element('result-stats').innerHTML = `<div><b>${world.player.kills}/6</b><small>HOSTILES</small></div><div><b>${world.time.toFixed(1)}s</b><small>SIM TIME</small></div><div><b>${world.player.hp}</b><small>HEALTH</small></div><div><b>${decisions.length}</b><small>DECISIONS</small></div>`;
  setScreen('result'); updateHUD();
  saveRecording();
}

function saveRecording() {
  if (recording) {
    recordingStatus.textContent = 'Saving gameplay and telemetry…';
    const captured = { date: new Date().toISOString(), state: { screen, controller, thinking, ...world.snapshot() },
      metrics: metrics(), decisions: structuredClone(decisions), events: structuredClone(world.events), frames: structuredClone(frames),
      error: screen === 'error' ? element('error-message').textContent : null,
      capture: 'MediaRecorder of live rendered canvas with telemetry overlay, normal speed, inference waits retained, 1280x720, no audio' };
    const capture = recording, capturedRunId = runId;
    setTimeout(async () => {
      if (recording !== capture) return;
      try {
        const blob = await capture.stop(); recording = null;
        if (runId !== capturedRunId) return;
        recordingStatus.replaceChildren(document.createTextNode('Recording ready · '));
        for (const [label, suffix, content] of [
          ['DOWNLOAD VIDEO', 'webm', blob],
          ['DOWNLOAD DECISIONS', 'json', new Blob([JSON.stringify(captured, null, 2)], { type: 'application/json' })],
        ] as const) {
          const link = document.createElement('a'); link.href = URL.createObjectURL(content);
          link.download = `${capturedRunId}.${suffix}`; link.textContent = label;
          recordingStatus.append(link, document.createTextNode(' · '));
        }
      } catch (error) {
        if (recording === capture) recording = null;
        if (runId === capturedRunId) recordingStatus.textContent = String(error);
      }
    }, 2000);
  }
}

function manualInput(): Input {
  const turn = (Number(keys.has('ArrowRight')) - Number(keys.has('ArrowLeft'))) * 2.1 * DT + mouseTurn;
  const fire = firePressed || keys.has('Space') || keys.has('Mouse0'); firePressed = false;
  mouseTurn = 0;
  return { forward: Number(keys.has('KeyW') || keys.has('ArrowUp')) - Number(keys.has('KeyS') || keys.has('ArrowDown')),
    strafe: Number(keys.has('KeyD')) - Number(keys.has('KeyA')), turn, fire };
}

function updateHUD() {
  const p = world.player;
  element('health').textContent = String(Math.ceil(p.hp)).padStart(3, '0');
  element('health-fill').style.width = `${p.hp}%`;
  element('health-fill').style.background = p.hp < 35 ? '#e7774b' : '#b8d68b';
  element('status-line').textContent = p.hp < 35 ? 'CRITICAL · FIND A MEDKIT' : 'SUIT INTEGRITY NOMINAL';
  element('ammo').textContent = String(p.ammo).padStart(2, '0');
  element('kills').textContent = String(p.kills);
  element('clock').textContent = `${String(Math.floor(world.time / 60)).padStart(2, '0')}:${String(Math.floor(world.time % 60)).padStart(2, '0')}`;
  element('objective').textContent = p.core ? 'REACH THE EXIT' : p.kills === 6 ? 'RECOVER THE CORE' : 'PURGE THE STATION';
  element('objective-detail').textContent = p.core ? 'Return to the southern airlock' : p.kills === 6 ? 'Reactor chamber · north end of station' : `${6 - p.kills} hostile signals detected`;
  for (; eventCursor < world.events.length; eventCursor++) {
    const event = world.events[eventCursor];
    if (event.type === 'kill') announce(p.kills === 6 ? 'ALL HOSTILES ELIMINATED · REACTOR UNLOCKED' : 'HOSTILE ELIMINATED');
    else if (event.type === 'med') announce('MEDKIT ACQUIRED · VITALS RESTORED');
    else if (event.type === 'ammo') announce('AMMUNITION ACQUIRED · +15');
    else if (event.type === 'core') announce('REACTOR CORE SECURED · RETURN TO EXIT');
  }
  if (!element('telemetry').hidden) drawMap();
}

function drawMap() {
  const g = element<HTMLCanvasElement>('minimap').getContext('2d')!;
  const s = 230 / GRID_SIZE;
  g.fillStyle = '#121a13'; g.fillRect(0, 0, 230, 230);
  g.fillStyle = '#44503a';
  world.map.forEach((row, z) => row.forEach((cell, x) => { if (cell) g.fillRect(x * s, z * s, s - 1, s - 1); }));
  for (const p of world.pickups) if (!p.taken) {
    g.fillStyle = p.kind === 'core' ? '#e7ba70' : p.kind === 'med' ? '#b7d38d' : '#859d74';
    g.fillRect(p.x * s - 2, p.z * s - 2, 4, 4);
  }
  g.fillStyle = '#ce7151'; for (const e of world.enemies) if (e.alive) g.fillRect(e.x * s - 2, e.z * s - 2, 4, 4);
  g.fillStyle = '#b8d68b'; g.fillRect(world.exit.x * s - 3, world.exit.z * s - 1, 6, 2);
  g.save(); g.translate(world.player.x * s, world.player.z * s); g.rotate(world.player.yaw);
  g.fillStyle = '#f5e8c9'; g.beginPath(); g.moveTo(0, -5); g.lineTo(3, 4); g.lineTo(-3, 4); g.closePath(); g.fill(); g.restore();
}

function frame(now: number) {
  const rawDt = (now - previousFrame) / 1000, dt = Math.min(0.1, rawDt); previousFrame = now;
  if (screen === 'playing') {
    if (controller !== 'manual' && actionTicks === 0 && !thinking) void nextAction();
    if (!thinking) {
      accumulator += dt;
      while (accumulator >= DT && screen === 'playing') {
        if (controller !== 'manual' && (!action || actionTicks === 0)) { accumulator = 0; break; }
        world.step(controller === 'manual' ? manualInput() : world.motor(action!));
        if (controller !== 'manual') actionTicks--;
        accumulator -= DT;
        if (world.status !== 'playing') finish();
      }
    } else accumulator = 0;
    audio.update(world);
  }
  view.render(dt, screen === 'menu');
  if (recording) {
    const last = decisions.at(-1);
    recording.draw(canvas, { controller: names[controller], seed: world.seed, simulationSeconds: world.time,
      wallSeconds: ((endedAt || performance.now()) - startedAt) / 1000, health: world.player.hp, ammo: world.player.ammo,
      kills: world.player.kills, core: world.player.core, status: screen === 'error' ? 'error' : world.status, action: last?.action.toUpperCase() ?? '—',
      latencyMs: last?.client_ms ?? 0, confidence: last?.jev_confidence ?? last?.entropy_confidence, decisions: decisions.length,
      escalations: decisions.filter(d => d.escalated).length, thinking });
  }
  if (screen === 'playing' && rawDt > 0) frames.push({ ms: rawDt * 1000, waiting: thinking, drawCalls: view.drawCalls, triangles: view.triangles });
  if (now - hudAt > 120) { updateHUD(); hudAt = now; }
  if (frames.length) {
    const recent = frames.slice(-30); element('fps').textContent = `${Math.round(1000 / (recent.reduce((sum, f) => sum + f.ms, 0) / recent.length))} FPS`;
  }
  element('damage').style.opacity = String(Math.max(0, 1 - (world.time - world.player.lastHurt) * 2.7) * 0.6);
  element('message').style.opacity = now < messageUntil && screen === 'playing' ? '1' : '0';
  requestAnimationFrame(frame);
}

function lockPointer() {
  if (matchMedia('(pointer:coarse)').matches) return;
  if (screen === 'playing' && controller === 'manual' && !document.pointerLockElement)
    Promise.resolve(canvas.requestPointerLock()).catch(() => announce('USE LEFT / RIGHT ARROWS TO AIM'));
}
element('play').addEventListener('click', () => { start('manual', missionSeed()); lockPointer(); });
element('watch').addEventListener('click', () => start(element<HTMLSelectElement>('controller').value as Controller, missionSeed()));
element('resume').addEventListener('click', () => { resume(); if (controller === 'manual') lockPointer(); });
for (const id of ['restart-pause', 'replay']) element(id).addEventListener('click', () => start(controller, world.seed));
for (const id of ['menu-pause', 'return-menu', 'error-menu']) element(id).addEventListener('click', showMenu);
for (const id of ['telemetry-toggle', 'telemetry-close']) element(id).addEventListener('click', () => element('telemetry').hidden = !element('telemetry').hidden);
element('audio-toggle').addEventListener('click', () => { audio.enabled = !audio.enabled; element('audio-toggle').textContent = audio.enabled ? '♪' : '×'; if (audio.enabled) audio.start(); });
element('pause-toggle').addEventListener('click', pause);
canvas.addEventListener('mousedown', e => { if (e.button === 0 && screen === 'playing' && controller === 'manual') { lockPointer(); keys.add('Mouse0'); firePressed = true; } });
window.addEventListener('mouseup', () => keys.delete('Mouse0'));
window.addEventListener('mousemove', e => { if (document.pointerLockElement === canvas && screen === 'playing') mouseTurn += e.movementX * 0.0024; });
window.addEventListener('keydown', e => {
  if (screen === 'playing' && ['Space', 'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
  if (e.code === 'Escape') { if (screen === 'playing') pause(); else if (screen === 'paused') resume(); }
  else if (e.code === 'Tab' && screen === 'playing' && !e.repeat) element('telemetry').hidden = !element('telemetry').hidden;
  else if (screen === 'playing') { keys.add(e.code); if (e.code === 'Space') firePressed = true; }
});
window.addEventListener('keyup', e => keys.delete(e.code));
window.addEventListener('blur', () => { keys.clear(); mouseTurn = 0; });
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
document.addEventListener('pointerlockchange', () => { if (!document.pointerLockElement && controller === 'manual') pause(); });
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-key]')) {
  button.addEventListener('pointerdown', e => { e.preventDefault(); button.setPointerCapture(e.pointerId); keys.add(button.dataset.key!); if (button.dataset.key === 'Space') firePressed = true; });
  for (const event of ['pointerup', 'pointercancel']) button.addEventListener(event, () => keys.delete(button.dataset.key!));
}

function metrics() {
  const describe = (rows: typeof frames) => {
    const values = rows.map(f => f.ms).sort((a, b) => a - b);
    const percentile = (q: number) => values[Math.min(values.length - 1, Math.floor((values.length - 1) * q))] ?? 0;
    const ms = values.reduce((a, b) => a + b, 0);
    return { frames: values.length, fps: ms ? 1000 * values.length / ms : 0, p50_ms: percentile(0.5), p95_ms: percentile(0.95),
      p99_ms: percentile(0.99), max_ms: values.at(-1) ?? 0,
      mean_draw_calls: rows.length ? rows.reduce((sum, f) => sum + f.drawCalls, 0) / rows.length : 0 };
  };
  return { runId, controller, seed: world.seed, status: screen === 'error' ? 'error' : world.status, simulation_seconds: world.time,
    wall_seconds: ((endedAt || performance.now()) - startedAt) / 1000,
    renderer: view.rendererName, resolution: { width: canvas.width, height: canvas.height },
    all: describe(frames), gameplay: describe(frames.filter(f => !f.waiting)), waiting: describe(frames.filter(f => f.waiting)),
    decisions: decisions.length, escalations: decisions.filter(d => d.escalated).length,
    total_inference_ms: decisions.reduce((sum, d) => sum + d.client_ms, 0) };
}

// Readable test/recording interface. It never replaces model outputs or changes game rules.
Object.assign(window, { __RLCD: { start, pause, resume, menu: showMenu,
  state: () => ({ screen, controller, thinking, ...world.snapshot() }),
  metrics, decisions: () => structuredClone(decisions), events: () => structuredClone(world.events), frames: () => structuredClone(frames) } });
setScreen('menu'); updateHUD(); requestAnimationFrame(frame);
fetch('/api/status').then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json(); }).then(data => {
  const select = element<HTMLSelectElement>('controller');
  for (const option of select.options) {
    option.disabled = !['rule', 'random'].includes(option.value) && !data.available_modes.includes(option.value);
  }
  const available = data.available_modes.join(', ');
  element('service-status').textContent = available ? `Models ready: ${available} · rule and manual play available` : 'Rule and manual play ready · start a model with scripts/play.py';
}).catch(() => { element('service-status').textContent = 'Manual play and rule pilot ready · start the optional model service for AI'; });
