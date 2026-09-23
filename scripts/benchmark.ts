/** Reproducible headless missions. Model failures are recorded; no fallback. */
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { World, ruleAction, rng, ACTIONS, type Controller } from '../game/src/simulation.ts';
import { decide, type ModelMode } from '../game/src/decision.ts';

const protocol = JSON.parse(readFileSync('data/protocol.json', 'utf8'));
const modes = (process.env.RLCD_BENCHMARK_MODES ?? 'rule,random').split(',') as Controller[];
if (modes.some(mode => !['rule', 'random', 'laya', 'decoder', 'jev'].includes(mode))) throw new Error('Unsupported benchmark mode');
const base = process.env.RLCD_API ?? 'http://127.0.0.1:8765';
const id = `missions-${new Date().toISOString().replace(/[:.]/g, '-')}`;
const folder = resolve('runs', id);
mkdirSync(folder, { recursive: true });
const results = [];
for (const controller of modes) for (const seed of protocol.episode_seeds as number[]) {
  const world = new World(seed), random = rng(seed ^ 0x1baf), decisions = [];
  const started = performance.now();
  let error: string | null = null;
  while (world.status === 'playing') {
    const observation = world.observe();
    try {
      const response = controller === 'rule' || controller === 'random'
        ? { action: controller === 'rule' ? ruleAction(observation) : ACTIONS[Math.floor(random() * ACTIONS.length)], source: controller, client_ms: 0 }
        : await decide(observation.text, controller as ModelMode, id, base);
      decisions.push({ time: world.time, state: observation.text, ...response });
      world.execute(response.action);
    } catch {
      error = 'Model request failed; mission stopped without a fallback';
      break;
    }
  }
  const result = { controller, seed, status: error ? 'error' : world.status,
    simulation_seconds: world.time, wall_seconds: (performance.now() - started) / 1000,
    kills: world.player.kills, health: world.player.hp, core: world.player.core,
    decision_count: decisions.length, error, decisions };
  results.push(result);
  console.log(`${controller} seed ${seed}: ${result.status} · ${world.time.toFixed(2)} simulated seconds`);
}
writeFileSync(resolve(folder, 'missions.json'), JSON.stringify({
  method: 'Headless shared motor; wall time excludes rendering, includes model waits; no fallback',
  episode_seeds: protocol.episode_seeds, results,
}, null, 2) + '\n');
console.log(`Saved local results in runs/${id}/missions.json`);
