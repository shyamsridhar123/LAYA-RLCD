/** Replay retained tactical decisions; never re-query a model or replace actions. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { World, ACTIONS, type Action } from '../game/src/simulation.ts';

const read = (name: string) => JSON.parse(readFileSync(name, 'utf8'));
const episodes = read('benchmarks/gameplay/episodes.json');
for (const summary of episodes) {
  const file = `benchmarks/gameplay/traces/${summary.controller}-${summary.seed}.json`;
  const recording = read(file), world = new World(recording.seed);
  assert.equal(recording.trace.length, summary.decisions);
  for (const decision of recording.trace) {
    assert.equal(world.status, 'playing', file);
    assert.equal(world.time, decision.time, file);
    assert.equal(world.observe().text, decision.state, file);
    assert.ok(ACTIONS.includes(decision.action));
    world.execute(decision.action as Action);
  }
  // JSON serialization strips absent optional fields consistently on both sides.
  assert.deepEqual(JSON.parse(JSON.stringify(world.snapshot())), recording.final, file);
  assert.deepEqual(JSON.parse(JSON.stringify(world.events)), recording.events, file);
  assert.equal(world.status, summary.status);
  assert.equal(world.player.hp, summary.hp);
  assert.equal(world.player.kills, summary.kills);
  assert.equal(world.player.core, summary.core);
  assert.equal(world.time, summary.simulation_seconds);
}
const jev = read('benchmarks/jev/demo.json');
const world = new World(jev.seed);
for (const decision of read('benchmarks/jev/decisions.json')) {
  assert.equal(world.time, decision.time);
  assert.equal(world.observe().text, decision.state);
  world.execute(decision.action);
}
assert.equal(world.status, jev.outcome);
assert.equal(world.time, jev.simulation_seconds);
assert.equal(world.player.hp, jev.health);
assert.equal(world.player.kills, jev.kills);
assert.equal(world.player.core, jev.core);
console.log('Verified exact state/event replays for 20 missions, plus the Jev demo outcome. No model inference run.');
