import assert from 'node:assert/strict';
import { test } from 'node:test';
import { World, ruleAction, DT, MAX_SECONDS } from '../game/src/simulation.ts';

test('reference controller can finish the complete combat / core / extraction loop', () => {
  for (const seed of [42, 73, 109]) {
    const w = new World(seed);
    while (w.status === 'playing') w.execute(ruleAction(w.observe()));
    assert.equal(w.status, 'won', JSON.stringify(w.snapshot()));
    assert.equal(w.player.kills, 6);
    assert.equal(w.player.core, true);
    assert.ok(w.time < MAX_SECONDS);
  }
});

test('the same seed and actions replay the exact simulation, including projectiles', () => {
  const first = new World(314), second = new World(314);
  for (let i = 0; i < 40; i++) {
    const action = ruleAction(first.observe());
    first.execute(action); second.execute(action);
  }
  assert.deepEqual(first.snapshot(), second.snapshot());
  assert.deepEqual(first.events, second.events);
});

test('walls block movement and bullets; ammo is consumed only when a shot fires', () => {
  const w = new World(42);
  w.player.x = 1.22; w.player.z = 1.22;
  for (let i = 0; i < 100; i++) w.step({ forward: 1, strafe: -1, turn: 0, fire: false }, DT);
  assert.ok(w.player.x >= 1.2 && w.player.z >= 1.2);
  assert.equal(w.lineOfSight({ x: 7.5, z: 2.5 }, { x: 9.5, z: 2.5 }), false);
  const ammo = w.player.ammo;
  w.step({ forward: 0, strafe: 0, turn: 0, fire: true });
  assert.equal(w.player.ammo, ammo - 1);
  w.step({ forward: 0, strafe: 0, turn: 0, fire: true });
  assert.equal(w.player.ammo, ammo - 1);
});

test('the core stays sealed while a hostile is alive', () => {
  const w = new World(73);
  w.player.x = 11.5; w.player.z = 2.5;
  w.step({ forward: 0, strafe: 0, turn: 0, fire: false });
  assert.equal(w.player.core, false);
  w.enemies.forEach(e => { e.alive = false; });
  w.step({ forward: 0, strafe: 0, turn: 0, fire: false });
  assert.equal(w.player.core, true);
});
