import assert from 'node:assert/strict';
import { test } from 'node:test';
import { decide } from '../game/src/decision.ts';

test('HTTP failures and action substitution stop the pilot', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response('provider error', { status: 502 });
    await assert.rejects(decide('state', 'jev', 'test'), /no fallback/);
    globalThis.fetch = async () => Response.json({ action: 'heal', source: 'rule', mode: 'jev', escalated: false });
    await assert.rejects(decide('state', 'jev', 'test'), /Invalid model response/);
    globalThis.fetch = async () => Response.json({ action: 'teleport', source: 'jev', mode: 'jev', escalated: false });
    await assert.rejects(decide('state', 'jev', 'test'), /Invalid model response/);
  } finally { globalThis.fetch = original; }
});
