import { ACTIONS, type Action } from './simulation.ts';

export type ModelMode = 'laya' | 'decoder' | 'jev';
export type Decision = {
  action: Action; source: string; mode: string; elapsed_ms: number; client_ms: number;
  laya_ms?: number; decoder_ms?: number; escalated: boolean;
  entropy_confidence?: number; top_probability?: number; probabilities?: Record<Action, number>;
  jev_ms?: number; jev_model?: string; jev_confidence?: number; confidence_kind?: string;
};

export async function decide(state: string, mode: ModelMode, runId: string, base = '', signal?: AbortSignal): Promise<Decision> {
  const started = performance.now();
  const response = await fetch(`${base}/api/decision`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state, mode, run_id: runId }),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(100_000)]) : AbortSignal.timeout(100_000),
  });
  if (!response.ok) throw new Error(`Decision service returned ${response.status}. Check the local model service; no fallback action was used.`);
  const decision = await response.json() as Decision;
  if (!ACTIONS.includes(decision.action) || decision.source !== mode || decision.mode !== mode || decision.escalated !== false)
    throw new Error('Invalid model response; the pilot stopped.');
  decision.client_ms = performance.now() - started;
  return decision;
}
