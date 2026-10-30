/**
 * The resource monitor (src/components/workbench/ResourceMonitor.tsx) says where the run computes its forces: the GPU,
 * N CPU threads, or one CPU thread with the reason (EngineEvent 'perf' path / threads / resident / why).
 */
import { describe, it, expect, afterEach } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import ResourceMonitor from '../src/components/workbench/ResourceMonitor';
import { getThemeTokens } from '../src/theme';
import type { FromEngine } from '../src/engine/protocol';
import type { EngineEvent } from '../src/engine/types';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Ready = Extract<FromEngine, { type: 'ready' }>;
type Perf = Extract<EngineEvent, { kind: 'perf' }>;

const READY: Ready = {
  type: 'ready', backend: 'WebGPU · nvidia', kind: 'webgpu', webgpuAvailable: true, cores: 16, threads: 8,
  device: { cores: 16, memoryGB: 8, workers: true, sharedMemory: true, gpu: { kind: 'hardware', name: 'nvidia', compatibility: false, maxStorageBufferMB: 2048 } },
  auto: { backend: 'webgpu', threads: 8, why: 'hardware GPU (nvidia) for lj/cut runs; 8 CPU threads for everything else' },
  commands: [], styles: {} as Ready['styles'],
};

const perf = (p: Partial<Perf>): Perf => ({
  kind: 'perf', step: 250, atoms: 500, stepsPerSec: 1000, elapsed: 0.25, path: 'serial', threads: 1, resident: false, why: null, ...p,
});

let root: Root | null = null;
let host: HTMLElement | null = null;

const show = (p: Perf, running = true): string => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(<ResourceMonitor ct={getThemeTokens('dark')} ready={READY} perf={p} history={[]} runInfo={{ dt: 0.005, units: 'lj' }} running={running} />));
  return host.textContent ?? '';
};

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null; host = null;
});

describe('resource monitor: where the run computes', () => {
  it('whole GPU steps', () => {
    expect(show(perf({ path: 'gpu', resident: true }))).toContain('whole steps on the GPU');
  });

  it('GPU forces with CPU fixes', () => {
    expect(show(perf({ path: 'gpu', resident: false }), false)).toContain('forces on the GPU · last run');
  });

  it('the pair threads', () => {
    expect(show(perf({ path: 'threads', threads: 8, atoms: 32000 }))).toContain('pair term on 8 CPU threads');
  });

  it('one thread, and why', () => {
    const text = show(perf({ why: '500 atoms; the pair threads start at 2000 atoms, below that they cost more than they save' }));
    expect(text).toContain('one CPU thread');
    expect(text).toContain('One thread: 500 atoms; the pair threads start at 2000 atoms, below that they cost more than they save.');
  });
});
