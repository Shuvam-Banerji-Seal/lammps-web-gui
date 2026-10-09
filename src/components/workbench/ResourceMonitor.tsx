import React, { useEffect, useState } from 'react';
import type { getThemeTokens } from '../../theme';
import type { FromEngine } from '../../engine/protocol';
import type { EngineEvent } from '../../engine/types';
import { deviceHints } from '../../engine/device';
import { simulatedPerDay } from '../../lammps/thermoUnits';

type Ready = Extract<FromEngine, { type: 'ready' }>;
type Perf = Extract<EngineEvent, { kind: 'perf' }>;

const fmt = (v: number, digits = 3): string =>
  v >= 1e6 ? `${(v / 1e6).toPrecision(digits)} M` : v >= 1e4 ? `${(v / 1e3).toPrecision(digits)} k` : v.toPrecision(digits);

/** The page's JS heap (Chromium's performance.memory); the engine worker's heap is not visible from here. */
const pageHeap = (): { used: number; limit: number } | null => {
  const m = (performance as Performance & { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
  return m ? { used: m.usedJSHeapSize, limit: m.jsHeapSizeLimit } : null;
};

const Spark: React.FC<{ values: number[]; className: string }> = ({ values, className }) => {
  if (values.length < 2) return null;
  let hi = 0;
  for (const v of values) if (v > hi) hi = v;
  const W = 120, H = 24;
  const pts = values.map((v, i) => `${((i / (values.length - 1)) * W).toFixed(1)},${(H - (hi > 0 ? (v / hi) * (H - 2) : 0) - 1).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`steps per second over the run, latest ${values[values.length - 1].toFixed(0)}, peak ${hi.toFixed(0)}`}>
      <polyline points={pts} fill="none" strokeWidth={1.5} className={className} stroke="currentColor" />
    </svg>
  );
};

/**
 * Resource monitor: what this device offers the engine (cores, memory, shared memory, WebGPU), what the
 * engine uses (backend, threads; and why, when Auto chose), and how fast the current run goes.
 */
const ResourceMonitor: React.FC<{
  ct: ReturnType<typeof getThemeTokens>;
  ready: Ready | null;
  perf: Perf | null;
  history: number[];
  runInfo: { dt: number; units: string } | null;
  running: boolean;
}> = ({ ct, ready, perf, history, runInfo, running }) => {
  const [heap, setHeap] = useState(pageHeap);
  useEffect(() => {
    const t = setInterval(() => setHeap(pageHeap()), 2000);
    return () => clearInterval(t);
  }, []);
  if (!ready) return null;
  const d = ready.device;
  const gpu = d.gpu.kind === 'hardware' ? `GPU ${d.gpu.name || 'hardware'}${d.gpu.compatibility ? ' (compatibility mode)' : ''}`
    : d.gpu.kind === 'software' ? 'GPU: software only' : 'no WebGPU';
  const chip = `rounded px-1.5 py-0.5 ${ct.chipIdle}`;
  const hints = deviceHints(d);
  const sps = perf?.stepsPerSec ?? 0;
  const perDay = runInfo && perf ? simulatedPerDay(runInfo.units, runInfo.dt, sps) : null;
  return (
    <section aria-label="Resource monitor" className={`rounded border p-2 text-[11px] ${ct.divider}`}>
      <p className={`font-semibold ${ct.muted}`}>Device and performance</p>
      <div className="mt-1 flex flex-wrap gap-1">
        <span className={chip}>{d.cores} cores</span>
        {d.memoryGB !== null && <span className={chip}>{d.memoryGB} GB</span>}
        <span className={chip} title="Shared memory (a cross-origin isolated page) lets every threaded pair style run on several cores">
          {d.sharedMemory ? 'shared-memory threads' : d.workers ? 'no shared memory' : 'no Web Workers'}
        </span>
        <span className={chip} title={d.gpu.maxStorageBufferMB ? `largest GPU storage buffer ${d.gpu.maxStorageBufferMB} MB` : undefined}>{gpu}</span>
      </div>
      <p className="mt-1">
        <span className={ct.muted}>Engine: </span>
        {ready.auto ? 'Auto → ' : ''}{ready.backend}
        {ready.kind === 'webgpu' && ready.threads > 1 ? ` · ${ready.threads} CPU threads for runs the GPU cannot take` : ''}
      </p>
      {ready.auto && <p className={ct.muted}>{ready.auto.why}</p>}
      {perf && (
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1" aria-live="off">
          <span><strong className="text-[13px]">{fmt(sps)}</strong> steps/s</span>
          {sps > 0 && <span>{(1000 / sps).toPrecision(3)} ms/step</span>}
          <span title="atoms × steps per second, comparable across system sizes">{fmt(sps * perf.atoms)} atom-steps/s</span>
          {perDay && <span>{perDay}</span>}
          <span className={ct.muted}>{perf.threaded ? 'pair term on threads' : 'one thread'}{running ? '' : ' · last run'}</span>
          <Spark values={history} className={ct.accentText} />
        </div>
      )}
      {heap && <p className={`mt-1 ${ct.muted}`}>Page memory {(heap.used / 2 ** 20).toFixed(0)} MB of {(heap.limit / 2 ** 20).toFixed(0)} MB</p>}
      {hints.length > 0 && (
        <ul className={`mt-1 list-disc pl-4 ${ct.muted}`}>{hints.map((h) => <li key={h}>{h}</li>)}</ul>
      )}
    </section>
  );
};

export default ResourceMonitor;
