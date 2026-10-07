import type { System } from '../system';
import type { ForceBackend, ForceResult, PairTable } from '../types';
import { PairLJCut } from '../force/pair/lj_cut';
import { newPairTable } from '../pairs';
import { clearAccum } from '../force/types';
import { isResident, MAX_CHUNK, type ResidentBackend } from '../md';
import { yieldNow } from './verlet';
import { CpuForceBackend } from '../cpu/forces';

/*
 * Accelerated force paths for the notebook's CPU-threads and WebGPU choices
 * (cpu/parallel.ts, gpu/webgpuForces.ts, gpu/resident.ts). They implement one
 * case exactly — plain pair_style lj/cut in a fully periodic orthogonal box,
 * no bonds, charges or kspace, and only fix nve / enforce2d on all atoms —
 * and return the pair energy and virial. Every other input runs on the
 * general fp64 engine; accelerator() says why, and the run log shows it.
 */

export interface Accel {
  backend: ForceBackend;
  table: PairTable;
  /** Whole NVE steps on the GPU (resident.ts) with readback only on output steps. */
  resident: boolean;
  enforce2d: boolean;
}

const PRESSURE_TENSOR = new Set(['pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']);

/** Whether the chosen backend can run this system, and how. */
export const accelerator = (sys: System, backend: ForceBackend): { accel: Accel | null; reason: string | null } => {
  if (backend instanceof CpuForceBackend) return { accel: null, reason: null };
  const s = sys.state;
  const why = (r: string) => ({ accel: null, reason: r });
  const p = sys.ff.pair;
  if (!(p instanceof PairLJCut) || p.name !== 'lj/cut') return why(`pair style ${p?.name ?? 'none'} (the ${backend.kind === 'webgpu' ? 'GPU' : 'threaded'} path implements lj/cut only)`);
  if (sys.ff.kspace || sys.ff.bond || sys.ff.angle || sys.ff.dihedral || sys.ff.improper) return why('bonded or long-range terms');
  if (s.topo.bonds.n || s.topo.angles.n || s.topo.dihedrals.n || s.topo.impropers.n) return why('molecular topology');
  if (s.box.triclinic || !s.box.periodic.every(Boolean)) return why('a triclinic or non-periodic box');
  if (p.tail) return why('pair_modify tail yes');
  if (sys.nb.excludes.length || sys.nb.includeBit) return why('neigh_modify exclude/include');
  const allBit = sys.groupBit('all');
  for (const f of sys.fixes) {
    if ((f.style !== 'nve' && f.style !== 'enforce2d') || f.groupBit !== allBit) return why(`fix ${f.style}${f.groupBit !== allBit ? ` on group ${f.group}` : ''}`);
  }
  if (!sys.fixes.some((f) => f.style === 'nve')) return why('no fix nve');
  if (sys.computes.some((c) => c.needsEatom || c.needsVatom)) return why('per-atom energy or stress computes');
  // the accelerated backends return a scalar virial only
  if (sys.thermo.keywords.some((k) => PRESSURE_TENSOR.has(k))) return why('pressure-tensor thermo keywords');
  if (sys.computes.some((c) => c.pressFlag && c.id !== 'thermo_press')) return why('a user pressure compute');
  // the table from the initialized (mixed) coefficients
  p.init(sys.styleContext());
  const t = newPairTable(s.ntypes, p.cutGlobal);
  t.shift = p.shift;
  for (let i = 1; i <= s.ntypes; i++) {
    for (let j = 1; j <= s.ntypes; j++) {
      const k = i * (s.ntypes + 1) + j;
      t.pairs[k] = { epsilon: p.p.get('epsilon', i, j), sigma: p.p.get('sigma', i, j), cutoff: p.p.get('cut', i, j) };
      t.explicit[k] = true;
    }
  }
  const enforce2d = sys.fixes.some((f) => f.style === 'enforce2d');
  const resident = isResident(backend) && backend.canAdvance(s, t);
  return { accel: { backend, table: t, resident, enforce2d }, reason: null };
};

/** Stores an accelerated force result as the system's current forces. */
const store = (sys: System, r: ForceResult): void => {
  const acc = sys.ff.acc;
  clearAccum(acc);
  acc.evdwl = r.pe;
  const w = r.virialTensor;
  if (w) for (let c = 0; c < 6; c++) acc.virial[c] = w[c];
  else {
    const d = sys.dimension;
    for (let c = 0; c < d; c++) acc.virial[c] = r.virial / d;
  }
  sys.forcesCurrent();
};

export interface AccelHooks {
  cancelled(): boolean;
  afterSetup(): void;
  afterStep(step: number): void;
  /** Steps on which output needs the state (thermo, dumps, frames). */
  hostStep(step: number): boolean;
}

/** The run loop on an accelerated backend. Returns steps taken. */
export const runAccelerated = async (sys: System, n: number, a: Accel, hooks: AccelHooks): Promise<number> => {
  const s = sys.state;
  const g = sys.geom;
  const remap = () => { for (let i = 0; i < s.n; i++) g.remap(s.x, s.image, i); };
  // setup forces through the backend (fix setup: enforce2d zeroes vz/fz)
  sys.ff.init(s, sys.nb, g, sys.styleContext());
  remap();
  sys.bump();
  store(sys, await a.backend.compute(s, a.table));
  for (const f of sys.fixes) f.setup();
  sys.refreshComputes();
  hooks.afterSetup();
  const end = s.step + n;
  let taken = 0;
  let lastYield = performance.now();
  const nve = sys.fixes.filter((f) => f.style === 'nve');
  const e2d = sys.fixes.filter((f) => f.style === 'enforce2d');
  while (s.step < end) {
    if (hooks.cancelled()) break;
    if (a.resident) {
      // whole steps on the GPU up to the next step the host needs
      let m = 1;
      while (m < MAX_CHUNK && s.step + m < end && !hooks.hostStep(s.step + m)) m++;
      const r = await (a.backend as ResidentBackend).advance(s, a.table, m, { enforce2d: a.enforce2d });
      s.step += m;
      taken += m;
      sys.bump();
      store(sys, r);
      sys.refreshComputes();
      hooks.afterStep(s.step);
      await yieldNow();
      lastYield = performance.now();
      continue;
    }
    s.step++;
    sys.refreshComputes();
    for (const f of nve) f.initialIntegrate!();
    remap();
    sys.bump();
    store(sys, await a.backend.compute(s, a.table));
    for (const f of e2d) f.postForce!();
    for (const f of nve) f.finalIntegrate!();
    sys.refreshComputes();
    taken++;
    hooks.afterStep(s.step);
    if (performance.now() - lastYield > 30) { await yieldNow(); lastYield = performance.now(); }
  }
  return taken;
};
