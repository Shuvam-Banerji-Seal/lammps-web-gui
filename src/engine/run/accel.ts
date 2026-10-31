import { GPU_PAIR_STYLES, gpuPairSpec, type GpuPairSpec } from '../gpu/pairSpec';
import { COMM, MODIFY, OUTPUT, PAIR } from '../timer';
import type { System } from '../system';
import type { ForceBackend, ForceResult, PairTable, SimState } from '../types';
import { PairLJCut } from '../force/pair/lj_cut';
import { newPairTable } from '../pairs';
import { clearAccum } from '../force/types';
import { isResident, MAX_CHUNK, type ResidentBackend } from '../md';
import { computeSetup, yieldNow } from './verlet';
import { haltRequested } from './halt';
import { CpuForceBackend } from '../cpu/forces';
import { runLabelCheck } from '../labelcheck';

/*
 * Accelerated force paths for the notebook's CPU-threads and WebGPU choices
 * (cpu/parallel.ts, gpu/webgpuForces.ts, gpu/resident.ts). They implement one
 * force field exactly — plain pair_style lj/cut in a fully periodic
 * orthogonal box, no bonds, charges or kspace — and return the pair forces,
 * energy and virial. The run then follows the usual velocity-Verlet hook
 * order with every fix (thermostats, force-modifying fixes, ...); only the
 * force evaluation is replaced. Whole steps stay on the GPU (resident.ts)
 * when the only fixes are nve / enforce2d on all atoms. Every other input
 * runs on the general fp64 engine; accelerator() says why, and the run log
 * shows it.
 */

export interface Accel {
  backend: ForceBackend;
  table: PairTable;
  /** A pair style of gpu/pairSpec.ts on the GPU (computeSpec); null for plain lj/cut (compute with the table). */
  spec: GpuPairSpec | null;
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
  const ljcut = p instanceof PairLJCut && p.name === 'lj/cut';
  // the GPU also runs the pair styles of gpu/pairSpec.ts; the lj/cut threads of cpu/parallel.ts only lj/cut
  const gpuStyle = backend.kind === 'webgpu' && !!p && GPU_PAIR_STYLES[p.name] !== undefined;
  if (!ljcut && !gpuStyle) {
    return why(`pair style ${p?.name ?? 'none'} (the ${backend.kind === 'webgpu' ? `GPU path implements lj/cut, ${Object.keys(GPU_PAIR_STYLES).join(', ')}` : 'threaded path implements lj/cut'} only)`);
  }
  if (sys.ff.kspace || sys.ff.bond || sys.ff.angle || sys.ff.dihedral || sys.ff.improper) return why('bonded or long-range terms');
  if (s.topo.bonds.n || s.topo.angles.n || s.topo.dihedrals.n || s.topo.impropers.n) return why('molecular topology');
  if (s.box.triclinic || !s.box.periodic.every(Boolean)) return why('a triclinic or non-periodic box');
  if (p.tail) return why('pair_modify tail yes');
  if (sys.nb.excludes.length || sys.nb.includeBit) return why('neigh_modify exclude/include');
  // the backends build their own cell lists, so fixes that act at neighbor-list time cannot run with them
  for (const f of sys.fixes) {
    if (f.preExchange || f.preNeighbor || f.postNeighbor || f.preReverse) return why(`fix ${f.style} (it acts when neighbor lists are rebuilt)`);
  }
  if (!sys.fixes.some((f) => f.timeIntegrate)) return why('no time-integration fix');
  if (sys.groups.dynamic.size) return why('a dynamic group (atoms are reassigned between force evaluations)');
  if (sys.computes.some((c) => c.needsEatom || c.needsVatom)) return why('per-atom energy or stress computes');
  // the accelerated backends return a scalar virial only
  if (sys.thermo.keywords.some((k) => PRESSURE_TENSOR.has(k))) return why('pressure-tensor thermo keywords');
  if (sys.computes.some((c) => c.pressFlag && c.id !== 'thermo_press')) return why('a user pressure compute');
  // the table from the initialized (mixed) coefficients
  p.init(sys.styleContext());
  if (!ljcut) {
    const qqrd2e = s.units.qqr2e * (p.coulConstScale ?? 1) / sys.ff.dielectric;
    const spec = gpuPairSpec(p, s.ntypes, qqrd2e);
    if (typeof spec === 'string') return why(spec);
    // the kernel's 27-cell scan needs at least 3 cutoffs along each periodic edge (webgpuForces.ts sortByCell)
    const short = [0, 1, 2].some((d) => !(d === 2 && sys.dimension === 2) && (s.box.hi[d] - s.box.lo[d]) < 3 * spec.maxCutoff);
    if (short) return why('a box shorter than 3 pair cutoffs along an edge');
    const enforce2d = sys.fixes.some((f) => f.style === 'enforce2d');
    return { accel: { backend, table: newPairTable(s.ntypes, 0), spec, resident: false, enforce2d }, reason: null };
  }
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
  const allBit = sys.groupBit('all');
  const onlyNve = sys.fixes.every((f) => (f.style === 'nve' || f.style === 'enforce2d') && f.groupBit === allBit);
  const resident = onlyNve && isResident(backend) && backend.canAdvance(s, t);
  return { accel: { backend, table: t, spec: null, resident, enforce2d }, reason: null };
};

/** Stores an accelerated force result as the system's current forces. */
const store = (sys: System, r: ForceResult & { evdwl?: number; ecoul?: number }): void => {
  const acc = sys.ff.acc;
  clearAccum(acc);
  acc.evdwl = r.evdwl ?? r.pe;
  acc.ecoul = r.ecoul ?? 0;
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

/** One force evaluation on the accelerated backend: the style kernel of computeSpec, or lj/cut with the table. */
const evaluate = (a: Accel, s: SimState): ForceResult | Promise<ForceResult> =>
  a.spec ? (a.backend as ForceBackend & { computeSpec(s: SimState, spec: GpuPairSpec): Promise<ForceResult> }).computeSpec(s, a.spec) : a.backend.compute(s, a.table);

/** The run loop on an accelerated backend. Returns steps taken. */
export const runAccelerated = async (sys: System, n: number, a: Accel, hooks: AccelHooks): Promise<number> => {
  const s = sys.state;
  const g = sys.geom;
  const remap = () => { for (let i = 0; i < s.n; i++) g.remap(s.x, s.image, i); };
  // setup forces through the backend (fix setup: enforce2d zeroes vz/fz)
  sys.ff.init(s, sys.nb, g, sys.styleContext());
  remap();
  sys.sortAtoms(true);
  for (const f of sys.fixes) f.setupPreForce?.();
  sys.bump();
  store(sys, await evaluate(a, s));
  computeSetup(sys);
  for (const f of sys.fixes) f.setup();
  // labelmap check_labels: once, at the end of the setup (native Modify::setup)
  runLabelCheck(sys);
  sys.refreshComputes();
  hooks.afterSetup();
  const end = s.step + n;
  let taken = 0;
  let lastYield = performance.now();
  const fixes = sys.fixes;
  const has = <K extends keyof (typeof fixes)[number]>(k: K) => fixes.filter((f) => typeof f[k] === 'function');
  const fInitial = has('initialIntegrate');
  const fPostInt = has('postIntegrate');
  const fPreForce = has('preForce');
  const fPostForce = has('postForce');
  const fFinal = has('finalIntegrate');
  const fEnd = has('endOfStep');
  const timer = sys.timer;
  while (s.step < end) {
    if (hooks.cancelled()) break;
    timer.mark();
    if (a.resident) {
      // whole steps on the GPU up to the next step the host needs
      let m = 1;
      while (m < MAX_CHUNK && s.step + m < end && !hooks.hostStep(s.step + m)) m++;
      const r = await (a.backend as ResidentBackend).advance(s, a.table, m, { enforce2d: a.enforce2d });
      s.step += m;
      taken += m;
      sys.bump();
      store(sys, r);
      // whole steps on the GPU: the force evaluation and the integration together
      timer.stamp(PAIR);
      sys.refreshComputes();
      hooks.afterStep(s.step);
      timer.stamp(OUTPUT);
      await yieldNow();
      lastYield = performance.now();
      continue;
    }
    // the hook order of verlet.ts runVerlet, with the backend's forces
    s.step++;
    sys.refreshComputes();
    for (const f of fInitial) f.initialIntegrate!();
    for (const f of fPostInt) f.postIntegrate!();
    timer.stamp(MODIFY);
    remap();
    sys.bump();
    timer.stamp(COMM);
    for (const f of fPreForce) f.preForce!();
    timer.stamp(MODIFY);
    store(sys, await evaluate(a, s));
    timer.stamp(PAIR);
    for (const f of fPostForce) f.postForce!();
    for (const f of fFinal) f.finalIntegrate!();
    for (const f of fEnd) if (s.step % f.nevery === 0) f.endOfStep!();
    timer.stamp(MODIFY);
    sys.refreshComputes();
    taken++;
    hooks.afterStep(s.step);
    timer.stamp(OUTPUT);
    if (haltRequested(sys)) break;
    if (performance.now() - lastYield > 30) { await yieldNow(); lastYield = performance.now(); }
  }
  for (const f of fixes) f.postRun?.();
  return taken;
};
