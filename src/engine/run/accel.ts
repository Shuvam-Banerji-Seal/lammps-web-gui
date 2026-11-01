import { GPU_PAIR_STYLES, gpuPairSpec, LONG_COUL_STYLES, TABLE_LAW, type GpuPairSpec } from '../gpu/pairSpec';
import { GPU_EAM_STYLES, gpuEamSpec, isGpuEam, type EamPair, type GpuEamSpec } from '../gpu/eamSpec';
import { GPU_SW_STYLES, gpuSwSpec, isGpuSw, type GpuSwSpec, type SwPair } from '../gpu/swSpec';
import { GPU_TERSOFF_STYLES, gpuTersoffSpec, isGpuTersoff, type GpuTersoffSpec, type TersoffPair } from '../gpu/tersoffSpec';
import { GPU_TIP4P_STYLES, gpuTip4pSpec, isGpuTip4p, type GpuTip4pSpec, type Tip4pPair, type Tip4pSites } from '../gpu/tip4pSpec';
import { buildOwnedSites } from '../force/pair/tip4p_sites';
import type { SpecialList } from '../atoms';
import type { LevelSelect } from '../force/forcefield';
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
 * (cpu/parallel.ts, gpu/webgpuForces.ts, gpu/resident.ts). The CPU threads
 * implement plain pair_style lj/cut without bonds; the GPU the pair styles of
 * gpu/pairSpec.ts, with the bonded terms and the kspace style computed on the
 * CPU beside it each step. All need a fully periodic orthogonal box. They
 * return the pair forces, energy and virial. The run then follows the usual velocity-Verlet hook
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
  /** Bonded terms: computed on the CPU each step and added to the GPU's pair forces (special bonds weighted there). */
  bonded: boolean;
  /** The kspace style: computed on the CPU each step beside the GPU's real-space pair term. */
  kspace: boolean;
  /** pair eam, eam/alloy or eam/fs on the GPU (computeEAM: densities and forces there, the embedding step on the CPU). */
  eam: GpuEamSpec | null;
  /** pair sw or sw/mod on the GPU (computeSW: neighbor lists and gathered three-body forces there). */
  sw: GpuSwSpec | null;
  /** pair tersoff on the GPU (computeTersoff: neighbor lists, bond orders and gathered forces there). */
  tersoff: GpuTersoffSpec | null;
  /** a TIP4P pair style on the GPU (computeTip4p: the M sites built here each step, their forces spread on the waters). */
  tip4p: GpuTip4pSpec | null;
  /** Whole NVE steps on the GPU (resident.ts) with readback only on output steps. */
  resident: boolean;
  enforce2d: boolean;
}

const PRESSURE_TENSOR = new Set(['pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']);
/** The fix styles that change the box during a run (barostats, fix deform). */
const BOX_CHANGING = /^(npt|nph|press\/|deform|rigid\/np[th]|nphug|msst|qbmsst|bocs)/;

/** Whether the chosen backend can run this system, and how. */
export const accelerator = (sys: System, backend: ForceBackend): { accel: Accel | null; reason: string | null } => {
  if (backend instanceof CpuForceBackend) return { accel: null, reason: null };
  const s = sys.state;
  const why = (r: string) => ({ accel: null, reason: r });
  const p = sys.ff.pair;
  const ljcut = p instanceof PairLJCut && p.name === 'lj/cut';
  // the GPU also runs the pair styles of gpu/pairSpec.ts; the lj/cut threads of cpu/parallel.ts only lj/cut
  const gpuStyle = backend.kind === 'webgpu' && !!p && GPU_PAIR_STYLES[p.name] !== undefined;
  const eamStyle = backend.kind === 'webgpu' && isGpuEam(p);
  const swStyle = backend.kind === 'webgpu' && isGpuSw(p);
  const tersoffStyle = backend.kind === 'webgpu' && isGpuTersoff(p);
  const tip4pStyle = backend.kind === 'webgpu' && isGpuTip4p(p);
  if (!ljcut && !gpuStyle && !eamStyle && !swStyle && !tersoffStyle && !tip4pStyle) {
    const names = [...Object.keys(GPU_PAIR_STYLES).filter((k) => k !== 'lj/cut'), ...GPU_EAM_STYLES, ...GPU_SW_STYLES, ...GPU_TERSOFF_STYLES, ...GPU_TIP4P_STYLES].join(', ');
    return why(`pair style ${p?.name ?? 'none'} (the ${backend.kind === 'webgpu' ? `GPU path implements lj/cut, ${names}` : 'threaded path implements lj/cut'} only)`);
  }
  // the long-range Coulomb styles' real-space term runs on the GPU, the kspace style on the CPU beside it
  const longStyle = (gpuStyle && LONG_COUL_STYLES.has(GPU_PAIR_STYLES[p.name])) || (tip4pStyle && p.name.endsWith('/long'));
  const kspace = !!sys.ff.kspace;
  if (kspace && !longStyle) return why('a kspace style (long-range terms)');
  if (!p.computeOn) return why('pair_modify compute no');
  // fix adapt changes the pair coefficients during the run; the backends upload them once
  const adapt = sys.fixes.find((f) => f.style.startsWith('adapt'));
  if (adapt) return why(`fix ${adapt.style} (it changes the pair coefficients during the run)`);
  // bonded terms run on the CPU beside the GPU's pair term (special bonds weighted in the kernel); the lj/cut threads
  // of cpu/parallel.ts have no special bonds
  const molecular = !!(sys.ff.bond || sys.ff.angle || sys.ff.dihedral || sys.ff.improper)
    || s.topo.bonds.n > 0 || s.topo.angles.n > 0 || s.topo.dihedrals.n > 0 || s.topo.impropers.n > 0;
  if (molecular && backend.kind !== 'webgpu') return why('bonded terms or molecular topology');
  if (s.box.triclinic || !s.box.periodic.every(Boolean)) return why('a triclinic or non-periodic box');
  if (sys.nb.excludes.length || sys.nb.includeBit) return why('neigh_modify exclude/include');
  // the backends build their own cell lists, so fixes that act at neighbor-list time cannot run with them
  for (const f of sys.fixes) {
    if (f.preExchange || f.preNeighbor || f.postNeighbor || f.preReverse) return why(`fix ${f.style} (it acts when neighbor lists are rebuilt)`);
  }
  if (!sys.fixes.some((f) => f.timeIntegrate)) return why('no time-integration fix');
  if (sys.computes.some((c) => c.needsEatom || c.needsVatom)) return why('per-atom energy or stress computes');
  // the accelerated backends return a scalar virial only
  // the pressure tensor: the GPU kernels return the six virial components with the forces (not the whole-step kernel
  // of resident.ts, nor the lj/cut threads of cpu/parallel.ts, which give the scalar virial only); a barostat's own
  // pressure compute is a user pressure compute here (fix npt, nph)
  const needTensor = sys.thermo.keywords.some((k) => PRESSURE_TENSOR.has(k)) || sys.computes.some((c) => c.pressFlag && c.id !== 'thermo_press');
  if (needTensor && backend.kind !== 'webgpu') {
    if (sys.thermo.keywords.some((k) => PRESSURE_TENSOR.has(k))) return why('pressure-tensor thermo keywords');
    return why('a user pressure compute');
  }
  // a fix that changes the box (a barostat, fix deform) may shrink it during the run: the kernels' minimum image needs
  // 2 cutoffs per periodic edge at every step, so such runs start with 3
  const boxMayChange = sys.fixes.some((f) => BOX_CHANGING.test(f.style));
  const edgeCutoffs = boxMayChange ? 3 : 2;
  const shortBox = (cut: number) => [0, 1, 2].some((d) => !(d === 2 && sys.dimension === 2) && (s.box.hi[d] - s.box.lo[d]) < edgeCutoffs * cut);
  const shortWhy = `a box shorter than ${edgeCutoffs} pair cutoffs along an edge${boxMayChange ? ' (a fix that changes the box)' : ''}`;
  // the table from the initialized (mixed) coefficients
  p.init(sys.styleContext());
  if (eamStyle) {
    if (molecular) return why('bonded terms with an EAM pair style');
    const eam = gpuEamSpec(p as EamPair);
    if (typeof eam === 'string') return why(eam);
    // the kernels' minimum image needs at least 2 cutoffs along each periodic edge (gpu/eamKernels.ts)
    if (shortBox(eam.maxCutoff)) return why(shortWhy);
    const enforce2d = sys.fixes.some((f) => f.style === 'enforce2d');
    return { accel: { backend, table: newPairTable(s.ntypes, 0), spec: null, bonded: false, kspace: false, eam, sw: null, tersoff: null, tip4p: null, resident: false, enforce2d }, reason: null };
  }
  if (swStyle) {
    if (molecular) return why('bonded terms with an sw pair style');
    const sw = gpuSwSpec(p as SwPair);
    if (typeof sw === 'string') return why(sw);
    // the kernels' minimum image needs at least 2 cutoffs along each periodic edge (gpu/swKernels.ts)
    if (shortBox(sw.maxCutoff)) return why(shortWhy);
    const enforce2d = sys.fixes.some((f) => f.style === 'enforce2d');
    return { accel: { backend, table: newPairTable(s.ntypes, 0), spec: null, bonded: false, kspace: false, eam: null, sw, tersoff: null, tip4p: null, resident: false, enforce2d }, reason: null };
  }
  if (tersoffStyle) {
    if (molecular) return why('bonded terms with a tersoff pair style');
    const tersoff = gpuTersoffSpec(p as TersoffPair);
    if (typeof tersoff === 'string') return why(tersoff);
    // the kernels' minimum image needs at least 2 cutoffs along each periodic edge (gpu/tersoffKernels.ts)
    if (shortBox(tersoff.maxCutoff)) return why(shortWhy);
    const enforce2d = sys.fixes.some((f) => f.style === 'enforce2d');
    return { accel: { backend, table: newPairTable(s.ntypes, 0), spec: null, bonded: false, kspace: false, eam: null, sw: null, tersoff, tip4p: null, resident: false, enforce2d }, reason: null };
  }
  if (tip4pStyle) {
    const qqrd2e = s.units.qqr2e * (p.coulConstScale ?? 1) / sys.ff.dielectric;
    const sb = sys.ff.special;
    const tip4p = gpuTip4pSpec(p as Tip4pPair, qqrd2e, molecular ? { lj: [1, ...sb.lj], coul: [1, ...sb.coul] } : null);
    if (typeof tip4p === 'string') return why(tip4p);
    // the kernel's minimum image needs at least 2 cutoffs along each periodic edge (gpu/tip4pKernels.ts)
    if (shortBox(tip4p.maxCutoff)) return why(shortWhy);
    const enforce2d = sys.fixes.some((f) => f.style === 'enforce2d');
    return { accel: { backend, table: newPairTable(s.ntypes, 0), spec: null, bonded: molecular, kspace, eam: null, sw: null, tersoff: null, tip4p, resident: false, enforce2d }, reason: null };
  }
  if (!ljcut || molecular) {
    const qqrd2e = s.units.qqr2e * (p.coulConstScale ?? 1) / sys.ff.dielectric;
    const sb = sys.ff.special;
    const spec = gpuPairSpec(p, s.ntypes, qqrd2e, molecular ? { lj: [1, ...sb.lj], coul: [1, ...sb.coul] } : null);
    if (typeof spec === 'string') return why(spec);
    // the kernel's minimum image needs at least 2 cutoffs along each periodic edge (webgpuForces.ts KERNEL_STYLES; the
    // whole-step kernel 3, resident.ts canAdvanceSpec): a shorter edge has two copies of a partner within the cutoff
    if (shortBox(spec.maxCutoff)) return why(shortWhy);
    const enforce2d = sys.fixes.some((f) => f.style === 'enforce2d');
    // whole steps on the GPU (resident.ts) with only nve / enforce2d on all atoms, as for lj/cut
    const allBit = sys.groupBit('all');
    const onlyNve = sys.fixes.every((f) => (f.style === 'nve' || f.style === 'enforce2d') && f.groupBit === allBit);
    const gb = backend as ForceBackend & { canAdvanceSpec?(s: SimState, spec: GpuPairSpec): boolean };
    const resident = !molecular && !kspace && !needTensor && !sys.groups.dynamic.size && spec.law !== TABLE_LAW && onlyNve && isResident(backend) && !!gb.canAdvanceSpec?.(s, spec);
    return { accel: { backend, table: newPairTable(s.ntypes, 0), spec, bonded: molecular, kspace, eam: null, sw: null, tersoff: null, tip4p: null, resident, enforce2d }, reason: null };
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
  // whole GPU steps cannot reassign a dynamic group between steps
  const resident = onlyNve && !needTensor && !sys.groups.dynamic.size && isResident(backend) && backend.canAdvance(s, t);
  return { accel: { backend, table: t, spec: null, bonded: false, kspace: false, eam: null, sw: null, tersoff: null, tip4p: null, resident, enforce2d }, reason: null };
};

/**
 * Stores an accelerated force result as the system's current forces. When the CPU ran terms beside it (bonded, kspace)
 * the accumulator already holds their energies and virials (evaluate() ran them first), and the pair energies of the
 * bonded styles that tally into them (dihedral charmm's 1-4 pairs), so the GPU's pair energies are added.
 */
const store = (sys: System, r: ForceResult & { evdwl?: number; ecoul?: number }, cpuTerms = false): void => {
  const acc = sys.ff.acc;
  if (cpuTerms) {
    acc.virial.fill(0);
    acc.evdwl += r.evdwl ?? r.pe;
    acc.ecoul += r.ecoul ?? 0;
  } else {
    clearAccum(acc);
    acc.evdwl = r.evdwl ?? r.pe;
    acc.ecoul = r.ecoul ?? 0;
  }
  // pair_modify tail yes: the energy correction etail / V, which ForceField.compute adds to evdwl when it computes every
  // term (its pressure term comes from the pressure compute, ForceField.ptail)
  if (sys.ff.etailV !== 0) acc.evdwl += sys.ff.etailV / sys.geom.volume(sys.dimension);
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

/** The terms the CPU computes beside the GPU's pair term (force/forcefield.ts LevelSelect). */
const cpuTerms = (a: Accel): LevelSelect => ({
  has: (t) => (a.bonded && (t === 'bond' || t === 'angle' || t === 'dihedral' || t === 'improper')) || (a.kspace && t === 'kspace'),
});

type SpecBackend = ForceBackend & { computeSpec(s: SimState, spec: GpuPairSpec, special?: SpecialList | null): Promise<ForceResult> };

/**
 * One force evaluation on the accelerated backend: the style kernel of computeSpec, or lj/cut with the table. With
 * bonded terms, they are computed first on the CPU (fp64; ForceField.compute clears the forces and the accumulator and
 * fills their energies and virials), then the GPU's pair forces are added to theirs.
 */
const evaluate = async (a: Accel, sys: System, step: boolean): Promise<ForceResult> => {
  const s = sys.state;
  if (a.sw) {
    const gb = a.backend as ForceBackend & { computeSW(s: SimState, spec: GpuSwSpec): Promise<ForceResult> };
    return gb.computeSW(s, a.sw);
  }
  if (a.tersoff) {
    const gb = a.backend as ForceBackend & { computeTersoff(s: SimState, spec: GpuTersoffSpec): Promise<ForceResult> };
    return gb.computeTersoff(s, a.tersoff);
  }
  if (a.eam) {
    const pair = sys.ff.pair as EamPair;
    const gb = a.backend as ForceBackend & { computeEAM(s: SimState, spec: GpuEamSpec, embed: (rho: Float64Array, fp: Float64Array) => number): Promise<ForceResult> };
    return gb.computeEAM(s, a.eam, (rho, fp) => pair.embedOwned(s.type, rho, s.n, fp));
  }
  if (a.tip4p) return evaluateTip4p(a, a.tip4p, sys, step);
  if (!a.spec) return a.backend.compute(s, a.table);
  const gb = a.backend as SpecBackend;
  if (!a.bonded && !a.kspace) return gb.computeSpec(s, a.spec);
  // the kspace style's G-ewald of this step (Ewald sets it again when the box changes)
  if (a.spec.gewald > 0) a.spec.gewald = sys.ff.pair!.gEwald;
  // the CPU neighbor list is not rebuilt on this path: the bonded partners by the minimum image
  sys.ff.compute(s, sys.nb, sys.geom, { step, minimumImage: true }, cpuTerms(a));
  const n3 = 3 * s.n;
  const fb = s.f.slice(0, n3);
  const r = await gb.computeSpec(s, a.spec, sys.ff.specialPartners);
  for (let k = 0; k < n3; k++) s.f[k] += fb[k];
  return r;
};

type Tip4pBackend = ForceBackend & { computeTip4p(s: SimState, spec: GpuTip4pSpec, special: SpecialList | null, sites: Tip4pSites): Promise<ForceResult> };

/**
 * A TIP4P style on the GPU: the CPU's bonded and kspace terms first (as for computeSpec), then the M sites of this step
 * (tip4p_sites.ts buildOwnedSites, fp64) and the GPU's pair forces with the site forces spread on the waters.
 */
const evaluateTip4p = async (a: Accel, spec: GpuTip4pSpec, sys: System, step: boolean): Promise<ForceResult> => {
  const s = sys.state;
  const pair = sys.ff.pair as Tip4pPair & { siteModel(): { otype: number; htype: number; alpha: number } | null };
  // the kspace style's G-ewald of this step (Ewald sets it again when the box changes)
  if (spec.gewald > 0) spec.gewald = pair.gEwald;
  const model = pair.siteModel();
  if (!model) throw new Error(`pair style ${pair.name} is not initialized`);
  const own = buildOwnedSites(s.n, s.x, s.type, s.id, sys.geom, model);
  const delta = new Float64Array(3 * s.n);
  for (let k = 0; k < 3 * s.n; k++) delta[k] = own.M[k] - s.x[k];
  const sites: Tip4pSites = { delta, h1: own.h1, h2: own.h2 };
  const gb = a.backend as Tip4pBackend;
  if (!a.bonded && !a.kspace) return gb.computeTip4p(s, spec, null, sites);
  sys.ff.compute(s, sys.nb, sys.geom, { step, minimumImage: true }, cpuTerms(a));
  const n3 = 3 * s.n;
  const fb = s.f.slice(0, n3);
  const r = await gb.computeTip4p(s, spec, sys.ff.specialPartners, sites);
  for (let k = 0; k < n3; k++) s.f[k] += fb[k];
  return r;
};

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
  store(sys, await evaluate(a, sys, false), a.bonded || a.kspace);
  // the dynamic groups at the setup, after its forces (as run/coupled.ts and native group.html)
  sys.assignDynamicGroups(true);
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
      const r = a.spec
        ? await (a.backend as ResidentBackend & { advanceSpec(s: SimState, spec: GpuPairSpec, n: number, o: { enforce2d: boolean }): Promise<ForceResult> }).advanceSpec(s, a.spec, m, { enforce2d: a.enforce2d })
        : await (a.backend as ResidentBackend).advance(s, a.table, m, { enforce2d: a.enforce2d });
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
    store(sys, await evaluate(a, sys, true), a.bonded || a.kspace);
    timer.stamp(PAIR);
    // group.html: "The point in the timestep at which atoms are assigned to a dynamic group is after interatomic forces
    // have been computed, but before any fixes which alter forces or otherwise update the system have been invoked."
    sys.assignDynamicGroups(false);
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
