import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import {
  PairThreadWorker, cloneableFields, pairFromFields, restrictCounts, sameShape, splitRanges, type PairThreadMessage, type PairWorkerLike,
} from '../src/engine/cpu/pairThreadsCore';
import { THREADED_PAIRS } from '../src/engine/cpu/threadedPairFactories';
import { THREADED_NAMES } from '../src/engine/cpu/threadedPairs';
import { THREADED as GPL_THREADED } from '../src/engine-gpl/registry/threaded_pairs';
import { GPL_THREADED_PAIRS } from '../src/engine/lazyStyles';
import { PairLJCutCoulLong } from '../src/engine/force/pair/coul_long';

/*
 * Shared-memory pair threads (src/engine/cpu/pairThreads.ts). The workers run in-process here
 * (each message is handled at once), which exercises the shared buffers, the range split, the
 * pair copies and the reduction; real threads are checked in a browser.
 */

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  /**
   * with `load`, the worker also builds the GPL-ported styles (the factories the browser's pair worker imports by URL,
   * threadedPairLoaders.ts; enginePairLoaders.test.ts checks that import), at once as for the others
   */
  constructor(load = false) { this.worker = new PairThreadWorker(load ? { ...THREADED_PAIRS, ...GPL_THREADED } : THREADED_PAIRS); }
  readonly worker: PairThreadWorker;
  computes = 0;
  postMessage(msg: unknown): void {
    const m = msg as PairThreadMessage;
    if (m.type === 'compute') this.computes++;
    this.worker.handle(m);
  }
  terminate(): void {}
}

const BOX = `atom_style charge
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 2 box
create_atoms 1 box
mass * 1.0
set group all type/fraction 2 0.5 4321
set type 1 charge 0.5
set type 2 charge -0.5
velocity all create 1.5 87287 loop geom
`;

/** Pair setup lines for every threaded style (units lj unless the style needs more). */
const STYLES: Record<string, string> = {
  'lj/cut': 'pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 0.8 1.1',
  'lj/cut/coul/cut': 'pair_style lj/cut/coul/cut 2.5\npair_coeff * * 1.0 1.0',
  'lj/cut/coul/debye': 'pair_style lj/cut/coul/debye 1.0 2.5\npair_coeff * * 1.0 1.0',
  'lj/cut/coul/long': 'pair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1e-4',
  'coul/cut': 'pair_style coul/cut 2.5\npair_coeff * *',
  'coul/debye': 'pair_style coul/debye 1.0 2.5\npair_coeff * *',
  'coul/long': 'pair_style coul/long 2.5\npair_coeff * *\nkspace_style ewald 1e-4',
  'lj/charmm/coul/charmm': 'pair_style lj/charmm/coul/charmm 2.0 2.5\npair_coeff * * 1.0 1.0',
  'lj/charmm/coul/charmm/implicit': 'pair_style lj/charmm/coul/charmm/implicit 2.0 2.5\npair_coeff * * 1.0 1.0',
  'lj/charmm/coul/long': 'pair_style lj/charmm/coul/long 2.0 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1e-4',
  'lj/charmmfsw/coul/charmmfsh': 'pair_style lj/charmmfsw/coul/charmmfsh 2.0 2.5\npair_coeff * * 1.0 1.0',
  'lj/charmmfsw/coul/long': 'pair_style lj/charmmfsw/coul/long 2.0 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1e-4',
  buck: 'pair_style buck 2.5\npair_coeff * * 100.0 0.3 1.0',
  'buck/coul/cut': 'pair_style buck/coul/cut 2.5\npair_coeff * * 100.0 0.3 1.0',
  'buck/coul/long': 'pair_style buck/coul/long 2.5\npair_coeff * * 100.0 0.3 1.0\nkspace_style ewald 1e-4',
  born: 'pair_style born 2.5\npair_coeff * * 1.0 0.3 1.0 1.0 1.0',
  'born/coul/long': 'pair_style born/coul/long 2.5\npair_coeff * * 1.0 0.3 1.0 1.0 1.0\nkspace_style ewald 1e-4',
  morse: 'pair_style morse 2.5\npair_coeff * * 1.0 2.0 1.1',
  'lj/cut/coul/dsf': 'pair_style lj/cut/coul/dsf 0.5 2.5\npair_coeff * * 1.0 1.0',
  'lj/cut/coul/wolf': 'pair_style lj/cut/coul/wolf 0.5 2.5\npair_coeff * * 1.0 1.0',
  'coul/dsf': 'pair_style coul/dsf 0.5 2.5\npair_coeff * *',
  'coul/wolf': 'pair_style coul/wolf 0.5 2.5\npair_coeff * *',
  'lj/class2': 'pair_style lj/class2 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 0.8 1.1',
  'lj/class2/coul/cut': 'pair_style lj/class2/coul/cut 2.5\npair_coeff * * 1.0 1.0',
  'lj/class2/coul/long': 'pair_style lj/class2/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1e-4',
  'lj96/cut': 'pair_style lj96/cut 2.5\npair_coeff * * 1.0 1.0',
  'lj/expand': 'pair_style lj/expand 2.5\npair_coeff * * 1.0 0.8 0.2',
  'lj/smooth': 'pair_style lj/smooth 2.0 2.5\npair_coeff * * 1.0 1.0',
  'lj/smooth/linear': 'pair_style lj/smooth/linear 2.5\npair_coeff * * 1.0 1.0',
  'lj/gromacs': 'pair_style lj/gromacs 2.0 2.5\npair_coeff * * 1.0 1.0',
  'lj/cubic': 'pair_style lj/cubic\npair_coeff * * 1.0 0.8908987',
  'mie/cut': 'pair_style mie/cut 2.5\npair_coeff * * 1.0 1.0 12.0 6.0',
  soft: 'pair_style soft 1.12\npair_coeff * * 10.0',
  yukawa: 'pair_style yukawa 2.0 2.5\npair_coeff * * 1.0',
  gauss: 'pair_style gauss 2.5\npair_coeff * * 1.0 1.0',
  table: 'pair_style table linear 1000\npair_coeff * * lj.table LJ',
  zbl: 'pair_style zbl 0.8 1.2\npair_coeff * * 1.0 1.0',
};

/** A tabulated 12-6 LJ (pair_table.html file format: keyword, N and R, then index r e f per line) for pair style table. */
const ljTable = (): string => {
  const n = 1000, lo = 0.5, hi = 2.5;
  const lines = ['# LJ for the threads test', 'LJ', `N ${n} R ${lo} ${hi}`, ''];
  for (let k = 0; k < n; k++) {
    const r = lo + (k * (hi - lo)) / (n - 1), r6 = r ** -6;
    lines.push(`${k + 1} ${r} ${4 * (r6 * r6 - r6)} ${(48 * r6 * r6 - 24 * r6) / r}`);
  }
  return lines.join('\n');
};

/** The EAM styles (two-pass styles, threadedPairs.ts TwoPassPair): a metal box and the potential files they read. */
const EAM_BOX = `units metal
atom_style atomic
lattice fcc 3.6
region box block 0 5 0 5 0 5
create_box 2 box
create_atoms 1 box
mass * 60.0
set group all type/fraction 2 0.5 4321
velocity all create 600.0 87287 loop geom
`;
const EAM_STYLES: Record<string, string> = {
  eam: 'pair_style eam\npair_coeff * * Cu_u3.eam',
  'eam/alloy': 'pair_style eam/alloy\npair_coeff * * CuNi.eam.alloy Cu Ni',
  'eam/fs': 'pair_style eam/fs\npair_coeff * * AlFe_mm.eam.fs Al Fe',
};
/** The full-list three-body styles: a displaced diamond Si lattice (two types for tersoff/zbl's Si-C file). */
const SI_BOX = `units metal
atom_style atomic
lattice diamond 5.431
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
mass * 28.06
set group all type/fraction 2 0.3 4321
displace_atoms all random 0.1 0.1 0.1 8191
velocity all create 1500.0 87287 loop geom
`;
const THREEBODY_STYLES: Record<string, string> = {
  tersoff: 'pair_style tersoff\npair_coeff * * Si.tersoff Si Si',
  'tersoff/mod': 'pair_style tersoff/mod\npair_coeff * * Si.tersoff.mod Si Si',
  'tersoff/mod/c': 'pair_style tersoff/mod/c\npair_coeff * * Si.tersoff.modc Si Si',
  'tersoff/zbl': 'pair_style tersoff/zbl\npair_coeff * * SiC.tersoff.zbl Si C',
  'sw/mod': 'pair_style sw/mod\npair_coeff * * Si.sw Si Si',
  // full-list many-body styles whose rows add only to their own atoms; the spline files' CubicSpline objects get their
  // class back on the thread's copy (Pair.threadAdopt)
  'meam/spline': 'pair_style meam/spline\npair_coeff * * TiO.meam.spline Ti O',
  'meam/sw/spline': 'pair_style meam/sw/spline\npair_coeff * * Si.b.meam.sw.spline Si Si',
  vashishta: 'pair_style vashishta\npair_coeff * * SiC.vashishta Si C',
};
/** pair snap (full list; each row's bispectrum energy, with its self term, is added once by the thread owning the row). */
const SNAP_BOX = `units metal
atom_style atomic
lattice bcc 3.316
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
mass * 180.88
set group all type/fraction 2 0.5 4321
displace_atoms all random 0.05 0.05 0.05 8191
velocity all create 1000.0 87287 loop geom
`;
const SNAP_STYLES: Record<string, string> = {
  snap: 'pair_style snap\npair_coeff * * Ta06A.snapcoeff Ta06A.snapparam Ta Ta',
  // chemflag 1 (the explicit multi-element bispectrum, built by each thread's copy on its first compute)
  'snap (chemflag)': 'pair_style snap\npair_coeff * * InP_JCPA2020.snapcoeff InP_JCPA2020.snapparam In P',
};
/** GPL-ported styles on the threads (lazyStyles.ts GPL_THREADED_PAIRS), in the Si box. */
const GPL_STYLES: Record<string, string> = {
  sw: 'pair_style sw\npair_coeff * * Si.sw Si Si',
};
/** GPL-ported pairwise styles on the threads, in the charged LJ box. */
const GPL_PAIR_STYLES: Record<string, string> = {
  'born/coul/long/cs': 'pair_style born/coul/long/cs 2.5\npair_coeff * * 1.0 0.3 1.0 1.0 1.0\nkspace_style ewald 1e-4',
  'buck/coul/long/cs': 'pair_style buck/coul/long/cs 2.5\npair_coeff * * 100.0 0.3 1.0\nkspace_style ewald 1e-4',
  'born/coul/dsf/cs': 'pair_style born/coul/dsf/cs 0.5 2.5\npair_coeff * * 1.0 0.3 1.0 1.0 1.0',
  'born/coul/wolf/cs': 'pair_style born/coul/wolf/cs 0.5 2.5\npair_coeff * * 1.0 0.3 1.0 1.0 1.0',
  'coul/wolf/cs': 'pair_style coul/wolf/cs 0.5 2.5\npair_coeff * *',
  'coul/long/cs': 'pair_style coul/long/cs 2.5\npair_coeff * *\nkspace_style ewald 1e-4',
  'lj/cut/coul/long/cs': 'pair_style lj/cut/coul/long/cs 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1e-4',
  'lj/class2/coul/long/cs': 'pair_style lj/class2/coul/long/cs 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1e-4',
  'buck/long/coul/long': 'pair_style buck/long/coul/long cut long 2.5\npair_coeff * * 100.0 0.3 1.0\nkspace_style ewald 1e-4',
};
const POTENTIALS = ['Cu_u3.eam', 'CuNi.eam.alloy', 'AlFe_mm.eam.fs', 'Si.tersoff', 'Si.tersoff.mod', 'Si.tersoff.modc', 'SiC.tersoff.zbl', 'Si.sw',
  'TiO.meam.spline', 'Si.b.meam.sw.spline', 'Ta06A.snapcoeff', 'Ta06A.snapparam', 'InP_JCPA2020.snapcoeff', 'InP_JCPA2020.snapparam', 'SiC.vashishta'];
const potential = (f: string): string => readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', f), 'utf8');

interface Outcome { rows: Record<string, number>[]; x: number[]; threadedComputes: number; pa: number[] }

const runCase = async (pairLines: string, threads: number, extra = '', box = BOX, load = false, steps = 30): Promise<Outcome> => {
  const workers: InProcessWorker[] = [];
  // load: the workers build their styles as the browser's pair worker does, and the input runs a run 0 first
  const backend = threads > 1 ? new SharedThreadsBackend(threads, () => { const w = new InProcessWorker(load); workers.push(w); return w; }) : undefined;
  if (backend) backend.pairThreads.minAtoms = 1;
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => files.set(n, t) }, backend);
  if (box === EAM_BOX || box === SI_BOX || box === SNAP_BOX) for (const f of POTENTIALS) session.addFile(f, potential(f));
  if (pairLines.includes('lj.table')) session.addFile('lj.table', ljTable());
  await session.execute(`${box}${pairLines}
neighbor 0.3 bin
fix 1 all nve
timestep 0.002
thermo_style custom step pe evdwl ecoul elong press pxx pyz
thermo_modify format float %.16g
thermo 10
${extra}
${load ? 'run 0' : ''}`);
  // a GPL port reaches the workers after the engine thread has loaded its registry module (asynchronously): a run 0
  // sends it, and a moment later every worker has built it
  if (load) await new Promise((r) => setTimeout(r, 200));
  await session.execute(`run ${steps}
write_dump all custom out.dump id x y z fx fy fz modify sort id format float %.17g
`);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const dump = files.get('out.dump')!.trim().split('\n');
  const k0 = dump.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const x = dump.slice(k0 + 1).flatMap((l) => l.trim().split(/\s+/).slice(1).map(Number));
  // a dump of per-atom values the extra lines define (every frame's columns after the ID)
  const pa = (files.get('pa.dump') ?? '').split('\n').filter((l) => /^\s*\d+\s/.test(l) && l.trim().split(/\s+/).length > 2)
    .flatMap((l) => l.trim().split(/\s+/).slice(1).map(Number));
  backend?.dispose();
  return { rows, x, threadedComputes: workers.reduce((s, w) => s + w.computes, 0), pa };
};

const expectClose = (a: Outcome, b: Outcome, rel: number): void => {
  expect(a.rows.length).toBe(b.rows.length);
  for (let r = 0; r < a.rows.length; r++) {
    for (const k of Object.keys(a.rows[r])) {
      const u = a.rows[r][k], v = b.rows[r][k];
      expect(Math.abs(u - v), `row ${r} ${k}: ${u} vs ${v}`).toBeLessThanOrEqual(rel * Math.max(1, Math.abs(u)));
    }
  }
  let worst = 0;
  for (let i = 0; i < a.x.length; i++) worst = Math.max(worst, Math.abs(a.x[i] - b.x[i]) / Math.max(1, Math.abs(a.x[i])));
  expect(worst).toBeLessThanOrEqual(rel);
  expect(a.pa.length).toBe(b.pa.length);
  let worstPa = 0;
  for (let i = 0; i < a.pa.length; i++) worstPa = Math.max(worstPa, Math.abs(a.pa[i] - b.pa[i]) / Math.max(1, Math.abs(a.pa[i])));
  expect(worstPa).toBeLessThanOrEqual(rel);
};

/** The pair term's per-atom energy and virial, dumped every 5 steps (they are tallied on every step of the run). */
const PERATOM = `compute pea all pe/atom pair
compute sa all stress/atom NULL pair
compute spe all reduce sum c_pea c_sa[1] c_sa[5]
thermo_style custom step pe press c_spe[1] c_spe[2] c_spe[3]
thermo_modify format float %.16g
dump pa all custom 5 pa.dump id c_pea c_sa[1] c_sa[2] c_sa[3] c_sa[4] c_sa[5] c_sa[6]
dump_modify pa sort id format float %.17g`;

describe('shared-memory pair threads: helpers', () => {
  it('splits the atoms into ranges of about equal work that cover every atom once', () => {
    const nn = Int32Array.from({ length: 1000 }, (_, i) => (i % 7) * 10);
    for (const k of [1, 2, 3, 4, 7]) {
      const b = splitRanges(nn, 1000, k);
      expect(b[0]).toBe(0);
      expect(b[k]).toBe(1000);
      for (let r = 0; r < k; r++) expect(b[r + 1]).toBeGreaterThanOrEqual(b[r]);
      const work = (lo: number, hi: number) => { let s = 0; for (let i = lo; i < hi; i++) s += nn[i] + 1; return s; };
      const total = work(0, 1000);
      for (let r = 0; r < k; r++) expect(Math.abs(work(b[r], b[r + 1]) - total / k)).toBeLessThan(80);
    }
    // more ranges than atoms: some ranges are empty, the ranges still cover every atom once
    const b = splitRanges(new Int32Array(3), 3, 5);
    expect(b[0]).toBe(0);
    expect(b[5]).toBe(3);
    for (let r = 0; r < 5; r++) expect(b[r + 1]).toBeGreaterThanOrEqual(b[r]);
  });

  it('restricts a list to a range of owned atoms', () => {
    const out = restrictCounts(Int32Array.from([3, 1, 4, 1, 5]), 5, 1, 3, new Int32Array(5));
    expect(Array.from(out)).toEqual([0, 1, 4, 0, 0]);
  });

  it('copies a pair style through structured cloning, keeping nested class instances', () => {
    const p = new PairLJCutCoulLong();
    p.settings(['2.5']);
    p.allocate(2);
    p.coeff(['*', '*', '1.0', '1.0']);
    const fields = cloneableFields(p);
    const q = pairFromFields('lj/cut/coul/long', structuredClone(fields), THREADED_PAIRS) as PairLJCutCoulLong;
    expect(sameShape(p, q)).toBe(true);
    expect(q.cutCoul).toBe(2.5);
    // PairParams keeps its methods
    expect(q.p.get('epsilon', 1, 2)).toBe(p.p.get('epsilon', 1, 2));
  });
});

describe('shared-memory pair threads: every threaded style matches the single-thread engine', () => {
  for (const [name, lines] of Object.entries(STYLES)) {
    it(name, async () => {
      const serial = await runCase(lines, 1);
      const threaded = await runCase(lines, 4);
      expect(threaded.threadedComputes).toBeGreaterThan(0);
      expectClose(serial, threaded, 1e-10);
    });
  }
  for (const [name, lines] of Object.entries(EAM_STYLES)) {
    it(`${name} (two passes)`, async () => {
      const serial = await runCase(lines, 1, '', EAM_BOX);
      const threaded = await runCase(lines, 4, '', EAM_BOX);
      // two compute messages per worker and force evaluation
      expect(threaded.threadedComputes).toBeGreaterThan(0);
      expect(threaded.threadedComputes % 6).toBe(0);
      expectClose(serial, threaded, 1e-10);
    });
  }
  for (const [name, lines] of Object.entries(THREEBODY_STYLES)) {
    it(`${name} (full list, three-body)`, async () => {
      const serial = await runCase(lines, 1, '', SI_BOX);
      const threaded = await runCase(lines, 4, '', SI_BOX);
      expect(threaded.threadedComputes).toBeGreaterThan(0);
      expectClose(serial, threaded, 1e-10);
    });
  }
  for (const [name, lines] of Object.entries(SNAP_STYLES)) {
    it(`${name} (full list, many-body)`, async () => {
      const serial = await runCase(lines, 1, '', SNAP_BOX);
      const threaded = await runCase(lines, 4, '', SNAP_BOX);
      expect(threaded.threadedComputes).toBeGreaterThan(0);
      expectClose(serial, threaded, 1e-10);
    });
  }
  // the GPL-ported styles (lazyStyles.ts GPL_THREADED_PAIRS): the worker loads them through the lazy registries, and
  // gets the neighbors in native order and the atom IDs (Pair.threadNeighbors, threadTags)
  for (const [name, lines] of Object.entries(GPL_STYLES)) {
    it(`${name} (GPL port, full list in native order, atom IDs)`, async () => {
      const serial = await runCase(lines, 1, '', SI_BOX, true);
      const threaded = await runCase(lines, 4, '', SI_BOX, true);
      expect(threaded.threadedComputes).toBeGreaterThan(0);
      expectClose(serial, threaded, 1e-10);
    });
  }
  for (const [name, lines] of Object.entries(GPL_PAIR_STYLES)) {
    it(`${name} (GPL port)`, async () => {
      const serial = await runCase(lines, 1, '', BOX, true);
      const threaded = await runCase(lines, 4, '', BOX, true);
      expect(threaded.threadedComputes).toBeGreaterThan(0);
      expectClose(serial, threaded, 1e-10);
    });
  }
  it('covers every style of THREADED_PAIRS', () => {
    // the TIP4P styles (M sites as shared inputs, Pair.threadExtras) have their own test, engineTip4pThreads.test.ts
    const tip4p = ['lj/cut/tip4p/cut', 'lj/cut/tip4p/long', 'tip4p/cut', 'tip4p/long'];
    expect([...Object.keys(STYLES), ...Object.keys(EAM_STYLES), ...Object.keys(THREEBODY_STYLES), 'snap', ...tip4p].sort()).toEqual(Object.keys(THREADED_PAIRS).sort());
    // eff/cut (per-atom inputs and outputs, Pair.threadExtras) has its own test, engineEffThreads.test.ts
    expect([...Object.keys(GPL_STYLES), ...Object.keys(GPL_PAIR_STYLES), 'eff/cut'].sort()).toEqual([...GPL_THREADED_PAIRS].sort());
    // the engine side knows the styles by name only (threadedPairs.ts), the worker by factory (threadedPairFactories.ts,
    // and for the GPL ports the lazy registries)
    expect([...THREADED_NAMES].sort()).toEqual([...Object.keys(THREADED_PAIRS), ...GPL_THREADED_PAIRS].sort());
  });
});

/*
 * Per-atom energy and virial (compute pe/atom, stress/atom): every thread tallies into its own arrays ('ebuffers'), the
 * engine thread adds them to its own in worker order.
 */
describe('shared-memory pair threads: per-atom energy and virial of every threaded style', () => {
  const cases: [string, string, string, boolean][] = [
    ...Object.entries(STYLES).map(([n, l]): [string, string, string, boolean] => [n, l, BOX, false]),
    ...Object.entries(EAM_STYLES).map(([n, l]): [string, string, string, boolean] => [n, l, EAM_BOX, false]),
    ...Object.entries(THREEBODY_STYLES).map(([n, l]): [string, string, string, boolean] => [n, l, SI_BOX, false]),
    ...Object.entries(SNAP_STYLES).map(([n, l]): [string, string, string, boolean] => [n, l, SNAP_BOX, false]),
    ...Object.entries(GPL_STYLES).map(([n, l]): [string, string, string, boolean] => [n, l, SI_BOX, true]),
    ...Object.entries(GPL_PAIR_STYLES).map(([n, l]): [string, string, string, boolean] => [n, l, BOX, true]),
  ];
  for (const [name, lines, box, load] of cases) {
    it(name, async () => {
      const serial = await runCase(lines, 1, PERATOM, box, load, 10);
      const threaded = await runCase(lines, 3, PERATOM, box, load, 10);
      expect(threaded.threadedComputes).toBeGreaterThan(0);
      expect(serial.pa.length).toBeGreaterThan(0);
      expect(serial.pa.some((v) => v !== 0)).toBe(true);
      expectClose(serial, threaded, 1e-10);
    });
  }
});

/**
 * pair_style hybrid and hybrid/overlay: each threaded sub-style runs on the threads in its own pair slot (force/pair/
 * hybrid.ts subRunner), the others on the engine thread.
 */
const HYBRID_CASES: [string, string, string, boolean][] = [
  ['hybrid/overlay lj/cut coul/cut', 'pair_style hybrid/overlay lj/cut 2.5 coul/cut 2.5\npair_coeff * * lj/cut 1.0 1.0\npair_coeff * * coul/cut', BOX, false],
  ['hybrid lj/cut morse (by type pair)', 'pair_style hybrid lj/cut 2.5 morse 2.5\npair_coeff 1 1 lj/cut 1.0 1.0\npair_coeff 2 2 lj/cut 1.0 1.0\npair_coeff 1 2 morse 1.0 2.0 1.1', BOX, false],
  ['hybrid/overlay eam/alloy lj/cut (two passes in a slot)', 'pair_style hybrid/overlay eam/alloy lj/cut 3.0\npair_coeff * * eam/alloy CuNi.eam.alloy Cu Ni\npair_coeff 1 2 lj/cut 0.01 2.5', EAM_BOX, false],
  ['hybrid/overlay tersoff lj/cut (full list in a slot)', 'pair_style hybrid/overlay tersoff lj/cut 3.0\npair_coeff * * tersoff Si.tersoff Si Si\npair_coeff * * lj/cut 0.001 2.0', SI_BOX, false],
  ['hybrid/overlay sw lj/cut (a GPL sub-style)', 'pair_style hybrid/overlay sw lj/cut 3.0\npair_coeff * * sw Si.sw Si Si\npair_coeff * * lj/cut 0.001 2.0', SI_BOX, true],
];

describe('shared-memory pair threads: the sub-styles of pair_style hybrid', () => {
  for (const [name, lines, box, gpl] of HYBRID_CASES) {
    it(name, async () => {
      const serial = await runCase(lines, 1, '', box, gpl);
      const threaded = await runCase(lines, 4, '', box, gpl);
      expect(threaded.threadedComputes).toBeGreaterThan(0);
      expectClose(serial, threaded, 1e-10);
    });
  }
});

describe('shared-memory pair threads: refresh and fallback', () => {
  it('follows fix adapt changing the pair coefficients during the run', async () => {
    const extra = 'variable e equal 1.0+0.01*elapsed\nfix ad all adapt 1 pair lj/cut epsilon * * v_e';
    const serial = await runCase(STYLES['lj/cut'], 1, extra);
    const threaded = await runCase(STYLES['lj/cut'], 3, extra);
    expect(threaded.threadedComputes).toBeGreaterThan(0);
    expectClose(serial, threaded, 1e-10);
  });

  it('follows fix adapt changing coul/long and eam/alloy scale during the run (the worker copies take the new table)', async () => {
    const extra = 'variable s equal 1.0-0.002*elapsed\nfix ad all adapt 2 pair coul/long scale 1 2 v_s';
    const serial = await runCase(STYLES['coul/long'], 1, extra);
    const threaded = await runCase(STYLES['coul/long'], 3, extra);
    expect(threaded.threadedComputes).toBeGreaterThan(0);
    expectClose(serial, threaded, 1e-10);
    const eamExtra = 'variable s equal 1.0+0.003*elapsed\nfix ad all adapt 2 pair eam/alloy scale 1 1 v_s';
    const eSerial = await runCase(EAM_STYLES['eam/alloy'], 1, eamExtra, EAM_BOX);
    const eThreaded = await runCase(EAM_STYLES['eam/alloy'], 3, eamExtra, EAM_BOX);
    expect(eThreaded.threadedComputes).toBeGreaterThan(0);
    expectClose(eSerial, eThreaded, 1e-10);
    // and the scale changed the energy
    const plain = await runCase(EAM_STYLES['eam/alloy'], 1, '', EAM_BOX);
    expect(Math.abs(plain.rows[3].pe - eSerial.rows[3].pe)).toBeGreaterThan(1e-3);
  });

  it('runs per-atom energy evaluations on the threads', async () => {
    const extra = 'compute pea all pe/atom\ncompute spe all reduce sum c_pea\nthermo_style custom step pe c_spe';
    const serial = await runCase(STYLES['lj/cut'], 1, extra);
    const threaded = await runCase(STYLES['lj/cut'], 3, extra);
    expect(threaded.threadedComputes).toBeGreaterThan(0);
    expectClose(serial, threaded, 1e-10);
  });

  it('runs on the engine thread until every worker has started (no Atomics.wait on an unstarted worker)', async () => {
    // a worker created inside the engine worker starts only when the engine thread returns to its event loop: until
    // then it handles no message, so the threads must not be used (they would block in Atomics.wait)
    class UnstartedWorker extends InProcessWorker {
      postMessage(msg: unknown): void { if ((msg as PairThreadMessage).type === 'compute') this.computes++; }
    }
    const workers: UnstartedWorker[] = [];
    const backend = new SharedThreadsBackend(3, () => { const w = new UnstartedWorker(); workers.push(w); return w; });
    backend.pairThreads.minAtoms = 1;
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, backend);
    await session.execute(`${BOX}${STYLES['lj/cut']}\nfix 1 all nve\nthermo 10\nrun 20\n`);
    expect(workers.reduce((s, w) => s + w.computes, 0)).toBe(0);
    expect(backend.pairThreads.calls).toBe(0);
    expect(events.filter((e) => e.kind === 'thermo').length).toBeGreaterThan(0);
    backend.dispose();
  });

  it('leaves the eam/alloy densities and F\' where fix pair reads them', async () => {
    const extra = 'fix fp all pair 10 eam/alloy rho 0 fp 0\ncompute rs all reduce sum f_fp[1] f_fp[2]\nthermo_style custom step pe c_rs[1] c_rs[2]';
    const serial = await runCase(EAM_STYLES['eam/alloy'], 1, extra, EAM_BOX);
    const threaded = await runCase(EAM_STYLES['eam/alloy'], 3, extra, EAM_BOX);
    expect(threaded.threadedComputes).toBeGreaterThan(0);
    expect(serial.rows[1]['c_rs[1]']).toBeGreaterThan(0);
    expectClose(serial, threaded, 1e-10);
  });

  it('runs eam/alloy per-atom energy evaluations on the threads (the embedding energy on the engine thread)', async () => {
    const extra = 'compute pea all pe/atom\ncompute spe all reduce sum c_pea\nthermo_style custom step pe c_spe';
    const serial = await runCase(EAM_STYLES['eam/alloy'], 1, extra, EAM_BOX);
    const threaded = await runCase(EAM_STYLES['eam/alloy'], 3, extra, EAM_BOX);
    expect(threaded.threadedComputes).toBeGreaterThan(0);
    expectClose(serial, threaded, 1e-10);
  });

  it('leaves styles outside the list (born/coul/dsf) to the engine thread', async () => {
    const lines = 'pair_style born/coul/dsf 0.8 2.5\npair_coeff * * 1.0 0.3 1.0 1.0 1.0';
    const serial = await runCase(lines, 1);
    const threaded = await runCase(lines, 3);
    expect(threaded.threadedComputes).toBe(0);
    expectClose(serial, threaded, 0);
  });
});
