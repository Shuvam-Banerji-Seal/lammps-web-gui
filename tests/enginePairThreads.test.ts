import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import {
  PairThreadWorker, cloneableFields, pairFromFields, restrictCounts, sameShape, splitRanges, type PairThreadMessage, type PairWorkerLike,
} from '../src/engine/cpu/pairThreadsCore';
import { THREADED_PAIRS } from '../src/engine/cpu/threadedPairs';
import { PairLJCutCoulLong } from '../src/engine/force/pair/coul_long';

/*
 * Shared-memory pair threads (src/engine/cpu/pairThreads.ts). The workers run in-process here
 * (each message is handled at once), which exercises the shared buffers, the range split, the
 * pair copies and the reduction; real threads are checked in a browser.
 */

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  readonly worker = new PairThreadWorker();
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
};

interface Outcome { rows: Record<string, number>[]; x: number[]; threadedComputes: number }

const runCase = async (pairLines: string, threads: number, extra = ''): Promise<Outcome> => {
  const workers: InProcessWorker[] = [];
  const backend = threads > 1 ? new SharedThreadsBackend(threads, () => { const w = new InProcessWorker(); workers.push(w); return w; }) : undefined;
  if (backend) backend.pairThreads.minAtoms = 1;
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => files.set(n, t) }, backend);
  await session.execute(`${BOX}${pairLines}
neighbor 0.3 bin
fix 1 all nve
timestep 0.002
thermo_style custom step pe evdwl ecoul elong press pxx pyz
thermo_modify format float %.16g
thermo 10
${extra}
run 30
write_dump all custom out.dump id x y z fx fy fz modify sort id format float %.17g
`);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const dump = files.get('out.dump')!.trim().split('\n');
  const k0 = dump.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const x = dump.slice(k0 + 1).flatMap((l) => l.trim().split(/\s+/).slice(1).map(Number));
  backend?.dispose();
  return { rows, x, threadedComputes: workers.reduce((s, w) => s + w.computes, 0) };
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
};

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
    const q = pairFromFields('lj/cut/coul/long', structuredClone(fields)) as PairLJCutCoulLong;
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
  it('covers every style of THREADED_PAIRS', () => {
    expect(Object.keys(STYLES).sort()).toEqual(Object.keys(THREADED_PAIRS).sort());
  });
});

describe('shared-memory pair threads: refresh and fallback', () => {
  it('follows fix adapt changing the pair coefficients during the run', async () => {
    const extra = 'variable e equal 1.0+0.01*elapsed\nfix ad all adapt 1 pair lj/cut epsilon * * v_e';
    const serial = await runCase(STYLES['lj/cut'], 1, extra);
    const threaded = await runCase(STYLES['lj/cut'], 3, extra);
    expect(threaded.threadedComputes).toBeGreaterThan(0);
    expectClose(serial, threaded, 1e-10);
  });

  it('runs per-atom energy evaluations on the engine thread', async () => {
    const extra = 'compute pea all pe/atom\ncompute spe all reduce sum c_pea\nthermo_style custom step pe c_spe';
    const serial = await runCase(STYLES['lj/cut'], 1, extra);
    const threaded = await runCase(STYLES['lj/cut'], 3, extra);
    expectClose(serial, threaded, 1e-10);
  });

  it('leaves styles outside the list (eam, dsf) to the engine thread', async () => {
    const lines = 'pair_style lj/cut/coul/dsf 0.8 2.5\npair_coeff * * 1.0 1.0';
    const serial = await runCase(lines, 1);
    const threaded = await runCase(lines, 3);
    expect(threaded.threadedComputes).toBe(0);
    expectClose(serial, threaded, 0);
  });
});
