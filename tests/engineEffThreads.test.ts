import { describe, expect, it, vi } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import { PairThreadWorker, type PairThreadMessage, type PairWorkerLike } from '../src/engine/cpu/pairThreadsCore';
import { THREADED_PAIRS } from '../src/engine/cpu/threadedPairFactories';
import { THREADED as GPL_THREADED } from '../src/engine-gpl/registry/threaded_pairs';

/*
 * pair eff/cut on the pair threads (Pair.threadExtras): the electron spin and radius go to the threads as shared
 * inputs, each thread adds the radial electron forces and the pvector terms into its own arrays, and the engine thread
 * adds them up and finishes (the reverse sum into the owned electrons, the radial virial with pressure/evirials). The
 * workers run in-process here and build the style from the GPL registry's factories.
 */

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  readonly worker = new PairThreadWorker({ ...THREADED_PAIRS, ...GPL_THREADED });
  computes = 0;
  postMessage(msg: unknown): void {
    const m = msg as PairThreadMessage;
    if (m.type === 'compute') this.computes++;
    this.worker.handle(m);
  }
  terminate(): void {}
}

/** Hydrogen: nuclei on a simple cubic lattice, an electron near each, spins up in the upper half and down below. */
const SETUP = (pairArgs: string) => `units electron
atom_style electron
lattice sc 2.4
region box block 0 5 0 5 0 5
create_box 2 box
create_atoms 1 box
lattice sc 2.4 origin 0.2 0.25 0.3
create_atoms 2 box
mass 1 1.007940
mass 2 1.000000
set type 1 charge 1.0
set type 1 spin/electron 0
set type 2 charge -1.0
set type 2 spin/electron 1
set type 2 radius/electron 1.1
region lower block INF INF INF INF INF 6.0 units box
group lower region lower
group electrons type 2
group down intersect lower electrons
set group down spin/electron -1
pair_style eff/cut ${pairArgs}
pair_coeff * *
comm_modify vel yes
neighbor 0.5 bin
fix 1 all nve/eff
timestep 0.001
thermo_style custom step pe ke etotal press
thermo_modify format float %.16g
thermo 5
`;

const run = async (pairArgs: string, threads: number) => {
  const workers: InProcessWorker[] = [];
  const backend = threads > 1 ? new SharedThreadsBackend(threads, () => { const w = new InProcessWorker(); workers.push(w); return w; }) : undefined;
  if (backend) backend.pairThreads.minAtoms = 1;
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => files.set(n, t) }, backend);
  await s.execute(`${SETUP(pairArgs)}run 0`);
  await s.execute(`run 20
compute pv all pair eff/cut
variable ke equal c_pv[1]
variable pauli equal c_pv[2]
variable ec equal c_pv[3]
variable rr equal c_pv[4]
thermo_style custom step pe v_ke v_pauli v_ec v_rr press
run 0
`);
  const before = workers.reduce((n, w) => n + w.computes, 0);
  // per-atom energy and virial (pe/atom, stress/atom), tallied on the threads too
  await s.execute(`compute ep all property/atom eradius erforce
compute pea all pe/atom
compute sa all stress/atom NULL
run 0
write_dump all custom out.dump id x y z fx fy fz c_ep[1] c_ep[2] c_pea c_sa[1] c_sa[2] c_sa[3] c_sa[4] c_sa[5] c_sa[6] modify sort id format float %.17g
`);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const peratomThreaded = workers.reduce((n, w) => n + w.computes, 0) - before;
  const dump = files.get('out.dump')!.trim().split('\n');
  const k0 = dump.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const vals = dump.slice(k0 + 1).flatMap((l) => l.trim().split(/\s+/).slice(1).map(Number));
  backend?.dispose();
  return { rows, vals, threaded: workers.reduce((n, w) => n + w.computes, 0), peratomThreaded };
};

const close = (a: number, b: number, rel: number) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(a));

describe('pair eff/cut on the pair threads', () => {
  for (const args of ['4.0', '4.0 limit/eradius pressure/evirials']) {
    it(`eff/cut ${args}: the one-thread run, with the radial forces and the pvector terms`, async () => {
      const serial = await run(args, 1);
      const threaded = await run(args, 4);
      expect(threaded.threaded).toBeGreaterThan(0);
      expect(threaded.peratomThreaded).toBeGreaterThan(0);
      expect(threaded.rows.length).toBe(serial.rows.length);
      for (let r = 0; r < serial.rows.length; r++) {
        for (const k of Object.keys(serial.rows[r])) {
          expect(close(serial.rows[r][k], threaded.rows[r][k], 1e-10), `row ${r} ${k}: ${serial.rows[r][k]} vs ${threaded.rows[r][k]}`).toBe(true);
        }
      }
      expect(threaded.vals.length).toBe(serial.vals.length);
      let worst = 0;
      for (let i = 0; i < serial.vals.length; i++) worst = Math.max(worst, Math.abs(serial.vals[i] - threaded.vals[i]) / Math.max(1, Math.abs(serial.vals[i])));
      expect(worst).toBeLessThan(1e-10);
      // the pvector and the radial forces are not zero (the test sees them)
      const last = serial.rows[serial.rows.length - 1];
      expect(Math.abs(last.v_ke)).toBeGreaterThan(0);
    });
  }
});

/*
 * A GPL-ported style's registry module is loaded on the engine thread before the run (SharedPairThreads.preload): a
 * load started inside the run advances only at the run's yields between steps, and in the browser an eff/cut run of
 * 13718 atoms (seconds per step) stayed on one thread for its whole length ("the threads were still starting").
 */
describe('the pair threads load a GPL style before the run', () => {
  const spawn = () => new InProcessWorker();
  it('run and minimize preload the pair style, or each hybrid sub-style', async () => {
    const backend = new SharedThreadsBackend(3, spawn);
    const preload = vi.spyOn(backend.pairThreads, 'preload');
    const s = new Session({ emit: () => {} }, backend);
    await s.execute(`${SETUP('4.0')}run 1\nmin_style cg\nminimize 0 0 1 1`);
    expect(preload.mock.calls.map((c) => [...c[0]])).toEqual([['eff/cut'], ['eff/cut']]);
    await s.execute(`clear
units lj
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style hybrid/overlay lj/cut 2.5 soft 1.0
pair_coeff * * lj/cut 1.0 1.0
pair_coeff * * soft 1.0
run 0`);
    expect([...preload.mock.calls[2][0]]).toEqual(['lj/cut', 'soft']);
    backend.dispose();
  });
  it('only a GPL style\'s module URL is kept', async () => {
    const backend = new SharedThreadsBackend(3, spawn);
    const pt = backend.pairThreads;
    await pt.preload(['eff/cut', 'lj/cut', 'sw']);
    const urls = (pt as unknown as { gplUrls: Map<string, string> }).gplUrls;
    expect([...urls.keys()].sort()).toEqual(['eff/cut', 'sw']);
    expect(urls.get('eff/cut')).toMatch(/threaded_pairs/);
    backend.dispose();
  });
});
