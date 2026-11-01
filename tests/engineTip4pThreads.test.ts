import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import { PairThreadWorker, type PairThreadMessage, type PairWorkerLike } from '../src/engine/cpu/pairThreadsCore';
import { THREADED_PAIRS } from '../src/engine/cpu/threadedPairFactories';

/*
 * The TIP4P pair styles on the pair threads (Pair.threadExtras): the engine thread places the M sites and finds each
 * oxygen's hydrogens and each ghost's owner, the threads read them as shared inputs. The workers run in-process here.
 */

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  readonly worker = new PairThreadWorker(THREADED_PAIRS);
  computes = 0;
  postMessage(msg: unknown): void {
    const m = msg as PairThreadMessage;
    if (m.type === 'compute') this.computes++;
    this.worker.handle(m);
  }
  terminate(): void {}
}

/** n^3 waters on a cubic grid (spacing a), each turned by a fixed pseudo-random rotation: a data file (atom_style full). */
const waterData = (n: number, a: number): string => {
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const r0 = 0.9572, half = (104.52 / 2) * Math.PI / 180;
  const atoms: string[] = [], bonds: string[] = [], angles: string[] = [];
  let id = 0, mol = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    mol++;
    const o = [(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a];
    // a random orthonormal frame (two angles) for the molecule's plane
    const th = Math.PI * rnd(), ph = 2 * Math.PI * rnd(), ps = 2 * Math.PI * rnd();
    const u = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    const t0 = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const v = [u[1] * t0[2] - u[2] * t0[1], u[2] * t0[0] - u[0] * t0[2], u[0] * t0[1] - u[1] * t0[0]];
    const vn = Math.hypot(v[0], v[1], v[2]);
    const w = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]].map((c) => c / vn);
    const e = v.map((c, m) => (c / vn) * Math.cos(ps) + w[m] * Math.sin(ps));
    const h1 = o.map((c, m) => c + r0 * (Math.cos(half) * u[m] + Math.sin(half) * e[m]));
    const h2 = o.map((c, m) => c + r0 * (Math.cos(half) * u[m] - Math.sin(half) * e[m]));
    const oi = ++id;
    atoms.push(`${oi} ${mol} 1 -1.0484 ${o.join(' ')}`);
    atoms.push(`${++id} ${mol} 2 0.5242 ${h1.join(' ')}`);
    atoms.push(`${++id} ${mol} 2 0.5242 ${h2.join(' ')}`);
    bonds.push(`${bonds.length + 1} 1 ${oi} ${oi + 1}`, `${bonds.length + 2} 1 ${oi} ${oi + 2}`);
    angles.push(`${angles.length + 1} 1 ${oi + 1} ${oi} ${oi + 2}`);
  }
  const L = n * a;
  return ['water box', '', `${id} atoms`, `${bonds.length} bonds`, `${angles.length} angles`, '2 atom types', '1 bond types', '1 angle types', '',
    `0 ${L} xlo xhi`, `0 ${L} ylo yhi`, `0 ${L} zlo zhi`, '', 'Masses', '', '1 15.9994', '2 1.008', '',
    'Atoms # full', '', ...atoms, '', 'Bonds', '', ...bonds, '', 'Angles', '', ...angles, ''].join('\n');
};

const STYLES: Record<string, string> = {
  'lj/cut/tip4p/long': 'pair_style lj/cut/tip4p/long 1 2 1 1 0.1546 6.0\npair_coeff 1 1 0.1852 3.1589\npair_coeff 2 2 0.0 1.0\npair_coeff 1 2 0.0 1.0\nkspace_style pppm/tip4p 1.0e-4',
  'lj/cut/tip4p/cut': 'pair_style lj/cut/tip4p/cut 1 2 1 1 0.1546 6.0\npair_coeff 1 1 0.1852 3.1589\npair_coeff 2 2 0.0 1.0\npair_coeff 1 2 0.0 1.0',
  'tip4p/long': 'pair_style tip4p/long 1 2 1 1 0.1546 6.0\npair_coeff * *\nkspace_style pppm/tip4p 1.0e-4',
  'tip4p/cut': 'pair_style tip4p/cut 1 2 1 1 0.1546 6.0\npair_coeff * *',
};

const run = async (lines: string, threads: number, peratom: boolean) => {
  const workers: InProcessWorker[] = [];
  const backend = threads > 1 ? new SharedThreadsBackend(threads, () => { const w = new InProcessWorker(); workers.push(w); return w; }) : undefined;
  if (backend) backend.pairThreads.minAtoms = 1;
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) }, backend);
  s.addFile('water.data', waterData(4, 3.15));
  await s.execute(`units real
atom_style full
read_data water.data
bond_style harmonic
bond_coeff 1 450.0 0.9572
angle_style harmonic
angle_coeff 1 55.0 104.52
${lines}
neighbor 2.0 bin
velocity all create 300.0 4928459 dist gaussian
fix 1 all nve
timestep 0.5
thermo_style custom step pe evdwl ecoul elong press pxx pyz
thermo_modify format float %.16g
thermo 5
${peratom ? 'compute pea all pe/atom pair\ncompute sa all stress/atom NULL pair\ndump pa all custom 5 pa.dump id c_pea c_sa[1] c_sa[2] c_sa[3] c_sa[4] c_sa[5] c_sa[6]\ndump_modify pa sort id format float %.17g' : ''}
run 10
write_dump all custom out.dump id x y z fx fy fz modify sort id format float %.17g
`);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const nums = (name: string) => (files.get(name) ?? '').split('\n').filter((l) => /^\s*\d+\s+\S/.test(l) && l.trim().split(/\s+/).length > 3)
    .flatMap((l) => l.trim().split(/\s+/).slice(1).map(Number));
  backend?.dispose();
  return { rows, x: nums('out.dump'), pa: nums('pa.dump'), threaded: workers.reduce((n, w) => n + w.computes, 0) };
};

const worst = (a: number[], b: number[]) => {
  expect(a.length).toBe(b.length);
  let w = 0;
  for (let i = 0; i < a.length; i++) w = Math.max(w, Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(a[i])));
  return w;
};

describe('the TIP4P pair styles on the pair threads', () => {
  for (const [name, lines] of Object.entries(STYLES)) {
    for (const peratom of [false, true]) {
      it(`${name}${peratom ? ' (per-atom energy and virial)' : ''}`, async () => {
        const serial = await run(lines, 1, peratom);
        const threaded = await run(lines, 3, peratom);
        expect(threaded.threaded).toBeGreaterThan(0);
        expect(threaded.rows.length).toBe(serial.rows.length);
        for (let r = 0; r < serial.rows.length; r++) {
          for (const k of Object.keys(serial.rows[r])) {
            const u = serial.rows[r][k], v = threaded.rows[r][k];
            expect(Math.abs(u - v), `row ${r} ${k}: ${u} vs ${v}`).toBeLessThanOrEqual(1e-10 * Math.max(1, Math.abs(u)));
          }
        }
        expect(worst(serial.x, threaded.x)).toBeLessThanOrEqual(1e-10);
        if (peratom) {
          expect(serial.pa.length).toBeGreaterThan(0);
          expect(worst(serial.pa, threaded.pa)).toBeLessThanOrEqual(1e-10);
        }
      });
    }
  }
});
