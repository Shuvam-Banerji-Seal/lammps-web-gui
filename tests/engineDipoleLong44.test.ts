import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * pair_style lj/cut/dipole/long with kspace_style ewald/dipole: forces and torques against central finite
 * differences of the total potential energy (real space + k space + self term), and the refusals.
 * Torque on a point dipole: tau = mu x E with E = -dU/dmu, so for a rotation of mu about a unit axis e,
 * e . tau = -dU/dtheta. Forces: F = -dU/dx. Energies are thermo pe of run 0.
 */

interface Atom { x: number[]; mu: number[]; type: number }

const ATOMS: Atom[] = [
  { x: [1.3, 1.2, 1.1], mu: [0.5, -0.3, 0.8], type: 1 },
  { x: [3.1, 2.0, 1.6], mu: [-0.6, 0.4, 0.2], type: 1 },
  { x: [2.1, 4.2, 2.5], mu: [0.3, 0.9, -0.5], type: 1 },
  { x: [4.4, 4.1, 4.7], mu: [0.2, -0.7, 0.4], type: 1 },
  { x: [1.4, 3.0, 4.3], mu: [0.7, 0.2, 0.4], type: 1 },
];

const L = 8;

interface Run {
  pe: number;
  elong: number;
  error?: string;
  rows: number[][];
}

/** Runs run 0 with the given setup lines; returns pe, elong and the per-atom fx fy fz tqx tqy tqz by id. */
const evaluate = async (setup: string[], atoms: Atom[], boundary = 'p p p', prism = false): Promise<Run> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  const lines = [
    'units lj', 'atom_style hybrid charge sphere dipole', `boundary ${boundary}`,
    prism ? `region box prism 0 ${L} 0 ${L} 0 ${L} 0.5 0 0` : `region box block 0 ${L} 0 ${L} 0 ${L}`, 'create_box 1 box',
    'mass * 1.0', 'set type * diameter 1.0 density 1.0', 'set type * charge 0.0',
  ];
  atoms.forEach((a, k) => {
    lines.push(`create_atoms ${a.type} single ${a.x.join(' ')} units box`);
    lines.push(`set atom ${k + 1} dipole ${a.mu.join(' ')}`);
  });
  lines.push(...setup, 'thermo_style custom step pe elong ecoul evdwl', 'thermo_modify format float %.17g norm no');
  lines.push('run 0', 'write_dump all custom dip44.dump id fx fy fz tqx tqy tqz modify format float %.17g sort id');
  try {
    await session.execute(`${lines.join('\n')}\n`);
  } catch (e) {
    return { pe: NaN, elong: NaN, error: e instanceof Error ? e.message : String(e), rows: [] };
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) return { pe: NaN, elong: NaN, error: String(err.message), rows: [] };
  const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').at(-1)!;
  const dump = files.get('dip44.dump')!.trim().split('\n');
  const body = dump.slice(dump.findIndex((l) => l.startsWith('ITEM: ATOMS')) + 1).map((l) => l.trim().split(/\s+/).map(Number));
  return { pe: row.row.pe as number, elong: row.row.elong as number, rows: body };
};

const SETUP = [
  'pair_style lj/cut/dipole/long 4.0',
  'pair_coeff * * 0.0 1.0',
  'kspace_style ewald/dipole 1.0e-4',
  'kspace_modify gewald 0.45 kmax/ewald 4 4 4',
];

/** Rotate v about the unit axis e by angle t (Rodrigues). */
const rotate = (v: number[], e: number[], t: number): number[] => {
  const c = Math.cos(t), s = Math.sin(t);
  const ev = e[0] * v[0] + e[1] * v[1] + e[2] * v[2];
  const cr = [e[1] * v[2] - e[2] * v[1], e[2] * v[0] - e[0] * v[2], e[0] * v[1] - e[1] * v[0]];
  return [0, 1, 2].map((d) => v[d] * c + cr[d] * s + e[d] * ev * (1 - c));
};

const H = 1e-6;

describe('lj/cut/dipole/long with kspace ewald/dipole', () => {
  it('forces agree with finite differences of the total energy', async () => {
    const base = await evaluate(SETUP, ATOMS);
    expect(base.error).toBeUndefined();
    for (let k = 0; k < ATOMS.length; k++) {
      for (let d = 0; d < 3; d++) {
        const plus = ATOMS.map((a) => ({ ...a, x: [...a.x], mu: [...a.mu] }));
        const minus = ATOMS.map((a) => ({ ...a, x: [...a.x], mu: [...a.mu] }));
        plus[k].x[d] += H; minus[k].x[d] -= H;
        const ep = (await evaluate(SETUP, plus)).pe, em = (await evaluate(SETUP, minus)).pe;
        const fd = -(ep - em) / (2 * H);
        expect(base.rows[k][1 + d]).toBeCloseTo(fd, 4);
      }
    }
  });

  it('torques agree with finite differences of the total energy (e . tau = -dU/dtheta)', async () => {
    const base = await evaluate(SETUP, ATOMS);
    for (let k = 0; k < ATOMS.length; k++) {
      for (let d = 0; d < 3; d++) {
        const e = [0, 0, 0]; e[d] = 1;
        const rot = (sign: number) => ATOMS.map((a, m) => (m === k ? { ...a, x: [...a.x], mu: rotate(a.mu, e, sign * H) } : { ...a, x: [...a.x], mu: [...a.mu] }));
        const ep = (await evaluate(SETUP, rot(1))).pe, em = (await evaluate(SETUP, rot(-1))).pe;
        const dUdtheta = (ep - em) / (2 * H);
        // e . tau for the torque component along the rotation axis
        const tau = base.rows[k].slice(4, 7);
        expect(tau[d]).toBeCloseTo(-dUdtheta, 4);
      }
    }
  });

  it('keeps the k-space self term in elong (one dipole, no pair interaction)', async () => {
    const one = [ATOMS[0]];
    const r = await evaluate(SETUP, one);
    // the self term is -(2 g^3 / (3 sqrt(pi))) mu^2 times the Coulomb constant (real units not used: lj units, C = 1)
    const g = 0.45, mu2 = ATOMS[0].mu.reduce((s, v) => s + v * v, 0);
    expect(r.elong).toBeLessThan(0);
    expect(r.elong).toBeCloseTo(-(2 * g ** 3 / (3 * Math.sqrt(Math.PI))) * mu2 + kOnly(one), 9);
  });

  it('refuses point charges, triclinic and non-periodic boxes, and unsupported keywords', async () => {
    const chargedRun = await evaluate(['set atom 1 charge 0.5', ...SETUP], ATOMS);
    expect(chargedRun.error).toMatch(/point charges/);
    const open = await evaluate(SETUP, ATOMS, 'f f f');
    expect(open.error).toMatch(/fully periodic/);
    const plain = await evaluate([...SETUP, 'kspace_modify force 0.01'], ATOMS);
    expect(plain.error).toMatch(/kspace_modify force is not supported by kspace_style ewald\/dipole/);
    const slab = await evaluate([...SETUP, 'kspace_modify slab 3.0'], ATOMS);
    expect(slab.error).toMatch(/kspace_modify slab is not supported/);
    const noKspaceDipole = await evaluate(['pair_style lj/cut/dipole/long 4.0', 'pair_coeff * * 0.0 1.0'], ATOMS);
    expect(noKspaceDipole.error).toMatch(/kspace/);
    const sixArg = await evaluate(['pair_style lj/cut/dipole/long 4.0', 'pair_coeff 1 1 0.0 1.0 3.0 3.5', ...SETUP.slice(2)], ATOMS);
    expect(sixArg.error).toMatch(/accepts only the LJ cutoff/);
    const tilt = await evaluate(SETUP, ATOMS, 'p p p', true);
    expect(tilt.error).toMatch(/orthogonal boxes only/);
  });
});

/** The k-space energy of a single dipole (the self term is removed) at the fixed G and kmax of SETUP. */
function kOnly(atoms: Atom[]): number {
  const g = 0.45, L8 = L, V = L8 * L8 * L8;
  let e = 0;
  const km = 4;
  for (let nx = 0; nx <= km; nx++) for (let ny = -km; ny <= km; ny++) for (let nz = -km; nz <= km; nz++) {
    if (nx === 0 && (ny < 0 || (ny === 0 && nz <= 0))) continue;
    if (nx * nx + ny * ny + nz * nz > km * km) continue;
    const K = [2 * Math.PI * nx / L8, 2 * Math.PI * ny / L8, 2 * Math.PI * nz / L8];
    const K2 = K[0] * K[0] + K[1] * K[1] + K[2] * K[2];
    const G = Math.exp(-K2 / (4 * g * g)) / K2;
    let mr = 0, mi = 0;
    for (const a of atoms) {
      const ph = K[0] * a.x[0] + K[1] * a.x[1] + K[2] * a.x[2];
      const mk = a.mu[0] * K[0] + a.mu[1] * K[1] + a.mu[2] * K[2];
      mr += mk * Math.cos(ph); mi += mk * Math.sin(ph);
    }
    e += (4 * Math.PI / V) * G * (mr * mr + mi * mi);
  }
  return e;
}

