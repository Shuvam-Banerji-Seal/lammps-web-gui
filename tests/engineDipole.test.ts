import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Point-dipole pair styles (lj/cut/dipole/cut, lj/sf/dipole/sf): forces and torques checked against central
 * finite differences of the pair energy. Torque on a dipole: tau = p x E, so for a rotation of p about a unit
 * axis e, e . tau = -dU/d(theta). Forces: F = -dU/dx. Energies come from thermo pe (pair only, run 0).
 */

interface Atom { x: number[]; q: number; mu: number[]; type: number }

const ATOMS: Atom[] = [
  { x: [0.3, 0.2, 0.1], q: 0.7, mu: [0.5, -0.3, 0.8], type: 1 },
  { x: [1.9, 0.8, 0.4], q: -0.4, mu: [-0.6, 0.4, 0.2], type: 2 },
  { x: [1.1, 2.2, 1.5], q: 0, mu: [0.3, 0.9, -0.5], type: 1 },
  { x: [2.4, 2.1, 2.7], q: 0.5, mu: [0, 0, 0], type: 2 },
  { x: [0.4, 1.0, 2.3], q: -0.6, mu: [0.7, 0.2, 0.4], type: 1 },
];

const L = 6;

/** Runs run 0 on the given configuration; returns pe and the per-atom force and torque by id. */
const evaluate = async (pairCmds: string[], atoms: Atom[]) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  const lines = [
    'units lj', 'atom_style hybrid sphere dipole', 'boundary p p p',
    `region box block 0 ${L} 0 ${L} 0 ${L}`, 'create_box 2 box',
    'mass * 1.0', 'set type * diameter 1.0 density 1.0',
  ];
  atoms.forEach((a, k) => {
    lines.push(`create_atoms ${a.type} single ${a.x.join(' ')} units box`);
    lines.push(`set atom ${k + 1} charge ${a.q}`);
    lines.push(`set atom ${k + 1} dipole ${a.mu.join(' ')}`);
  });
  lines.push(...pairCmds, 'thermo_style custom step pe');
  lines.push('thermo_modify format float %.17g norm no', 'timestep 0.0001', 'run 0');
  lines.push('write_dump all custom dip.dump id fx fy fz tqx tqy tqz mux muy muz modify format float %.17g sort id');
  await session.execute(`${lines.join('\n')}\n`);
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) throw new Error(String(err.message));
  const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').at(-1)!;
  const dump = files.get('dip.dump')!.trim().split('\n');
  const body = dump.slice(dump.findIndex((l) => l.startsWith('ITEM: ATOMS')) + 1).map((l) => l.trim().split(/\s+/).map(Number));
  return { pe: row.row.pe as number, rows: body };
};

const H = 1e-5;
const cfg = (pair: string[], k: number, d: number[] | null, coord: 'x' | 'dip', dir: number, sign: number): Atom[] => {
  const atoms = ATOMS.map((a) => ({ ...a, x: [...a.x], mu: [...a.mu] }));
  if (coord === 'x') atoms[k].x[dir] += sign * H;
  else {
    // rotate mu of atom k about unit axis e = dir (0,1,2) by sign * H (exact rotation)
    const e = [0, 0, 0]; e[dir] = 1;
    const p = atoms[k].mu;
    const c = Math.cos(sign * H), s = Math.sin(sign * H);
    const ex = e[0], ey = e[1], ez = e[2];
    const cross = [ey * p[2] - ez * p[1], ez * p[0] - ex * p[2], ex * p[1] - ey * p[0]];
    const dotep = ex * p[0] + ey * p[1] + ez * p[2];
    atoms[k].mu = p.map((v, i) => v * c + cross[i] * s + e[i] * dotep * (1 - c));
  }
  void d; void pair;
  return atoms;
};

const styles: Array<{ label: string; pair: string[] }> = [
  { label: 'lj/cut/dipole/cut', pair: ['pair_style lj/cut/dipole/cut 3.0 2.5', 'pair_coeff * * 0.6 1.0', 'pair_coeff 1 2 0.4 1.1 2.6 2.9'] },
  { label: 'lj/sf/dipole/sf', pair: ['pair_style lj/sf/dipole/sf 3.0 2.5', 'pair_coeff * * 0.6 1.0', 'pair_coeff 1 2 0.4 1.1 2.6 2.9'] },
];

describe('lj/cut/dipole/cut and lj/sf/dipole/sf: forces and torques vs finite differences', () => {
  for (const st of styles) {
    it(`${st.label}: forces equal -dU/dx`, async () => {
      const ref = await evaluate(st.pair, ATOMS);
      for (let k = 0; k < ATOMS.length; k++) {
        for (let d = 0; d < 3; d++) {
          const up = await evaluate(st.pair, cfg(st.pair, k, null, 'x', d, 1));
          const dn = await evaluate(st.pair, cfg(st.pair, k, null, 'x', d, -1));
          const fd = -(up.pe - dn.pe) / (2 * H);
          const an = ref.rows[k][1 + d];
          expect(Math.abs(an - fd), `k=${k} d=${d} an=${an} fd=${fd}`).toBeLessThan(1e-6 * (1 + Math.abs(fd)));
        }
      }
    });
    it(`${st.label}: torques satisfy e . tau = -dU/dtheta`, async () => {
      const ref = await evaluate(st.pair, ATOMS);
      for (let k = 0; k < ATOMS.length; k++) {
        for (let d = 0; d < 3; d++) {
          const up = await evaluate(st.pair, cfg(st.pair, k, null, 'dip', d, 1));
          const dn = await evaluate(st.pair, cfg(st.pair, k, null, 'dip', d, -1));
          const fd = -(up.pe - dn.pe) / (2 * H);
          const an = ref.rows[k][4 + d];
          expect(Math.abs(an - fd), `k=${k} d=${d} an=${an} fd=${fd}`).toBeLessThan(1e-6 * (1 + Math.abs(fd)));
        }
      }
    });
    it(`${st.label}: total force is zero (Newton's third law)`, async () => {
      const ref = await evaluate(st.pair, ATOMS);
      for (let d = 1; d <= 3; d++) {
        const s = ref.rows.reduce((acc, r) => acc + r[d], 0);
        expect(Math.abs(s)).toBeLessThan(1e-10);
      }
    });
  }
});

describe('pair_coeff scale (measured with native LAMMPS, black box)', () => {
  // Two atoms (charges 0.7 and -0.4, dipoles, separation sqrt(1.1)), lj/sf/dipole/sf 3.0, pair_coeff 1 2 ... scale 0.5.
  // Measured with native LAMMPS (black box), thermo norm no: E_vdwl -0.436484799227275, E_coul 0.117134670127853
  // (the unscaled Coulomb energy is 0.234269340255705: scale multiplies only the q and p terms).
  const pair = ['pair_style lj/sf/dipole/sf 3.0', 'pair_coeff * * 0.6 1.0', 'pair_coeff 1 2 0.6 1.0 scale 0.5'];
  const two: Atom[] = [
    { x: [1.0, 1.0, 1.0], q: 0.7, mu: [0.5, -0.3, 0.8], type: 1 },
    { x: [2.0, 1.3, 1.1], q: -0.4, mu: [-0.6, 0.4, 0.2], type: 2 },
  ];
  it('lj/sf/dipole/sf: scale multiplies the Coulomb part only', async () => {
    const r = await evaluate(pair, two);
    expect(Math.abs(r.pe - (-0.436484799227275 + 0.117134670127853))).toBeLessThan(1e-9);
  });
  it('lj/cut/dipole/cut rejects the scale keyword (native: "Expected floating point parameter instead of scale")', async () => {
    await expect(evaluate(['pair_style lj/cut/dipole/cut 3.0', 'pair_coeff 1 2 0.6 1.0 scale 0.5'], two)).rejects.toThrow(/scale/);
  });
});
