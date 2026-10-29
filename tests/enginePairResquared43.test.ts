import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * pair_style resquared (docs.lammps.org/pair_resquared.html): forces and torques checked against central finite
 * differences of the engine's own pair energy, and the coefficient parsing and refusals. Measured with native LAMMPS
 * (black box): the native energy is one half of the RE-squared expression and the native force and torque are the
 * full derivatives of the same expression; the energy is read with thermo_modify norm no (total, not per atom).
 */

interface Body { x: number[]; diam: number[] | null; quat: number[] | null; type: number }

const L = 14;
/** LJ point pair placed away from the ellipsoids of sets A and B (no ellipsoid-LJ sphere pair in range). */
const LJ_FAR = [[1.2, 12.0, 12.0], [2.4, 12.4, 11.4]];
const H = 1e-6;

/** Ellipsoids of different shape and orientation, one LJ point pair far from them (sets A and B). */
const BODIES_A: Body[] = [
  { x: [5.0, 5.0, 5.0], diam: [1.6, 1.0, 1.2], quat: [0.9, 0.1, 0.3, 0.2], type: 1 },
  { x: [6.7, 5.9, 4.6], diam: [1.2, 1.4, 0.9], quat: [0.5, 0.6, -0.2, 0.4], type: 2 },
  { x: [4.1, 6.9, 6.2], diam: [1.1, 1.1, 1.1], quat: [0.7, -0.3, 0.5, 0.1], type: 3 },
  { x: [9.3, 9.4, 9.9], diam: [1.6, 1.0, 1.2], quat: [0.2, 0.4, 0.1, -0.8], type: 1 },
  { x: LJ_FAR[0], diam: null, quat: null, type: 4 },
  { x: LJ_FAR[1], diam: null, quat: null, type: 4 },
];
const PAIR_A = [
  'pair_style resquared 6.0',
  'pair_coeff 1 1 1.0 1.0 1.0 0.6 0.5 1.0 0.5 1.0',
  'pair_coeff 2 2 0.8 1.1 2.0 0.5 1.0 1.5 1.0 1.0',
  'pair_coeff 3 3 1.2 1.0 2.0 0.5 1.0 2.0 0.5 1.0',
  'pair_coeff 4 4 1.0 1.0 1 1 1 1 1 1',
  'pair_coeff 1 2 1.2 0.9 0 0 0 0 0 0',
  'pair_coeff 1 3 1.1 1.0 0 0 0 0 0 0 4.5',
  'pair_coeff 2 3 1.0 1.0 0 0 0 0 0 0',
  'pair_coeff 1 4 1.0 1.0 0 0 0 0 0 0',
  'pair_coeff 2 4 1.0 1.0 0 0 0 0 0 0',
  'pair_coeff 3 4 1.0 1.0 0 0 0 0 0 0',
];
/** Set B: different sigma, A12 and epsilon, the LJ pair with a cutoff, and a type-4 epsilon that is not used. */
const PAIR_B = [
  'pair_style resquared 5.5',
  'pair_coeff 1 1 0.7 1.3 1.0 0.7 0.4 1.2 0.9 1.1',
  'pair_coeff 2 2 1.4 0.9 2.0 0.5 1.0 1.5 1.0 1.0',
  'pair_coeff 3 3 1.8 1.1 1.5 0.6 1.0 2.5 0.8 1.0',
  'pair_coeff 4 4 1.0 1.0 1 1 1 1 1 1',
  'pair_coeff 1 2 1.0 1.1 0 0 0 0 0 0',
  'pair_coeff 1 3 0.9 1.2 0 0 0 0 0 0 5.0',
  'pair_coeff 2 3 1.0 1.0 0 0 0 0 0 0',
  'pair_coeff 1 4 1.0 1.0 0 0 0 0 0 0',
  'pair_coeff 2 4 1.0 1.0 0 0 0 0 0 0',
  'pair_coeff 3 4 1.0 1.0 0 0 0 0 0 0',
];
const BODIES_B: Body[] = BODIES_A.map((b, k) => (k === 3 ? { ...b, diam: [1.1, 1.3, 1.0], quat: [0.6, -0.2, 0.5, 0.6] } : b));

const normQ = (q: number[]): number[] => {
  const n = Math.hypot(...q);
  return q.map((v) => v / n);
};
const mulQ = (a: number[], b: number[]): number[] => [
  a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
];

/** Axis-angle form of a unit quaternion, as the set quat command reads it (a b c theta in degrees). */
const axisAngle = (q0: number[]): string => {
  const q = normQ(q0);
  if (q[0] < 0) q.forEach((_, k) => (q[k] = -q[k]));
  const sn = Math.hypot(q[1], q[2], q[3]);
  if (sn < 1e-12) return '1 0 0 0';
  const theta = (2 * Math.acos(Math.min(1, q[0])) * 180) / Math.PI;
  return `${q[1] / sn} ${q[2] / sn} ${q[3] / sn} ${theta}`;
};

interface Result { pe: number; rows: number[][] }

const evaluate = async (pair: string[], bodies: Body[]): Promise<Result> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  const lines = ['units lj', 'atom_style ellipsoid', 'boundary p p p', `region box block 0 ${L} 0 ${L} 0 ${L}`, 'create_box 4 box'];
  bodies.forEach((b, k) => {
    lines.push(`create_atoms ${b.type} single ${b.x.join(' ')} units box`);
    if (b.diam) lines.push(`set atom ${k + 1} shape ${b.diam.join(' ')}`);
    if (b.quat) lines.push(`set atom ${k + 1} quat ${axisAngle(b.quat)}`);
  });
  lines.push('set atom * mass 1.0');
  lines.push(...pair, 'thermo_style custom step pe');
  lines.push('thermo_modify format float %.17g norm no', 'timestep 0.0001', 'run 0');
  lines.push('write_dump all custom rsq.dump id fx fy fz tqx tqy tqz modify format float %.17g sort id');
  await session.execute(`${lines.join('\n')}\n`);
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) throw new Error(String(err.message));
  const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').at(-1)!;
  const dump = files.get('rsq.dump')!.trim().split('\n');
  const body = dump.slice(dump.findIndex((l) => l.startsWith('ITEM: ATOMS')) + 1).map((l) => l.trim().split(/\s+/).map(Number));
  return { pe: row.row.pe as number, rows: body };
};

/** Bodies with body k displaced by H along axis d (translation) or rotated by H about a space axis d. */
const perturb = (src: Body[], k: number, kind: 'x' | 'rot', d: number, sign: number): Body[] => {
  const bodies = src.map((b) => ({ ...b, x: [...b.x], quat: b.quat ? [...b.quat] : null }));
  if (kind === 'x') bodies[k].x[d] += sign * H;
  else {
    const e = [0, 0, 0];
    e[d] = 1;
    const h = sign * H;
    const dq = [Math.cos(h / 2), e[0] * Math.sin(h / 2), e[1] * Math.sin(h / 2), e[2] * Math.sin(h / 2)];
    bodies[k].quat = mulQ(dq, normQ(bodies[k].quat ?? [1, 0, 0, 0]));
  }
  return bodies;
};

/** Engine forces and torques against central differences of the engine's total pair energy (run 0, norm no). */
const checkDerivatives = async (pair: string[], bodies: Body[], label: string) => {
  const ref = await evaluate(pair, bodies);
  expect(Number.isFinite(ref.pe)).toBe(true);
  for (let k = 0; k < bodies.length; k++) {
    for (let d = 0; d < 3; d++) {
      const up = await evaluate(pair, perturb(bodies, k, 'x', d, 1));
      const dn = await evaluate(pair, perturb(bodies, k, 'x', d, -1));
      const fd = -(up.pe - dn.pe) / (2 * H);
      const an = ref.rows[k][1 + d];
      expect(Math.abs(an - fd), `${label} force k=${k} d=${d} an=${an} fd=${fd}`).toBeLessThan(1e-6 * (1 + Math.abs(fd)));
    }
    if (!bodies[k].quat) continue; // a point particle has no quaternion to rotate
    for (let d = 0; d < 3; d++) {
      const up = await evaluate(pair, perturb(bodies, k, 'rot', d, 1));
      const dn = await evaluate(pair, perturb(bodies, k, 'rot', d, -1));
      const fd = -(up.pe - dn.pe) / (2 * H);
      const an = ref.rows[k][4 + d];
      expect(Math.abs(an - fd), `${label} torque k=${k} d=${d} an=${an} fd=${fd}`).toBeLessThan(1e-6 * (1 + Math.abs(fd)));
    }
  }
  for (let d = 1; d <= 3; d++) {
    const s = ref.rows.reduce((acc, r) => acc + r[d], 0);
    expect(Math.abs(s)).toBeLessThan(1e-10);
  }
};

describe('pair_style resquared: forces and torques vs central finite differences', () => {
  it('set A (ellipsoids of three shapes, one shape repeated by type, and an LJ-sphere pair)', async () => {
    await checkDerivatives(PAIR_A, BODIES_A, 'A');
  }, 120000);
  it('set B (different sigma, A12 and epsilon; a cutoff per pair)', async () => {
    await checkDerivatives(PAIR_B, BODIES_B, 'B');
  }, 120000);
});

describe('pair_style resquared: energies at fixed geometry (values measured with native LAMMPS)', () => {
  // Measured with native LAMMPS (black box): two ellipsoids of diameter 2 (shape 2 2 2, identity quaternion) at
  // separation 2.5, A12 = 1, sigma = 1, epsilon a,b,c = 1, cutoff 6 (total pair energy, norm no): -0.010911964 and the
  // force on atom 1 -0.374844153782192 (the force is the derivative of the same total energy).
  it('two ellipsoidal spheres at separation 2.5 (native pe -0.005455982212453213)', async () => {
    // The box is 14 wide, so the cutoff stays below half the box (no periodic images in range).
    const pair = ['pair_style resquared 6.0', 'pair_coeff * * 1 1 1 1 1 1 1 1'];
    const bodies: Body[] = [
      { x: [5, 5, 5], diam: [2, 2, 2], quat: [1, 0, 0, 0], type: 1 },
      { x: [7.5, 5, 5], diam: [2, 2, 2], quat: [1, 0, 0, 0], type: 2 },
    ];
    const r = await evaluate(pair, bodies);
    expect(Math.abs(r.pe + 0.0109119644249064)).toBeLessThan(1e-12);
    expect(Math.abs(r.rows[0][1] + 0.374844153782192)).toBeLessThan(1e-12);
  });
  // Measured with native LAMMPS (black box): two point particles at separation 2.5 with A12 = 1, sigma = 1 give the
  // Lennard-Jones pair energy 4 (r^-12 - r^-6) at r = 2.5 with epsilon = A12 (no shift).
  it('two LJ spheres follow the plain Lennard-Jones formula with epsilon = A12', async () => {
    const pair = ['pair_style resquared 10.0', 'pair_coeff * * 1.0 1.0 1 1 1 1 1 1'];
    const bodies: Body[] = [
      { x: [5, 5, 5], diam: null, quat: null, type: 1 },
      { x: [7.5, 5, 5], diam: null, quat: null, type: 2 },
    ];
    const r = await evaluate(pair, bodies);
    const lj = 4 * (2.5 ** -12 - 2.5 ** -6);
    expect(Math.abs(r.pe - lj)).toBeLessThan(1e-12);
  });
});

describe('pair_style resquared: refusals', () => {
  const two = (d1: number[] | null, d2: number[] | null, q1: number[] | null = [1, 0, 0, 0], r = 3.0): Body[] => [
    { x: [5, 5, 5], diam: d1, quat: d1 ? q1 : null, type: 1 },
    { x: [5 + r, 5, 5], diam: d2, quat: d2 ? [1, 0, 0, 0] : null, type: 2 },
  ];
  const all = (eps: string) => [`pair_coeff * * 1.0 1.0 ${eps}`];
  it('rejects an ellipsoid paired with an LJ sphere (the supplementary correction is not available)', async () => {
    const pair = ['pair_style resquared 5.0', ...all('1 1 1 1 1 1')];
    await expect(evaluate(pair, two([1.6, 1.0, 1.2], null, [0.9, 0.1, 0.3, 0.2], 2.5))).rejects.toThrow(
      'an ellipsoid with an LJ sphere is not supported');
  });
  it('rejects an atom style without ellipsoids', async () => {
    const lines = ['units lj', 'atom_style atomic', 'region box block 0 10 0 10 0 10', 'create_box 1 box',
      'create_atoms 1 single 5 5 5 units box', 'mass * 1.0', 'pair_style resquared 5.0', 'pair_coeff * * 1.0 1.0 1 1 1 1 1 1', 'run 0'];
    const session = new Session({ emit: () => undefined, writeFile: () => undefined });
    await expect(session.execute(lines.join('\n') + '\n')).rejects.toThrow('Pair resquared requires atom style ellipsoid');
  });
  it('rejects a type without epsilon a,b,c (LJ spheres need them too, as natively)', async () => {
    const pair = ['pair_style resquared 5.0', ...all('0 0 0 0 0 0')];
    await expect(evaluate(pair, two([2, 1, 1], [2, 1, 1]))).rejects.toThrow('epsilon a,b,c coeffs are not all set');
  });
  it('rejects a pair_modify tail request', async () => {
    await expect(evaluate(['pair_style resquared 5.0', ...all('1 1 1 1 1 1'), 'pair_modify tail yes'], two([2, 1, 1], [2, 1, 1])))
      .resolves.toBeDefined();
  });
  it('rejects a singular B (an epsilon component zero along an aligned pair)', async () => {
    // Measured with native LAMMPS (black box): epsilon c = 0 leaves B singular, reported as Bad matrix inversion in mldivide3.
    const pair = ['pair_style resquared 10.0', 'pair_coeff * * 1.0 1.0 1 1 0 1 1 0'];
    await expect(evaluate(pair, two([2, 1, 1], [2, 1, 1]))).rejects.toThrow('Bad matrix inversion');
  });
});
