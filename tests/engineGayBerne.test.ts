import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * pair_style gayberne (docs.lammps.org/pair_gayberne.html): forces and torques checked against central finite
 * differences of the pair energy. Forces: F = -dU/dx. Torques: tau = -dU/dphi, where phi is a rotation of the
 * body about a space-frame unit axis e; the rotation is applied to the quaternion as dq * q (dq = (cos h/2,
 * e sin h/2)), so the body's space matrix turns by exp(h [e]x). Energies come from thermo pe (run 0).
 */

interface Body { x: number[]; diam: number[] | null; quat: number[] | null; type: number }

const L = 12;
/**
 * Set A (mu = 1): two ellipsoids of different shape, a non-LJ sphere with anisotropic epsilon, and a point particle
 * whose epsilon a,b,c are equal (an LJ sphere). The point has no quaternion, so its rotation is not perturbed.
 */
const BODIES_A: Body[] = [
  { x: [5.0, 5.0, 5.0], diam: [1.6, 1.0, 1.2], quat: [0.9, 0.1, 0.3, 0.2], type: 1 },
  { x: [6.7, 5.9, 4.6], diam: [1.2, 1.4, 0.9], quat: [0.5, 0.6, -0.2, 0.4], type: 2 },
  { x: [4.1, 6.9, 6.2], diam: [1.1, 1.1, 1.1], quat: [0.7, -0.3, 0.5, 0.1], type: 3 },
  { x: [6.3, 7.4, 6.9], diam: null, quat: null, type: 4 },
];
const PAIR_A = [
  'pair_style gayberne 1.0 1.0 1.0 4.5',
  'pair_coeff 1 1 1.0 1.0 1.0 0.6 0.5 1.0 0.5 1.0',
  'pair_coeff 2 2 0.8 1.1 2.0 0.5 1.0 1.5 1.0 1.0',
  'pair_coeff 3 3 1.2 1.0 2.0 0.5 1.0 2.0 0.5 1.0',
  'pair_coeff 4 4 1.0 1.0 1.5 1.5 1.5 1.5 1.5 1.5 4.0',
  'pair_coeff 1 2 1.2 0.9 0 0 0 0 0 0',
];
/**
 * Set B (mu, upsilon, gamma not 1, no LJ spheres): ellipsoids and a sphere with anisotropic epsilon. Body 0 and
 * body 3 share a type, hence a shape (per-type shape rule), and differ in orientation.
 */
const BODIES_B: Body[] = [
  { x: [5.0, 5.0, 5.0], diam: [1.6, 1.0, 1.2], quat: [0.9, 0.1, 0.3, 0.2], type: 1 },
  { x: [6.7, 5.9, 4.6], diam: [1.2, 1.4, 0.9], quat: [0.5, 0.6, -0.2, 0.4], type: 2 },
  { x: [4.1, 6.9, 6.2], diam: [1.1, 1.1, 1.1], quat: [0.7, -0.3, 0.5, 0.1], type: 3 },
  { x: [6.3, 7.4, 6.9], diam: [1.6, 1.0, 1.2], quat: [0.2, 0.4, 0.1, -0.8], type: 1 },
];
const PAIR_B = [
  'pair_style gayberne 1.3 0.8 1.5 4.5',
  'pair_coeff 1 1 1.0 1.0 1.0 0.6 0.5 1.0 0.5 1.0',
  'pair_coeff 2 2 0.8 1.1 2.0 0.5 1.0 1.5 1.0 1.0',
  'pair_coeff 3 3 1.2 1.0 2.0 0.5 1.0 2.0 0.5 1.0',
  'pair_coeff 1 2 1.2 0.9 0 0 0 0 0 0',
  'pair_coeff 1 3 1.1 1.0 0 0 0 0 0 0 4.0',
  'pair_coeff 4 4 1.0 1.0 1 1 1 1 1 1',
];

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

const evaluate = async (pair: string[], bodies: Body[], style = 'ellipsoid'): Promise<Result> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  const lines = [
    `units lj`, `atom_style ${style}`, 'boundary p p p', `region box block 0 ${L} 0 ${L} 0 ${L}`, 'create_box 4 box',
  ];
  bodies.forEach((b, k) => {
    lines.push(`create_atoms ${b.type} single ${b.x.join(' ')} units box`);
    if (b.diam) lines.push(`set atom ${k + 1} shape ${b.diam.join(' ')}`);
    if (b.quat) lines.push(`set atom ${k + 1} quat ${axisAngle(b.quat)}`);
  });
  lines.push(style === 'ellipsoid' ? 'set atom * mass 1.0' : 'mass * 1.0');
  lines.push(...pair, 'thermo_style custom step pe');
  lines.push('thermo_modify format float %.17g norm no', 'timestep 0.0001', 'run 0');
  lines.push('write_dump all custom gb.dump id fx fy fz tqx tqy tqz modify format float %.17g sort id');
  await session.execute(`${lines.join('\n')}\n`);
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) throw new Error(String(err.message));
  const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').at(-1)!;
  const dump = files.get('gb.dump')!.trim().split('\n');
  const body = dump.slice(dump.findIndex((l) => l.startsWith('ITEM: ATOMS')) + 1).map((l) => l.trim().split(/\s+/).map(Number));
  return { pe: row.row.pe as number, rows: body };
};

const H = 1e-6;

/** Bodies with body k displaced by H along axis d (translation) or rotated by H about space axis d. */
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

/** Central differences of the energy of every body, for forces and for rotations (torque = -dU/dtheta). */
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
    // A point particle cannot be given a quaternion (set quat refuses it), so its rotation is not perturbed.
    if (!bodies[k].quat) continue;
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

describe('pair_style gayberne: forces and torques vs central finite differences', () => {
  it('set A (mu = 1, ellipsoids, LJ sphere, non-LJ sphere, point particle)', async () => {
    await checkDerivatives(PAIR_A, BODIES_A, 'A');
  });
  it('set B (mu, upsilon, gamma not 1; ellipsoids and a sphere with anisotropic epsilon)', async () => {
    await checkDerivatives(PAIR_B, BODIES_B, 'B');
  });
});

describe('pair_style gayberne: refusals (measured native failures or unreproduced native behaviour)', () => {
  const withBodies = (bodies: Body[], pair: string[]) => evaluate(pair, bodies);
  it('rejects an atom style without ellipsoids', async () => {
    const plain = BODIES_B.map((b) => ({ ...b, diam: null, quat: null }));
    await expect(evaluate(PAIR_B, plain, 'atomic')).rejects.toThrow('Pair gayberne requires atom style ellipsoid');
  });
  it('rejects two atoms of one type with different shapes', async () => {
    const bad = BODIES_B.map((b, k) => (k === 3 ? { ...b, diam: [1.0, 1.0, 1.0] } : b));
    await expect(withBodies(bad, PAIR_B)).rejects.toThrow('same type have same shape');
  });
  it('rejects an LJ sphere paired with an ellipsoid when mu is not 1', async () => {
    const pair = ['pair_style gayberne 1.0 1.0 1.5 4.5', ...PAIR_A.slice(1)];
    await expect(withBodies(BODIES_A, pair)).rejects.toThrow('needs mu = 1');
  });
  it('rejects a point particle with anisotropic epsilon', async () => {
    const pair = PAIR_A.map((l) => (l.startsWith('pair_coeff 4 4') ? 'pair_coeff 4 4 1.0 1.0 1.5 0.5 1.5 1.5 0.5 1.5 4.0' : l));
    await expect(withBodies(BODIES_A, pair)).rejects.toThrow('point particles need equal epsilon');
  });
  it('rejects a type without epsilon a,b,c', async () => {
    const pair = PAIR_B.filter((l) => !l.startsWith('pair_coeff 3 3'));
    await expect(withBodies(BODIES_B, pair)).rejects.toThrow('epsilon a,b,c coeffs are not all set');
  });
});
