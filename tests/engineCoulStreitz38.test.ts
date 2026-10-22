import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { parseStreitzFile, slaterShort } from '../src/engine/force/pair/coul_streitz';
import { erfcExact } from '../src/engine/force/erfc';

/*
 * pair_style coul/streitz: the Slater integrals checked against direct numerical quadrature
 * (a radial and angular grid, independent of the closed form), the equal-zeta limit, the
 * large-distance 1/r limit, forces against finite differences of the energy, and the error
 * cases. Bit-level agreement with native LAMMPS is covered by the w38streitz_* oracle cases
 * (engineOracle.test.ts).
 */

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');

/** Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on the Legendre polynomial). */
const gaussLegendre = (n: number): { x: number[]; w: number[] } => {
  const x: number[] = [], w: number[] = [];
  for (let i = 0; i < n; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let pp = 0;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      pp = (n * (z * p1 - p2)) / (z * z - 1);
      const z1 = z;
      z = z1 - p1 / pp;
      if (Math.abs(z - z1) < 1e-15) break;
    }
    x.push(z);
    w.push(2 / ((1 - z * z) * pp * pp));
  }
  return { x, w };
};

/**
 * [f_a|f_b](R) by quadrature. f_b is spherical about the origin, so the 3D integral is a radial
 * integral (4 pi r^2 f_b(r) dr) of the angular average of V_a(|x - R|) over mu = cos(theta):
 * V_a(s) = 1/s - (1/s + za) exp(-2 za s) is the potential of f_a, f_b = zb^3/pi exp(-2 zb r).
 * The radial range is split at r = R, where the average has a kink.
 */
const slaterQuad = (R: number, za: number, zb: number, n = 240, m = 240): number => {
  const g = gaussLegendre(n), h = gaussLegendre(m);
  const V = (s: number) => 1 / s - (1 / s + za) * Math.exp(-2 * za * s);
  const avg = (r: number): number => {
    let acc = 0;
    for (let k = 0; k < m; k++) {
      const s = Math.sqrt(Math.max(r * r + R * R - 2 * r * R * h.x[k], 1e-300));
      acc += h.w[k] * V(s);
    }
    return 0.5 * acc;
  };
  const piece = (a: number, b: number): number => {
    let acc = 0;
    for (let k = 0; k < n; k++) {
      const r = 0.5 * (b - a) * g.x[k] + 0.5 * (a + b);
      acc += 0.5 * (b - a) * g.w[k] * 4 * Math.PI * r * r * ((zb ** 3) / Math.PI) * Math.exp(-2 * zb * r) * avg(r);
    }
    return acc;
  };
  const L = 40 / zb; // f_b is below 1e-30 relative beyond this radius
  return piece(0, R) + piece(R, L);
};

/** Equal-zeta closed form of [f|f](R) (Slater 1s self-type integral). */
const fEqual = (R: number, z: number): number =>
  1 / R - (1 / R + (11 * z) / 8 + (3 * z * z * R) / 4 + (z ** 3 * R * R) / 6) * Math.exp(-2 * z * R);

const run = async (text: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [name, body] of Object.entries(files)) session.addFile(name, body);
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  const errEvent = events.find((e) => e.kind === 'error');
  return { thermo, logs, error: error ?? (errEvent && 'message' in errEvent ? errEvent.message : null) };
};

const streitzFile = readFileSync(join(POT, 'AlO.streitz'), 'utf8');

describe('Slater integral [f_a|f_b](R) = 1/R + S(R)', () => {
  const cases: [number, number, number][] = [
    [2.0, 0.968438, 2.143957],
    [0.8, 2.143957, 0.968438],
    [5.0, 1.3, 0.7],
    [1.5, 1.0, 1.5],
    [12.0, 0.968438, 2.143957],
  ];
  for (const [R, za, zb] of cases) {
    it(`matches direct quadrature at R=${R}, za=${za}, zb=${zb} to 1e-8 relative`, () => {
      const exact = slaterQuad(R, za, zb);
      const closed = 1 / R + slaterShort(R, za, zb).s;
      expect(Math.abs(closed - exact) / exact).toBeLessThan(1e-8);
    });
  }

  it('matches the equal-zeta form, and quadrature for equal zeta', () => {
    for (const [R, z] of [[0.9, 1.1], [2.5, 2.1436], [7.0, 0.968438]] as [number, number][]) {
      const eq = fEqual(R, z);
      expect(Math.abs(1 / R + slaterShort(R, z, z).s - eq) / eq).toBeLessThan(1e-13);
      expect(Math.abs(slaterQuad(R, z, z) - eq) / eq).toBeLessThan(1e-8);
    }
  });

  it('approaches the equal-zeta form as zeta_b -> zeta_a, and is symmetric', () => {
    const R = 2.0, z = 1.3;
    const eq = 1 / R + slaterShort(R, z, z).s;
    for (const dz of [1e-3, 1e-5, 1e-7]) {
      const near = 1 / R + slaterShort(R, z, z + dz).s;
      expect(Math.abs(near - eq) / eq).toBeLessThan(1e-3 * dz * 100 + 1e-12);
    }
    expect(slaterShort(R, 0.9, 1.6).s).toBeCloseTo(slaterShort(R, 1.6, 0.9).s, 14);
  });

  it('is accurate on both sides of the series / closed-form switch', () => {
    // |za - zb| R = 0.2 at zb = za + 0.4 / R: the points below and above use the series and the closed form
    const R = 1.2, za = 1.0;
    for (const f of [0.999, 1.001]) {
      const zb = za + (0.4 / R) * f;
      const exact = slaterQuad(R, za, zb);
      const val = 1 / R + slaterShort(R, za, zb).s;
      expect(Math.abs(val - exact) / exact).toBeLessThan(1e-8);
    }
  });

  it('tends to the bare 1/R at large distance', () => {
    for (const [za, zb] of [[0.968438, 2.143957], [2.0, 2.0]] as [number, number][]) {
      const R = 60;
      const F = 1 / R + slaterShort(R, za, zb).s;
      expect(Math.abs(F * R - 1)).toBeLessThan(1e-12);
    }
  });

  it('gives dS/dR matching a finite difference of S', () => {
    const cases2: [number, number, number][] = [[2.0, 0.968438, 2.143957], [1.0, 1.4, 1.4], [3.0, 2.0, 2.0005], [0.5, 0.9, 0.9]];
    for (const [R, za, zb] of cases2) {
      const h = 1e-5 * R;
      const num = (slaterShort(R + h, za, zb).s - slaterShort(R - h, za, zb).s) / (2 * h);
      const an = slaterShort(R, za, zb).ds;
      expect(Math.abs(an - num) / Math.abs(an)).toBeLessThan(1e-6);
    }
  });
});

describe('coul/streitz forces', () => {
  /** Two atoms along x: returns the force on atom 2 (printed) and the total pe at separation r. */
  // fx[2] needs the atom map: native refuses an indexed per-atom vector without one (Variable::peratom2global)
  const pairText = (ta: number, tb: number, qa: number, qb: number, r: number, pair: string[], extra: string[] = []) => `
units metal
atom_style charge
atom_modify map yes
boundary p p p
region box block 0 40 0 40 0 40
create_box 2 box
mass * 26.98
create_atoms ${ta} single 10 10 10
create_atoms ${tb} single ${10 + r} 10 10
set atom 1 charge ${qa}
set atom 2 charge ${qb}
${pair.join('\n')}
${extra.join('\n')}
pair_coeff * * AlO.streitz Al O
thermo_style custom step pe evdwl ecoul elong
thermo_modify format float %.17g
run 0
print "F $(fx[2]:%.17g)"
`;
  const energyAt = async (ta: number, tb: number, qa: number, qb: number, r: number, pair: string[], extra: string[] = []) => {
    const out = await run(pairText(ta, tb, qa, qb, r, pair, extra), { 'AlO.streitz': streitzFile });
    expect(out.error).toBeNull();
    return { pe: out.thermo[0].pe as number, f2: Number(/F (\S+)/.exec(out.logs.join('\n'))![1]) };
  };

  it('Slater-only pair (q_i = 0): force is -dE/dr to 1e-6 relative for Al-O and Al-Al', async () => {
    for (const [ta, tb] of [[1, 2], [1, 1]] as [number, number][]) {
      for (const r of [1.6, 3.1, 6.0]) {
        const h = 1e-6;
        const p = ['pair_style coul/streitz 12.0 wolf 0.31'];
        const c = await energyAt(ta, tb, 0, -0.8, r, p);
        const plus = await energyAt(ta, tb, 0, -0.8, r + h, p);
        const minus = await energyAt(ta, tb, 0, -0.8, r - h, p);
        const dEdr = (plus.pe - minus.pe) / (2 * h);
        // the 1/r part carries no charge product here, so the whole force is the shifted-force part
        expect(Math.abs(c.f2 + dEdr)).toBeLessThan(1e-6 * Math.max(1e-3, Math.abs(dEdr)));
      }
    }
  });

  it('ewald: total force on atom 2 equals -d(pe)/dr (pair plus reciprocal part)', async () => {
    const p = ['pair_style coul/streitz 12.0 ewald'];
    const k = ['kspace_style ewald 1e-6'];
    const r = 2.4, h = 1e-5;
    const c = await energyAt(1, 2, 1.2, -1.2, r, p, k);
    const plus = await energyAt(1, 2, 1.2, -1.2, r + h, p, k);
    const minus = await energyAt(1, 2, 1.2, -1.2, r - h, p, k);
    const dEdr = (plus.pe - minus.pe) / (2 * h);
    expect(Math.abs(c.f2 + dEdr)).toBeLessThan(1e-6 * Math.max(1, Math.abs(dEdr)));
  });

  it('wolf: the 1/r part carries the coul/wolf force shift (measured with native LAMMPS, black box)', async () => {
    // Measured with native LAMMPS (black box): the force on the atom at 11 A exceeds -dE/dr by
    // the coul/wolf f_shift term C q_i q_j f_shift; the energy itself has no force shift.
    const p = ['pair_style coul/streitz 12.0 wolf 0.31'];
    const r = 11.0, h = 1e-4;
    const c = await energyAt(2, 2, 1.0, -1.0, r, p);
    const plus = await energyAt(2, 2, 1.0, -1.0, r + h, p);
    const minus = await energyAt(2, 2, 1.0, -1.0, r - h, p);
    const dEdr = (plus.pe - minus.pe) / (2 * h);
    const C = 14.399645;
    const alpha = 0.31, rc = 12.0;
    const arc = alpha * rc;
    const eShift = erfcExact(arc) / rc;
    const fShift = -(eShift + (2 / Math.sqrt(Math.PI)) * alpha * Math.exp(-arc * arc)) / rc;
    const shiftForce = C * (1.0 * -1.0) * fShift;
    expect(Math.abs(c.f2 + dEdr - shiftForce)).toBeLessThan(1e-3 * Math.abs(shiftForce));
  });
});

describe('coul/streitz errors', () => {
  const head = `units metal
atom_style charge
boundary p p p
region box block 0 40 0 40 0 40
create_box 2 box
mass * 26.98
create_atoms 1 single 10 10 10
create_atoms 2 single 12 10 10
set atom 1 charge 1.0
set atom 2 charge -1.0
`;
  it('wolf without a damping parameter names the parameter', async () => {
    const out = await run(`${head}pair_style coul/streitz 12.0 wolf\npair_coeff * * AlO.streitz Al O\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(out.error).toMatch(/damping parameter/);
  });

  it('ewald without a kspace style names the kspace requirement', async () => {
    const out = await run(`${head}pair_style coul/streitz 12.0 ewald\npair_coeff * * AlO.streitz Al O\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(out.error).toMatch(/kspace/);
    expect(out.error).toMatch(/coul\/streitz/);
  });

  it('an unknown keyword is an error naming the keyword', async () => {
    const out = await run(`${head}pair_style coul/streitz 12.0 foo 0.3\npair_coeff * * AlO.streitz Al O\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(out.error).toMatch(/unknown keyword 'foo'/);
  });

  it('an element missing from the potential file is an error naming it', async () => {
    const out = await run(`${head}pair_style coul/streitz 12.0 wolf 0.31\npair_coeff * * AlO.streitz Al Fe\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(out.error).toMatch(/element 'Fe' is not in coul\/streitz potential file/);
  });

  it('fix qeq/slater with qfile coul/streitz runs on this pair style (wave 40; parity in w40qslat_*)', async () => {
    const out = await run(`${head}pair_style coul/streitz 12.0 wolf 0.31\npair_coeff * * AlO.streitz Al O\nfix 1 all qeq/slater 1 12.0 1.0e-6 100 coul/streitz\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(out.error).toBeFalsy();
  });

  it('the potential file parser rejects short lines and reads the five columns', () => {
    expect(() => parseStreitzFile('Al 0 10 0 0.9\n', 'bad.streitz')).toThrow(/expected 6 words/);
    const map = parseStreitzFile(streitzFile, 'AlO.streitz');
    expect(map.get('O')).toEqual({ chi: 5.484763, J: 14.035715, gamma: 0, zeta: 2.143957, Z: 0 });
  });
});
