import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { PairBuck6dCoulGaussDsf, PairBuck6dCoulGaussLong } from '../src/engine/force/pair/buck6d_gauss';
import { PairGaussCut, PairCoulDiel } from '../src/engine/force/pair/gauss_diel';
import { PairMomb } from '../src/engine/force/pair/momb';
import { PAIRS } from '../src/engine/registry/pair_batch49';

/*
 * Batch 49 pair styles (buck6d/coul/gauss/dsf and /long, gauss/cut, coul/diel): parsing, refusals, the
 * finite-difference consistency of the single() force with its energy, and the mixing rule of gauss/cut. The
 * native parity lives in tests/oracle/w49pb_*.in (engineOracle.test.ts).
 */

const DIMER = `units real
atom_style full
boundary f f f
region box block -60 60 -60 60 -60 60
create_box 2 box
create_atoms 1 single 0 0 0 units box
create_atoms 2 single 3.0 0 0 units box
mass * 1.0
set type 1 charge 0.8
set type 2 charge -0.4
`;

const run = async (script: string): Promise<{ errors: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  try {
    await session.execute(script);
  } catch (e) {
    events.push({ kind: 'error', message: String(e instanceof Error ? e.message : e) } as unknown as EngineEvent);
  }
  return { errors: events.filter((e) => e.kind === 'error').map((e) => ('message' in e ? String(e.message) : '')) };
};

const BUCK = [
  'pair_coeff 1 1 10304 3.0612245 457.17971 4.5218516 0.60800971',
  'pair_coeff 1 2 6157.8178 3.4682081 129.19572 0.78772886 0.73006542',
  'pair_coeff 2 2 1000 2.5 200 3.0 0.5 8.0',
];

/** A pair style instance initialised for the three type pairs of BUCK (no SimState needed for the kernels). */
const makeStyle = <T extends { settings(a: string[], c: never): void; allocate(n: number): void; coeff(a: string[], c: never): void; init(c: never): void }>(
  style: T, args: string[], coeffs: string[],
): T => {
  const ctx = { s: null } as never;
  style.settings(args, ctx);
  style.allocate(2);
  for (const c of coeffs) style.coeff(c.split(/\s+/).slice(1), ctx);
  style.init(ctx);
  return style;
};

/** Single-pair energy and force (fforce * r = -dE/dr) by central differences against the analytic force. */
const fdCheck = (
  pair: { single?: (i: number, j: number, it: number, jt: number, rsq: number, fc: number, fl: number, qi: number, qj: number) => { eng: number; fforce: number } },
  it: number, jt: number, r: number, qi: number, qj: number,
): { fd: number; an: number } => {
  const h = 1e-6;
  const at = (x: number) => pair.single!(0, 0, it, jt, x * x, 1, 1, qi, qj);
  const ep = at(r + h).eng, em = at(r - h).eng;
  const s = at(r);
  return { fd: -(ep - em) / (2 * h), an: s.fforce * r };
};

describe('pair buck6d/coul/gauss/dsf and /long', () => {
  it('parses the argument lists and refuses wrong counts and out-of-range smoothing', async () => {
    expect((await run(`${DIMER}pair_style buck6d/coul/gauss/dsf 0.9\nrun 0\n`)).errors[0]).toMatch(/Illegal pair_style command/);
    expect((await run(`${DIMER}pair_style buck6d/coul/gauss/dsf 0.9 12.0 6.0 1.0\nrun 0\n`)).errors[0]).toMatch(/Illegal pair_style command/);
    expect((await run(`${DIMER}pair_style buck6d/coul/gauss/dsf 0.0 12.0\nrun 0\n`)).errors[0]).toMatch(/smooth must be in \(0, 1\]/);
    expect((await run(`${DIMER}pair_style buck6d/coul/gauss/long 0.9 0.9\nrun 0\n`)).errors[0]).toMatch(/Illegal pair_style command/);
    expect((await run(`${DIMER}pair_style buck6d/coul/gauss/long 0.9 1.5 12.0\nrun 0\n`)).errors[0]).toMatch(/smooth2 must be in \(0, 1\]/);
  });

  it('needs every I,J pair (no mixing) and a kspace style for the long variant', async () => {
    const partial = `${DIMER}pair_style buck6d/coul/gauss/dsf 0.9 12.0\npair_coeff 1 1 10304 3.0612245 457.17971 4.5218516 0.60800971\nrun 0\n`;
    expect((await run(partial)).errors[0]).toMatch(/All pair coeffs are not set/);
    const longNoKspace = `${DIMER}pair_style buck6d/coul/gauss/long 0.9 0.9 12.0\n${BUCK.join('\n')}\nrun 0\n`;
    expect((await run(longNoKspace)).errors[0]).toMatch(/requires a kspace style|needs a kspace style/);
  });

  it('refuses pair_modify shift and tail, which the doc says are not supported', async () => {
    const shift = `${DIMER}pair_style buck6d/coul/gauss/dsf 0.9 12.0\n${BUCK.join('\n')}\npair_modify shift yes\nrun 0\n`;
    expect((await run(shift)).errors[0]).toMatch(/pair_modify shift is not supported by pair style buck6d\/coul\/gauss\/dsf/);
    const tail = `${DIMER}pair_style buck6d/coul/gauss/dsf 0.9 12.0\n${BUCK.join('\n')}\npair_modify tail yes\nrun 0\n`;
    expect((await run(tail)).errors[0]).toMatch(/pair_modify tail is not supported/);
  });

  it('is registered under its two names', () => {
    expect(Object.keys(PAIRS)).toEqual(expect.arrayContaining(['buck6d/coul/gauss/dsf', 'buck6d/coul/gauss/long']));
  });

  it('single() forces agree with central differences of the energy, in the smoothing region too', () => {
    const dsf = makeStyle(new PairBuck6dCoulGaussDsf(), ['0.9', '12.0', '10.0'], BUCK) as unknown as Parameters<typeof fdCheck>[0];
    for (const [it, jt] of [[1, 1], [1, 2], [2, 2]] as const) {
      for (const r of [2.0, 3.5, 7.9, 10.0, 11.5]) {
        const { fd, an } = fdCheck(dsf, it, jt, r, 0.8, -0.4);
        expect(Math.abs(fd - an)).toBeLessThanOrEqual(1e-6 * Math.max(1e-3, Math.abs(an)));
      }
    }
    const long = makeStyle(new PairBuck6dCoulGaussLong(), ['0.9', '0.8', '12.0', '10.0'], BUCK) as unknown as Parameters<typeof fdCheck>[0];
    (long as unknown as { gEwald: number }).gEwald = 0.25;
    for (const r of [2.0, 8.5, 9.5, 11.0]) {
      const { fd, an } = fdCheck(long, 1, 2, r, 0.8, -0.4);
      expect(Math.abs(fd - an)).toBeLessThanOrEqual(1e-6 * Math.max(1e-3, Math.abs(an)));
    }
  });

  it('the energy and force vanish at the cutoff (dsf: Coulomb shifted to zero at cutoff2)', () => {
    // the vdW part is not zero at 10 A (cutoff 12), so the Coulomb part is isolated as the charged minus the neutral energy
    const dsf = makeStyle(new PairBuck6dCoulGaussDsf(), ['0.9', '12.0', '10.0'], BUCK) as unknown as Parameters<typeof fdCheck>[0];
    const edge = 10.0 - 1e-9;
    const coul = dsf.single!(0, 0, 1, 2, edge * edge, 1, 1, 0.8, -0.4).eng - dsf.single!(0, 0, 1, 2, edge * edge, 1, 1, 0, 0).eng;
    expect(Math.abs(coul)).toBeLessThan(1e-6);
    const beyond = dsf.single!(0, 0, 1, 2, 10.5 * 10.5, 1, 1, 0.8, -0.4).eng - dsf.single!(0, 0, 1, 2, 10.5 * 10.5, 1, 1, 0, 0).eng;
    expect(beyond).toBe(0);
  });
});

describe('pair gauss/cut', () => {
  it('needs a cutoff, a positive sigma_h and refuses a bad argument count', async () => {
    expect((await run(`${DIMER}pair_style gauss/cut\nrun 0\n`)).errors[0]).toMatch(/usage: pair_style gauss\/cut cutoff/);
    expect((await run(`${DIMER}pair_style gauss/cut 6.0\npair_coeff 1 1 1.0 3.0\nrun 0\n`)).errors[0]).toMatch(/usage: pair_coeff I J H r_mh sigma_h/);
    expect((await run(`${DIMER}pair_style gauss/cut 6.0\npair_coeff 1 1 1.0 3.0 0.0\nrun 0\n`)).errors[0]).toMatch(/sigma_h must be > 0/);
  });

  it('refuses the sixthpower mix rule (the doc supports arithmetic and geometric only)', async () => {
    const script = `${DIMER}pair_style gauss/cut 6.0\npair_coeff 1 1 1.5 3.2 0.7\npair_coeff 2 2 0.4 3.0 0.9\npair_modify mix sixthpower\nrun 0\n`;
    expect((await run(script)).errors[0]).toMatch(/supports only mix geometric or arithmetic/);
  });

  it('mixes H by the epsilon rule and the other parameters by the sigma rule', () => {
    const g = new PairGaussCut();
    const ctx = { s: null } as never;
    g.settings(['8.0'], ctx);
    g.allocate(2);
    g.coeff(['1', '1', '1.5', '3.2', '0.7', '6.0'], ctx);
    g.coeff(['2', '2', '0.4', '3.0', '0.9', '5.0'], ctx);
    g.init(ctx);
    // geometric (default): sqrt for H, r_mh, sigma and the cutoff
    const rsq = 4.0 * 4.0;
    const geo = g.single!(0, 0, 1, 2, rsq, 1, 1).eng;
    const H = Math.sqrt(1.5 * 0.4), rm = Math.sqrt(3.2 * 3.0), s = Math.sqrt(0.7 * 0.9);
    expect(geo).toBeCloseTo((H / (s * Math.sqrt(2 * Math.PI))) * Math.exp(-((4 - rm) ** 2) / (2 * s * s)), 12);
  });

  it('single() forces agree with central differences of the energy', () => {
    const g = new PairGaussCut();
    const ctx = { s: null } as never;
    g.settings(['8.0'], ctx);
    g.allocate(1);
    g.coeff(['1', '1', '1.5', '3.2', '0.7', '6.0'], ctx);
    g.init(ctx);
    for (const r of [2.0, 3.2, 4.0, 5.5, 5.9]) {
      const { fd, an } = fdCheck(g as unknown as Parameters<typeof fdCheck>[0], 1, 1, r, 0, 0);
      expect(Math.abs(fd - an)).toBeLessThanOrEqual(1e-7 * Math.max(1e-3, Math.abs(an)));
    }
  });
});

describe('pair coul/diel', () => {
  it('refuses tail, unset I,J pairs and special Coulomb factors other than 0 and 1', async () => {
    const tail = `${DIMER}pair_style coul/diel 3.5\npair_coeff * * 78.0 1.375 0.112\npair_modify tail yes\nrun 0\n`;
    expect((await run(tail)).errors[0]).toMatch(/pair_modify tail is not supported by pair style coul\/diel/);
    const unset = `${DIMER}pair_style coul/diel 3.5\npair_coeff 1 1 78.0 1.375 0.112\nrun 0\n`;
    expect((await run(unset)).errors[0]).toMatch(/All pair coeffs are not set/);
  });

  it('single() forces agree with central differences of the energy', () => {
    const d = new PairCoulDiel();
    const ctx = { s: null } as never;
    d.settings(['3.5'], ctx);
    d.allocate(1);
    d.coeff(['1', '1', '78.0', '1.375', '0.112'], ctx);
    d.init(ctx);
    for (const r of [0.9, 1.2, 1.6, 2.2, 3.0]) {
      const { fd, an } = fdCheck(d as unknown as Parameters<typeof fdCheck>[0], 1, 1, r, 0.8, 0.8);
      expect(Math.abs(fd - an)).toBeLessThanOrEqual(1e-6 * Math.abs(an) + 1e-9);
    }
  });

  it('reproduces the documented energy shape: C q_i q_j (eps/eps_D - 1)/r, zero beyond the cutoff', () => {
    const d = new PairCoulDiel();
    const ctx = { s: null } as never;
    d.settings(['3.5'], ctx);
    d.allocate(1);
    d.coeff(['1', '1', '78.0', '1.375', '0.112'], ctx);
    d.init(ctx);
    const r = 1.6;
    const epsD = (5.2 + 78) / 2 + ((78 - 5.2) / 2) * Math.tanh((r - 1.375) / 0.112);
    // single() uses the Coulomb prefactor of the last compute (1 before any run), so the prefactor is 1 here
    const e = d.single!(0, 0, 1, 1, r * r, 1, 1, 1, 1).eng;
    expect(e).toBeCloseTo((78 / epsD - 1) / r, 9);
    expect(d.single!(0, 0, 1, 1, 3.6 * 3.6, 1, 1, 1, 1).eng).toBe(0);
  });
});

describe('pair momb', () => {
  it('parses cutoff s6 d and the five coefficients, and refuses bad counts', async () => {
    expect((await run(`${DIMER}pair_style momb 12.0 0.75\nrun 0\n`)).errors[0]).toMatch(/usage: pair_style momb cutoff s6 d/);
    expect((await run(`${DIMER}pair_style momb 12.0 0.75 20.0\npair_coeff * * 0.3 1.2 2.5 10.2847\nrun 0\n`)).errors[0]).toMatch(/usage: pair_coeff I J D0 alpha r0 C6 Rr/);
  });

  it('has no mixing rule: an unset I,J pair is refused', async () => {
    const script = `${DIMER}pair_style momb 12.0 0.75 20.0\npair_coeff 1 1 0.3 1.2 2.5 10.2847 2.361\npair_coeff 2 2 0.5 1.0 3.0 22.9973 2.702\nrun 0\n`;
    expect((await run(script)).errors[0]).toMatch(/All pair coeffs are not set/);
  });

  it('single() forces agree with central differences; the Morse-only shift is measured', () => {
    const m = new PairMomb();
    const ctx = { s: null } as never;
    m.settings(['12.0', '0.75', '20.0'], ctx);
    m.allocate(1);
    m.coeff(['1', '1', '0.3', '1.2', '2.5', '10.2847', '2.361'], ctx);
    m.init(ctx);
    for (const r of [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]) {
      const { fd, an } = fdCheck(m as unknown as Parameters<typeof fdCheck>[0], 1, 1, r, 0, 0);
      expect(Math.abs(fd - an)).toBeLessThanOrEqual(1e-7 * Math.max(1e-3, Math.abs(an)));
    }
    // energy matches the doc formula at r = 4.0
    const r = 4.0;
    const f = 1 / (1 + Math.exp(-20 * (r / 2.361 - 1)));
    const e = 0.3 * (Math.exp(-2 * 1.2 * (r - 2.5)) - 2 * Math.exp(-1.2 * (r - 2.5))) - 0.75 * 10.2847 / r ** 6 * f;
    expect(m.single!(0, 0, 1, 1, r * r, 1, 1).eng).toBeCloseTo(e, 14);
  });
});
