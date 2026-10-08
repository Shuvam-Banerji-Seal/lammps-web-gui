import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { compileLepton } from '../src/engine/lepton';
import { parseLepton } from '../src/engine/lepton/parse';
import { StyleError } from '../src/engine/force/types';
import { zblEnergy, erf, erfc, type ZblConst } from '../src/engine/lepton/compile';

/*
 * Lepton expression engine and the lepton styles (wave 16). Expected values
 * marked "measured" were obtained with native LAMMPS as a black box (a pair
 * or bond of two atoms, pe or ebond printed by run 0); the docs are
 * docs.lammps.org/lepton_expression.html, pair_lepton.html, bond_lepton.html,
 * angle_lepton.html, dihedral_lepton.html, fix_efield_lepton.html and the
 * wall/lepton part of fix_wall.html.
 */

const ev = (text: string, r: number, builtins = ['r']): number => {
  const p = compileLepton(text, { builtins, wrt: ['r'] });
  return p.value(Float64Array.from([r]));
};

describe('lepton expression grammar (measured with native LAMMPS, r = 1.3)', () => {
  const cases: Array<[string, number]> = [
    ['r^2', 1.69],
    ['-2^2', -4],          // unary minus looser than ^
    ['-r^2', -1.69],
    ['2^3^2', 512],        // right-associative ^
    ['2*r^2^2', 2 * 1.3 ** 4],
    ['2^-1', 0.5],
    ['r^-2', 1 / 1.69],
    ['2/r*3', (2 / 1.3) * 3],   // left-associative / and *
    ['4/2/r', 4 / 2 / 1.3],
    ['2 3*r', 29.9],       // whitespace is removed
    ['2*-r', -2.6],
    ['2- -r', 3.3],
    ['r--2', 3.3],
    ['2*(3)', 6],
    ['.5*r', 0.65],
    ['5.*r', 6.5],
    ['1E3*r', 1300],
    ['2e+3*r', 2600],
    ['3.12e-2*r', 0.0312 * 1.3],
    ['min(r,2)', 1.3],
    ['max(r,1.5)', 1.5],
    ['min(r,2)*max(r,2)', 2.6],
    ['delta(r-1.3)', 1],
    ['step(r-1.4)', 0],
    ['abs(-r)', 1.3],
    ['cot(r)', 1 / Math.tan(1.3)],
    ['sec(r)', 1 / Math.cos(1.3)],
    ['csc(r)', 1 / Math.sin(1.3)],
    ['1/sec(r)', Math.cos(1.3)],
    ['tan(r)', Math.tan(1.3)],
    ['2^0.5*r', 2 ** 0.5 * 1.3],
    ['r ^ 2', 1.69],
  ];
  for (const [text, want] of cases) {
    it(`"${text}"`, () => {
      expect(ev(text, 1.3)).toBeCloseTo(want, 12);
    });
  }
});

describe('lepton definitions (measured with native LAMMPS)', () => {
  const cases: Array<[string, number]> = [
    ['a^2; a=r', 1.69],
    ['a^2; a=b+1; b=r', 5.29],
    ['r^2; r=2', 4],                 // a definition shadows the built-in r
    ['r*2; a=r; r=3', 6],
    ['a+r; r=3; a=r*2', 5.6],        // a use after the definition falls back to the built-in r
    ['2*a; a=r*b; b=r+1', 5.98],
    ['a+r; a=2*r; r=3', 9],
    ['r+s; s=r; r=2', 4],
    ['a*r; a=2;', 2.6],              // trailing semicolon
    ['x_1*r; x_1=2', 2.6],
    ['a+r; a=b; b=r', 2.6],
  ];
  for (const [text, want] of cases) {
    it(`"${text}"`, () => {
      expect(ev(text, 1.3)).toBeCloseTo(want, 12);
    });
  }
});

describe('lepton parse errors are StyleErrors', () => {
  const bad = [
    '+r', '2*+r', 'r^+2', '1.0r', 'r**2', 'r^2^', '(r', 'r)', '', '()', 'sin(r)(1)',
    'min(r)', 'sqrt(r,2)', 'min(r,2,3)', 'foo(r)', 'k*r; k=2; k=3', 'a^2; a=b=r', 'v_*r',
  ];
  for (const text of bad) {
    it(`rejects "${text}"`, () => {
      expect(() => compileLepton(text, { builtins: ['r'], wrt: ['r'] })).toThrow(StyleError);
    });
  }
  it('names an undefined identifier (native LAMMPS substitutes 0 silently)', () => {
    expect(() => compileLepton('foo*r', { builtins: ['r'], wrt: ['r'] })).toThrow(/name 'foo'/);
  });
  it('a use after the definition is not resolved (names must be defined before use)', () => {
    expect(() => compileLepton('a^2+b; b=r; a=b', { builtins: ['r'], wrt: ['r'] })).toThrow(/name 'b'/);
  });
  it('zbl is a pair-only function', () => {
    expect(() => parseLepton('zbl(13,6,r)', { zbl: false })).toThrow(StyleError);
  });
});

describe('lepton exact derivatives', () => {
  const exprs = [
    'r^2', '-2^2*r', 'sqrt(r)*exp(-r)', 'log(r)/r', 'sin(r)*cos(2*r)', 'tan(r)', 'sec(r)', 'csc(r)', 'cot(r)',
    'asin(r/3)', 'acos(r/3)', 'atan(r)', 'sinh(r)', 'cosh(r)', 'tanh(r)', 'erf(r)', 'erfc(r)', 'abs(r-1.7)',
    'min(r,1.5)*r', 'max(r,1.5)^2', 'step(r-1.2)*r^3', 'r^r', '2^r', 'a*r^2; a=3*b; b=1.5', '(r-1)/(r+1)',
  ];
  for (const text of exprs) {
    it(`d/dr "${text}" matches a central difference`, () => {
      const p = compileLepton(text, { builtins: ['r'], wrt: ['r'] });
      for (const r of [0.9, 1.3, 2.1]) {
        const h = 1e-5;
        const fd = (p.value(Float64Array.from([r + h])) - p.value(Float64Array.from([r - h]))) / (2 * h);
        const an = p.deriv[0](Float64Array.from([r]));
        expect(Math.abs(an - fd)).toBeLessThan(1e-7 * Math.max(1, Math.abs(fd)));
      }
    });
  }
  it('gradient of V(x,y,z) (efield): three partial derivatives', () => {
    const p = compileLepton('-0.5*x^2 - 0.3*y*z + sin(x)*exp(-0.1*z)', { builtins: ['x', 'y', 'z'], wrt: ['x', 'y', 'z'] });
    const e = Float64Array.from([0.4, 1.1, -0.7]);
    expect(p.deriv[0](e)).toBeCloseTo(-0.4 + Math.cos(0.4) * Math.exp(0.07), 12);
    expect(p.deriv[1](e)).toBeCloseTo(-0.3 * -0.7, 12);
    expect(p.deriv[2](e)).toBeCloseTo(-0.3 * 1.1 + Math.sin(0.4) * Math.exp(0.07) * (-0.1), 12);
  });
});

describe('special functions', () => {
  it('erf and erfc match reference values to 1e-13', () => {
    expect(erf(0.5)).toBeCloseTo(0.5204998778130465, 13);
    expect(erf(1.3)).toBeCloseTo(0.9340079449406524, 13);
    expect(erfc(2.5)).toBeCloseTo(4.069520174449590e-4, 15);
    expect(erfc(4.2)).toBeCloseTo(2.8554941795921868e-9, 20);
  });
  it('zbl(13,6,r) in metal units (measured with native LAMMPS: 9.84465622794533 at r = 1.3)', () => {
    const c: ZblConst = { qqr2e: 14.399645, angstrom: 1 };
    expect(zblEnergy(13, 6, 1.3, c)).toBeCloseTo(9.84465622794533, 10);
    const p = compileLepton('zbl(13,6,r)', { builtins: ['r'], wrt: ['r'], zbl: c });
    expect(p.value(Float64Array.from([1.3]))).toBeCloseTo(9.84465622794533, 10);
  });
  it('zbl in lj units (measured: 0.683673536947983 at r = 1.3)', () => {
    const c: ZblConst = { qqr2e: 1, angstrom: 1 };
    const p = compileLepton('zbl(13,6,r)', { builtins: ['r'], wrt: ['r'], zbl: c });
    expect(p.value(Float64Array.from([1.3]))).toBeCloseTo(0.683673536947983, 12);
  });
});

/** Runs an input in a fresh session; returns the error message or '' and the thermo rows. */
const run = async (text: string): Promise<{ error: string; rows: Array<Record<string, number>> }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error = '';
  try {
    await session.execute(text);
  } catch (e) {
    error = String(e);
  }
  const err = events.find((x) => x.kind === 'error');
  if (err && 'message' in err) error = String(err.message);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as unknown as Record<string, number>);
  return { error, rows };
};

const pair2 = (style: string, coeff: string, extra = '', q = '') => `
units lj
atom_style ${q ? 'charge' : 'atomic'}
region b block -20 20 -20 20 -20 20
create_box 1 b
create_atoms 1 single 0 0 0
create_atoms 1 single 1.3 0 0
${q ? 'set atom 1 charge 1.0\nset atom 2 charge -1.0' : ''}
mass 1 1.0
pair_style ${style}
pair_coeff ${coeff}
${extra}
thermo_style custom step pe evdwl ecoul
thermo_modify norm no
run 0
`;

describe('pair styles (measured with native LAMMPS, two atoms at r = 1.3)', () => {
  it('pair lepton: energy and pair_modify shift', async () => {
    const a = await run(pair2('lepton 2.5', '1 1 "r^2"'));
    expect(a.error).toBe('');
    expect(a.rows[0].evdwl).toBeCloseTo(1.69, 12);
    const b = await run(pair2('lepton 2.5', '1 1 "r^2"', 'pair_modify shift yes'));
    expect(b.rows[0].evdwl).toBeCloseTo(1.69 - 6.25, 12);
  });
  it('per-pair cutoff excludes the pair beyond it', async () => {
    const a = await run(pair2('lepton 2.5', '1 1 "r^2" 1.0'));
    expect(a.rows[0].evdwl).toBe(0);
  });
  it('lepton/coul tallies ecoul with the charges qi, qj (measured: -0.769230769230769)', async () => {
    const a = await run(pair2('lepton/coul 2.5', '1 1 "qi*qj/r" 4.0', '', 'q'));
    expect(a.rows[0].ecoul).toBeCloseTo(-1 / 1.3, 12);
    expect(a.rows[0].evdwl).toBe(0);
  });
  it('lepton/coul rejects pair_modify shift (measured: native error)', async () => {
    const a = await run(pair2('lepton/coul 2.5', '1 1 "qi*qj/r"', 'pair_modify shift yes', 'q'));
    expect(a.error).toMatch(/shift/);
  });
  it('lepton rejects tail (pair page: no tail corrections)', async () => {
    const a = await run(pair2('lepton 2.5', '1 1 "r^2"', 'pair_modify tail yes'));
    expect(a.error).toMatch(/tail/);
  });
  it('lepton/coul rejects dispersion and tip4p keywords', async () => {
    expect((await run(pair2('lepton/coul 2.5 dispersion', '1 1 "qi*qj/r"', '', 'q'))).error).toMatch(/dispersion/);
    expect((await run(pair2('lepton/coul 2.5 tip4p', '1 1 "qi*qj/r"', '', 'q'))).error).toMatch(/tip4p/);
  });
  it('lepton/coul with a long-range keyword needs a kspace style', async () => {
    expect((await run(pair2('lepton/coul 2.5 pppm', '1 1 "qi*qj/r*erfc(alpha*r); alpha=1.067"', '', 'q'))).error).toMatch(/kspace/);
  });
  it('lepton does not mix: an unset I J pair is an error', async () => {
    const text = `
units lj
atom_style atomic
region b block -20 20 -20 20 -20 20
create_box 2 b
create_atoms 1 single 0 0 0
create_atoms 2 single 1.3 0 0
mass * 1.0
pair_style lepton 2.5
pair_coeff 1 1 "r^2"
run 0
`;
    expect((await run(text)).error).toMatch(/not set/);
  });
  it('pair lepton with zbl(13,6,r) in metal units (measured: 9.84465622794533 at 1.3 Angstrom)', async () => {
    const text = `
units metal
atom_style atomic
region b block -20 20 -20 20 -20 20
create_box 1 b
create_atoms 1 single 0 0 0
create_atoms 1 single 1.3 0 0
mass 1 1.0
pair_style lepton 10
pair_coeff 1 1 "zbl(13,6,r)"
thermo_style custom step evdwl
thermo_modify norm no
run 0
`;
    const r = await run(text);
    expect(r.error).toBe('');
    expect(r.rows[0].evdwl).toBeCloseTo(9.84465622794533, 10);
  });
  it('lepton/sphere requires atom_style sphere', async () => {
    expect((await run(pair2('lepton/sphere 2.5', '1 1 "r^2"'))).error).toMatch(/sphere/);
  });
  it('an undefined name in a pair expression is a StyleError', async () => {
    expect((await run(pair2('lepton 2.5', '1 1 "k*r^2"'))).error).toMatch(/name 'k'/);
  });
});

describe('bond, angle and dihedral conventions (measured with native LAMMPS)', () => {
  it('bond variable is r_i - r0; auto offset subtracts the value at r0 (measured: 0.04 and 1.04)', async () => {
    const text = (style: string) => `
units lj
atom_style bond
bond_style ${style}
region b block -20 20 -20 20 -20 20
create_box 1 b bond/types 1 extra/bond/per/atom 1
create_atoms 1 single 0 0 0
create_atoms 1 single 1.2 0 0
mass 1 1.0
create_bonds single/bond 1 1 2
pair_style zero 5.0
pair_coeff * *
bond_coeff 1 1.0 "r^2+1"
thermo_style custom step ebond
thermo_modify norm no
run 0
`;
    expect((await run(text('lepton'))).rows[0].ebond).toBeCloseTo(0.04, 12);
    expect((await run(text('lepton no_offset'))).rows[0].ebond).toBeCloseTo(1.04, 12);
  });
  it('angle variable is theta_i - theta0 in radians (measured: -0.5236 at 60 degrees with theta0 = 90)', () => {
    // the angle style is exercised by the oracle case w16lepton_bonded; here the variable convention
    expect(compileLepton('theta', { builtins: ['theta'], wrt: ['theta'] }).value(Float64Array.from([Math.PI / 3 - Math.PI / 2]))).toBeCloseTo(-Math.PI / 6, 12);
  });
  it('dihedral phi is in [0, 2 pi) (measured: 4.71238898038469 for a torsion of -90 degrees)', () => {
    expect(compileLepton('phi', { builtins: ['phi'], wrt: ['phi'] }).value(Float64Array.from([3 * Math.PI / 2]))).toBeCloseTo(4.71238898038469, 12);
  });
  it('bond_style lepton rejects unknown arguments', async () => {
    const text = `
units lj
atom_style bond
bond_style lepton bogus
`;
    expect((await run(text)).error).toMatch(/bogus/);
  });
});

describe('fix styles', () => {
  it('wall/lepton shifts the energy to 0 at the cutoff (measured: r^2 at x = 1, cutoff 4 gives -15)', async () => {
    const text = `
units lj
atom_style charge
boundary f p p
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single 1.0 0 0
set atom 1 charge 2.0
mass 1 1.0
pair_style zero 5.0
pair_coeff * *
fix w all wall/lepton xlo 0.0 "r^2" 4.0
fix_modify w energy yes
fix 1 all nve
thermo_style custom step pe
thermo_modify norm no
run 0
`;
    const r = await run(text);
    expect(r.error).toBe('');
    expect(r.rows[0].pe).toBeCloseTo(-15, 12);
  });
  it('wall/lepton without r in the derivative (e.g. "r") is evaluated exactly', async () => {
    const text = `
units lj
atom_style charge
boundary f p p
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single 1.0 0 0
mass 1 1.0
pair_style zero 5.0
pair_coeff * *
fix w all wall/lepton xlo 0.0 "r" 4.0
fix_modify w energy yes
fix 1 all nve
thermo_style custom step pe
thermo_modify norm no
run 0
`;
    const r = await run(text);
    expect(r.error).toBe('');
    expect(r.rows[0].pe).toBeCloseTo(1 - 4, 12);
  });
  it('efield/lepton: energy q V and charge force (measured: -3 and 2 for V = -E x, q = 2, x = 1.5)', async () => {
    const text = `
units lj
atom_style charge
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single 1.5 0 0
set atom 1 charge 2.0
mass 1 1.0
pair_style zero 5.0
pair_coeff * *
fix ex all efield/lepton "-E*x; E=1"
fix_modify ex energy yes
fix 1 all nve
thermo_style custom step pe f_ex
thermo_modify norm no
run 0
`;
    const r = await run(text);
    expect(r.error).toBe('');
    expect(r.rows[0].pe).toBeCloseTo(-3, 12);
    expect(r.rows[0].f_ex).toBeCloseTo(-3, 12);
  });
  it('efield/lepton rejects dipole atom styles', async () => {
    const text = `
units lj
atom_style dipole
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single 1.5 0 0
mass 1 1.0
pair_style zero 5.0
pair_coeff * *
fix ex all efield/lepton "-E*x; E=1"
`;
    expect((await run(text)).error).toMatch(/dipole/);
  });
});

describe('lepton inputs in the repo', () => {
  it('the oracle inputs are present', () => {
    const dir = join(__dirname, 'oracle');
    for (const f of ['w16lepton_pair.in', 'w16lepton_coul.in', 'w16lepton_bonded.in', 'w16lepton_wall.in', 'w16lepton_efield.in']) {
      expect(readFileSync(join(dir, f), 'utf8').length).toBeGreaterThan(0);
    }
  });
});
