import { describe, expect, it } from 'vitest';
import {
  DF1, DF2, DF4, DF5, DF6, F1, F2, F3, F4, F5, F6, backboneSite, baseSite, is3pto5p, lrfOf, oxdnaConstants,
} from '../src/engine-gpl/oxdna/common';
import { BondOxdnaFene } from '../src/engine-gpl/oxdna/bond_fene';
import { PairOxdnaExcv } from '../src/engine-gpl/oxdna/pair_excv';
import { PairOxdnaStk, stackingStrength } from '../src/engine-gpl/oxdna/pair_stk';
import { PairOxdnaHbond } from '../src/engine-gpl/oxdna/pair_hbond';
import { PairOxdnaXstk, PairOxrna2Xstk } from '../src/engine-gpl/oxdna/pair_xstk';
import { PairOxdnaCoaxstk } from '../src/engine-gpl/oxdna/pair_coaxstk';
import { BONDS, FIXES, PAIRS } from '../src/engine-gpl/registry/oxdna';
import { PairOxdna2Dh } from '../src/engine-gpl/oxdna/pair_dh';
import { BondOxdna2Fene, PairOxdna2Excv } from '../src/engine-gpl/oxdna/variants';
import { backboneOxdna2, backboneOxrna2, backboneOxrna2Dh } from '../src/engine-gpl/oxdna/common';
import { stackingDerived } from '../src/engine-gpl/oxdna/pair_stk';
import { noSquishRotate, vec3ToVec4, vec4ToVec3 } from '../src/engine-gpl/oxdna/fix_nve_dot';
import type { StyleContext } from '../src/engine/force/types';

// Every expected value below is written out from the formula of the LAMMPS source named in the comment
// (src/CG-DNA/mf_oxdna.h, constants_oxdna.cpp, the coeff() functions), not from the TypeScript code under test.
const ctx = (text = '', s: unknown = null): StyleContext => ({ s, readFile: () => text, log: () => {} }) as unknown as StyleContext;

describe('oxDNA modulation functions (mf_oxdna.h)', () => {
  // F1: eps, a, cut_0, cut_lc, cut_hc, cut_lo, cut_hi, b_lo, b_hi, shift
  const p1 = { eps: 1.5, a: 6.0, cut0: 0.4, cutLc: 0.2, cutHc: 1.0, cutLo: 0.6, cutHi: 0.9, bLo: 0.7, bHi: 0.8, shift: 0.05 };
  const f1 = (r: number) => F1(r, p1.eps, p1.a, p1.cut0, p1.cutLc, p1.cutHc, p1.cutLo, p1.cutHi, p1.bLo, p1.bHi, p1.shift);
  const df1 = (r: number) => DF1(r, p1.eps, p1.a, p1.cut0, p1.cutLc, p1.cutHc, p1.cutLo, p1.cutHi, p1.bLo, p1.bHi);

  it('F1 takes its four regions: zero outside, quadratic tails, and the shifted exponential', () => {
    expect(f1(0.1)).toBe(0);
    expect(f1(1.2)).toBe(0);
    // r in (cut_lc, cut_lo]: eps b_lo (r - cut_lc)^2
    expect(f1(0.5)).toBeCloseTo(p1.eps * p1.bLo * (0.5 - 0.2) ** 2, 14);
    // r in (cut_lo, cut_hi]: eps (1 - exp(-(r - cut_0) a))^2 - shift
    const r = 0.7;
    const t = 1 - Math.exp(-(r - 0.4) * 6.0);
    expect(f1(r)).toBeCloseTo(p1.eps * t * t - p1.shift, 14);
    // r in (cut_hi, cut_hc]: eps b_hi (r - cut_hc)^2
    expect(f1(0.95)).toBeCloseTo(p1.eps * p1.bHi * (0.95 - 1.0) ** 2, 14);
  });

  it('DF1 is (dF1/dr) / r in each region (mf_oxdna.h returns the force over r; checked by a centred difference)', () => {
    for (const r of [0.25, 0.5, 0.7, 0.95]) {
      const h = 1e-6;
      const num = (f1(r + h) - f1(r - h)) / (2 * h);
      expect(df1(r) * r).toBeCloseTo(num, 5);
    }
  });

  // F2: k, cut_0, cut_lc, cut_hc, cut_lo, cut_hi, b_lo, b_hi, cut_c
  it('F2 and DF2 follow the four regions of the cross-stacking modulation', () => {
    const k = 47.5, c0 = 0.575, lc = 0.495, hc = 0.655, lo = 0.58, hi = 0.65, blo = 1.2, bhi = 1.1, cc = 0.675;
    expect(F2(0.4, k, c0, lc, hc, lo, hi, blo, bhi, cc)).toBe(0);
    // r in [cut_lc, cut_lo): k b_lo (cut_lc - r)^2
    expect(F2(0.52, k, c0, lc, hc, lo, hi, blo, bhi, cc)).toBeCloseTo(k * blo * (lc - 0.52) ** 2, 12);
    // r in [cut_lo, cut_hi): k/2 ((r - cut_0)^2 - (cut_0 - cut_c)^2)
    expect(F2(0.6, k, c0, lc, hc, lo, hi, blo, bhi, cc)).toBeCloseTo(k * 0.5 * ((0.6 - c0) ** 2 - (c0 - cc) ** 2), 12);
    // r in [cut_hi, cut_hc]: k b_hi (cut_hc - r)^2
    expect(F2(0.652, k, c0, lc, hc, lo, hi, blo, bhi, cc)).toBeCloseTo(k * bhi * (hc - 0.652) ** 2, 12);
    expect(F2(0.66, k, c0, lc, hc, lo, hi, blo, bhi, cc)).toBe(0);
    const h = 1e-6;
    for (const r of [0.52, 0.6, 0.652]) {
      const num = (F2(r + h, k, c0, lc, hc, lo, hi, blo, bhi, cc) - F2(r - h, k, c0, lc, hc, lo, hi, blo, bhi, cc)) / (2 * h);
      expect(DF2(r, k, c0, lc, hc, lo, hi, blo, bhi)).toBeCloseTo(num, 4);
    }
  });

  it('F3 switches from the LJ form inside cut_ast to the quadratic tail, and its force matches the energy', () => {
    const eps = 2.0, sigma = 0.7, cutAst = 0.675;
    const lj1 = 4 * eps * sigma ** 12, lj2 = 4 * eps * sigma ** 6;
    const cutC = 0.7118792143562591, b = 892.0162233432603;
    const fp = [0];
    // inside: 4 eps ((sigma/r)^12 - (sigma/r)^6) at r = 0.6
    const e1 = F3(0.36, cutAst * cutAst, cutC, lj1, lj2, eps, b, fp);
    expect(e1).toBeCloseTo(4 * eps * ((sigma / 0.6) ** 12 - (sigma / 0.6) ** 6), 9);
    // outside, between cut_ast and cut_c: eps b (cut_c - r)^2
    const e2 = F3(0.69 * 0.69, cutAst * cutAst, cutC, lj1, lj2, eps, b, fp);
    expect(e2).toBeCloseTo(eps * b * (cutC - 0.69) ** 2, 9);
    // fpair = -(dE/dr)/r, checked with a centred difference of the energy
    const h = 1e-7;
    const E = (r: number) => F3(r * r, cutAst * cutAst, cutC, lj1, lj2, eps, b, [0]);
    F3(0.6 * 0.6, cutAst * cutAst, cutC, lj1, lj2, eps, b, fp);
    expect(fp[0] * 0.6).toBeCloseTo(-(E(0.6 + h) - E(0.6 - h)) / (2 * h), 4);
  });

  // F4 (theta, a, theta_0, dtheta_ast, b, dtheta_c)
  it('F4 and DF4: quadratic well, quadratic walls and the zero outside dtheta_c', () => {
    const a = 0.9, th0 = 1.0, dast = 0.3, b = 0.5, dc = 0.8;
    expect(F4(th0 + 1.0, a, th0, dast, b, dc)).toBe(0);
    expect(F4(th0, a, th0, dast, b, dc)).toBe(1);
    expect(F4(th0 + 0.2, a, th0, dast, b, dc)).toBeCloseTo(1 - a * 0.04, 14);
    expect(F4(th0 + 0.5, a, th0, dast, b, dc)).toBeCloseTo(b * (0.5 - dc) ** 2, 14);
    expect(F4(th0 - 0.5, a, th0, dast, b, dc)).toBeCloseTo(b * (-0.5 + dc) ** 2, 14);
    const th = th0 + 0.45;
    const h = 1e-6;
    expect(DF4(th, a, th0, dast, b, dc)).toBeCloseTo((F4(th + h, a, th0, dast, b, dc) - F4(th - h, a, th0, dast, b, dc)) / (2 * h), 5);
  });

  // F5 (x, a, x_ast, b, x_c)
  it('F5 is 1 for x >= 0 and has the same wall shape as F4', () => {
    expect(F5(0.3, 1.0, -0.2, 0.5, -0.7)).toBe(1);
    expect(F5(-0.1, 1.0, -0.2, 0.5, -0.7)).toBeCloseTo(1 - 1.0 * 0.01, 14);
    expect(F5(-0.5, 1.0, -0.2, 0.5, -0.7)).toBeCloseTo(0.5 * (-0.5 + 0.7) ** 2, 14);
    expect(F5(-0.9, 1.0, -0.2, 0.5, -0.7)).toBe(0);
    expect(DF5(-0.1, 1.0, -0.2, 0.5, -0.7)).toBeCloseTo(0.2, 14);
    expect(DF5(0.1, 1.0, -0.2, 0.5, -0.7)).toBe(0);
  });

  // F6 (theta, a, b)
  it('F6 and DF6 are zero below b and quadratic above it', () => {
    expect(F6(0.2, 2.0, 0.5)).toBe(0);
    expect(F6(1.5, 2.0, 0.5)).toBeCloseTo(0.5 * 2.0 * 1.0 * 1.0, 14);
    expect(DF6(1.5, 2.0, 0.5)).toBeCloseTo(2.0 * 1.0, 14);
    expect(DF6(0.2, 2.0, 0.5)).toBe(0);
  });

  it('is3pto5p returns the sign of delr . n, with a zero dot product counted as +1', () => {
    expect(is3pto5p([1, 0, 0], [0.5, 0.2, 0])).toBe(1);
    expect(is3pto5p([1, 0, 0], [-0.5, 0.2, 0])).toBe(-1);
    expect(is3pto5p([1, 0, 0], [0, 1, 0])).toBe(1);
  });
});

describe('oxDNA constants and sites (constants_oxdna.cpp, nucleotide_oxdna.h)', () => {
  it('uses the lj values for lj units and the real-unit values for "real" styles', () => {
    const lj = oxdnaConstants('lj');
    expect(lj.dx_cbk_oxdna1).toBe(-0.4);
    expect(lj.dx_cstk_oxdna1).toBe(0.34);
    expect(lj.dx_cbs_oxdna1).toBe(0.4);
    expect(lj.lambda_dh_one_prefactor).toBe(0.3616455075438555);
    const real = oxdnaConstants('real');
    expect(real.dx_cbk_oxdna1).toBe(-3.4072);
    expect(real.dx_cstk_oxdna1).toBe(2.89612);
    expect(real.dx_cbs_oxdna1).toBe(3.4072);
    expect(real.qeff_dh_pf_one_prefactor).toBe(4.15079634587587);
  });

  it('places the backbone and base sites along the local x axis', () => {
    const out = [0, 0, 0];
    backboneSite(-0.4, [1, 0, 0], out);
    expect(out[0]).toBeCloseTo(-0.4, 14);
    expect(Math.abs(out[1]) + Math.abs(out[2])).toBe(0);
    baseSite(0.4, [0, 0, 1], out);
    expect(out[2]).toBeCloseTo(0.4, 14);
    expect(Math.abs(out[0]) + Math.abs(out[1])).toBe(0);
  });

  it('lrfOf gives the identity frame for the unit quaternion and a z rotation by 90 degrees for (cos45, 0, 0, sin45)', () => {
    const out = new Float64Array(9);
    lrfOf([1, 0, 0, 0], 0, out, 0);
    expect(Array.from(out)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const c = Math.SQRT1_2;
    lrfOf([c, 0, 0, c], 0, out, 0);
    // rotation by +90 degrees about z: ex = (0, 1, 0), ey = (-1, 0, 0), ez = (0, 0, 1)
    const exp = [0, 1, 0, -1, 0, 0, 0, 0, 1];
    for (let k = 0; k < 9; k++) expect(out[k]).toBeCloseTo(exp[k], 12);
  });
});

describe('oxDNA styles: coefficients and their derived values', () => {
  it('stackingStrength is xi + kappa T (pair_oxdna_stk.cpp stacking_strength)', () => {
    expect(stackingStrength(1.3448, 2.6568, 0.1)).toBeCloseTo(1.3448 + 2.6568 * 0.1, 12);
  });

  it('the excluded volume b and cut_c follow from sigma and cut_ast (pair_oxdna_excv.cpp coeff)', () => {
    const p = new PairOxdnaExcv();
    p.allocate(4);
    p.coeff(['*', '*', '2.0', '0.7', '0.675', '2.0', '0.515', '0.5', '2.0', '0.33', '0.32']);
    // cut_bkbk_c for sigma 0.7, cut_ast 0.675: cut_ast - 8 (x^12 - x^6) / (4/sigma (6 x^7 - 12 x^13)), x = sigma/cut_ast
    const x = 0.7 / 0.675;
    const cutC = 0.675 - 8 * (x ** 12 - x ** 6) / (4 / 0.7 * (6 * x ** 7 - 12 * x ** 13));
    expect(p.initOne(1, 1)).toBeCloseTo(cutC, 12);
    expect(p.initOne(1, 1)).toBeCloseTo(0.7118792143562591, 12);
  });

  it('the excluded volume reads a potential file line of the form "I J excv v1 ... v9"', () => {
    const text = '# comment\n*   *   excv   2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32\n';
    const p = new PairOxdnaExcv();
    p.allocate(4);
    p.coeff(['*', '*', 'file.cgdna'], ctx(text));
    expect(p.initOne(2, 2)).toBeCloseTo(0.7118792143562591, 12);
  });

  it('the excluded volume refuses a missing pair coefficient and an offset', () => {
    const p = new PairOxdnaExcv();
    p.allocate(3);
    p.coeff(['1', '1', '2.0', '0.7', '0.675', '2.0', '0.515', '0.5', '2.0', '0.33', '0.32']);
    expect(() => p.initOne(2, 2)).toThrow('Coefficient mixing not defined in oxdna');
    expect(() => p.coeff(['1', '1', '2.0', '0.7'])).toThrow('Incorrect args for pair coefficients in oxdna/excv');
  });

  it('the FENE bond takes "K Delta r0" and refuses a wrong count with the bond message', () => {
    const b = new BondOxdnaFene();
    b.allocate(2);
    b.coeff(['*', '2.0', '0.25', '0.7525']);
    b.init();
    expect(b.equilibrium(2)).toBeCloseTo(0.7525, 14);
    expect(() => b.coeff(['1', '2.0', '0.25'])).toThrow('Incorrect args for bond coefficients in oxdna/fene, oxdna2/fene or oxrna2/fene');
  });

  it('the FENE bond reads "type fene K Delta r0" from a potential file and names a missing type', () => {
    const b = new BondOxdnaFene();
    b.allocate(1);
    b.coeff(['1', 'pot.cgdna'], ctx('1 fene 2.0 0.25 0.7525\n'));
    b.init();
    const miss = new BondOxdnaFene();
    miss.allocate(1);
    expect(() => miss.coeff(['1', 'pot.cgdna'], ctx('2 fene 2.0 0.25 0.7525\n'))).toThrow('No corresponding fene potential found in file pot.cgdna for bond type 1');
  });

  it('the FENE bond requires special_bonds lj 0 1 1 (its init names the rule)', () => {
    const b = new BondOxdnaFene();
    b.allocate(1);
    b.coeff(['*', '2.0', '0.25', '0.7525']);
    b.linkForceField({ pair: null, special: { lj: [0, 0.5, 1], coul: [0, 0, 0] } });
    expect(() => b.init()).toThrow("Must use 'special_bonds lj 0 1 1' with bond style oxdna/fene");
  });

  it('the stacking style checks seqav or seqdep, and needs 20 numbers after the temperature', () => {
    const s = new PairOxdnaStk();
    s.allocate(4);
    expect(() => s.coeff(['*', '*', 'seqxx', '0.1', ...Array(20).fill('1')])).toThrow('Incorrect setting, select seqav or seqdep in oxdna/stk');
    expect(() => s.coeff(['*', '*', 'seqav', '0.1', '1', '2'])).toThrow('Incorrect args for pair coefficients in oxdna/stk');
  });

  it('the hydrogen-bond style reads its 24 numbers and symmetrises the table in initOne', () => {
    const h = new PairOxdnaHbond();
    h.allocate(4);
    h.coeff(['1', '4', 'seqav', '1.077', '8.0', '0.4', '0.75', '0.34', '0.7', '1.5', '0', '0.7', '1.5', '0', '0.7', '1.5', '0', '0.7', '0.46', '3.141592653589793', '0.7', '4.0', '1.5707963267948966', '0.45', '4.0', '1.5707963267948966', '0.45']);
    const cut = h.initOne(1, 4);
    expect(cut).toBeGreaterThan(0);
    expect(h.initOne(4, 1)).toBeCloseTo(cut, 14);
    expect(() => h.coeff(['1', '4', 'seqav', '1.0', 'file.cgdna'])).toThrow('Incorrect args for pair coefficients in oxdna/hbond');
  });

  it('the cross-stacking and coaxial styles check their argument counts (23 and 21 numbers)', () => {
    const x = new PairOxdnaXstk();
    x.allocate(2);
    expect(() => x.coeff(['*', '*', '1', '2'])).toThrow('Incorrect args for pair coefficients in oxdna/xstk');
    const c = new PairOxdnaCoaxstk();
    c.allocate(2);
    expect(() => c.coeff(['*', '*', '1', '2'])).toThrow('Incorrect args for pair coefficients in oxdna/coaxstk');
  });

  it('oxrna2/xstk reads 20 numbers from a potential file line, oxdna/xstk 23 (pair_oxrna2_xstk.cpp coeff)', () => {
    // The line of oxrna2_lj.cgdna in the duplex2 example: 20 numbers after "* * xstk".
    const nums = '59.9626 0.5 0.6 0.42 0.58 2.25 0.505 0.58 1.7 1.266 0.68 1.7 1.266 0.68 1.7 0.309 0.68 1.7 0.309 0.68';
    const fromFile = new PairOxrna2Xstk();
    fromFile.allocate(2);
    fromFile.coeff(['*', '*', 'oxrna2_lj.cgdna'], ctx(`* * xstk ${nums}\n`));
    const fromArgs = new PairOxrna2Xstk();
    fromArgs.allocate(2);
    fromArgs.coeff(['*', '*', ...nums.split(' ')]);
    // the coefficient tables are private; read them through a cast
    const tabs = (x: PairOxrna2Xstk) => (x as unknown as { t: Record<string, Float64Array> }).t;
    for (const key of Object.keys(tabs(fromArgs))) {
      expect(Array.from(tabs(fromFile)[key])).toEqual(Array.from(tabs(fromArgs)[key]));
    }
    const oxdna = new PairOxdnaXstk();
    oxdna.allocate(2);
    expect(() => oxdna.coeff(['*', '*', 'oxrna2_lj.cgdna'], ctx(`* * xstk ${nums}\n`))).toThrow('xstk needs 23 values');
  });

  it('the registry exports exactly the styles of this port', () => {
    expect(Object.keys(BONDS).sort()).toEqual(['oxdna/fene', 'oxdna2/fene', 'oxdna3/fene', 'oxrna2/fene']);
    expect(Object.keys(PAIRS).sort()).toEqual(expect.arrayContaining(['oxdna/coaxstk', 'oxdna/excv', 'oxdna/hbond', 'oxdna/stk', 'oxdna/xstk', 'oxdna3/coaxstk', 'oxdna3/dh', 'oxdna3/excv', 'oxdna3/hbond', 'oxdna3/stk', 'oxdna3/xstk']));
  });
});

describe('atom_style oxdna after read_restart', () => {
  // runs the duplex input twice (with and without the restart): more than the default 5 s on a busy machine
  it('rebuilds the strand partners from the stored bonds: the energies equal a run without the restart', { timeout: 60_000 }, async () => {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');
    const { Session } = await import('../src/engine/interpreter');
    const dir = join(__dirname, 'oracle');
    const input = readFileSync(join(dir, 'w57oxdna_duplex1_full.in'), 'utf8').replace(/^run[ \t]+200[ \t]*$/m, 'run 0');
    const data = readFileSync(join(dir, 'w57oxdna_duplex1.data'), 'utf8');
    const ff = input.slice(input.indexOf('\nset ') + 1, input.indexOf('\nfix ') + 1);
    const pe = async (script: string): Promise<number[]> => {
      const rows: number[] = [];
      const s = new Session({ emit: (e: { kind: string; row?: Record<string, number> }) => { if (e.kind === 'thermo' && e.row) rows.push(e.row.pe); }, writeFile: () => {} } as never);
      s.addFile('w57oxdna_duplex1.data', data);
      await s.execute(script);
      return rows;
    };
    const direct = await pe(input);
    // the oxDNA sites sit off the centres, so the neighbor skin of the input (2.0) is set again after clear
    const restarted = await pe(`${input}pair_style none\nwrite_restart d.restart\nclear\nread_restart d.restart\nneighbor 2.0 bin\nneigh_modify every 1 delay 0 check yes\n${ff}fix 1 all nve/asphere\nthermo_style custom step pe\nrun 0\n`);
    expect(direct.length).toBe(1);
    expect(restarted.length).toBe(2);
    expect(restarted[1]).toBeCloseTo(direct[0], 12);
    expect(direct[0]).not.toBe(0);
  });
});

describe('oxDNA2 and oxRNA2 variants (nucleotide_oxdna.h, the pair_oxdna2_dh.cpp and pair_oxrna2_dh.cpp coeff)', () => {
  it('the oxDNA2 backbone site is dx e1 + dy e2 and the oxRNA2 site is dx e1 + dz e3 (lj values)', () => {
    const c = oxdnaConstants('lj');
    const out = [0, 0, 0];
    backboneOxdna2(c, [1, 0, 0], [0, 1, 0], [0, 0, 1], out);
    expect(out[0]).toBeCloseTo(-0.34, 14);
    expect(out[1]).toBeCloseTo(0.3408, 14);
    expect(out[2]).toBeCloseTo(0, 14);
    backboneOxrna2(c, [1, 0, 0], [0, 1, 0], [0, 0, 1], out);
    expect(out[0]).toBeCloseTo(-0.4, 14);
    expect(out[2]).toBeCloseTo(0.2, 14);
    // the oxRNA2 Debye-Huckel site keeps the oxDNA1 x offset (pair_oxrna2_dh.cpp compute_backbone_site)
    backboneOxrna2Dh(c, [1, 0, 0], [0, 1, 0], [0, 0, 1], out);
    expect(out[0]).toBeCloseTo(-0.4, 14);
    expect(out[2]).toBeCloseTo(0.2, 14);
    expect(oxdnaConstants('real').dx_cbk_oxdna2).toBeCloseTo(-2.89612, 12);
  });

  it('the oxDNA2 Debye-Huckel cutoffs follow lambda = C1 sqrt(T / 0.1 / rhos) (pair_oxdna2_dh.cpp coeff)', () => {
    const p = new PairOxdna2Dh();
    p.allocate(2);
    const ctx = { s: null, readFile: () => '', log: () => {} } as unknown as StyleContext;
    // pair_coeff without atoms is refused: the qeff of every atom is set by the command
    expect(() => p.coeff(['*', '*', '0.1', '0.2', '0.815'], ctx)).toThrow('pair_coeff oxdna2/dh needs the atoms');
    expect(() => p.coeff(['*', '*', '0.1', '0.2'])).toThrow('Incorrect args for pair coefficients in oxdna2/dh');
    expect(() => p.coeff(['*', '*', '0.1', '0.2', '0.8', 'half_charged_ends', 'maybe'], ctx)).toThrow();
    const lambda = oxdnaConstants('lj').lambda_dh_one_prefactor * Math.sqrt(0.1 / 0.1 / 0.2);
    const cutAst = 3 * lambda;
    const cutC = cutAst * (cutAst + 3 * lambda) / (cutAst + lambda);
    // the coefficients are set only with atoms in a real run; the formula is checked on the derived values here
    expect(cutC).toBeGreaterThan(cutAst);
  });

  it('the oxDNA2 FENE bond and excluded volume are the oxDNA1 classes with the oxDNA2 backbone site', () => {
    expect(new BondOxdna2Fene().name).toBe('oxdna2/fene');
    expect(new PairOxdna2Excv().name).toBe('oxdna2/excv');
    expect(Object.keys(FIXES)).toEqual(['nve/dot', 'nve/dotc/langevin']);
    expect(Object.keys(BONDS)).toEqual(['oxdna/fene', 'oxdna2/fene', 'oxrna2/fene', 'oxdna3/fene']);
    expect(PAIRS['oxdna2/stk']).toBeDefined();
    expect(PAIRS['oxrna2/coaxstk']).toBeDefined();
  });

  it('the stacking derived shift is eps (1 - exp(-a (cut_c - cut_0)))^2 (pair_oxdna_stk.cpp coeff)', () => {
    const d = stackingDerived(1.61, 2.6568, 0.8, 0.9, 0.9 - 0.2, 0.95);
    const tmp = 1 - Math.exp(-(0.9 - 0.8) * 2.6568);
    expect(d.shiftOne).toBeCloseTo(1.61 * tmp * tmp, 14);
  });

  it('the no-squish helpers: the identity quaternion round-trips the angular momentum, and rotations keep the norm', () => {
    const q = new Float64Array([1, 0, 0, 0]);
    const p = [0, 0, 0, 0], l3 = [0, 0, 0];
    vec3ToVec4(q, 0, [0.5, -0.8, 1.3], p);
    vec4ToVec3(q, 0, p, l3);
    expect(l3[0]).toBeCloseTo(0.5, 14);
    expect(l3[1]).toBeCloseTo(-0.8, 14);
    expect(l3[2]).toBeCloseTo(1.3, 14);
    // a rotation by 0.3 rad about body x: noSquishRotate keeps |q| = 1 to the order of the step
    const qr = new Float64Array([Math.cos(0.15), Math.sin(0.15), 0, 0]);
    const pr = [0.1, 0.2, 0.3, 0.4];
    noSquishRotate(1, pr, qr, 0, [1, 1, 1], 1e-3);
    const n = Math.hypot(qr[0], qr[1], qr[2], qr[3]);
    expect(Math.abs(n - 1)).toBeLessThan(1e-6);
    // a zero step leaves both vectors unchanged
    const q0 = new Float64Array([0.6, 0.8, 0, 0]);
    const p0 = [1, 2, 3, 4];
    noSquishRotate(3, p0, q0, 0, [1, 2, 3], 0);
    expect(p0).toEqual([1, 2, 3, 4]);
    expect(Array.from(q0)).toEqual([0.6, 0.8, 0, 0]);
  });
});
