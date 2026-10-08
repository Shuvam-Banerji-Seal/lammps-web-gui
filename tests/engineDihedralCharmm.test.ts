import { describe, expect, it } from 'vitest';
import { DihedralCharmm, DihedralCharmmfsw } from '../src/engine/force/dihedral/charmm';
import { PairLJCharmmCoulCharmm, PairLJCharmmCoulCharmmImplicit, PairLJCharmmfswCoulCharmmfsh } from '../src/engine/force/pair/charmm';

import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { StyleError, newAccum, type BondedCompute, type StyleContext } from '../src/engine/force/types';

/*
 * dihedral_style charmm / charmmfsw (docs.lammps.org/dihedral_charmm.html):
 * E = K [1 + cos(n phi - d)] per quadruplet, plus the 1-4 LJ and Coulomb terms
 * scaled by the weighting factor. Forces must equal -grad E by finite
 * differences of the total (dihedral + 1-4) energy. Constants of the 1-4 terms are
 * the measured forms documented in src/engine/force/dihedral/charmm.ts.
 */

const L = 40;
const ctx = (s: ReturnType<typeof emptyState>): StyleContext => ({ s, readFile: () => '', log: () => {} });

type Pair = PairLJCharmmCoulCharmm | PairLJCharmmfswCoulCharmmfsh | PairLJCharmmCoulCharmmImplicit;

const makePair = (p: Pair, settings: string[]) => {
  const s = emptyState(UNIT_SYSTEMS.real, 3, { lo: [0, 0, 0], hi: [L, L, L] }, 3, 'full');
  p.settings(settings);
  p.allocate(3);
  p.coeff(['1', '1', '0.1', '3.0', '0.05', '3.5']);
  p.coeff(['2', '2', '0.15', '3.2', '0.1', '3.1']);
  p.coeff(['3', '3', '0.12', '3.1', '0.08', '3.3']);
  p.init(ctx(s));
  return p;
};

/** Four atoms (types 1 1 2 2, charges), one dihedral of type dtype. */
const system = (pts: number[][], dtype: number) => {
  const s = emptyState(UNIT_SYSTEMS.real, 3, { lo: [0, 0, 0], hi: [L, L, L] }, 3, 'full');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  const types = [1, 1, 2, 2], qs = [0.25, -0.3, 0.1, -0.2];
  for (let i = 0; i < 4; i++) { s.type[i] = types[i]; s.q[i] = qs[i]; }
  pushTopo(s.topo.dihedrals, dtype, [1, 2, 3, 4]);
  return s;
};

/** Total energy (dihedral + 1-4) and forces for the given positions. */
const evaluate = (s: ReturnType<typeof system>, dih: DihedralCharmm) => {
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [L, L, L] }));
  s.f.fill(0);
  const acc = newAccum();
  const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc, virial: new Float64Array(6), eatom: null, vatom: null };
  dih.compute(bc);
  return { e: acc.edihed + acc.evdwl + acc.ecoul, f: Float64Array.from(s.f), acc };
};

const PTS = [[18.1, 20.2, 19.7], [19.4, 20.1, 20.05], [20.2, 21.1, 19.8], [21.4, 21.3, 21.1]];

const setupDih = (name: 'charmm' | 'charmmfsw', pair: Pair, coeffs: string[], special = { lj: [0, 0, 0] as [number, number, number], coul: [0, 0, 0] as [number, number, number] }) => {
  const dih = name === 'charmm' ? new DihedralCharmm() : new DihedralCharmmfsw();
  dih.allocate(3);
  // types not given by the test get K = 0 with weight 1 (only type 1 is used by the quadruplet)
  const given = new Set(coeffs.map((c) => c.split(' ')[0]));
  for (const c of coeffs) dih.coeff(c.split(' '));
  for (const t of ['1', '2', '3']) if (!given.has(t)) dih.coeff([t, '0', '1', '0', '1.0']);
  dih.linkForceField({ pair, special });
  dih.init(undefined);
  return dih;
};

describe('dihedral charmm: forces are -grad of the total energy', () => {
  const cases: Array<{ name: string; dih: 'charmm' | 'charmmfsw'; pair: () => Pair; settings: string[] }> = [
    { name: 'charmm + lj/charmm/coul/charmm', dih: 'charmm', pair: () => makePair(new PairLJCharmmCoulCharmm(), ['8.0', '10.0']), settings: [] },
    { name: 'charmm + lj/charmmfsw/coul/charmmfsh', dih: 'charmm', pair: () => makePair(new PairLJCharmmfswCoulCharmmfsh(), ['8.0', '10.0', '9.0']), settings: [] },
    { name: 'charmmfsw + lj/charmmfsw/coul/charmmfsh', dih: 'charmmfsw', pair: () => makePair(new PairLJCharmmfswCoulCharmmfsh(), ['8.0', '10.0', '9.0']), settings: [] },
  ];
  for (const c of cases) {
    it(c.name, () => {
      const pair = c.pair();
      const dih = setupDih(c.dih, pair, ['1 0.2 1 180 1.0', '2 1.8 1 0 1.0', '3 3.1 2 180 0.5']);
      const s = system(PTS, 1);
      const base = evaluate(s, dih);
      const h = 1e-6;
      for (let a = 0; a < 4; a++) {
        for (let d = 0; d < 3; d++) {
          const k = 3 * a + d, x0 = s.x[k];
          s.x[k] = x0 + h; const ep = evaluate(s, dih).e;
          s.x[k] = x0 - h; const em = evaluate(s, dih).e;
          s.x[k] = x0;
          expect(base.f[k]).toBeCloseTo(-(ep - em) / (2 * h), 4);
        }
      }
    });
  }
});

describe('dihedral charmm: 1-4 terms', () => {
  it('weight 0 leaves only K[1 + cos(n phi - d)] and the weight scales the 1-4 terms', () => {
    const pair = makePair(new PairLJCharmmCoulCharmm(), ['8.0', '10.0']);
    const s = system(PTS, 1);
    const dih0 = setupDih('charmm', pair, ['1 0.7 1 0 0.0']);
    const r0 = evaluate(s, dih0);
    expect(r0.acc.evdwl).toBe(0);
    expect(r0.acc.ecoul).toBe(0);
    const dih1 = setupDih('charmm', pair, ['1 0.7 1 0 1.0']);
    const r1 = evaluate(s, dih1);
    expect(r1.acc.edihed).toBeCloseTo(r0.acc.edihed, 12);
    const dihH = setupDih('charmm', pair, ['1 0.7 1 0 0.5']);
    const rH = evaluate(s, dihH);
    expect(rH.acc.evdwl).toBeCloseTo(0.5 * r1.acc.evdwl, 12);
    expect(rH.acc.ecoul).toBeCloseTo(0.5 * r1.acc.ecoul, 12);
  });

  it('1-4 LJ is the plain 4 eps14 [(sigma14/r)^12 - (sigma14/r)^6] at any separation (charmm)', () => {
    const pair = makePair(new PairLJCharmmCoulCharmm(), ['8.0', '10.0']);
    const s = system([[10, 10, 10], [11.5, 10, 10], [11.5, 11.5, 10], [11.5, 11.5, 11.5]], 1);
    s.q.fill(0);
    const r = Math.sqrt(3 * 1.5 * 1.5);
    // types 1 and 2 are not given an explicit 1-2 pair_coeff: arithmetic mixing of the 14 values
    const epsMix = Math.sqrt(0.05 * 0.1), sigMix = 0.5 * (3.5 + 3.1);
    const dih = setupDih('charmm', pair, ['1 0.0 1 0 1.0']);
    const res = evaluate(s, dih);
    const sr6 = (sigMix / r) ** 6;
    expect(res.acc.evdwl).toBeCloseTo(4 * epsMix * (sr6 * sr6 - sr6), 10);
  });

  it('charmm 1-4 Coulomb is qq C / r with C = qqr2e (real) for lj/charmm pairs', () => {
    const pair = makePair(new PairLJCharmmCoulCharmm(), ['8.0', '10.0']);
    const s = system([[10, 10, 10], [11.5, 10, 10], [11.5, 11.5, 10], [11.5, 11.5, 11.5]], 1);
    const dih = setupDih('charmm', pair, ['1 0.0 1 0 1.0']);
    const res = evaluate(s, dih);
    const r = Math.sqrt(3 * 1.5 * 1.5);
    expect(res.acc.ecoul).toBeCloseTo(332.06371 * (0.25 * -0.2) / r, 6);
  });

  it('charmmfsw adds the force-switch constant to the 1-4 LJ and uses the (b-r)^2/(r b^2) Coulomb form', () => {
    const pair = makePair(new PairLJCharmmfswCoulCharmmfsh(), ['8.0', '10.0', '9.0']);
    const s = system([[10, 10, 10], [11.5, 10, 10], [11.5, 11.5, 10], [11.5, 11.5, 11.5]], 1);
    const dih = setupDih('charmmfsw', pair, ['1 0.0 1 0 1.0']);
    const res = evaluate(s, dih);
    const r = Math.sqrt(3 * 1.5 * 1.5), a = 8, b = 9;
    const sigMix = 0.5 * (3.5 + 3.1), epsMix = Math.sqrt(0.05 * 0.1), s6 = sigMix ** 6;
    const sr6 = s6 / r ** 6;
    const shift = 4 * epsMix * s6 * (1 / (a ** 3 * 10 ** 3) - s6 / (a ** 6 * 10 ** 6));
    expect(res.acc.evdwl).toBeCloseTo(4 * epsMix * (sr6 * sr6 - sr6) + shift, 10);
    const C = 332.0716; // charmmfsw pair styles use the CHARMM conversion factor
    expect(res.acc.ecoul).toBeCloseTo(C * (0.25 * -0.2) * (1 / r - 2 / b + r / (b * b)), 4);
  });
});

describe('dihedral charmm: argument and link errors', () => {
  const pair = () => makePair(new PairLJCharmmCoulCharmm(), ['8.0', '10.0']);
  it('rejects malformed dihedral_coeff', () => {
    const d = new DihedralCharmm();
    d.allocate(2);
    expect(() => d.coeff(['1', '0.2', '1', '180'])).toThrow(StyleError);
    expect(() => d.coeff(['1', '0.2', '-1', '180', '1.0'])).toThrow(StyleError);
    expect(() => d.coeff(['1', '0.2', '1.5', '180', '1.0'])).toThrow(StyleError);
    expect(() => d.coeff(['1', '0.2', '-1', '0', '1.0'])).toThrow(StyleError);
    expect(() => d.coeff(['1', '0.2', '1', '180', '1.5'])).toThrow(StyleError);
  });
  it('rejects init without a linked pair style', () => {
    const d = new DihedralCharmm();
    d.allocate(1);
    d.coeff(['1', '0.2', '1', '180', '1.0']);
    expect(() => d.init(undefined)).toThrow(/linked/);
  });
  it('rejects pair styles without the charmm 1-4 terms', () => {
    const d = new DihedralCharmm();
    d.allocate(1);
    d.coeff(['1', '0.2', '1', '180', '1.0']);
    d.linkForceField({ pair: pair(), special: { lj: [0, 0, 0], coul: [0, 0, 0] } });
    const implicit = makePair(new PairLJCharmmCoulCharmmImplicit(), ['8.0', '10.0']);
    d.linkForceField({ pair: implicit, special: { lj: [0, 0, 0], coul: [0, 0, 0] } });
    expect(() => d.init(undefined)).toThrow(StyleError);
  });
  it('charmmfsw needs an lj/charmmfsw pair', () => {
    const d = new DihedralCharmmfsw();
    d.allocate(1);
    d.coeff(['1', '0.2', '1', '180', '1.0']);
    d.linkForceField({ pair: pair(), special: { lj: [0, 0, 0], coul: [0, 0, 0] } });
    expect(() => d.init(undefined)).toThrow(/Dihedral charmmfsw is incompatible/);
  });
  it('rejects non-zero 1-4 special_bonds when a weight is non-zero, but allows them with weight 0', () => {
    const d = new DihedralCharmm();
    d.allocate(1);
    d.coeff(['1', '0.2', '1', '180', '1.0']);
    d.linkForceField({ pair: pair(), special: { lj: [0, 0, 0.5], coul: [0, 0, 0] } });
    expect(() => d.init(undefined)).toThrow(/special_bonds charmm/);
    const z = new DihedralCharmm();
    z.allocate(1);
    z.coeff(['1', '0.2', '1', '180', '0.0']);
    z.linkForceField({ pair: pair(), special: { lj: [0, 0, 0.5], coul: [0, 0, 0] } });
    expect(() => z.init(undefined)).not.toThrow();
  });
});
