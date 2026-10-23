import { describe, expect, it } from 'vitest';
import { BondFeneNm } from '../src/engine-gpl/xmolecule/bond_fene_nm';
import { BondHarmonicRestrain } from '../src/engine-gpl/xmolecule/bond_harmonic_restrain';
import { BondQuarticExp } from '../src/engine-gpl/xmolecule/bond_quartic_exp';
import { AngleCosineShiftExp } from '../src/engine-gpl/xmolecule/angle_cosine_shift_exp';
import { AngleMWLC } from '../src/engine-gpl/xmolecule/angle_mwlc';
import { DihedralSpherical } from '../src/engine-gpl/xmolecule/dihedral_spherical';
import { ImproperRing, powint } from '../src/engine-gpl/xmolecule/improper_ring';
import { BONDS, ANGLES, DIHEDRALS, IMPROPERS } from '../src/engine-gpl/registry/xmolecule';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, StyleError, type Bonded, type BondedCompute, type StyleContext } from '../src/engine/force/types';

/*
 * The EXTRA-MOLECULE bonded styles of src/engine-gpl/xmolecule: force = -dE/dx by central differences for every
 * style, the argument errors and the native messages, the coefficient lines of write_data (as native prints them),
 * and the registry names. The energies are the documented expressions (docs.lammps.org pages cited in the sources).
 */

const CTX_NOSYS: StyleContext = { s: null, readFile: () => '', log: () => {} };

type Kind = 'bond' | 'angle' | 'dihedral' | 'improper';
interface Rig {
  s: ReturnType<typeof emptyState>;
  geom: Geometry;
  map: Int32Array;
}

const L = 30;
const rig = (pts: number[][]): Rig => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [L, L, L] }, 2, 'molecular');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [L, L, L] }));
  return { s, geom, map: buildAtomMap(s) };
};

/** A rig with one term of the given kind over atoms 1..n (topology ids as listed). */
const termRig = (pts: number[][], kind: Kind, ids: number[], type = 1): Rig => {
  const r = rig(pts);
  const list = { bond: r.s.topo.bonds, angle: r.s.topo.angles, dihedral: r.s.topo.dihedrals, improper: r.s.topo.impropers }[kind];
  pushTopo(list, type, ids);
  return r;
};

const makeBc = (r: Rig, kind: Kind): BondedCompute => {
  const acc = newAccum();
  r.s.f.fill(0);
  const virial = { bond: acc.vbond, angle: acc.vangle, dihedral: acc.vdihed, improper: acc.vimp }[kind];
  return { s: r.s, geom: r.geom, map: r.map, f: r.s.f, acc, virial, eatom: null, vatom: null };
};

const energyOf = (bc: BondedCompute, kind: Kind): number => ({
  bond: bc.acc.ebond, angle: bc.acc.eangle, dihedral: bc.acc.edihed, improper: bc.acc.eimp,
})[kind];

/** Central differences: the largest |f - (-dE/dx)| over every coordinate of the term's atoms. */
const fdError = (style: Bonded, kind: Kind, r: Rig, ids: number[]): number => {
  const bc = makeBc(r, kind);
  style.compute(bc);
  const f0 = r.s.f.slice();
  const h = 1e-6;
  let worst = 0;
  for (const id of ids) {
    const a = r.map[id];
    for (let d = 0; d < 3; d++) {
      const x0 = r.s.x[3 * a + d];
      r.s.x[3 * a + d] = x0 + h;
      const bp = makeBc(r, kind);
      style.compute(bp);
      const ep = energyOf(bp, kind);
      r.s.x[3 * a + d] = x0 - h;
      const bm = makeBc(r, kind);
      style.compute(bm);
      const em = energyOf(bm, kind);
      r.s.x[3 * a + d] = x0;
      const num = -(ep - em) / (2 * h);
      worst = Math.max(worst, Math.abs(f0[3 * a + d] - num) / Math.max(1, Math.abs(num)));
    }
  }
  return worst;
};

const ctxFor = (r: Rig): StyleContext => ({ ...CTX_NOSYS, s: r.s });

/** Two bonds of a chain of 4 atoms, distorted, with a periodic-free geometry. */
const CHAIN4 = [[5, 5, 5], [6.1, 5.4, 5.2], [7.0, 6.3, 5.1], [8.3, 6.6, 6.2]];
/** Three atoms, bent, for the angles. */
const BENT3 = [[5, 5, 5], [6.2, 5.3, 5.1], [6.9, 6.4, 5.6]];
/** The ring improper: atom 2 has three neighbours 1, 3, 4. */
const STAR4 = [[5.1, 5.0, 4.9], [5.0, 5.0, 5.0], [5.8, 5.9, 5.2], [4.6, 5.7, 5.5]];

describe('bond_style fene/nm', () => {
  it('force = -dE/dr and the native coefficient count', () => {
    const st = new BondFeneNm();
    st.allocate(1);
    st.coeff(['1', '30.0', '2.0', '1.0', '1.6', '12', '6'], CTX_NOSYS);
    st.init(CTX_NOSYS);
    // r = 1.4 (inside sigma = 1.6, the n-m term active) and r = 1.9 (outside)
    expect(fdError(st, 'bond', termRig([[5, 5, 5], [6.4, 5.1, 5]], 'bond', [1, 2]), [1, 2])).toBeLessThan(2e-6);
    expect(fdError(st, 'bond', termRig([[5, 5, 5], [6.9, 5.2, 5]], 'bond', [1, 2]), [1, 2])).toBeLessThan(2e-6);
  });

  it('refuses a coefficient line that is not 7 words (native message)', () => {
    const st = new BondFeneNm();
    st.allocate(1);
    expect(() => st.coeff(['1', '30.0', '2.0', '1.0', '1.6', '12'], CTX_NOSYS)).toThrow(new StyleError('Incorrect args for bond coefficients'));
  });

  it('equilibrium is 0.97 sigma and write_data prints K R0 eps sigma n m', () => {
    const st = new BondFeneNm();
    st.allocate(1);
    st.coeff(['1', '30.0', '2.0', '1.0', '1.6', '12', '6'], CTX_NOSYS);
    st.init(CTX_NOSYS);
    expect(st.equilibrium(1)).toBeCloseTo(0.97 * 1.6, 15);
    expect(st.dataCoeffs()).toEqual(['1 30 2 1 1.6 12 6']);
  });
});

describe('bond_style harmonic/restrain', () => {
  it('stores the positions at the first init and keeps them afterwards', () => {
    const r = termRig(CHAIN4.slice(0, 2), 'bond', [1, 2]);
    const st = new BondHarmonicRestrain();
    st.allocate(1);
    st.coeff(['1', '50.0'], CTX_NOSYS);
    st.init(ctxFor(r));
    // the bond at its initial length has no energy; moving atom 2 gives K (r - r0)^2
    const bc0 = makeBc(r, 'bond');
    st.compute(bc0);
    expect(bc0.acc.ebond).toBeCloseTo(0, 14);
    r.s.x[3 + 0] += 0.2;
    const bc1 = makeBc(r, 'bond');
    st.compute(bc1);
    expect(bc1.acc.ebond).toBeGreaterThan(0);
    // a second init (the second run) keeps the stored positions
    st.init(ctxFor(r));
    const bc2 = makeBc(r, 'bond');
    st.compute(bc2);
    expect(bc2.acc.ebond).toBeCloseTo(bc1.acc.ebond, 14);
  });

  it('force = -dE/dx after the atoms move', () => {
    const r = termRig(CHAIN4.slice(0, 2), 'bond', [1, 2]);
    const st = new BondHarmonicRestrain();
    st.allocate(1);
    st.coeff(['1', '50.0'], CTX_NOSYS);
    st.init(ctxFor(r));
    r.s.x[3] += 0.25;
    r.s.x[4] -= 0.1;
    expect(fdError(st, 'bond', r, [1, 2])).toBeLessThan(2e-5);
  });

  it('writes no Bond Coeffs section (native writedata = 0) and refuses two-word-short coeff lines', () => {
    const st = new BondHarmonicRestrain();
    st.allocate(1);
    expect(() => st.coeff(['1'], CTX_NOSYS)).toThrow(new StyleError('Incorrect args for bond coefficients'));
    st.coeff(['1', '50.0'], CTX_NOSYS);
    expect(st.dataCoeffs()).toBeNull();
  });
});

describe('bond_style quartic/exp', () => {
  it('force = -dE/dr with and without the exponential', () => {
    const st = new BondQuarticExp();
    st.allocate(2);
    st.coeff(['1', '1.1', '50.0', '-10.0', '5.0', '2.0', '0.3'], CTX_NOSYS);
    st.coeff(['2', '1.2', '40.0', '-5.0', '2.0', '0.0', '0.5'], CTX_NOSYS);
    st.init(CTX_NOSYS);
    expect(fdError(st, 'bond', termRig([[5, 5, 5], [6.3, 5.2, 5.1]], 'bond', [1, 2], 1), [1, 2])).toBeLessThan(2e-6);
    expect(fdError(st, 'bond', termRig([[5, 5, 5], [6.3, 5.2, 5.1]], 'bond', [1, 2], 2), [1, 2])).toBeLessThan(2e-6);
    expect(st.equilibrium(2)).toBe(1.2);
    expect(st.dataCoeffs()).toEqual(['1 1.1 50 -10 5 2 0.3', '2 1.2 40 -5 2 0 0.5']);
  });
});

describe('angle_style cosine/shift/exp', () => {
  it('force = -dE/dx on the exponential and on the expansion branch', () => {
    const st = new AngleCosineShiftExp();
    st.allocate(2);
    st.coeff(['1', '10.0', '110.0', '2.0'], CTX_NOSYS);
    st.coeff(['2', '5.0', '100.0', '0.0005'], CTX_NOSYS);
    st.init(CTX_NOSYS);
    for (const t of [1, 2]) {
      expect(fdError(st, 'angle', termRig(BENT3, 'angle', [1, 2, 3], t), [1, 2, 3])).toBeLessThan(2e-6);
    }
  });

  it('write_data prints Umin, theta0 in degrees and a; equilibrium in degrees and radians', () => {
    const st = new AngleCosineShiftExp();
    st.allocate(1);
    st.coeff(['1', '10.0', '110.0', '2.0'], CTX_NOSYS);
    st.init(CTX_NOSYS);
    const line = st.dataCoeffs()[0].split(' ');
    expect(line[0]).toBe('1');
    expect(line[1]).toBe('10');
    expect(Number(line[2])).toBeCloseTo(110, 10);
    expect(line[3]).toBe('2');
    expect(st.equilibrium(1)).toBeCloseTo(110, 10);
    expect(st.equilibriumRadians(1)).toBeCloseTo((110 * Math.PI) / 180, 14);
  });

  it('refuses a coefficient line that is not 4 words', () => {
    const st = new AngleCosineShiftExp();
    st.allocate(1);
    expect(() => st.coeff(['1', '10.0', '110.0'], CTX_NOSYS)).toThrow(new StyleError('Incorrect args for angle coefficients'));
  });
});

describe('angle_style mwlc', () => {
  it('force = -dE/dx', () => {
    const st = new AngleMWLC();
    st.allocate(1);
    st.coeff(['1', '2.0', '1.0', '1.5', '300.0'], CTX_NOSYS);
    st.init(CTX_NOSYS);
    const r = termRig(BENT3, 'angle', [1, 2, 3]);
    expect(fdError(st, 'angle', r, [1, 2, 3])).toBeLessThan(2e-6);
  });

  it('equilibrium is pi and write_data prints k1 k2 mu T', () => {
    const st = new AngleMWLC();
    st.allocate(1);
    st.coeff(['1', '2.0', '1.0', '1.5', '300.0'], CTX_NOSYS);
    expect(st.equilibrium(1)).toBe(180);
    expect(st.equilibriumRadians(1)).toBe(Math.PI);
    expect(st.dataCoeffs()).toEqual(['1 2 1 1.5 300']);
  });
});

describe('dihedral_style spherical', () => {
  const coeffs = (t: string[]): string[] => t;
  it('force = -dE/dx for a two-term and a one-term type', () => {
    const st = new DihedralSpherical();
    st.allocate(2);
    st.coeff(coeffs(['1', '2', '286.1', '1', '124', '1', '1', '90.0', '0', '1', '90.0', '0', '-50.0', '2', '60', '0.5', '2', '45', '1', '1', '30', '0.2']), CTX_NOSYS);
    st.coeff(coeffs(['2', '1', '69.3', '1', '93.9', '1', '1', '90', '0', '1', '90', '0']), CTX_NOSYS);
    st.init(CTX_NOSYS);
    const pts = [[5, 5, 5], [6.1, 5.4, 5.2], [7.0, 6.3, 5.1], [7.9, 6.2, 6.4]];
    for (const t of [1, 2]) {
      expect(fdError(st, 'dihedral', termRig(pts, 'dihedral', [1, 2, 3, 4], t), [1, 2, 3, 4])).toBeLessThan(2e-5);
    }
  });

  it('write_data lines: type, nterms, then 10 values per term each followed by a space', () => {
    const st = new DihedralSpherical();
    st.allocate(1);
    st.coeff(['1', '1', '286.1', '1', '124', '1', '1', '90.0', '0', '1', '90.0', '0'], CTX_NOSYS);
    st.init(CTX_NOSYS);
    expect(st.dataCoeffs()).toEqual(['1 1 286.1 1 124 1 1 90 0 1 90 0 ']);
  });

  it('refuses the wrong number of words (native message) and a non-positive nterms', () => {
    const st = new DihedralSpherical();
    st.allocate(1);
    // As native LAMMPS does (src/EXTRA-MOLECULE/dihedral_spherical.cpp coeff at b332f28): too many words is the count
    // error, too few reads past the end with utils::numeric
    expect(() => st.coeff(['1', '1', ...Array(11).fill('1')], CTX_NOSYS)).toThrow(new StyleError('Incorrect number of arguments for dihedral coefficients'));
    expect(() => st.coeff(['1', '1', '1', '1'], CTX_NOSYS)).toThrow(new StyleError('Expected floating point parameter instead of NULL or empty string in input script or data file'));
    expect(() => st.coeff(['1', '0', '1', '1'], CTX_NOSYS)).toThrow(new StyleError('Incorrect number of terms arg for dihedral coefficients'));
    expect(() => st.coeff(['1', '1'], CTX_NOSYS)).toThrow(new StyleError('Incorrect args for dihedral coefficients'));
  });
});

describe('improper_style ring', () => {
  it('powint is repeated multiplication with the native special cases', () => {
    expect(powint(0.5, 6)).toBeCloseTo(0.015625, 15);
    expect(powint(0, 5)).toBe(0);
    expect(powint(3, 0)).toBe(1);
    expect(powint(2, -2)).toBe(0.25);
  });

  it('force = -dE/dx for the three bend angles of the quadruplet', () => {
    const st = new ImproperRing();
    st.allocate(1);
    st.coeff(['1', '80.0', '70.5'], CTX_NOSYS);
    st.init(CTX_NOSYS);
    const r = termRig(STAR4, 'improper', [1, 2, 3, 4]);
    expect(fdError(st, 'improper', r, [1, 2, 3, 4])).toBeLessThan(2e-5);
  });

  it('refuses a coefficient line that is not 3 words (native message); no Impropers Coeffs section', () => {
    const st = new ImproperRing();
    st.allocate(1);
    expect(() => st.coeff(['1', '80.0'], CTX_NOSYS)).toThrow(new StyleError('Incorrect args for improper coefficients'));
    st.coeff(['1', '80.0', '70.5'], CTX_NOSYS);
    expect(st.dataCoeffs()).toBeNull();
  });
});

describe('registry', () => {
  it('names each style under its kind', () => {
    expect(Object.keys(BONDS).sort()).toEqual(['fene/nm', 'harmonic/restrain', 'quartic/exp']);
    expect(Object.keys(ANGLES).sort()).toEqual(['cosine/shift/exp', 'mwlc']);
    expect(Object.keys(DIHEDRALS)).toEqual(['spherical']);
    expect(Object.keys(IMPROPERS)).toEqual(['ring']);
  });
});
