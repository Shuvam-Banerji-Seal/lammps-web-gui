import { describe, expect, it } from 'vitest';
import { AngleClass2P6, AngleClass2 } from '../src/engine/force/angle/class2';
import { ImproperInversionHarmonic } from '../src/engine/force/improper/inversion';
import { AngleCosineBuck6d } from '../src/engine/force/angle/buck6d';
import { PairBuck6dCoulGaussDsf } from '../src/engine/force/pair/buck6d_gauss';
import { KSpacePPPM, KSpacePPPMCG } from '../src/engine/force/kspace/pppm';
import { DihedralCharmm } from '../src/engine/force/dihedral/charmm';
import { PairLJCharmmCoulCharmm } from '../src/engine/force/pair/charmm';
import { PairHybrid } from '../src/engine/force/pair/hybrid';
import { StyleError, type Pair } from '../src/engine/force/types';
import { ANGLES as BONDED50_ANGLES, IMPROPERS as BONDED50_IMPROPERS } from '../src/engine/registry/bonded50';
import { LAZY_GROUPS } from '../src/engine/lazyStyles';

/*
 * Wave 50 bonded and k-space styles: class2/p6, inversion/harmonic, cosine/buck6d, pppm/cg and the charmm 1-4 terms
 * under hybrid/overlay. Parsing and refusals here; the numbers are checked against native LAMMPS by the w50bd_* oracle
 * cases (tests/oracle/w50bd_*.in).
 */

const ctx = (): never => ({ s: undefined, readFile: () => '', log: () => {} }) as never;

describe('angle_style class2/p6', () => {
  it('takes theta0 and K2..K6 on the plain line and writes all six back', () => {
    const a = new AngleClass2P6();
    expect(a.name).toBe('class2/p6');
    a.allocate(2);
    a.coeff(['1', '110.0', '40.0', '-5.0', '2.0', '-0.5', '0.1']);
    a.coeff(['1', 'bb', '0.0', '1.0', '1.0']);
    a.coeff(['1', 'ba', '0.0', '0.0', '1.0', '1.0']);
    a.coeff(['2', '100.0', '35.0', '3.0', '-1.0', '0.4', '-0.05']);
    a.coeff(['2', 'bb', '1.0', '1.0', '1.2']);
    a.coeff(['2', 'ba', '0.5', '-0.3', '1.0', '1.2']);
    a.init(undefined);
    const line = a.dataCoeffs()[0].split(' ').map(Number);
    expect(line.length).toBe(7);
    expect(line[1]).toBeCloseTo(110, 12);
    expect(line.slice(2)).toEqual([40, -5, 2, -0.5, 0.1]);
  });

  it('rejects the five-value class2 line for class2/p6 and the six-value line for class2', () => {
    const p6 = new AngleClass2P6();
    p6.allocate(1);
    expect(() => p6.coeff(['1', '110.0', '40.0', '-5.0', '2.0'])).toThrow(StyleError);
    const c2 = new AngleClass2();
    c2.allocate(1);
    expect(() => c2.coeff(['1', '110.0', '40.0', '-5.0', '2.0', '0.1'])).toThrow(StyleError);
  });

  it('still needs the bb and ba lines (native: All angle coeffs are not set)', () => {
    const p6 = new AngleClass2P6();
    p6.allocate(1);
    p6.coeff(['1', '110.0', '40.0', '-5.0', '2.0', '-0.5', '0.1']);
    expect(() => p6.init(undefined)).toThrow(StyleError);
  });

  it('is registered in the lazy bonded50 group', () => {
    expect(Object.keys(BONDED50_ANGLES)).toContain('class2/p6');
    expect(LAZY_GROUPS.find((g) => g.id === 'bonded50')?.names.ANGLES).toContain('class2/p6');
  });
});

describe('improper_style inversion/harmonic', () => {
  it('parses K and omega0 (degrees) and writes them back in degrees', () => {
    const d = new ImproperInversionHarmonic();
    d.allocate(2);
    d.coeff(['1', '10.0', '20.0']);
    d.coeff(['2', '18.776340', '0.0']);
    expect(d.dataCoeffs()).toEqual(['1 10 20', '2 18.7763 0']);
  });

  it('rejects a wrong number of coefficients', () => {
    const d = new ImproperInversionHarmonic();
    d.allocate(1);
    expect(() => d.coeff(['1', '10.0'])).toThrow(StyleError);
  });

  it('is registered in the lazy bonded50 group', () => {
    expect(Object.keys(BONDED50_IMPROPERS)).toContain('inversion/harmonic');
  });
});

describe('angle_style cosine/buck6d', () => {
  it('parses K, n and theta0 (degrees)', () => {
    const a = new AngleCosineBuck6d();
    a.allocate(1);
    a.coeff(['1', '1.978350', '4', '180.0']);
    expect(a.dataCoeffs()).toEqual(['1 1.97835 4 180']);
    expect(() => a.coeff(['1', '1.9', '4.5', '180.0'])).toThrow(StyleError);
  });

  it('needs a buck6d pair style, named in the error', () => {
    const a = new AngleCosineBuck6d();
    a.allocate(1);
    a.coeff(['1', '1.978350', '4', '180.0']);
    const lj = new PairLJCharmmCoulCharmm();
    a.linkForceField({ pair: lj as unknown as Pair, special: { lj: [1, 0, 0], coul: [1, 1, 1] } });
    expect(() => a.init(undefined)).toThrow(/needs pair_style buck6d\/coul\/gauss\/dsf or buck6d\/coul\/gauss\/long \(got lj\/charmm\/coul\/charmm\)/);
  });

  it('refuses a non-zero 1-3 special_bonds lj weight (the buck6d 1-3 term would be counted twice)', () => {
    const a = new AngleCosineBuck6d();
    a.allocate(1);
    a.coeff(['1', '1.978350', '4', '180.0']);
    a.linkForceField({ pair: new PairBuck6dCoulGaussDsf() as unknown as Pair, special: { lj: [1, 0.5, 0], coul: [1, 1, 1] } });
    expect(() => a.init(undefined)).toThrow(/1-3 weight 0.0/);
    a.linkForceField({ pair: new PairBuck6dCoulGaussDsf() as unknown as Pair, special: { lj: [1, 0, 0], coul: [1, 1, 1] } });
    expect(() => a.init(undefined)).not.toThrow();
  });
});

describe('kspace_style pppm/cg', () => {
  it('takes the accuracy and an optional smallq (default 1e-5)', () => {
    const k = new KSpacePPPMCG();
    k.settings(['1.0e-5']);
    expect(k.smallq).toBe(1e-5);
    k.settings(['1.0e-5', '0.1']);
    expect(k.smallq).toBe(0.1);
    expect(() => k.settings([])).toThrow(StyleError);
    expect(() => k.settings(['1.0e-5', '-1'])).toThrow(StyleError);
  });

  it('keeps charges above smallq on the mesh and zeroes the rest (|q| <= smallq is excluded)', () => {
    const k = new KSpacePPPMCG();
    k.settings(['1.0e-5', '0.1']);
    const g = (k as unknown as { gridCharges(s: { n: number; q: Float64Array }): ArrayLike<number> }).gridCharges({ n: 4, q: Float64Array.from([0.4, 3e-6, -0.1, -0.2]) });
    expect(Array.from(g)).toEqual([0.4, 0, 0, -0.2]);
  });

  it('refuses kspace_modify slab (the slab correction was not measured with the optimisation)', () => {
    const k = new KSpacePPPMCG();
    k.settings(['1.0e-5']);
    expect(() => k.modify('slab', ['3.0'])).toThrow(/slab/);
  });

  it('is a pppm solver (plain pppm keeps every charge on the mesh)', () => {
    const p = new KSpacePPPM();
    expect((p as unknown as { gridCharges(s: { q: Float64Array }): ArrayLike<number> }).gridCharges({ q: Float64Array.from([0, 1e-9]) })).toEqual(Float64Array.from([0, 1e-9]));
  });
});

describe('dihedral_style charmm with pair_style hybrid/overlay', () => {
  const ljSub = () => {
    const lj = new PairLJCharmmCoulCharmm();
    lj.settings(['8.0', '10.0']);
    lj.allocate(3);
    lj.coeff(['1', '1', '0.1', '3.0', '0.05', '3.5']);
    lj.coeff(['2', '2', '0.15', '3.2', '0.1', '3.1']);
    lj.coeff(['3', '3', '0.12', '3.1', '0.08', '3.3']);
    lj.coeff(['1', '2', '0.12', '3.1', '0.07', '3.3']);
    lj.coeff(['1', '3', '0.11', '3.05', '0.06', '3.4']);
    lj.coeff(['2', '3', '0.13', '3.15', '0.09', '3.2']);
    return lj;
  };
  const stub = (name: string) => ({ name }) as unknown as Pair;
  const hybrid = (subs: { style: Pair; name: string }[]) => {
    const h = new PairHybrid('hybrid/overlay', {} as never);
    h.subs = subs.map((s, k) => ({ ...s, instance: 1, scale: 1, specialLJ: null, specialCoul: null, used: true, half: null, full: null, idx: k }) as never);
    return h;
  };
  const dihedral = () => {
    const d = new DihedralCharmm();
    d.allocate(1);
    d.coeff(['1', '0.2', '1', '180', '1.0']);
    return d;
  };

  it('uses the lj/charmm sub-style with momb sub-styles (measured combination)', () => {
    const lj = ljSub();
    const h = hybrid([{ style: stub('momb'), name: 'momb' }, { style: lj, name: 'lj/charmm/coul/charmm' }]);
    const d = dihedral();
    d.linkForceField({ pair: h, special: { lj: [0, 0, 0], coul: [0, 0, 0] } });
    expect(() => d.init(ctx())).not.toThrow();
  });

  it('accepts morse and eam/fs next to the lj/charmm sub-style (measured: no 1-4 term from them)', () => {
    for (const other of ['morse', 'eam/fs']) {
      const d = dihedral();
      d.linkForceField({ pair: hybrid([{ style: ljSub(), name: 'lj/charmm/coul/charmm' }, { style: stub(other), name: other }]) as never, special: { lj: [0, 0, 0], coul: [0, 0, 0] } });
      expect(() => d.init(ctx())).not.toThrow();
    }
  });

  it('refuses other sub-styles next to the lj/charmm one, naming the pair style', () => {
    const d = dihedral();
    d.linkForceField({ pair: hybrid([{ style: ljSub(), name: 'lj/charmm/coul/charmm' }, { style: stub('lj/cut'), name: 'lj/cut' }]) as never, special: { lj: [0, 0, 0], coul: [0, 0, 0] } });
    expect(() => d.init(ctx())).toThrow(/not measured with pair style hybrid\/overlay/);
  });

  it('refuses two lj/charmm sub-styles', () => {
    const d = dihedral();
    d.linkForceField({ pair: hybrid([{ style: ljSub(), name: 'lj/charmm/coul/charmm' }, { style: ljSub(), name: 'lj/charmm/coul/charmm' }]) as never, special: { lj: [0, 0, 0], coul: [0, 0, 0] } });
    expect(() => d.init(ctx())).toThrow(/not measured with pair style hybrid\/overlay/);
  });

  it('refuses charmmfsw under hybrid/overlay (not measured)', () => {
    const d = new (class extends DihedralCharmm { override readonly name = 'charmmfsw'; protected override readonly forceSwitched = true; })();
    d.allocate(1);
    d.coeff(['1', '0.2', '1', '180', '1.0']);
    d.linkForceField({ pair: hybrid([{ style: stub('momb'), name: 'momb' }]) as never, special: { lj: [0, 0, 0], coul: [0, 0, 0] } });
    expect(() => d.init(ctx())).toThrow(StyleError);
  });
});
