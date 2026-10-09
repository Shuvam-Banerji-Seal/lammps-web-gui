import { describe, expect, it } from 'vitest';
import {
  PairMeam,
  eroseE,
  eroseDeriv,
  parseMeamParams,
  type MeamElement,
} from '../src/engine/force/pair/meam';
import {
  alloyAtomEnergyGrad,
  makeAlloyModel,
  pairErose,
  type AlloyElement,
  type AlloyModel,
  type AlloyNeighbor,
  type AlloyPair,
} from '../src/engine/force/pair/meam_alloy';
import { StyleError } from '../src/engine/force/types';

/*
 * Multi-element MEAM, meam29 scope: erose_form = 2 with per-pair attrac/repuls, and the diamond
 * (zincblende) alloy reference lattce(I,J) = dia. The engine models are checked against native
 * LAMMPS (black box) energies measured on non-periodic clusters with the synthetic entries of
 * tests/oracle/w29meama_erose2_* and tests/oracle/w29meama_dia_*; the periodic crystal parity is in
 * tests/engineOracle.test.ts (w29meama_erose2, w29meama_dia). All numbers below are native output.
 */

/** Non-periodic cluster energy and F = -grad E of an alloy model (box large enough that rc cuts it). */
const clusterAlloy = (model: AlloyModel, types: number[], X: number[]) => {
  const n = types.length;
  const F = new Float64Array(3 * n);
  let E = 0;
  for (let i = 0; i < n; i++) {
    const nb: AlloyNeighbor[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = X[3 * j] - X[3 * i], dy = X[3 * j + 1] - X[3 * i + 1], dz = X[3 * j + 2] - X[3 * i + 2];
      const r = Math.hypot(dx, dy, dz);
      if (r >= model.opts.rc) continue;
      nb.push({ e: types[j], j, dx, dy, dz, r });
    }
    const g = new Float64Array(3 * nb.length);
    E += alloyAtomEnergyGrad(model, types[i], nb, g);
    nb.forEach((p, m) => {
      for (let c = 0; c < 3; c++) {
        F[3 * p.j + c] -= g[3 * m + c];
        F[3 * i + c] += g[3 * m + c];
      }
    });
  }
  return { E, F };
};

const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

describe('multi-element MEAM: erose_form = 2 with per-pair attrac/repuls', () => {
  it('pairErose reproduces the single-element eroseE/eroseDeriv for forms 0, 1 and 2', () => {
    const el: MeamElement = {
      z: 8, re: 2.7, alpha: 5.0, Ec: 8.9, A: 1, beta: [2, 1.5, 2.5, 4], t: [1, 2, 1, 1], ibar: 0, lat: 'bcc',
    };
    for (const form of [0, 1, 2]) {
      const e = { form, attrac: 0.4, repuls: 0.9 };
      const pr: AlloyPair = { Ec: 8.9, re: 2.7, alpha: 5.0, lat: 'self', attrac: 0.4, repuls: 0.9 };
      for (const r of [2.0, 2.7, 3.2, 3.95]) {
        const a = pairErose(pr, form, r);
        expect(a.E).toBeCloseTo(eroseE({ ...el, erose: e }, r), 12);
        expect(a.dE).toBeCloseTo(eroseDeriv({ ...el, erose: e }, r), 12);
      }
    }
  });

  const A_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.5, alpha: 5.0, Ec: 3.5, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0] };
  const B_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.7, alpha: 4.0, Ec: 4.0, A: 1, beta: [3.0, 1.0, 2.0, 3.0], t: [1, 1.5, 0.5, 2.0] };
  const EROSE2_MODEL: AlloyModel = makeAlloyModel(
    [A_EL, B_EL],
    [
      [
        { Ec: 3.5, re: 2.5, alpha: 5.0, lat: 'self', attrac: 0.4, repuls: 0.9 },
        { Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b1', attrac: 0.3, repuls: 0.6 },
      ],
      [
        { Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b1', attrac: 0.3, repuls: 0.6 },
        { Ec: 4.0, re: 2.7, alpha: 4.0, lat: 'self', attrac: 0.2, repuls: 0.5 },
      ],
    ] as AlloyPair[][],
    { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8, eroseForm: 2 },
    true,
  );

  // Measured with native LAMMPS (black box): dimers with w29meama_erose2_lib.meam + w29meama_erose2.meam.
  const NATIVE: Array<[number[], number, number]> = [
    [[0, 1], 1.8, 7.085753163255], [[0, 1], 2.0, -0.055971213385], [[0, 1], 2.4, -3.210705990202], [[0, 1], 3.0, -2.711249732667], [[0, 1], 3.5, -2.291520174189],
    [[0, 0], 1.8, 3.656328263965], [[0, 0], 2.0, -1.402657276419], [[0, 0], 2.4, -2.846419350895], [[0, 0], 3.0, -2.168008695256], [[0, 0], 3.5, -1.767502316916],
    [[1, 1], 1.8, -0.914612347745], [[1, 1], 2.0, -2.917303688195], [[1, 1], 2.4, -3.533576507627], [[1, 1], 3.0, -2.752103176795], [[1, 1], 3.5, -2.257816382071],
  ];
  for (const [types, r, pe] of NATIVE) {
    it(`dimer ${types[0]}-${types[1]} at ${r} A matches native pe to 1e-9 relative`, () => {
      const E = clusterAlloy(EROSE2_MODEL, types, [0, 0, 0, r, 0, 0]).E;
      expect(rel(E, pe)).toBeLessThan(1e-9);
    });
  }

  it('mixed trimer and cluster match native pe to 1e-9 relative', () => {
    // Measured with native LAMMPS (black box): A-B-A at (0,0,0),(2.4,0,0),(1.2,2.0,0.3) is -6.080252753316 eV;
    // A-B-A-A at (0,0,0),(2.4,0.1,0),(1.2,2.0,0.3),(1.0,-0.2,2.2) is -8.342851416058 eV.
    expect(rel(clusterAlloy(EROSE2_MODEL, [0, 1, 0], [0, 0, 0, 2.4, 0, 0, 1.2, 2.0, 0.3]).E, -6.080252753316)).toBeLessThan(1e-9);
    expect(rel(clusterAlloy(EROSE2_MODEL, [0, 1, 0, 0], [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2]).E, -8.342851416058)).toBeLessThan(1e-9);
  });

  it('forces equal -grad E to 1e-6 relative (mixed cluster)', () => {
    const X = [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2];
    const types = [0, 1, 0, 1];
    const { F } = clusterAlloy(EROSE2_MODEL, types, X);
    const h = 1e-5;
    let worst = 0;
    for (let q = 0; q < X.length; q++) {
      const xp = X.slice(), xm = X.slice();
      xp[q] += h;
      xm[q] -= h;
      const fd = -(clusterAlloy(EROSE2_MODEL, types, xp).E - clusterAlloy(EROSE2_MODEL, types, xm).E) / (2 * h);
      worst = Math.max(worst, Math.abs(F[q] - fd) / Math.max(Math.abs(fd), 1e-2));
    }
    expect(worst).toBeLessThan(1e-6);
  });
});

describe('multi-element MEAM: diamond (zincblende) alloy, lattce(I,J) = dia', () => {
  const RE_A = (5.4 * Math.sqrt(3)) / 4;
  const RE_B = (3.4 * Math.sqrt(3)) / 4;
  const A_EL: AlloyElement = { z: 4, lat: 'dia', re: RE_A, alpha: 4.5, Ec: 4.0, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0] };
  const B_EL: AlloyElement = { z: 4, lat: 'dia', re: RE_B, alpha: 4.0, Ec: 4.5, A: 1, beta: [3.0, 1.0, 2.0, 3.0], t: [1, 1.5, 0.5, 2.0] };
  const DIA_MODEL: AlloyModel = makeAlloyModel(
    [A_EL, B_EL],
    [
      [
        { Ec: 4.0, re: RE_A, alpha: 4.5, lat: 'self' },
        { Ec: 4.2, re: 2.0, alpha: 4.2, lat: 'dia' },
      ],
      [
        { Ec: 4.2, re: 2.0, alpha: 4.2, lat: 'dia' },
        { Ec: 4.5, re: RE_B, alpha: 4.0, lat: 'self' },
      ],
    ] as AlloyPair[][],
    { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8 },
    true,
  );

  // Measured with native LAMMPS (black box): dimers with w29meama_dia_lib.meam + w29meama_dia.meam.
  const NATIVE: Array<[number[], number, number]> = [
    [[0, 1], 1.8, -5.058657065583], [[0, 1], 2.0, -5.046022752973], [[0, 1], 2.4, -4.292944624419], [[0, 1], 3.0, -3.058933085371], [[0, 1], 3.5, -2.328643009059],
    [[0, 0], 1.8, -3.411611730081], [[0, 0], 2.0, -4.635928335455], [[0, 0], 2.4, -4.806102367449], [[0, 0], 3.0, -3.564721609794], [[0, 0], 3.5, -2.627769469148],
    [[1, 1], 1.8, -4.383094642469], [[1, 1], 2.0, -3.760535863967], [[1, 1], 2.4, -2.974789726846], [[1, 1], 3.0, -2.370346499956], [[1, 1], 3.5, -2.008359258368],
  ];
  for (const [types, r, pe] of NATIVE) {
    it(`dimer ${types[0]}-${types[1]} at ${r} A matches native pe to 1e-9 relative`, () => {
      const E = clusterAlloy(DIA_MODEL, types, [0, 0, 0, r, 0, 0]).E;
      expect(rel(E, pe)).toBeLessThan(1e-9);
    });
  }

  it('mixed trimer and cluster match native pe to 1e-9 relative', () => {
    // Measured with native LAMMPS (black box): A-B-A at (0,0,0),(2.4,0,0),(1.2,2.0,0.3) is -7.758690128429 eV;
    // A-B-A-A at (0,0,0),(2.4,0.1,0),(1.2,2.0,0.3),(1.0,-0.2,2.2) is -9.910487748140 eV.
    expect(rel(clusterAlloy(DIA_MODEL, [0, 1, 0], [0, 0, 0, 2.4, 0, 0, 1.2, 2.0, 0.3]).E, -7.758690128429)).toBeLessThan(1e-9);
    expect(rel(clusterAlloy(DIA_MODEL, [0, 1, 0, 0], [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2]).E, -9.910487748140)).toBeLessThan(1e-9);
  });

  it('forces equal -grad E to 1e-6 relative (mixed cluster)', () => {
    const X = [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2];
    const types = [0, 1, 0, 1];
    const { F } = clusterAlloy(DIA_MODEL, types, X);
    const h = 1e-5;
    let worst = 0;
    for (let q = 0; q < X.length; q++) {
      const xp = X.slice(), xm = X.slice();
      xp[q] += h;
      xm[q] -= h;
      const fd = -(clusterAlloy(DIA_MODEL, types, xp).E - clusterAlloy(DIA_MODEL, types, xm).E) / (2 * h);
      worst = Math.max(worst, Math.abs(F[q] - fd) / Math.max(Math.abs(fd), 1e-2));
    }
    expect(worst).toBeLessThan(1e-6);
  });

  it('the parser accepts lattce(1,2) = dia and the coeff accepts dia elements with a dia pair', () => {
    const par = parseMeamParams(
      ['rc = 4.0', 'delr = 0.1', 'Ec(1,2) = 4.2', 're(1,2) = 2.0', 'alpha(1,2) = 4.2', 'lattce(1,2) = dia', 'zbl(1,1) = 0', 'zbl(2,2) = 0', 'zbl(1,2) = 0', ''].join('\n'),
      'par',
      2,
    );
    expect(par.pair.get('1,2')?.lattce).toBe('dia');
  });
});

describe('multi-element MEAM: parser subset and StyleErrors', () => {
  const ctxWith = (files: Record<string, string>) => ({
    s: null,
    readFile: (n: string) => {
      if (!(n in files)) throw new Error(`no file ${n}`);
      return files[n];
    },
    log: () => {},
  });
  const LIB_DIA = "'A' 'dia' 4 14 28.086 4.5 2.0 1.5 2.5 4.0 5.4 4.0 1.0 1.0 2.0 1.0 1.0 1.0 0\n'B' 'dia' 4 6 12.011 4.0 3.0 1.0 2.0 3.0 3.4 4.5 1.0 1.0 1.5 0.5 2.0 1.0 0\n";
  const PAR_DIA = [
    'rc = 4.0', 'delr = 0.1', 'Ec(1,1) = 4.0', 'alpha(1,1) = 4.5', 'Ec(2,2) = 4.5', 'alpha(2,2) = 4.0',
    'Ec(1,2) = 4.2', 're(1,2) = 2.0', 'alpha(1,2) = 4.2', 'lattce(1,2) = dia', 'zbl(1,1) = 0', 'zbl(2,2) = 0', 'zbl(1,2) = 0', '',
  ].join('\n');
  const coeffWith = (lib: string, par: string, ntypes = 2) => {
    const files = { 'lib.meam': lib, 'par.meam': par };
    const p = new PairMeam();
    p.settings([], ctxWith(files));
    p.allocate(ntypes);
    p.coeff(['*', '*', 'lib.meam', 'A', 'B', 'par.meam', 'A', 'B'], ctxWith(files));
    return p;
  };

  it('erose_form = 2 with per-pair attrac/repuls is accepted in a multi-element potential', () => {
    const par = parseMeamParams('rc = 4\ndelr = 0.1\nerose_form = 2\nattrac(1,1) = 0.4\nrepuls(1,1) = 0.9\nattrac(1,2) = 0.3\n', 'par', 2);
    expect(par.erose.form).toBe(2);
    expect(par.pair.get('1,1')?.attrac).toBe(0.4);
    expect(par.pair.get('1,2')?.attrac).toBe(0.3);
  });

  it('attrac/repuls in a multi-element potential with erose_form = 0 is a StyleError naming attrac', () => {
    expect(() => parseMeamParams('rc = 4\ndelr = 0.1\nattrac(2,2) = 0.1\n', 'par', 2)).toThrow(/attrac/);
    expect(() => parseMeamParams('rc = 4\ndelr = 0.1\nattrac(2,2) = 0.1\n', 'par', 2)).toThrow(StyleError);
  });

  it('erose_form = 3 is still a StyleError naming erose_form', () => {
    expect(() => parseMeamParams('erose_form = 3\n', 'par', 2)).toThrow(/erose_form/);
  });

  it('per-triplet Cmin(I,J,K)/Cmax(I,J,K) in a multi-element potential are kept per triplet (w33meamtrip_* oracle cases)', () => {
    const par = parseMeamParams('rc = 4\ndelr = 0.1\nCmax(1,1,2) = 3.0\nCmin(1,2,1) = 0.5\n', 'par', 2);
    expect(par.cmax3.get('1,1,2')).toBe(3.0);
    expect(par.cmin3.get('1,2,1')).toBe(0.5);
    expect(par.opts.Cmin).toBe(2.0);
    expect(par.opts.Cmax).toBe(2.8);
  });

  it('a dia two-element alloy is accepted; a bcc element and lattce(1,2) = c11 are StyleErrors', () => {
    expect(() => coeffWith(LIB_DIA, PAR_DIA)).not.toThrow();
    const libBcc = LIB_DIA.replace("'A' 'dia'", "'A' 'bcc'");
    expect(() => coeffWith(libBcc, PAR_DIA)).toThrow(StyleError);
    expect(() => coeffWith(LIB_DIA, PAR_DIA.replace('lattce(1,2) = dia', 'lattce(1,2) = c11'))).toThrow(/c11/);
  });
});
