import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { PODModel, jacobiEigen, parsePODCoeffFile, parsePODParams } from '../src/engine-gpl/pod/eapod';
import { StyleError } from '../src/engine/force/types';

/* pair_style pod (src/engine-gpl/pod, LAMMPS ML-POD at b332f28): parsing, the eigen solver, and the forces as the
 * derivative of the energy (finite differences), for the Ta and InP models of examples/PACKAGES/pod. */

const ORACLE = join(__dirname, 'oracle');
const read = (f: string) => readFileSync(join(ORACLE, f), 'utf8');

const taModel = () => new PODModel(parsePODParams(read('w128pod_Ta_param.pod')), parsePODCoeffFile(read('w128pod_Ta_coeff.pod')));
const inpModel = () => new PODModel(parsePODParams(read('w128pod_InP_param.pod')), parsePODCoeffFile(read('w128pod_InP_coeff.pod')));

/** Deterministic neighbour shell: n vectors of length 1.8 to 4.6 with pseudo-random directions. */
const shell = (n: number, seed: number): Float64Array => {
  const out = new Float64Array(3 * n);
  let s = seed;
  const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  for (let k = 0; k < n; k++) {
    const x = 2 * rnd() - 1, y = 2 * rnd() - 1, z = 2 * rnd() - 1;
    const len = Math.sqrt(x * x + y * y + z * z);
    const r = 1.8 + 2.8 * rnd();
    out[3 * k] = (r * x) / len; out[3 * k + 1] = (r * y) / len; out[3 * k + 2] = (r * z) / len;
  }
  return out;
};

describe('pair_style pod parameter and coefficient files', () => {
  it('reads the Ta parameter file keywords', () => {
    const p = parsePODParams(read('w128pod_Ta_param.pod'));
    expect(p.species).toEqual(['Ta']);
    expect([p.rin, p.rcut]).toEqual([1.0, 5.0]);
    expect([p.besseldegree, p.inversedegree]).toEqual([3, 6]);
    expect([p.nrbf2, p.nrbf3, p.P3, p.nrbf4, p.P4]).toEqual([6, 5, 4, 0, 0]);
  });

  it('reads the coefficient count and rejects a wrong count', () => {
    const c = parsePODCoeffFile(read('w128pod_Ta_coeff.pod'));
    expect(c.coeff.length).toBe(32);
    expect(c.coeff[0]).toBeCloseTo(-4.45745, 12);
    expect(() => new PODModel(parsePODParams('species Ta\nrcut 5\n'), { coeff: new Float64Array(3), proj: new Float64Array(0), centroids: new Float64Array(0) }))
      .toThrow(StyleError);
  });

  it('refuses the five-, six- and seven-body terms and environment clusters by name', () => {
    const base = read('w128pod_Ta_param.pod');
    const c = parsePODCoeffFile(read('w128pod_Ta_coeff.pod'));
    expect(() => new PODModel(parsePODParams(`${base}\nfivebody_number_radial_basis_functions 2\n`), c)).toThrow(/fivebody/);
    expect(() => new PODModel(parsePODParams(base.replace('number_of_environment_clusters 1', 'number_of_environment_clusters 2')), c))
      .toThrow(/environment-adaptive/);
  });
});

describe('jacobi eigen decomposition', () => {
  it('diagonalises a symmetric matrix', () => {
    const n = 7;
    const a = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
      const v = Math.sin(1.7 * (i + 1) * (j + 2)) + (i === j ? 3 : 0);
      a[i + n * j] = v; a[j + n * i] = v;
    }
    const { values, vectors } = jacobiEigen(a, n);
    let worst = 0;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        let av = 0;
        for (let k = 0; k < n; k++) av += a[i + n * k] * vectors[k + n * j];
        worst = Math.max(worst, Math.abs(av - values[j] * vectors[i + n * j]));
      }
    }
    expect(worst).toBeLessThan(1e-12);
  });
});

describe('pair_style pod forces are the energy derivative', () => {
  const check = (model: PODModel, ti: number, tjs: number[], seed: number) => {
    const nj = tjs.length;
    const rij = shell(nj, seed);
    const tj = Int32Array.from(tjs);
    const fij = new Float64Array(3 * nj);
    const e0 = model.atomEnergyForce(ti, nj, rij, tj, fij);
    expect(Number.isFinite(e0)).toBe(true);
    // five-point central difference: truncation O(h^4), round-off O(eps |E| / h)
    const h = 1e-3;
    const tmp = new Float64Array(3 * nj);
    const at = (k: number, d: number) => {
      const r = Float64Array.from(rij); r[k] += d;
      return model.atomEnergyForce(ti, nj, r, tj, tmp);
    };
    let worst = 0;
    for (let k = 0; k < 3 * nj; k++) {
      const fd = (-at(k, 2 * h) + 8 * at(k, h) - 8 * at(k, -h) + at(k, -2 * h)) / (12 * h);
      worst = Math.max(worst, Math.abs(fd - fij[k]) / Math.max(1, Math.abs(fd)));
    }
    return worst;
  };

  it('Ta single element, 24 neighbours', () => {
    expect(check(taModel(), 1, Array.from({ length: 24 }, () => 1), 12345)).toBeLessThan(1e-7);
  });

  it('InP two elements, mixed neighbour types, 30 neighbours', () => {
    const tjs = Array.from({ length: 30 }, (_, k) => (k % 3 === 0 ? 1 : 2));
    expect(check(inpModel(), 2, tjs, 777)).toBeLessThan(1e-7);
    expect(check(inpModel(), 1, tjs, 4242)).toBeLessThan(1e-7);
  });

  it('an isolated atom has the one-body energy and no force', () => {
    const m = taModel();
    const fij = new Float64Array(0);
    const e = m.atomEnergyForce(1, 0, new Float64Array(0), new Int32Array(0), fij);
    expect(e).toBeCloseTo(-4.45745, 12);
  });
});
