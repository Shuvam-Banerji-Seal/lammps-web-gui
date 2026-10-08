import { describe, expect, it } from 'vitest';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, clearAccum, StyleError, type BondedCompute } from '../src/engine/force/types';
import type { SimpleBonded } from '../src/engine/force/bonded_util';
import { DihedralCosineSquaredRestricted } from '../src/engine/force/dihedral/misc18';
import { ImproperDistharm, ImproperSqdistharm } from '../src/engine/force/improper/misc18';

/*
 * Unit tests for the wave-18 dihedral/improper styles:
 * dihedral_style cosine/squared/restricted, improper_style distharm and
 * improper_style sqdistharm.
 *
 * Measured with native LAMMPS (black box): for all three styles the reported
 * energy equals the formula on its docs page with no extra factor, and the
 * forces are the exact gradient of that energy. The finite-difference check
 * below therefore expects f = -dE/dx for the documented energy E. The native
 * values measured for the same geometries and coefficients are hard coded
 * below as literals.
 */

const GEOMS: number[][][] = [
  [[5.1, 5.2, 4.7], [6.0, 5.1, 5.05], [6.6, 6.1, 4.8], [7.4, 6.3, 5.7]],
  [[5, 5, 5], [6, 5, 5], [6.5, 5.9, 5.1], [7.5, 6.0, 4.0]],
  [[19.6, 5.2, 4.7], [0.5, 5.1, 5.05], [1.1, 6.1, 4.8], [1.9, 6.3, 5.7]],
  [[5, 5, 5], [6.4, 4.6, 5.3], [5.1, 6.2, 4.8], [6, 5.4, 6.6]],
];

const KIND = { dihedral: 'dihedrals', improper: 'impropers' } as const;

const make = (style: SimpleBonded, pts: number[][], kind: keyof typeof KIND) => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [20, 20, 20] }, 1, 'molecular');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  pushTopo(s.topo[KIND[kind]], 1, [1, 2, 3, 4]);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [20, 20, 20] }));
  const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
  return { s, bc };
};

/** Zeroes accumulators, runs compute, returns the energy (forces left in bc.f). */
const energy = (style: SimpleBonded, bc: BondedCompute): number => {
  clearAccum(bc.acc);
  bc.f.fill(0);
  bc.virial.fill(0);
  style.compute(bc);
  return style.kind === 'dihedral' ? bc.acc.edihed : bc.acc.eimp;
};

/** Points of a BondedCompute's current positions. */
const points = (s: ReturnType<typeof make>['s']): number[][] => {
  const out: number[][] = [];
  for (let a = 0; a < 4; a++) out.push([s.x[3 * a], s.x[3 * a + 1], s.x[3 * a + 2]]);
  return out;
};

/**
 * f = -dE/dx by central differences at the given geometry; the reported
 * energy must equal the documented energy E (measured: no extra factor).
 */
const checkForces = (style: SimpleBonded, kind: keyof typeof KIND, pts: number[][], docEnergy: (p: number[][]) => number) => {
  const { s, bc } = make(style, pts, kind);
  expect(energy(style, bc)).toBeCloseTo(docEnergy(pts), 12);
  const fAna = Array.from(bc.f);
  const h = 1e-6;
  for (let a = 0; a < 4; a++) {
    for (let d = 0; d < 3; d++) {
      const k = 3 * a + d;
      const x0 = s.x[k];
      s.x[k] = x0 + h;
      const ep = docEnergy(points(s));
      s.x[k] = x0 - h;
      const em = docEnergy(points(s));
      s.x[k] = x0;
      expect(fAna[k], `atom ${a} dim ${d}`).toBeCloseTo(-(ep - em) / (2 * h), 5);
    }
  }
  // the virial is sum_k (r_k - r_0)_c f_k,d component-wise (minimum image)
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [20, 20, 20] }));
  const ref = new Float64Array(6);
  const rel = [0, 0, 0];
  for (let a = 0; a < 4; a++) {
    for (let c = 0; c < 3; c++) rel[c] = s.x[3 * a + c] - s.x[c];
    geom.minimumImage(rel);
    const [rx, ry, rz] = rel, [fx, fy, fz] = [bc.f[3 * a], bc.f[3 * a + 1], bc.f[3 * a + 2]];
    ref[0] += rx * fx; ref[1] += ry * fy; ref[2] += rz * fz;
    ref[3] += rx * fy; ref[4] += rx * fz; ref[5] += ry * fz;
  }
  for (let c = 0; c < 6; c++) expect(bc.virial[c], `virial ${c}`).toBeCloseTo(ref[c], 6);
};

// --- documented energies (the formula on each docs page) ---

/** 20-box: wrap into [-10, 10) so the model sees the same minimum-image deltas as the engine. */
const wrap = (p: number[][]): number[][] => p.map((q) => q.map((c) => c - 20 * Math.round(c / 20)));

const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: number[]) => Math.sqrt(dot(a, a));

const phiOf = (raw: number[][]): number => {
  const p = wrap(raw);
  const b1 = sub(p[1], p[0]), b2 = sub(p[2], p[1]), b3 = sub(p[3], p[2]);
  return Math.atan2(norm(b2) * dot(b1, cross(b2, b3)), dot(cross(b1, b2), cross(b2, b3)));
};

/** K [cos(phi) - cos(phi0)]^2 / sin^2(phi) with phi0 in degrees. */
const csrDoc = (K: number, phi0deg: number) => (p: number[][]) => {
  const phi = phiOf(p);
  const u = Math.cos(phi) - Math.cos((phi0deg * Math.PI) / 180);
  return (K * u * u) / (Math.sin(phi) * Math.sin(phi));
};

/** d = oriented distance of the first atom from the plane of the other three. */
export const orientedDistance = (raw: number[][]): number => {
  const p = wrap(raw);
  const b1 = sub(p[2], p[1]), b2 = sub(p[3], p[1]);
  const n = cross(b1, b2);
  return dot(sub(p[0], p[1]), n) / norm(n);
};

/** K (d - d0)^2 */
const distHarmDoc = (K: number, d0: number) => (p: number[][]) => {
  const d = orientedDistance(p);
  return K * (d - d0) ** 2;
};

/** K (d^2 - d0^2)^2 */
const sqDistHarmDoc = (K: number, d02: number) => (p: number[][]) => {
  const d = orientedDistance(p);
  return K * (d * d - d02) ** 2;
};

const fresh = (make2: () => SimpleBonded, coeffs: string[]) => {
  const style = make2();
  style.allocate(1);
  if (coeffs.length) style.coeff(['1', ...coeffs]);
  style.init();
  return style;
};

describe('dihedral_style cosine/squared/restricted: f = -dE/dx by central finite differences', () => {
  it('all four geometries (K 10, phi0 120)', () => {
    const style = fresh(() => new DihedralCosineSquaredRestricted(), ['10.0', '120']);
    for (const pts of GEOMS) checkForces(style, 'dihedral', pts, csrDoc(10.0, 120));
  });
  it('other coefficients (K 2.5, phi0 240 and K 1, phi0 0)', () => {
    const a = fresh(() => new DihedralCosineSquaredRestricted(), ['2.5', '240']);
    const b = fresh(() => new DihedralCosineSquaredRestricted(), ['1.0', '0']);
    for (const pts of GEOMS) {
      checkForces(a, 'dihedral', pts, csrDoc(2.5, 240));
      checkForces(b, 'dihedral', pts, csrDoc(1.0, 0));
    }
  });
  it('reproduces native LAMMPS energy and forces (K 10, phi0 120)', () => {
    // native: improper-free four-atom run, thermo edihed, write_dump fx fy fz
    const style = fresh(() => new DihedralCosineSquaredRestricted(), ['10.0', '120']);
    const { bc } = make(style, GEOMS[3], 'dihedral');
    expect(energy(style, bc)).toBeCloseTo(1.30008783898178, 13);
    const native = [
      1.2959055073783121, -1.4347525260259815, -7.9605624024667616,
      -3.2530930822359503, -1.461781412322734, 3.780341494380731,
      -1.6815161448787137, 0.12348957129157823, 4.7671086048177038,
      3.6387037197363519, 2.7730443670571372, -0.58688769673167274,
    ];
    for (let k = 0; k < 12; k++) expect(bc.f[k]).toBeCloseTo(native[k], 12);
  });
  it('phi0 is read in degrees and written back in degrees', () => {
    const style = fresh(() => new DihedralCosineSquaredRestricted(), ['10.0', '120']);
    expect(style.dataCoeffs()).toEqual(['1 10 120']);
  });
  it('a near-cis or planar quadruplet is singular and throws', () => {
    const style = fresh(() => new DihedralCosineSquaredRestricted(), ['10.0', '120']);
    // exactly coplanar I,J,K,L: sin(phi) = 0
    const { bc } = make(style, [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, -2, 0]], 'dihedral');
    expect(() => energy(style, bc)).toThrow(StyleError);
    // |sin(phi)| = 1.1e-8 (native reports nan here, see the header)
    const near = make(style, [[0, 0, 0], [1, 0, 0], [1.5, 1, 0], [2.5, 1, 1e-8]], 'dihedral');
    expect(() => energy(style, near.bc)).toThrow(StyleError);
  });
});

describe('improper_style distharm / sqdistharm: f = -dE/dx by central finite differences', () => {
  it('distharm (K 1, d0 0.3) at all four geometries', () => {
    const style = fresh(() => new ImproperDistharm(), ['1.0', '0.3']);
    for (const pts of GEOMS) checkForces(style, 'improper', pts, distHarmDoc(1.0, 0.3));
  });
  it('distharm (K 25, d0 0.15)', () => {
    const style = fresh(() => new ImproperDistharm(), ['25.0', '0.15']);
    for (const pts of GEOMS) checkForces(style, 'improper', pts, distHarmDoc(25.0, 0.15));
  });
  it('sqdistharm (K 1, d0^2 0.25) at all four geometries', () => {
    const style = fresh(() => new ImproperSqdistharm(), ['1.0', '0.25']);
    for (const pts of GEOMS) checkForces(style, 'improper', pts, sqDistHarmDoc(1.0, 0.25));
  });
  it('sqdistharm (K 50, d0^2 0.01)', () => {
    const style = fresh(() => new ImproperSqdistharm(), ['50.0', '0.01']);
    for (const pts of GEOMS) checkForces(style, 'improper', pts, sqDistHarmDoc(50.0, 0.01));
  });
  it('distharm reproduces native LAMMPS energy and forces (K 1, d0 0.3)', () => {
    const style = fresh(() => new ImproperDistharm(), ['1.0', '0.3']);
    const { bc } = make(style, GEOMS[3], 'improper');
    expect(energy(style, bc)).toBeCloseTo(1.267410219001696, 13);
    const native = [
      1.776261354950909, 1.3536830487327494, -0.28649376692756645,
      -0.8526054503764362, -0.64976786339171944, 0.13751700812523193,
      -1.0668172675108147, -0.81301799822396759, 0.172067301211422,
      0.14316136293634196, 0.10910281288293766, -0.023090542409087478,
    ];
    for (let k = 0; k < 12; k++) expect(bc.f[k]).toBeCloseTo(native[k], 12);
  });
  it('sqdistharm reproduces native LAMMPS energy and forces (K 1, d0^2 0.25)', () => {
    const style = fresh(() => new ImproperSqdistharm(), ['1.0', '0.25']);
    const { bc } = make(style, GEOMS[3], 'improper');
    expect(energy(style, bc)).toBeCloseTo(0.1865672640142568, 13);
    const native = [
      1.1255563918912883, 0.85778289543328046, -0.18154135353085329,
      -0.54026706810781833, -0.41173578980797443, 0.087139849694809701,
      -0.67600580910008157, -0.51518184645127174, 0.10903319501614228,
      0.090716485316611584, 0.069134740825965713, -0.014631691180098688,
    ];
    for (let k = 0; k < 12; k++) expect(bc.f[k]).toBeCloseTo(native[k], 12);
  });
  it('d = 0 keeps the plain forces (native parity)', () => {
    // native: improper_coeff 1 1.0 0.5, I=(0,0,0) J=(1,0,0) K=(2,0,0)
    // L=(0.3,0.2,0.7) -> eimp 0.25 = K d0^2 (measured) and the forces below
    const pts = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0.3, 0.2, 0.7]];
    const style = fresh(() => new ImproperDistharm(), ['1.0', '0.5']);
    const { bc } = make(style, pts, 'improper');
    expect(energy(style, bc)).toBeCloseTo(0.25, 14);
    const native = [
      0, -0.96152394764082327, 0.27472112789737813,
      -2.7755575615628926e-17, 1.9230478952816465, -0.54944225579475636,
      2.7755575615628926e-17, -0.96152394764082327, 0.27472112789737818,
      0, 0, 5.5511151231257827e-17,
    ];
    for (let k = 0; k < 12; k++) expect(bc.f[k]).toBeCloseTo(native[k], 12);
  });
  it('a degenerate (collinear) plane gives d = 0 with no force instead of native nan', () => {
    // native reports nan here (see the header of improper/misc18.ts)
    const pts = [[0.3, 0.2, 0.7], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    const dh = fresh(() => new ImproperDistharm(), ['1.0', '0.5']);
    const sq = fresh(() => new ImproperSqdistharm(), ['1.0', '0.25']);
    for (const [style, e] of [[dh, 0.25], [sq, 0.0625]] as const) {
      const { bc } = make(style, pts, 'improper');
      expect(energy(style, bc)).toBeCloseTo(e, 14);
      for (let k = 0; k < 12; k++) expect(bc.f[k]).toBe(0);
    }
  });
});

describe('wave-18 dihedral/improper styles: missing or bad coefficients throw', () => {
  it('init without coefficients throws for every style', () => {
    const makers = [
      () => new DihedralCosineSquaredRestricted(), () => new ImproperDistharm(), () => new ImproperSqdistharm(),
    ];
    for (const make2 of makers) {
      const style = make2();
      style.allocate(1);
      expect(() => style.init()).toThrow(StyleError);
    }
  });
  it('wrong coefficient counts throw', () => {
    const dh = new ImproperDistharm(); dh.allocate(1);
    const sq = new ImproperSqdistharm(); sq.allocate(1);
    const cs = new DihedralCosineSquaredRestricted(); cs.allocate(1);
    expect(() => dh.coeff(['1', '1.0'])).toThrow(StyleError);
    expect(() => dh.coeff(['1', '1.0', '0.5', '2'])).toThrow(StyleError);
    expect(() => sq.coeff(['1', '1.0'])).toThrow(StyleError);
    expect(() => sq.coeff(['1', '1.0', '0.5', '2'])).toThrow(StyleError);
    expect(() => cs.coeff(['1', '1.0'])).toThrow(StyleError);
    expect(() => cs.coeff(['1', '1.0', '120', '3'])).toThrow(StyleError);
  });
  it('non-numeric coefficients throw', () => {
    const dh = new ImproperDistharm(); dh.allocate(1);
    expect(() => dh.coeff(['1', 'k', 'x'])).toThrow(StyleError);
  });
});