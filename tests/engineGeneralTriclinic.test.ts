import { describe, expect, it } from 'vitest';
import { makeLattice } from '../src/engine/lattice';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { StyleError } from '../src/engine/force/types';
import {
  generalAtomSites, generalBoxFromRestricted, generalCreateBox, generalFrame, rotateVector, rotationFromEdges,
  toGeneralPoint, toRestrictedPoint, unrotateVector, type GeneralBox, type V3,
} from '../src/engine/triclinic_general';

/*
 * General triclinic boxes (docs.lammps.org/Howto_triclinic.html, create_box.html,
 * lattice.html). The rotation maps the general edge vectors onto the restricted
 * form a = (ax 0 0), b = (bx by 0), c = (cx cy cz). Reference values are measured
 * with native LAMMPS (black box) on the same input; see the oracle cases
 * tests/oracle/w10gentri_*.in for the full parity checks.
 */

const A: V3 = [1, 1, 0], B: V3 = [0, 1, 0.5], C: V3 = [0.2, 0.1, 1];
const ORIGIN: V3 = [0.5, 0.2, 0.1];
const near = (a: readonly number[], b: readonly number[], tol = 1e-9) => {
  expect(a.length).toBe(b.length);
  a.forEach((v, i) => expect(Math.abs(v - b[i]), `component ${i}`).toBeLessThan(tol));
};
const dot = (p: V3, q: V3) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];

describe('rotation general -> restricted', () => {
  const { Q } = rotationFromEdges(A, B, C);

  it('is orthonormal with determinant +1', () => {
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      expect(Math.abs(dot(Q[i], Q[j]) - (i === j ? 1 : 0))).toBeLessThan(1e-12);
    }
    const det = dot(Q[0], [Q[1][1] * Q[2][2] - Q[1][2] * Q[2][1], Q[1][2] * Q[2][0] - Q[1][0] * Q[2][2], Q[1][0] * Q[2][1] - Q[1][1] * Q[2][0]]);
    expect(det).toBeCloseTo(1, 12);
  });

  it('maps A onto the +x axis and B into the xy plane with +y, C with +z', () => {
    const a = rotateVector(Q, A), b = rotateVector(Q, B), c = rotateVector(Q, C);
    near(a, [Math.SQRT2, 0, 0], 1e-12);
    expect(b[1]).toBeGreaterThan(0);
    expect(Math.abs(b[2])).toBeLessThan(1e-12);
    expect(c[2]).toBeGreaterThan(0);
  });

  it('gives the restricted box and tilts measured with native LAMMPS', () => {
    const f = generalFrame({ origin: ORIGIN, A, B, C });
    near(f.lo, ORIGIN);
    // hi = origin + (ax, by, cz): measured xhi 1.914213562373095, yhi 1.0660254037844383, zhi 0.9573214099741123
    near(f.hi, [0.5 + Math.SQRT2, 0.2 + Math.sqrt(0.75), 0.9573214099741123], 1e-9);
    near(f.tilt, [0.7071067811865476, 0.2121320343559643, 0.5196152422706631], 1e-12);
  });

  it('rotates atoms about the box origin (measured: (1.15 1.025 0.975) -> (1.5429825 0.8062178 0.7429911))', () => {
    const { Q: q } = rotationFromEdges(A, B, C);
    const x = toRestrictedPoint(q, ORIGIN, [1.15, 1.025, 0.975]);
    near(x, [1.5429825022501573, 0.8062177826491066, 0.7429910574805841], 1e-12);
    // the origin itself is fixed
    near(toRestrictedPoint(q, ORIGIN, ORIGIN), ORIGIN, 1e-12);
  });

  it('rotates velocities without an origin (measured: (1 2 3) -> (2.1213203 2.3094011 2.0412415))', () => {
    near(rotateVector(Q, [1, 2, 3]), [2.121320343559643, 2.3094010767585025, 2.0412414523193148], 1e-12);
  });

  it('round trips positions and velocities', () => {
    const pts: V3[] = [[0.1, -2, 3.5], [1.7, 0.3, -0.9], [-4, 2, 0]];
    for (const p of pts) {
      near(toGeneralPoint(Q, ORIGIN, toRestrictedPoint(Q, ORIGIN, p)), p, 1e-12);
      near(unrotateVector(Q, rotateVector(Q, p)), p, 1e-12);
    }
  });

  it('converts a restricted box back to the general edge vectors', () => {
    const f = generalFrame({ origin: ORIGIN, A, B, C });
    const back = generalBoxFromRestricted(f.Q, f.lo, f.hi, f.tilt);
    near(back.origin, ORIGIN, 1e-12);
    near(back.A, A, 1e-12);
    near(back.B, B, 1e-12);
    near(back.C, C, 1e-12);
  });

  it('is the identity for an already restricted box', () => {
    const { Q: q } = rotationFromEdges([2, 0, 0], [0.5, 1.5, 0], [0.1, 0.2, 3]);
    near(q[0], [1, 0, 0], 1e-12);
    near(q[1], [0, 1, 0], 1e-12);
    near(q[2], [0, 0, 1], 1e-12);
  });
});

describe('argument errors', () => {
  it('rejects zero, co-planar and left-handed edge vectors with StyleError', () => {
    expect(() => rotationFromEdges([0, 0, 0], B, C)).toThrow(StyleError);
    expect(() => rotationFromEdges([1, 0, 0], [0, 1, 0], [1, 1, 0])).toThrow(/co-planar/);
    expect(() => rotationFromEdges([0, 1, 0], [1, 0, 0], [0, 0, 1])).toThrow(/right-handed/);
  });

  it('create_box NULL needs a triclinic/general lattice and six valid bounds', () => {
    expect(() => generalCreateBox(null, ['0', '1', '0', '1', '0', '1'])).toThrow(/triclinic\/general/);
    const lat = makeLattice('custom', 1, UNIT_SYSTEMS.metal, 3, ['a1', '1', '0', '0', 'a2', '0', '1', '0', 'a3', '0', '0', '1', 'basis', '0', '0', '0', 'triclinic/general']);
    expect(() => generalCreateBox(lat, ['0', '1', '0', '1'])).toThrow(StyleError);
    expect(() => generalCreateBox(lat, ['0', 'x', '0', '1', '0', '1'])).toThrow(/expected a number/);
    expect(() => generalCreateBox(lat, ['1', '1', '0', '1', '0', '1'])).toThrow(/exceed/);
    const r = generalCreateBox(lat, ['0', '2', '0', '2', '0', '2']);
    near(r.lo, [0, 0, 0]);
    near(r.hi, [2, 2, 2]);
    near(r.tilt, [0, 0, 0]);
  });
});

describe('lattice triclinic/general', () => {
  const units = UNIT_SYSTEMS.metal;
  const kw = ['a1', '1', '1', '0', 'a2', '0', '1', '0.5', 'a3', '0.2', '0.1', '1', 'basis', '0', '0', '0', 'basis', '0.5', '0.25', '0.75', 'triclinic/general'];

  it('reports the lattice spacings measured with native LAMMPS', () => {
    const lat = makeLattice('custom', 1, units, 3, kw);
    expect(lat.general).toBeDefined();
    // "Lattice spacing in x,y,z = 2.3334524 1.3856406 0.85732141"
    near(lat.spacing, [2.3334524, 1.3856406, 0.85732141], 1e-6);
  });

  it('places basis atoms in general coordinates and rotates them into the restricted frame', () => {
    const lat = makeLattice('custom', 1, units, 3, kw);
    near(lat.general!.toGeneral([0, 0, 0]), [0, 0, 0]);
    // basis 2 at (0.5 0.25 0.75) in general coordinates (measured: 0.65 0.825 0.875)
    near(lat.general!.toGeneral([0.5, 0.25, 0.75]), [0.65, 0.825, 0.875], 1e-12);
    near(lat.toBox([0.5, 0.25, 0.75]), toRestrictedPoint(lat.general!.Q, [0, 0, 0], [0.65, 0.825, 0.875]), 1e-12);
    near(lat.fromBox(lat.toBox([0.3, 0.6, 0.9])), [0.3, 0.6, 0.9], 1e-12);
  });

  it('create_atoms sites of a general box match the measured atom positions', () => {
    const lat = makeLattice('custom', 1, units, 3, kw);
    // "create_box 1 NULL 0 1 0 1 0 1" is the unit cell itself: one site per basis atom
    const cell = generalCreateBox(lat, ['0', '1', '0', '1', '0', '1']);
    expect(cell.hi[0]).toBeCloseTo(Math.SQRT2, 12);
    const general: GeneralBox = { origin: [0, 0, 0], A: [1, 1, 0], B: [0, 1, 0.5], C: [0.2, 0.1, 1] };
    const sites = generalAtomSites(lat.general!, general);
    expect(sites.x.length / 3).toBe(2);
    expect(sites.basis).toEqual([0, 1]);
    near(sites.x.slice(3, 6), toRestrictedPoint(lat.general!.Q, [0, 0, 0], [0.65, 0.825, 0.875]), 1e-9);
    // the restricted box of the same cell gives the same sites (the box is converted, not re-derived)
    const back = generalBoxFromRestricted(lat.general!.Q, cell.lo, cell.hi, cell.tilt);
    near(back.A, [1, 1, 0], 1e-9);
    near(back.C, [0.2, 0.1, 1], 1e-9);
  });

  it('rejects orient, 2d use and a left-handed cell with StyleError', () => {
    expect(() => makeLattice('custom', 1, units, 3, [...kw, 'orient', 'x', '1', '0', '0'])).toThrow(/orient/);
    // measured with native LAMMPS (black box): a 2d general lattice with a1 = (1 0 0), a2 = (0 1 0) and the default a3 = (0 0 1) is accepted
    expect(() => makeLattice('custom', 1, units, 2, ['a1', '1', '0', '0', 'a2', '0', '1', '0', 'basis', '0', '0', '0', 'triclinic/general'])).not.toThrow();
    // measured with native LAMMPS (black box): a3 = (0 0 2) in 2d stops with Lattice triclinic/general a3 vector for a 2d simulation must be (0,0,1)
    expect(() => makeLattice('custom', 1, units, 2, ['a1', '1', '0', '0', 'a2', '0', '1', '0', 'a3', '0', '0', '2', 'basis', '0', '0', '0', 'triclinic/general'])).toThrow(/must be \(0,0,1\)/);
    expect(() => makeLattice('custom', 1, units, 3, ['a1', '0', '1', '0', 'a2', '1', '0', '0', 'a3', '0', '0', '1', 'basis', '0', '0', '0', 'triclinic/general'])).toThrow(/right-handed/);
    expect(() => makeLattice('fcc', 1, units, 3, ['triclinic/general'])).toThrow(/custom/);
  });
});
