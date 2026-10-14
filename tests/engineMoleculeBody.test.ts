import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseMoleculeFile } from '../src/engine/molecule';
import { bodyRadiusBody, checkBodyParams, nativeFrame } from '../src/engine/bodies';

/*
 * Body molecules (src/engine/molecule.ts Body Integers / Body Doubles; bodies.ts radius and checks). docs.lammps.org/molecule.html:
 * "defining a *body* particle, which requires setting the number of *atoms* to 1, and setting the *inertia* in a specific
 * section"; "For *rounded/polyhedron*, the number of edges E and faces F is required, and Ninteger should have a value of 3."
 * The insertion is checked against native LAMMPS in oracle case w152body_pour_cube.
 */

const O = { toff: 0, boff: 0, aoff: 0, doff: 0, ioff: 0, scale: 1 };
const CUBE = readFileSync(join(__dirname, 'oracle', 'w152body_pour_cube.mol'), 'utf8');

describe('body molecules', () => {
  it('reads Body Integers and Body Doubles of a rounded polyhedron body', () => {
    const t = parseMoleculeFile('cube', 'cube.mol', CUBE, O);
    expect(t.natoms).toBe(1);
    expect(t.body?.ints).toEqual([8, 12, 6]);
    expect(t.body?.dbls).toHaveLength(79);
    expect(t.mass?.[0]).toBe(1);
  });

  it('refuses a body molecule with more than one atom', () => {
    const two = CUBE.replace('1 atoms', '2 atoms');
    expect(() => parseMoleculeFile('cube', 'cube.mol', two, O)).toThrow(/natoms must be 1 for body particle/);
  });

  it('refuses a body molecule without a Masses section', () => {
    const noMass = CUBE.replace('Masses\n\n1 1.0\n', '');
    expect(() => parseMoleculeFile('cube', 'cube.mol', noMass, O)).toThrow(/Masses section for body particle/);
  });

  it('refuses a Body Doubles section with the wrong number of values', () => {
    const short = CUBE.replace('0.5\n', '');
    expect(() => parseMoleculeFile('cube', 'cube.mol', short, O)).toThrow(/Body Doubles/);
  });

  it('radius of the cube: enclosing radius of the vertices plus the rounded radius', () => {
    const t = parseMoleculeFile('cube', 'cube.mol', CUBE, O);
    const r = bodyRadiusBody('rounded/polyhedron', t.body!.ints, t.body!.dbls);
    expect(r).toBeCloseTo(Math.sqrt(3) + 0.25, 12);
  });

  it('checks the integer count of the body style', () => {
    const t = parseMoleculeFile('cube', 'cube.mol', CUBE, O);
    expect(() => checkBodyParams('rounded/polyhedron', [8], t.body!.dbls)).toThrow(/integer values/);
    expect(() => checkBodyParams('nparticle', t.body!.ints, t.body!.dbls)).toThrow(/one integer/);
  });

  it('principal axes in the native order (degenerate rod inertia 1 1 1.333: z first, then y, then x)', () => {
    const f = nativeFrame([1, 1, 1.333, 0, 0, 0]);
    expect(f.moments).toEqual([1.333, 1, 1]);
    // columns are the axes: ex = z, ey = y, ez = -x (the third axis flips for a right-handed frame)
    expect([f.E[0], f.E[3], f.E[6]].map((x) => x + 0)).toEqual([0, 0, 1]);
    expect([f.E[1], f.E[4], f.E[7]].map((x) => x + 0)).toEqual([0, 1, 0]);
    expect([f.E[2], f.E[5], f.E[8]].map((x) => x + 0)).toEqual([-1, 0, 0]);
  });

  it('refuses an inertia tensor with off-diagonal terms', () => {
    expect(() => nativeFrame([1, 1, 1, 0.5, 0, 0])).toThrow(/off-diagonal inertia/);
  });
});
