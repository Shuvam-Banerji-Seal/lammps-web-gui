import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseCIFFile, parseSymmetryOperation } from '../src/services/cifParser';

const root = resolve(__dirname, '..');
const readFixture = (rel: string) => readFileSync(resolve(root, rel), 'utf8');

const elementOf = (
  mol: ReturnType<typeof parseCIFFile>,
  a: { type: number }
): string => mol.atomTypes[a.type].element;

describe('CIF symmetry expansion — real COD fixtures', () => {
  it('expands cod-1000041 (NaCl, F m -3 m, 192 ops) to exactly 8 atoms: 4 Na + 4 Cl', () => {
    const mol = parseCIFFile(readFixture('tests/fixtures/cif/cod-1000041.cif'));
    expect(mol.atoms).toHaveLength(8);
    const na = mol.atoms.filter(a => elementOf(mol, a) === 'Na');
    const cl = mol.atoms.filter(a => elementOf(mol, a) === 'Cl');
    expect(na).toHaveLength(4);
    expect(cl).toHaveLength(4);
  });

  it('NaCl: every Na-Cl nearest-neighbour distance is a/2 = 2.81 A (a = 5.62)', () => {
    const mol = parseCIFFile(readFixture('tests/fixtures/cif/cod-1000041.cif'));
    const na = mol.atoms.filter(a => elementOf(mol, a) === 'Na');
    const cl = mol.atoms.filter(a => elementOf(mol, a) === 'Cl');
    for (const n of na) {
      let nearest = Infinity;
      for (const c of cl) {
        const d = Math.hypot(n.x - c.x, n.y - c.y, n.z - c.z);
        if (d < nearest) nearest = d;
      }
      expect(Math.abs(nearest - 5.62 / 2)).toBeLessThan(0.01);
    }
  });

  it('NaCl: all 8 generated positions are distinct and ids run 1..8', () => {
    const mol = parseCIFFile(readFixture('tests/fixtures/cif/cod-1000041.cif'));
    expect(mol.atoms.map(a => a.id)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    for (let i = 0; i < mol.atoms.length; i++) {
      for (let j = i + 1; j < mol.atoms.length; j++) {
        const a = mol.atoms[i];
        const b = mol.atoms[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        expect(d).toBeGreaterThan(0.1); // well above float noise, far below a/2
      }
    }
  });

  it('expands cod-1011176 (alpha-quartz, P 32 2 1, 6 ops) to exactly 9 atoms: 3 Si + 6 O', () => {
    const mol = parseCIFFile(readFixture('tests/fixtures/cif/cod-1011176.cif'));
    expect(mol.atoms).toHaveLength(9);
    const si = mol.atoms.filter(a => elementOf(mol, a) === 'Si');
    const o = mol.atoms.filter(a => elementOf(mol, a) === 'O');
    expect(si).toHaveLength(3);
    expect(o).toHaveLength(6);
  });

  it('cod-9008800 (AgZn, P m -3 m, 48 ops) is the dedup control: still exactly 2 atoms', () => {
    const mol = parseCIFFile(readFixture('tests/fixtures/cif/cod-9008800.cif'));
    expect(mol.atoms).toHaveLength(2);
    const ag = mol.atoms.filter(a => elementOf(mol, a) === 'Ag');
    const zn = mol.atoms.filter(a => elementOf(mol, a) === 'Zn');
    expect(ag).toHaveLength(1);
    expect(zn).toHaveLength(1);
  });

  it('public/examples/nacl.cif (explicit 8 sites, no symmetry loop) still yields 8 atoms', () => {
    const mol = parseCIFFile(readFixture('public/examples/nacl.cif'));
    expect(mol.atoms).toHaveLength(8);
    expect(mol.atoms.map(a => a.id)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });
});

describe('symmetry operation parser', () => {
  it('parses the identity "x,y,z"', () => {
    const op = parseSymmetryOperation('x,y,z');
    expect(op.r).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    expect(op.t).toEqual([0, 0, 0]);
  });

  it('parses "-x+1/2,y,-z": R = diag(-1,1,-1), t = (0.5,0,0)', () => {
    const op = parseSymmetryOperation('-x+1/2,y,-z');
    expect(op.r).toEqual([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]);
    expect(op.t[0]).toBeCloseTo(0.5, 12);
    expect(op.t[1]).toBe(0);
    expect(op.t[2]).toBe(0);
  });

  it('parses "1/2+z,y,1/2-x" (leading-fraction and trailing-variable terms)', () => {
    const op = parseSymmetryOperation('1/2+z,y,1/2-x');
    expect(op.r).toEqual([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]);
    expect(op.t[0]).toBeCloseTo(0.5, 12);
    expect(op.t[1]).toBe(0);
    expect(op.t[2]).toBeCloseTo(0.5, 12);
  });

  it('parses hexagonal "x-y,x,z+1/6": row 0 = [1,-1,0]', () => {
    const op = parseSymmetryOperation('x-y,x,z+1/6');
    expect(op.r[0]).toEqual([1, -1, 0]);
    expect(op.r[1]).toEqual([1, 0, 0]);
    expect(op.r[2]).toEqual([0, 0, 1]);
    expect(op.t[2]).toBeCloseTo(1 / 6, 12);
  });

  it('handles quoted values with spaces and upper-case variables', () => {
    const op = parseSymmetryOperation("-X, 1/2 + Y, -Z");
    expect(op.r).toEqual([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]);
    expect(op.t[1]).toBeCloseTo(0.5, 12);
  });

  it('reads a quoted symop loop with an id column and applies the ops', () => {
    const cif = `data_q
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x, y, z'
2 '-X, -Y, -Z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
`;
    const mol = parseCIFFile(cif);
    expect(mol.atoms).toHaveLength(2);
    expect(mol.atoms[0].x).toBeCloseTo(0.4, 6);
    // (-0.1, -0.2, -0.3) wraps to (0.9, 0.8, 0.7) -> 0.9 * 4 = 3.6
    expect(mol.atoms[1].x).toBeCloseTo(3.6, 6);
    expect(mol.atoms[1].y).toBeCloseTo(3.2, 6);
    expect(mol.atoms[1].z).toBeCloseTo(2.8, 6);
  });

  it('throws a clear error naming the operation on garbage input', () => {
    expect(() => parseSymmetryOperation('q,r,s')).toThrow(/q,r,s/);
    expect(() => parseSymmetryOperation('x,y')).toThrow(/x,y/);
    expect(() => parseSymmetryOperation('x,1/0,z')).toThrow(/1\/0/);
  });
});
