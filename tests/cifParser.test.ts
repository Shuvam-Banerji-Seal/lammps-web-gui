import { describe, it, expect } from 'vitest';
import {
  parseCIFFile,
  cifCellToBoxBounds,
  fractionalToCartesian,
  elementFromTypeSymbol,
  elementFromLabel,
} from '../src/services/cifParser';

const SIMPLE_CIF = `data_simple
_cell_length_a 4.05
_cell_length_b 4.05
_cell_length_c 4.05
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al1 Al 0.0 0.0 0.0
Al2 Al 0.5 0.5 0.5
`;

const TRICLINIC_CIF = `data_tri
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 80
_cell_angle_beta 95
_cell_angle_gamma 100

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.1 0.2 0.3
O1 O 0.4 0.5 0.6
`;

describe('CIF parser', () => {
  it('parses cubic cell atoms into Cartesian coords', () => {
    const result = parseCIFFile(SIMPLE_CIF);
    expect(result.atoms).toHaveLength(2);
    // cubic a=4.05: frac (0.5,0.5,0.5) -> (2.025, 2.025, 2.025)
    expect(result.atoms[1].x).toBeCloseTo(2.025, 5);
    expect(result.atoms[1].y).toBeCloseTo(2.025, 5);
    expect(result.atoms[1].z).toBeCloseTo(2.025, 5);
  });

  it('maps elements via type_symbol', () => {
    const result = parseCIFFile(SIMPLE_CIF);
    expect(result.atomTypes[result.atoms[0].type].element).toBe('Al');
  });

  it('produces LAMMPS-convention box bounds for cubic cell', () => {
    const result = parseCIFFile(SIMPLE_CIF);
    const box = result.box!;
    expect(box.xlo).toBe(0);
    expect(box.xhi).toBeCloseTo(4.05, 10);
    expect(box.yhi).toBeCloseTo(4.05, 10);
    expect(box.zhi).toBeCloseTo(4.05, 10);
    expect(Math.abs(box.xy ?? 0)).toBeLessThan(1e-12); // trig residue of cos(90 deg)
    expect(Math.abs(box.xz ?? 0)).toBeLessThan(1e-12);
    expect(Math.abs(box.yz ?? 0)).toBeLessThan(1e-12);
  });

  it('handles triclinic cells with correct tilt factors', () => {
    const result = parseCIFFile(TRICLINIC_CIF);
    const box = result.box!;
    expect(box.xhi).toBeCloseTo(5.0, 5);
    expect(box.xy).toBeCloseTo(6.0 * Math.cos(100 * Math.PI / 180), 5);
    expect(box.xz).toBeCloseTo(7.0 * Math.cos(95 * Math.PI / 180), 5);
    expect(box.yhi).toBeCloseTo(6.0 * Math.sin(100 * Math.PI / 180), 5);
    expect(box.zhi).toBeGreaterThan(0);
  });

  it('infers element from label when type_symbol missing', () => {
    const cif = `data_x
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cl1 0.0 0.0 0.0
C12 0.1 0.2 0.3
`;
    const result = parseCIFFile(cif);
    expect(result.atoms).toHaveLength(2);
    const types = result.atoms.map(a => result.atomTypes[a.type].element);
    expect(types).toContain('Cl');
    expect(types).toContain('C');
  });

  it('rejects fractional coordinates without cell parameters', () => {
    const cif = `data_bad
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.0 0.0 0.0
`;
    expect(() => parseCIFFile(cif)).toThrow(/_cell_length/);
  });

  it('strips charge notation from type symbols ("O2-", "Fe3+")', () => {
    expect(elementFromTypeSymbol('O2-')).toBe('O');
    expect(elementFromTypeSymbol('Fe3+')).toBe('Fe');
    expect(elementFromLabel('OW32')).toBe('O');
    expect(elementFromLabel('N1')).toBe('N');
  });

  it('supports Cartn_* coordinates', () => {
    const cif = `data_cart
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_Cartn_x
_atom_site_Cartn_y
_atom_site_Cartn_z
H1 H 0.0 0.0 0.0
H2 H 0.74 0.0 0.0
`;
    const result = parseCIFFile(cif);
    expect(result.atoms[1].x).toBeCloseTo(0.74, 5);
  });

  it('throws on missing atom sites', () => {
    expect(() => parseCIFFile('data_empty\n_cell_length_a 1\n')).toThrow(/_atom_site/);
  });

  it('fractionalToCartesian preserves volume orientation', () => {
    const p = fractionalToCartesian(
      { a: 5, b: 6, c: 7, alphaDeg: 90, betaDeg: 90, gammaDeg: 90 }, 1, 0, 0
    );
    expect(p).toEqual({ x: 5, y: 0, z: 0 });
    const q = fractionalToCartesian(
      { a: 5, b: 6, c: 7, alphaDeg: 90, betaDeg: 90, gammaDeg: 90 }, 0, 1, 0
    );
    expect(q.y).toBeCloseTo(6, 10);
  });
});
