import { describe, expect, it } from 'vitest';
import { parseDumpFile } from '../src/services/dumpParser';
import { parseFile } from '../src/services/fileParser';
import { applyTypeElements, defaultTypeColor, typeAtomicNumbers, withTypeElement } from '../src/services/elementTypes';
import { atomDisplayRadius } from '../src/services/atomStyle';
import { ATOM_COLORS, COVALENT_RADII } from '../src/constants';
import type { VisualizationConfig } from '../src/types';

/*
 * The element of an atom is its type's element (atomTypes), never its type number: a LAMMPS dump without element
 * information drew type 2 (hydrogen) as helium and type 3 (nitrogen) as lithium.
 */

// methylamine-like fragment, "dump custom 1 all custom 1 x.lammpstrj id type x y z": types 1 = C, 2 = H, 3 = N
const DUMP = `ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0 20
0 20
0 20
ITEM: ATOMS id type x y z
1 1 10.0 10.0 10.0
2 2 11.09 10.0 10.0
3 3 9.53 11.38 10.0
4 2 9.64 9.49 10.89
`;

describe('type elements', () => {
  it('a dump without element information gives distinct colours, not He/Li CPK colours', () => {
    const d = parseDumpFile(DUMP);
    const colours = [1, 2, 3].map((t) => defaultTypeColor(d.atomTypes, t));
    expect(new Set(colours).size).toBe(3);
    expect(colours).not.toContain(ATOM_COLORS[2]); // helium
    expect(colours).not.toContain(ATOM_COLORS[3]); // lithium
    expect(typeAtomicNumbers(d.atomTypes)).toEqual({});
  });

  it('setting the elements of the types gives CPK colours, element radii and guessed bonds', () => {
    let d = parseDumpFile(DUMP);
    d = withTypeElement(d, 1, 'C');
    d = withTypeElement(d, 2, 'H');
    d = withTypeElement(d, 3, 'N');
    expect(typeAtomicNumbers(d.atomTypes)).toEqual({ 1: 6, 2: 1, 3: 7 });
    expect(defaultTypeColor(d.atomTypes, 2)).toBe(ATOM_COLORS[1]);
    expect(d.atomTypes[3].mass).toBeCloseTo(14.007);
    // C-H, C-N and C-H: three bonds once the elements are known
    expect(d.bonds).toHaveLength(3);
    const config = { atomScale: 1, visualizationMode: 'element-balls', typeAtomicNumbers: typeAtomicNumbers(d.atomTypes) } as unknown as VisualizationConfig;
    // the radius of type 2 is hydrogen's (not helium's), of type 3 nitrogen's
    expect(atomDisplayRadius(d.atoms[1], config)).toBeCloseTo(COVALENT_RADII[1] * 0.6);
    expect(atomDisplayRadius(d.atoms[2], config)).toBeCloseTo(COVALENT_RADII[7] * 0.6);
    // back to unknown: no bonds again
    expect(withTypeElement(d, 3, 'X').bonds).toHaveLength(2);
  });

  it('applies remembered elements by type label to a newly loaded file whose types are unknown', () => {
    const remembered = new Map([['Type 1', 'C'], ['Type 2', 'H'], ['Type 3', 'N']]);
    const d = applyTypeElements(parseDumpFile(DUMP), remembered);
    expect([1, 2, 3].map((t) => d.atomTypes[t].element)).toEqual(['C', 'H', 'N']);
    // a type the file names keeps the file's element
    const named = parseDumpFile(DUMP.replace('ITEM: ATOMS id type x y z', 'ITEM: ATOMS id type mass x y z')
      .replace('1 1 10.0', '1 1 15.999 10.0').replace('2 2 11.09', '2 2 1.008 11.09').replace('3 3 9.53', '3 3 14.007 9.53').replace('4 2 9.64', '4 2 1.008 9.64'));
    expect(applyTypeElements(named, remembered).atomTypes[1].element).toBe('O');
  });

  it('a LAMMPS dump xyz (type numbers in the species column) names no element either', () => {
    const d = parseFile('3\nAtoms. Timestep: 0\n1 10.0 10.0 10.0\n2 11.09 10.0 10.0\n3 9.53 11.38 10.0\n', 'xyz');
    expect(d.atoms.map((a) => a.type)).toEqual([1, 2, 3]);
    expect([1, 2, 3].map((t) => d.atomTypes[t].element)).toEqual(['X', 'X', 'X']);
    expect(d.atomTypes[2].label).toBe('Type 2');
    expect(d.bonds).toHaveLength(0);
  });
});
