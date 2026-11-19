import { describe, expect, it } from 'vitest';
import { parseDumpFile } from '../src/services/dumpParser';
import { parseFile } from '../src/services/fileParser';
import { applyTypeElements, defaultTypeColor, inheritTypeInfo, typeAtomicNumbers, typeDisplayName, withTypeElement } from '../src/services/elementTypes';
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

describe('type names across files', () => {
  // a data file of the same C/H/N system: Masses name the elements, Atom Type Labels name the types
  const DATA = `methylamine fragment

4 atoms
3 atom types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Atom Type Labels

1 c3
2 hn
3 n3

Masses

1 12.011
2 1.008
3 14.007

Atoms # atomic

1 1 10.0 10.0 10.0
2 2 11.09 10.0 10.0
3 3 9.53 11.38 10.0
4 2 9.64 9.49 10.89
`;

  it('a dump that only numbers its types takes element, label and mass from the data file of the same system', () => {
    const data = parseFile(DATA, 'lammps');
    expect([1, 2, 3].map((t) => data.atomTypes[t].element)).toEqual(['C', 'H', 'N']);
    const d = inheritTypeInfo(parseDumpFile(DUMP), { atomTypes: data.atomTypes, natoms: data.atoms.length });
    expect([1, 2, 3].map((t) => d.atomTypes[t].element)).toEqual(['C', 'H', 'N']);
    expect(d.atomTypes[2].mass).toBeCloseTo(1.008);
    expect(d.bonds).toHaveLength(3);
    // the atom labels name the elements
    expect([1, 2, 3].map((t) => typeDisplayName(d.atomTypes, t))).toEqual(['C', 'H', 'N']);
  });

  it('does not take them from another system (atom count or types differ) or over what the dump names itself', () => {
    const data = parseFile(DATA, 'lammps');
    const from = { atomTypes: data.atomTypes, natoms: data.atoms.length };
    expect(inheritTypeInfo(parseDumpFile(DUMP), { ...from, natoms: 5 }).atomTypes[2].element).toBe('X');
    const fourTypes = DUMP.replace('4 2 9.64', '4 4 9.64');
    expect(inheritTypeInfo(parseDumpFile(fourTypes), from).atomTypes[2].element).toBe('X');
    const labelled = parseDumpFile(DUMP.replace('ITEM: ATOMS id type x y z', 'ITEM: ATOMS id type element x y z')
      .replace('1 1 10.0 10.0 10.0', '1 1 O 10.0 10.0 10.0').replace('2 2 11.09', '2 2 H 11.09').replace('3 3 9.53', '3 3 N 9.53').replace('4 2 9.64', '4 2 H 9.64'));
    // the dump's element column wins (type 8 is its oxygen; the data file has no type 8, so nothing is taken)
    expect(inheritTypeInfo(labelled, from).atomTypes[8].element).toBe('O');
  });

  it("labels a type of unknown element by the file's type label, else by its number", () => {
    const byLabel = parseDumpFile(DUMP.replace('1 1 10.0', '1 OW 10.0').replace('2 2 11.09', '2 HW 11.09').replace('3 3 9.53', '3 NX 9.53').replace('4 2 9.64', '4 HW 9.64'));
    expect([1, 2, 3].map((t) => typeDisplayName(byLabel.atomTypes, t))).toEqual(['OW', 'HW', 'NX']);
    expect(typeDisplayName(parseDumpFile(DUMP).atomTypes, 2)).toBe('2');
  });
});

describe('data-file type names', () => {
  const data = (masses: string, labels = '') => parseFile(`gaff fragment

3 atoms
3 atom types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi
${labels}
Masses

${masses}

Atoms # atomic

1 1 10.0 10.0 10.0
2 2 11.09 10.0 10.0
3 3 9.53 11.38 10.0
`, 'lammps');

  it('a force-field type name in a Masses comment is the label, and the mass names the element (ca is no calcium)', () => {
    const d = data('1 12.01 # ca\n2 1.008 # hc\n3 14.01 # na');
    expect([1, 2, 3].map((t) => d.atomTypes[t].element)).toEqual(['C', 'H', 'N']);
    expect([1, 2, 3].map((t) => d.atomTypes[t].label)).toEqual(['ca', 'hc', 'na']);
    // an element symbol in its own case still names the element
    expect(data('1 12.011 # C\n2 1.008 # H\n3 35.45 # Cl').atomTypes[3].element).toBe('Cl');
  });

  it('keeps the Atom Type Labels as the type labels while Masses name the elements', () => {
    const d = data('1 12.011\n2 1.008\n3 14.007', '\nAtom Type Labels\n\n1 c3\n2 hn\n3 n3\n');
    expect([1, 2, 3].map((t) => d.atomTypes[t].label)).toEqual(['c3', 'hn', 'n3']);
    expect([1, 2, 3].map((t) => d.atomTypes[t].element)).toEqual(['C', 'H', 'N']);
  });
});
