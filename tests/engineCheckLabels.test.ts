import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// A molecular system with type labels and deliberately mislabelled bonds, angles and a dihedral, checked by labelmap
// check_labels and molecule check_labels with newton_bond on and off. The expected lines are native LAMMPS's (b332f28)
// warnings and log lines, in its order (measured with native LAMMPS: the same inputs, warnings with the file and line
// suffix removed).
const DATA = "check_labels probe\n\n5 atoms\n3 atom types\n4 bonds\n2 bond types\n3 angles\n2 angle types\n1 dihedrals\n1 dihedral types\n1 impropers\n1 improper types\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nAtom Type Labels\n\n1 c\n2 h\n3 o\n\nBond Type Labels\n\n1 c-h\n2 c-o\n\nAngle Type Labels\n\n1 h-c-h\n2 h-c-o\n\nDihedral Type Labels\n\n1 h-c-o-h\n\nImproper Type Labels\n\n1 c-h-h-o\n\nMasses\n\n1 12.0\n2 1.0\n3 16.0\n\nAtoms # molecular\n\n1 1 c 5.0 5.0 5.0\n2 1 h 6.0 5.0 5.0\n3 1 h 4.5 5.9 5.0\n4 1 o 4.5 4.1 5.0\n5 1 h 4.5 3.5 5.8\n\nBonds\n\n1 c-h 1 2\n2 c-h 3 1\n3 c-o 1 4\n4 c-h 4 5\n\nAngles\n\n1 h-c-h 2 1 3\n2 h-c-o 4 1 2\n3 h-c-h 3 1 4\n\nDihedrals\n\n1 h-c-o-h 5 4 1 2\n\nImpropers\n\n1 c-h-h-o 1 2 3 4\n";
const MOL = "probe molecule\n\n5 atoms\n4 bonds\n3 angles\n1 dihedrals\n1 impropers\n\nCoords\n\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 -0.5 0.9 0.0\n4 -0.5 -0.9 0.0\n5 -0.5 -1.5 0.8\n\nTypes\n\n1 c\n2 h\n3 h\n4 o\n5 h\n\nBonds\n\n1 c-h 1 2\n2 c-h 3 1\n3 c-o 1 4\n4 c-h 4 5\n\nAngles\n\n1 h-c-h 2 1 3\n2 h-c-o 4 1 2\n3 h-c-h 3 1 4\n\nDihedrals\n\n1 h-c-o-h 5 4 1 2\n\nImpropers\n\n1 c-h-h-o 1 2 3 4\n";
const INPUTS: Record<string, string> = {
  "in.cl": "units real\natom_style molecular\nread_data cl.data\npair_style zero 5.0\npair_coeff * *\nbond_style zero\nbond_coeff *\nangle_style zero\nangle_coeff *\ndihedral_style zero\ndihedral_coeff *\nimproper_style harmonic\nimproper_coeff * 0.0 0.0\nlabelmap check_labels badi\nrun 0\nrun 0\n",
  "in.cl2": "units real\natom_style molecular\nread_data cl.data\npair_style zero 5.0\npair_coeff * *\nbond_style zero\nbond_coeff *\nangle_style zero\nangle_coeff *\ndihedral_style zero\ndihedral_coeff *\nimproper_style harmonic\nimproper_coeff * 0.0 0.0\nlabelmap check_labels ad\nmolecule m1 mol.txt check_labels bi\nmolecule m2 mol.txt check_labels adi\nrun 0\nrun 0\n",
  "in.cl3": "units real\nnewton on off\natom_style molecular\nread_data cl.data\npair_style zero 5.0\npair_coeff * *\nbond_style zero\nbond_coeff *\nangle_style zero\nangle_coeff *\ndihedral_style zero\ndihedral_coeff *\nimproper_style harmonic\nimproper_coeff * 0.0 0.0\nlabelmap check_labels ad\nmolecule m1 mol.txt check_labels bi\nmolecule m2 mol.txt check_labels adi\nrun 0\nrun 0\n"
};
const NATIVE: Record<string, string[]> = {
  "in.cl": [
    "WARNING: Bond between atoms 4, 5 has constituent atom types (o, h) that do not match its type label (c-h)",
    "WARNING: Bond between atoms 3, 1 has constituent atom types (h, c) in reverse order compared to its bond type label (c-h)",
    "WARNING: Angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: Angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: Dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "All impropers in the simulation have self-consistent type labels"
  ],
  "in.cl2": [
    "WARNING: In molecule template 'm1', the bond between atoms 3, 1 has constituent atom types (h, c) in reverse order compared to its bond type label (c-h)",
    "WARNING: In molecule template 'm1', the bond between atoms 4, 5 has constituent atom types (o, h) that do not match its type label (c-h)",
    "All impropers in molecule 'm1' have self-consistent type labels",
    "WARNING: In molecule template 'm2', the angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: In molecule template 'm2', the angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: In molecule template 'm2', the dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "All impropers in molecule 'm2' have self-consistent type labels",
    "WARNING: Angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: Angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: Dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)"
  ],
  "in.cl3": [
    "All impropers in molecule 'm1' have self-consistent type labels",
    "WARNING: In molecule template 'm2', the angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: In molecule template 'm2', the angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: In molecule template 'm2', the angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: In molecule template 'm2', the angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: In molecule template 'm2', the angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: In molecule template 'm2', the angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: In molecule template 'm2', the dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "WARNING: In molecule template 'm2', the dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "WARNING: In molecule template 'm2', the dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "WARNING: In molecule template 'm2', the dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "All impropers in molecule 'm2' have self-consistent type labels",
    "WARNING: Angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: Angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: Angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: Angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: Angle between atoms 3, 1, 4 has constituent atom types (h, c, o) that do not match its type label (h-c-h)",
    "WARNING: Angle between atoms 4, 1, 2 has constituent atom types (o, c, h) in reverse order compared to its angle type label (h-c-o)",
    "WARNING: Dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "WARNING: Dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "WARNING: Dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)",
    "WARNING: Dihedral between atoms 5, 4, 1, 2 has constituent atom types (h, o, c, h) in reverse order compared to its dihedral type label (h-c-o-h)"
  ]
};

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  session.addFile('cl.data', DATA);
  session.addFile('mol.txt', MOL);
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const lines = events
    .filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log')
    .map((e) => e.text)
    .filter((t) => /type label|self-consistent/.test(t));
  return { error, lines };
};

describe('check_labels (native warnings and order)', () => {
  for (const name of Object.keys(INPUTS)) {
    it(name, async () => {
      const r = await run(INPUTS[name]);
      expect(r.error).toBeNull();
      expect(r.lines).toEqual(NATIVE[name]);
    });
  }
  it('molecule check_labels argument errors (native texts)', async () => {
    const base = 'units real\natom_style molecular\nread_data cl.data\n';
    expect((await run(base + 'molecule m1 mol.txt check_labels bz')).error).toContain('Illegal check_labels option z');
    expect((await run(base + 'molecule m1 mol.txt check_labels')).error).toContain('Illegal check_labels command: missing argument(s)');
  });
});
