import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

// molecule ... auto i with improper styles of different symmetry. As native LAMMPS does (src/molecule.cpp at b332f28),
// the central atom takes the place of the style's atom of symmetry (Improper::symmatoms) before the type is inferred
// from the labels. The expected Impropers lines of write_data were measured with native LAMMPS (b332f28).
const MOL = "auto improper probe\n\n5 atoms\n4 bonds\n\nCoords\n\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 -0.5 0.9 0.0\n4 -0.5 -0.9 0.0\n5 -0.5 -1.5 0.8\n\nTypes\n\n1 c\n2 h\n3 h\n4 o\n5 h\n\nBonds\n\n1 c-h 1 2\n2 c-h 1 3\n3 c-o 1 4\n4 c-h 4 5\n";
const INPUTS: Record<string, string> = {
  "harmonic": "units real\natom_style molecular\nregion b block 0 10 0 10 0 10\ncreate_box 3 b bond/types 2 angle/types 3 dihedral/types 1 improper/types 2 extra/bond/per/atom 4 extra/angle/per/atom 6 extra/dihedral/per/atom 6 extra/improper/per/atom 4 extra/special/per/atom 10\nlabelmap atom 1 c 2 h 3 o\nlabelmap bond 1 c-h 2 c-o\nlabelmap angle 1 h-c-h 2 h-c-o 3 c-o-h\nlabelmap dihedral 1 h-c-o-h\nlabelmap improper 1 c-h-h-o 2 h-c-h-o\nmass * 1.0\nimproper_style harmonic\nimproper_coeff * 1.0 0.0\nmolecule m1 mol_auto.txt auto adi\ncreate_atoms 0 single 5 5 5 mol m1 4321\nwrite_data out_harmonic.data nocoeff\n",
  "class2": "units real\natom_style molecular\nregion b block 0 10 0 10 0 10\ncreate_box 3 b bond/types 2 angle/types 3 dihedral/types 1 improper/types 2 extra/bond/per/atom 4 extra/angle/per/atom 6 extra/dihedral/per/atom 6 extra/improper/per/atom 4 extra/special/per/atom 10\nlabelmap atom 1 c 2 h 3 o\nlabelmap bond 1 c-h 2 c-o\nlabelmap angle 1 h-c-h 2 h-c-o 3 c-o-h\nlabelmap dihedral 1 h-c-o-h\nlabelmap improper 1 c-h-h-o 2 h-c-h-o\nmass * 1.0\nimproper_style class2\nimproper_coeff * 1.0 0.0\nimproper_coeff * aa 0 0 0 0 0 0\nmolecule m1 mol_auto.txt auto adi\ncreate_atoms 0 single 5 5 5 mol m1 4321\nwrite_data out_class2.data nocoeff\n",
  "umbrella": "units real\natom_style molecular\nregion b block 0 10 0 10 0 10\ncreate_box 3 b bond/types 2 angle/types 3 dihedral/types 1 improper/types 2 extra/bond/per/atom 4 extra/angle/per/atom 6 extra/dihedral/per/atom 6 extra/improper/per/atom 4 extra/special/per/atom 10\nlabelmap atom 1 c 2 h 3 o\nlabelmap bond 1 c-h 2 c-o\nlabelmap angle 1 h-c-h 2 h-c-o 3 c-o-h\nlabelmap dihedral 1 h-c-o-h\nlabelmap improper 1 c-h-h-o 2 h-c-h-o\nmass * 1.0\nimproper_style umbrella\nimproper_coeff * 1.0 0.0\nmolecule m1 mol_auto.txt auto adi\ncreate_atoms 0 single 5 5 5 mol m1 4321\nwrite_data out_umbrella.data nocoeff\n",
  "cvff": "units real\natom_style molecular\nregion b block 0 10 0 10 0 10\ncreate_box 3 b bond/types 2 angle/types 3 dihedral/types 1 improper/types 2 extra/bond/per/atom 4 extra/angle/per/atom 6 extra/dihedral/per/atom 6 extra/improper/per/atom 4 extra/special/per/atom 10\nlabelmap atom 1 c 2 h 3 o\nlabelmap bond 1 c-h 2 c-o\nlabelmap angle 1 h-c-h 2 h-c-o 3 c-o-h\nlabelmap dihedral 1 h-c-o-h\nlabelmap improper 1 c-h-h-o 2 h-c-h-o\nmass * 1.0\nimproper_style cvff\nimproper_coeff * 1.0 1 2\nmolecule m1 mol_auto.txt auto adi\ncreate_atoms 0 single 5 5 5 mol m1 4321\nwrite_data out_cvff.data nocoeff\n"
};
const NATIVE: Record<string, string> = {
  "harmonic": "1 1 1 2 3 4",
  "class2": "1 2 2 1 3 4",
  "umbrella": "1 1 1 2 3 4",
  "cvff": "1 1 1 2 3 4"
};

describe('molecule auto impropers follow the improper style symmetry', () => {
  for (const style of Object.keys(INPUTS)) {
    it(style, async () => {
      const files = new Map<string, string>();
      const session = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
      session.addFile('mol_auto.txt', MOL);
      await session.execute(INPUTS[style]);
      const lines = (files.get(`out_${style}.data`) ?? '').split('\n');
      const k = lines.indexOf('Impropers');
      expect(lines[k + 2].trim()).toBe(NATIVE[style]);
    });
  }
});
