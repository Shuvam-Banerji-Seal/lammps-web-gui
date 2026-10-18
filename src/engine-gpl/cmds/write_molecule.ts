/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/write_molecule.cpp and
 * src/write_molecule.h (WriteMolecule::command) and the native-format part of Molecule::print (src/molecule.cpp at
 * b332f28; LAMMPS development team, Sandia National Laboratories). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * write_molecule ID file — write_molecule.html: "Write a molecule template ID to a file in the molecule file format"
 * (the native format; the JSON format is refused). A template with several sets (molecule files given in turn) writes
 * one file per set; the file name must then hold a '%', which is replaced by the set number.
 *
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import type { MoleculeTemplate } from '../../engine/molecule';
import type { System } from '../../engine/system';

/** The text of Molecule::print (native), the molecule's sections in its order. */
const printMolecule = (sys: System, t: MoleculeTemplate): string => {
  const out: string[] = [];
  const p = (s: string) => out.push(s);
  const n = t.natoms;
  // header counts: nbonds, nangles, ... (zero counts are left out)
  p(`  ${n} atoms\n`);
  if (t.bonds.length) p(`  ${t.bonds.length} bonds\n`);
  if (t.angles.length) p(`  ${t.angles.length} angles\n`);
  if (t.dihedrals.length) p(`  ${t.dihedrals.length} dihedrals\n`);
  if (t.impropers.length) p(`  ${t.impropers.length} impropers\n`);
  if (t.fragments.length) p(`  ${t.fragments.length} fragments\n`);
  if (t.body) throw new StyleError('write_molecule of a body molecule is not supported by the browser engine');
  if (t.shake) throw new StyleError('write_molecule of a molecule with shake data is not supported by the browser engine');
  if (t.lines || t.tris) throw new StyleError('write_molecule of a surface molecule (lines or triangles) is not supported by the browser engine');
  const f = (v: number) => String(v);
  // the type label of a type when the label map is complete for that kind (native has_typelabels), else the number
  const topo = sys.state.topo;
  const ntypesOf: Record<'atom' | 'bond' | 'angle' | 'dihedral' | 'improper', number> = {
    atom: sys.state.ntypes, bond: topo.nbondtypes, angle: topo.nangletypes, dihedral: topo.ndihedraltypes, improper: topo.nimpropertypes,
  };
  const typeText = (kind: 'atom' | 'bond' | 'angle' | 'dihedral' | 'improper', type: number): string => {
    if (sys.labels.complete(kind, ntypesOf[kind])) {
      const lb = sys.labels.labelOf(kind, type);
      if (lb !== undefined) return lb;
    }
    return String(type);
  };
  // Coords (always present: the Coords section is required in a molecule file)
  p('\nCoords\n\n');
  for (let i = 0; i < n; i++) p(` ${i + 1}  ${f(t.x[3 * i])} ${f(t.x[3 * i + 1])} ${f(t.x[3 * i + 2])}\n`);
  // Types, with the type labels when the label map is complete for the atom types
  p('\nTypes\n\n');
  const atomLabels = sys.labels.complete('atom', ntypesOf.atom);
  for (let i = 0; i < n; i++) p(atomLabels ? ` ${i + 1} ${typeText('atom', t.type[i])}\n` : ` ${i + 1}  ${t.type[i]}\n`);
  if (t.mol) {
    p('\nMolecules\n\n');
    for (let i = 0; i < n; i++) p(` ${i + 1}  ${t.mol[i]}\n`);
  }
  if (t.fragments.length) {
    p('\nFragments\n\n');
    for (const fr of t.fragments) p(` ${fr.name} ${fr.atoms.map((a) => ` ${a}`).join('')}\n`);
  }
  if (t.q) {
    p('\nCharges\n\n');
    for (let i = 0; i < n; i++) p(` ${i + 1}  ${f(t.q[i])}\n`);
  }
  if (t.diam) {
    p('\nDiameters\n\n');
    for (let i = 0; i < n; i++) p(` ${i + 1}  ${f(t.diam[i])}\n`);
  }
  if (t.mu) {
    p('\nDipoles\n\n');
    for (let i = 0; i < n; i++) p(` ${i + 1}  ${f(t.mu[3 * i])} ${f(t.mu[3 * i + 1])} ${f(t.mu[3 * i + 2])}\n`);
  }
  if (t.mass) {
    p('\nMasses\n\n');
    for (let i = 0; i < n; i++) p(` ${i + 1}  ${f(t.mass[i])}\n`);
  }
  // topology: stored with the first atom of a bond and the second (central) atom of an angle, dihedral and improper
  // (newton_bond on, the default); each section is listed by its owning atom, in file order within an atom
  const section = (title: string, list: number[][], owner: number, kind: 'bond' | 'angle' | 'dihedral' | 'improper', width: number) => {
    if (!list.length) return;
    p(`\n${title}\n\n`);
    let idx = 0;
    for (let i = 1; i <= n; i++) {
      for (const e of list) {
        if (e[owner] !== i) continue;
        idx++;
        p(` ${idx}  ${typeText(kind, e[0])}`);
        p(` ${e.slice(1, 1 + width).join(' ')}\n`);
      }
    }
  };
  section('Bonds', t.bonds, 1, 'bond', 2);
  section('Angles', t.angles, 2, 'angle', 3);
  section('Dihedrals', t.dihedrals, 2, 'dihedral', 4);
  section('Impropers', t.impropers, 2, 'improper', 4);
  if (t.special) {
    p('\nSpecial Bond Counts\n\n');
    for (let i = 0; i < n; i++) {
      const [s12, s13, s14] = t.special[i];
      p(` ${i + 1}  ${s12.length} ${s13.length} ${s14.length}\n`);
    }
    p('\nSpecial Bonds\n\n');
    for (let i = 0; i < n; i++) {
      const [s12, s13, s14] = t.special[i];
      p(` ${i + 1} `);
      for (const a of [...s12, ...s13, ...s14]) p(` ${a}`);
      p('\n');
    }
  }
  return out.join('');
};

/** write_molecule ID file. */
const writeMolecule: Handler = ({ sys }, a) => {
  if (a.length < 2) throw new StyleError('Illegal write_molecule command: missing argument(s)');
  const id = a[0];
  const sets = sys.molecules.get(id);
  if (!sets) throw new StyleError(`Molecule template ID ${id} for write_molecule does not exist`);
  const nset = sets.length;
  if (nset < 1) throw new StyleError(`Molecule template ${id} has invalid data (nset = ${nset})`);
  if (nset > 1 && !a[1].includes('%')) throw new StyleError(`Output filename must contain a '%' for molecule template ${id} with multiple data sets`);
  if (/\.json$/.test(a[1])) throw new StyleError('write_molecule to a JSON file is not supported by the browser engine (use the native molecule file format)');
  for (let i = 0; i < nset; i++) {
    const t = sets[i];
    const filename = nset > 1 ? a[1].replace(/%(?=[^%]*$)/, String(i + 1)) : a[1];
    sys.log(`Writing molecule ${id} to file ${filename} in native format\n`);
    const head = `# MOLECULE ${id}, units = ${sys.units.style}, set ${i + 1} of ${nset}, ${t.title ?? '(no title)'}\n\n`;
    sys.writeFile(filename, `${head}${printMolecule(sys, t)}\n`, false);
  }
};

export const COMMANDS = { write_molecule: writeMolecule };
