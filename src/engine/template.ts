import type { SimState } from './types';
import type { MoleculeTemplate } from './molecule';
import { pushTopo } from './atoms';
import { StyleError } from './force/types';

/*
 * atom_style template: the bonded topology of template atoms comes from the
 * molecule templates. atom_style.html: "The template stores one or more molecules with a single copy"
 * and "store a template index and template atom to identify which molecule and" (the rest of the
 * sentence is on the next line). The engine keeps bonded topology by atom ID (SimState.topo), so each
 * template molecule is expanded into it when its atoms are created or read; the per-atom template index
 * and template atom stay beside it so write_data and write_restart reproduce the template format.
 *
 * read_data.html: "set the *bonds*, *angles*, etc header keywords in the data file," (the Bonds,
 * Angles, Dihedrals and Impropers sections are refused for the same reason).
 */

/**
 * Expands the template topology of the atoms in [from, to): atoms sharing a
 * molecule ID and a template index form one molecule, and their template atoms
 * 1..Natoms map to its atom IDs. Throws StyleError when a molecule is incomplete
 * or an index is outside the template.
 */
export const expandTemplateTopology = (s: SimState, templates: MoleculeTemplate[], templateId: string, from: number, to: number): void => {
  const tIdx = s.tmplIndex!, tAt = s.tmplAtom!;
  const groups = new Map<string, { index: number; atoms: Map<number, number> }>();
  for (let i = from; i < to; i++) {
    const idx = tIdx[i], at = tAt[i];
    if (idx === 0 && at === 0) continue;
    if (idx < 1 || idx > templates.length) {
      throw new StyleError(`atom ${s.id[i]}: template index ${idx} is outside 1..${templates.length} of molecule template ${templateId}`);
    }
    const t = templates[idx - 1];
    if (at < 1 || at > t.natoms) throw new StyleError(`atom ${s.id[i]}: template atom ${at} is outside 1..${t.natoms} of molecule template ${templateId}`);
    const key = `${s.molecule[i]}:${idx}`;
    let g = groups.get(key);
    if (!g) groups.set(key, g = { index: idx, atoms: new Map() });
    if (g.atoms.has(at)) throw new StyleError(`molecule ${s.molecule[i]}: template atom ${at} appears twice`);
    g.atoms.set(at, s.id[i]);
  }
  for (const [key, g] of groups) {
    const t = templates[g.index - 1];
    if (g.atoms.size !== t.natoms) {
      throw new StyleError(`molecule ${key.split(':')[0]} (template index ${g.index}) has ${g.atoms.size} of the ${t.natoms} atoms of molecule template ${templateId}`);
    }
    for (const [list, kind] of [[t.bonds, 'bonds'], [t.angles, 'angles'], [t.dihedrals, 'dihedrals'], [t.impropers, 'impropers']] as const) {
      for (const e of list) pushTopo(s.topo[kind], e[0], e.slice(1).map((k) => g.atoms.get(k)!));
    }
  }
};
