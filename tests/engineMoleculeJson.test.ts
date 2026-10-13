import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseMoleculeFile, type MoleculeTemplate } from '../src/engine/molecule';
import { LabelMap, type LabelKind } from '../src/engine/labelmap';

/*
 * JSON molecule files and the molecule auto keyword (src/engine/molecule.ts). The JSON template must equal the
 * text template of the same molecule, and auto must reproduce the angles, dihedrals and impropers that the
 * JSON file lists (examples/PACKAGES/reaction/tiny_nylon_json: written by native LAMMPS).
 */

const EX = join(__dirname, '..', 'third_party', 'lammps', 'examples', 'PACKAGES', 'reaction');
const O = { toff: 0, boff: 0, aoff: 0, doff: 0, ioff: 0, scale: 1, units: 'real' };

/** The Type Labels sections of a data file (read_data), into a label map. */
const labelsOf = (file: string): LabelMap => {
  const map = new LabelMap();
  const kinds: Record<string, LabelKind> = { 'Atom Type Labels': 'atom', 'Bond Type Labels': 'bond', 'Angle Type Labels': 'angle', 'Dihedral Type Labels': 'dihedral', 'Improper Type Labels': 'improper' };
  let kind: LabelKind | null = null;
  for (const raw of readFileSync(file, 'utf8').split('\n')) {
    const l = raw.trim();
    if (kinds[l]) { kind = kinds[l]; continue; }
    if (!l) continue;
    if (kind && /^\d+\s+\S+$/.test(l)) { const [t, lab] = l.split(/\s+/); map.set(kind, Number(t), lab); }
    else if (/^[A-Z]/.test(l)) kind = null;
  }
  return map;
};

/** Order-free key of a template's topology: each entry with its type and atoms in the stored order. */
const topoKeys = (t: MoleculeTemplate) => ({
  types: Array.from(t.type),
  bonds: t.bonds.map((e) => e.join(' ')).sort(),
  angles: t.angles.map((e) => e.join(' ')).sort(),
  dihedrals: t.dihedrals.map((e) => e.join(' ')).sort(),
  impropers: t.impropers.map((e) => e.join(' ')).sort(),
  q: t.q ? Array.from(t.q) : null,
});

describe('JSON molecule files', () => {
  const labels = labelsOf(join(EX, 'tiny_nylon', 'tiny_nylon.data'));
  const pairs: [string, string][] = [
    ['rxn1_stp1_unreacted.json', 'rxn1_stp1_unreacted.molecule_template'],
    ['rxn1_stp1_reacted.json', 'rxn1_stp1_reacted.molecule_template'],
    ['rxn1_stp2_unreacted.json', 'rxn1_stp2_unreacted.molecule_template'],
    ['rxn1_stp2_reacted.json', 'rxn1_stp2_reacted.molecule_template'],
  ];
  for (const [json, text] of pairs) {
    it(`${json} reads as the text template ${text}`, () => {
      const j = parseMoleculeFile('m', json, readFileSync(join(EX, 'tiny_nylon_json', json), 'utf8'), O, labels);
      const x = parseMoleculeFile('m', text, readFileSync(join(EX, 'tiny_nylon', text), 'utf8'), O, labels);
      expect(j.natoms).toBe(x.natoms);
      expect(topoKeys(j)).toEqual(topoKeys(x));
      expect(Array.from(j.x)).toEqual(Array.from(x.x));
    });
    it(`${json} with auto adi regenerates its angles, dihedrals and impropers`, () => {
      const jtext = readFileSync(join(EX, 'tiny_nylon_json', json), 'utf8');
      const file = parseMoleculeFile('m', json, jtext, O, labels);
      const gen = parseMoleculeFile('m', json, jtext, { ...O, auto: 'adi', improperSymmatoms: [0, 1, 0, 0] }, labels);
      expect(gen.bonds).toEqual(file.bonds);
      // two angle types can carry the same labels in opposite order (c_1-c2-hc is type 9, hc-c2-c_1 is type 20 in
      // this file), and native takes the first exact match, so the angle types may differ; the atoms must not
      const byAtoms = (t: MoleculeTemplate) => t.angles.map((e) => { const [a, b, c] = [e[1], e[2], e[3]]; return a <= c ? `${a} ${b} ${c}` : `${c} ${b} ${a}`; }).sort();
      expect(byAtoms(gen)).toEqual(byAtoms(file));
      expect(topoKeys(gen).dihedrals).toEqual(topoKeys(file).dihedrals);
      expect(topoKeys(gen).impropers).toEqual(topoKeys(file).impropers);
    });
  }

  it('refuses malformed JSON and names the bad field', () => {
    expect(() => parseMoleculeFile('m', 'x.json', '{', O)).toThrow(/Error parsing JSON/);
    expect(() => parseMoleculeFile('m', 'x.json', '{"application":"LAMMPS","format":"molecule","revision":2,"types":{"format":["atom-id","type"],"data":[]}}', O)).toThrow(/unsupported revision 2/);
  });
});
