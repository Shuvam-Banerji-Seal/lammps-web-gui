import { StyleError } from './force/types';

/*
 * Type labels — docs.lammps.org/Howto_type_labels.html and labelmap.html.
 *
 * Howto_type_labels.html: "A collection of type labels for all type-kinds (atom types, bond types,
 * etc.) is stored as a "label map" which is simply a list of numeric types and their associated type
 * labels. Within a type-kind, each type label must be unique. It can be assigned to only one numeric
 * type."
 *
 * labelmap.html: "Valid type labels can contain any alphanumeric character, but must not start with a
 * number, a '#', or a '*' character. ... They must not contain blanks or any other whitespace." "If a
 * type label already exists for the same numeric type, it will be overwritten. Type labels must be
 * unique; assigning the same type label to multiple numeric types within the same type-kind is not
 * allowed."
 *
 * Type words of commands (mass, pair_coeff, bond_coeff, set, group, create_atoms, ...): "Or I can be a
 * type label, which is an alphanumeric string defined by the labelmap command or in a section of a data
 * file read by the read_data command, and which converts internally to a numeric type." A numeric type
 * or a wildcard passes through unchanged; a label is replaced by its numeric type.
 *
 * Measured with native LAMMPS (black box): the label map exists once a labelmap command or a Type Labels
 * section has defined it; "labelmap clear" empties it, and label2type() then reports an invalid label
 * rather than a missing map. Wildcards do not combine with labels ("mass C* 1.0" is refused).
 */

export const LABEL_KINDS = ['atom', 'bond', 'angle', 'dihedral', 'improper'] as const;
export type LabelKind = (typeof LABEL_KINDS)[number];

/** A kind name of the labelmap command, label2type() and the Type Labels sections. */
export const isLabelKind = (k: string): k is LabelKind => (LABEL_KINDS as readonly string[]).includes(k);

/** read_data.html section titles of the label maps, one per kind. */
export const LABEL_SECTION: Record<LabelKind, string> = {
  atom: 'Atom Type Labels', bond: 'Bond Type Labels', angle: 'Angle Type Labels',
  dihedral: 'Dihedral Type Labels', improper: 'Improper Type Labels',
};

/** The kind named by a label section title (undefined for any other title). */
export const labelKindOfSection = (title: string): LabelKind | undefined =>
  LABEL_KINDS.find((k) => LABEL_SECTION[k] === title);

/** Capitalised kind name as native messages print it ("Atom type string Z not found in labelmap"). */
const KIND_NAME: Record<LabelKind, string> = { atom: 'Atom', bond: 'Bond', angle: 'Angle', dihedral: 'Dihedral', improper: 'Improper' };

/** labelmap.html: a valid label does not start with a number, '#' or '*' and has no whitespace. */
export const isValidLabel = (label: string): boolean => label.length > 0 && !/^[\d#*]/.test(label) && !/\s/.test(label);

/**
 * A type word that is not a label: a numeric type, a range (1:3 style words are numeric) or a wildcard. A leading
 * minus is numeric too: turned-off interactions carry negative types (delete_bonds).
 */
export const isNumericTypeWord = (w: string): boolean => w === '' || /^-?[\d*]/.test(w);

/** A label map: per kind, numeric type -> label and label -> numeric type. */
export class LabelMap {
  /** true once a labelmap command or a Type Labels section has defined a map (native: "Cannot use label2type() function without a labelmap") */
  defined = false;
  /**
   * labelmap check_labels: the kinds (bond, angle, dihedral, improper) to check at the next run's setup, or null (native
   * LabelMap::checkflag and check_which_labels; the check runs once, then the flag is cleared).
   */
  checkWhich: CheckWhich | null = null;
  private readonly byType = new Map<LabelKind, Map<number, string>>();
  private readonly byLabel = new Map<LabelKind, Map<string, number>>();

  constructor() {
    for (const k of LABEL_KINDS) {
      this.byType.set(k, new Map());
      this.byLabel.set(k, new Map());
    }
  }

  /**
   * Labels numeric type `type` of `kind`. The label must not name another type of the kind; a label
   * already on this type is replaced (labelmap.html: "it will be overwritten").
   */
  set(kind: LabelKind, type: number, label: string): void {
    if (!isValidLabel(label)) throw new StyleError(`Type label string ${label} for ${kind} type ${type} is invalid`);
    const bt = this.byType.get(kind)!, bl = this.byLabel.get(kind)!;
    const other = bl.get(label);
    if (other !== undefined && other !== type) throw new StyleError(`The ${kind} type label ${label} is already in use for type ${other}`);
    const old = bt.get(type);
    if (old !== undefined) bl.delete(old);
    bt.set(type, label);
    bl.set(label, type);
    this.defined = true;
  }

  /** labelmap clear: drops every label of every kind (the map itself stays defined). */
  clear(): void {
    for (const k of LABEL_KINDS) {
      this.byType.get(k)!.clear();
      this.byLabel.get(k)!.clear();
    }
  }

  /** The numeric type of a label, or undefined. */
  typeOf(kind: LabelKind, label: string): number | undefined {
    return this.byLabel.get(kind)!.get(label);
  }

  /** The label of a numeric type, or undefined. */
  labelOf(kind: LabelKind, type: number): string | undefined {
    return this.byType.get(kind)!.get(type);
  }

  /** True when every numeric type 1..n of the kind has a label (the condition to write its section). */
  complete(kind: LabelKind, n: number): boolean {
    for (let t = 1; t <= n; t++) if (this.labelOf(kind, t) === undefined) return false;
    return true;
  }

  /** The (type, label) pairs of a kind, by increasing type. */
  entries(kind: LabelKind): [number, string][] {
    return [...this.byType.get(kind)!.entries()].sort((a, b) => a[0] - b[0]);
  }
}

/**
 * A type word of a command, as native converts it: a numeric type or wildcard passes through, a label
 * becomes its numeric type. Native message: "Atom type string Z not found in labelmap".
 */
export const typeWord = (map: LabelMap, kind: LabelKind, word: string): string => {
  if (isNumericTypeWord(word)) return word;
  const t = map.typeOf(kind, word);
  if (t === undefined) throw new StyleError(`${KIND_NAME[kind]} type string ${word} not found in labelmap`);
  return String(t);
};

/**
 * The labelmap command's (type, label) pairs: "list of one or more numeric-type/type-label pairs"
 * (labelmap.html). The numeric type must lie in 1..n (native: "Labelmap atom type 9 must be within 1-3").
 */
export const parseLabelPairs = (args: string[], kind: LabelKind, n: number): [number, string][] => {
  if (args.length === 0 || args.length % 2 !== 0) throw new StyleError(`Illegal labelmap ${kind} command: missing argument(s)`);
  const out: [number, string][] = [];
  for (let k = 0; k < args.length; k += 2) {
    if (!/^[+-]?\d+$/.test(args[k])) throw new StyleError(`Expected integer parameter instead of '${args[k]}' in input script or data file`);
    const t = Number(args[k]);
    if (t < 1 || t > n) throw new StyleError(`Labelmap ${kind} type ${t} must be within 1-${n}`);
    out.push([t, args[k + 1]]);
  }
  return out;
};

/**
 * labelmap write: the label map as labelmap commands, one line per kind that has labels, in the order
 * atom, bond, angle, dihedral, improper. Measured with native LAMMPS (black box): every label is written
 * between triple quotes with a blank on each side ("labelmap atom 1 """ C """ 2 """ H """").
 */
export const labelmapText = (map: LabelMap): string => {
  let out = '';
  for (const k of LABEL_KINDS) {
    const e = map.entries(k);
    if (e.length === 0) continue;
    out += `labelmap ${k} ${e.map(([t, l]) => `${t} """ ${l} """`).join(' ')}\n`;
  }
  return out;
};

/**
 * Type inference from labels (Howto_type_labels.html and the Symmetry convention of the bonded styles).
 * The bond, angle, dihedral and improper type labels are "a-b", "a-b-c", "a-b-c-d" lists of atom type labels.
 * A bonded type is found from the labels of the constituent atom types; the result is the numeric type, or
 * its negative when the labels match only in reverse order (as native LAMMPS does: "returns negative of
 * numeric type if constituent atoms types in reverse order"). 0 means no type matches.
 *
 * Howto_type_labels.html (and the molecule.html auto keyword): "Type labels for bonds, angles, and dihedrals
 * are assumed to be equivalent to those written in reverse order." Improper labels follow the symmetry of the
 * improper style: "For the *class2* improper style, the second atom in the quadruplet is the atom of symmetry;
 * all other atoms are considered interchangeable." (improper_class2.html, Symmetry convention).
 */

/** The atom type labels of a bonded type label, or null when it has another number of parts. */
const splitTypeLabel = (label: string | undefined, n: number): string[] | null => {
  if (!label) return null;
  const parts = label.split('-').filter((p) => p.length > 0);
  return parts.length === n ? parts : null;
};

/** Numeric atom type's label, or '' when the type has none. */
const atomLabel = (map: LabelMap, t: number): string => map.labelOf('atom', t) ?? '';

/** The signed bond type of two atom types (labels "a-b"; reverse order gives the negative). */
export const inferBondType = (map: LabelMap, t1: number, t2: number): number => {
  const a = atomLabel(map, t1), b = atomLabel(map, t2);
  if (!a || !b) return 0;
  let out = 0;
  for (const [ty, label] of map.entries('bond')) {
    const bt = splitTypeLabel(label, 2);
    if (!bt) continue;
    if (a === bt[0] && b === bt[1]) return ty;
    if (a === bt[1] && b === bt[0]) out = -ty;
  }
  return out;
};

/** The signed angle type of three atom types (labels "a-b-c"; the centre b must match, ends in either order). */
export const inferAngleType = (map: LabelMap, t1: number, t2: number, t3: number): number => {
  const a = atomLabel(map, t1), b = atomLabel(map, t2), c = atomLabel(map, t3);
  if (!a || !b || !c) return 0;
  let out = 0;
  for (const [ty, label] of map.entries('angle')) {
    const at = splitTypeLabel(label, 3);
    if (!at || b !== at[1]) continue;
    if (a === at[0] && c === at[2]) return ty;
    if (a === at[2] && c === at[0]) out = -ty;
  }
  return out;
};

/** The signed dihedral type of four atom types (labels "a-b-c-d"; the whole chain may be reversed). */
export const inferDihedralType = (map: LabelMap, t1: number, t2: number, t3: number, t4: number): number => {
  const m = [atomLabel(map, t1), atomLabel(map, t2), atomLabel(map, t3), atomLabel(map, t4)];
  if (m.some((x) => !x)) return 0;
  let out = 0;
  for (const [ty, label] of map.entries('dihedral')) {
    const dt = splitTypeLabel(label, 4);
    if (!dt) continue;
    if (m[0] === dt[0] && m[1] === dt[1] && m[2] === dt[2] && m[3] === dt[3]) return ty;
    if (m[3] === dt[0] && m[2] === dt[1] && m[1] === dt[2] && m[0] === dt[3]) out = -ty;
  }
  return out;
};

/**
 * The signed improper type of four atom types and the atom order that maps the improper's labels onto the
 * atoms (iorder[j] = index of the atom that plays label j). `symmatoms[j] === 1` marks the atom of symmetry,
 * which must match its label exactly; the others may be permuted. Native: improper labels are matched with
 * the symmetry of the improper style (infer_impropertype); iorder is returned as native sets it.
 */
export const inferImproperType = (
  map: LabelMap, t: [number, number, number, number], symmatoms: readonly number[],
): { type: number; iorder: [number, number, number, number] } => {
  const m = t.map((x) => atomLabel(map, x));
  let iorder: [number, number, number, number] = [0, 1, 2, 3];
  if (m.some((x) => !x)) return { type: 0, iorder };
  let out = 0;
  for (const [ty, label] of map.entries('improper')) {
    const it = splitTypeLabel(label, 4);
    if (!it) continue;
    if (m[0] === it[0] && m[1] === it[1] && m[2] === it[2] && m[3] === it[3]) return { type: ty, iorder };
    const avail: string[] = [...m];
    let navail = 4;
    let ok = true;
    for (let j = 0; j < 4; j++) {
      if (symmatoms[j] === 1) {
        if (m[j] !== it[j]) { ok = false; break; }
        avail[j] = '';
        navail--;
      }
    }
    if (!ok) continue;
    iorder = [0, 1, 2, 3];
    for (let j = 0; j < 4; j++) {
      if (symmatoms[j] === 1) continue;
      for (let k = 0; k < 4; k++) {
        if (it[j] === avail[k]) {
          avail[k] = '';
          navail--;
          iorder[j] = k;
          break;
        }
      }
    }
    if (navail === 0) out = -ty;
  }
  return { type: out, iorder };
};

/** The kinds a check_labels option selects: bonds, angles, dihedrals, impropers (letters b, a, d, i). */
export type CheckWhich = [boolean, boolean, boolean, boolean];

/**
 * The letters of a check_labels option (molecule.html and labelmap.html: "b" bonds, "a" angles, "d" dihedrals, "i"
 * impropers). As native LAMMPS does (src/molecule.cpp and src/label_map.cpp at b332f28), another letter is an error,
 * reported by bad(letter).
 */
export const parseCheckWhich = (word: string, bad: (c: string) => string): CheckWhich => {
  const out: CheckWhich = [false, false, false, false];
  for (const c of word) {
    const k = 'badi'.indexOf(c);
    if (k < 0) throw new StyleError(bad(c));
    out[k] = true;
  }
  return out;
};

/**
 * The atoms of symmetry of an improper style (native Improper::symmatoms: 1 marks the atom whose type label must match
 * in place; the others may be permuted when a type is inferred from labels). As native LAMMPS does (src/improper.cpp
 * sets all four to 0; the styles below set theirs in their constructors at b332f28: MOLECULE/improper_harmonic.cpp,
 * improper_cvff.cpp, improper_umbrella.cpp, EXTRA-MOLECULE/improper_cossq.cpp, improper_distance.cpp,
 * improper_fourier.cpp, improper_ring.cpp, MOFFF/improper_inversion_harmonic.cpp, CLASS2/improper_class2.cpp,
 * AMOEBA/improper_amoeba.cpp, YAFF/improper_distharm.cpp, improper_sqdistharm.cpp). No improper style gives no
 * symmetry (improper_style none); an accelerator suffix keeps the style's symmetry.
 */
export const improperSymmatoms = (style: string | undefined): readonly number[] => {
  if (style === undefined) return [0, 0, 0, 0];
  const base = style.replace(/\/(omp|kk|gpu|intel|opt)$/, '');
  const table: Record<string, readonly number[]> = {
    harmonic: [1, 0, 0, 0], cvff: [1, 0, 0, 0], umbrella: [1, 0, 0, 2], cossq: [1, 0, 0, 0], distance: [1, 0, 0, 0],
    fourier: [1, 0, 0, 2], 'inversion/harmonic': [1, 0, 0, 0], ring: [0, 1, 0, 0], class2: [0, 1, 0, 0],
    amoeba: [0, 1, 0, 0], distharm: [0, 0, 0, 1], sqdistharm: [0, 0, 0, 1],
  };
  return table[base] ?? [0, 0, 0, 0];
};

/** One topology entry for a check: its type, its atoms (IDs or template indices) and their atom types. */
export interface CheckEntry { type: number; atoms: number[]; types: number[] }

const KIND_NAMES = ['bond', 'angle', 'dihedral', 'improper'] as const;

/**
 * The warnings and log lines of a check_labels pass over the entries of each selected kind, in the order given. As
 * native LAMMPS does (src/label_map.cpp check_labels and src/molecule.cpp check_labels at b332f28): an entry whose
 * inferred type differs from its type warns, with the "reverse order" (or, for impropers, "different order") text when
 * the inferred type is its negative; a kind with no such entry logs that its labels are self-consistent. Bonds are
 * checked only with newton_bond on. The molecule check (template set) prefixes "In molecule template 'ID', the ...".
 */
export const checkLabelsReport = (
  map: LabelMap, which: CheckWhich, entries: CheckEntry[][], newtonBond: boolean, symmatoms: readonly number[],
  template: string | null,
): { warnings: string[]; logs: string[] } => {
  const warnings: string[] = [], logs: string[] = [];
  const lab = (t: number): string => map.labelOf('atom', t) ?? '';
  for (let k = 0; k < 4; k++) {
    if (!which[k] || (k === 0 && !newtonBond)) continue;
    const kind = KIND_NAMES[k];
    let perfect = true;
    for (const e of entries[k]) {
      const t = e.types;
      const inferred = k === 0 ? inferBondType(map, t[0], t[1])
        : k === 1 ? inferAngleType(map, t[0], t[1], t[2])
        : k === 2 ? inferDihedralType(map, t[0], t[1], t[2], t[3])
        : inferImproperType(map, [t[0], t[1], t[2], t[3]], symmatoms).type;
      if (inferred === e.type) continue;
      perfect = false;
      const ids = e.atoms.join(', ');
      const labels = t.map(lab).join(', ');
      const own = map.labelOf(kind, e.type) ?? '';
      const reversed = inferred === -e.type;
      let text: string;
      if (k === 0) text = `${template ? 'the bond' : 'Bond'} between atoms ${ids} has constituent atom types (${labels}) ${reversed ? 'in reverse order compared to its bond type label' : 'that do not match its type label'} (${own})`;
      else if (k === 1) text = `${template ? 'the angle' : 'Angle'} between atoms ${ids} has constituent atom types (${labels}) ${reversed ? 'in reverse order compared to its angle type label' : template ? 'that do not match its type label' : 'that do not match its type label'} (${own})`;
      else if (k === 2) text = `${template ? 'the dihedral' : 'Dihedral'} between atoms ${ids} has constituent atom types (${labels}) ${reversed ? 'in reverse order compared to its dihedral type label' : 'that do not match its dihedral label'} (${own})`;
      else text = `${template ? 'the improper' : 'Improper'} containing atoms ${ids} has constituent atom types (${labels}) ${reversed ? 'in a different order compared to its improper type label' : 'that do not match its improper label'} (${own})`;
      warnings.push(template ? `In molecule template '${template}', ${text}` : text);
    }
    if (perfect) logs.push(template ? `All ${kind}s in molecule '${template}' have self-consistent type labels` : `All ${kind}s in the simulation have self-consistent type labels`);
  }
  return { warnings, logs };
};
