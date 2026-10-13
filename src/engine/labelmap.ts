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
