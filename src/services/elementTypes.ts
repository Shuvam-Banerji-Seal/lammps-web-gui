import type { Atom, AtomTypeInfo, MoleculeData } from '../types';
import { ATOM_COLORS, DEFAULT_ATOM_COLOR, ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';
import { inferBonds } from './bondInference';

/*
 * The element of an atom comes from its TYPE's entry in MoleculeData.atomTypes (`element`: a symbol, or 'X' when the
 * file does not say). A type number is never read as an atomic number: a LAMMPS dump or data file numbers its types
 * 1, 2, 3, ... in whatever order the input chose, so type 2 is as likely hydrogen as helium.
 */

/** The atomic number of each type whose element is known; types whose element is unknown ('X') are left out. */
export const typeAtomicNumbers = (atomTypes: Readonly<Record<number, AtomTypeInfo>>): Record<number, number> => {
  const out: Record<number, number> = {};
  for (const info of Object.values(atomTypes)) {
    const z = info.element && info.element !== 'X' ? getAtomicNumberFromSymbol(info.element) : undefined;
    if (z !== undefined) out[info.id] = z;
  }
  return out;
};

/** An atom's atomic number by its type (for bond inference), or undefined when the type's element is unknown. */
export const atomicNumberByType = (atomTypes: Readonly<Record<number, AtomTypeInfo>>): ((a: Atom) => number | undefined) => {
  const z = typeAtomicNumbers(atomTypes);
  return (a) => z[a.type];
};

/** Colours for types whose element is unknown, in type order, so that every type stays distinguishable. */
const UNKNOWN_TYPE_COLORS = ['#7aa6d8', '#e07a5f', '#81b29a', '#f2cc8f', '#b28dff', '#ef8fb3', '#5fc9c4', '#c9a26b'];

/** The default colour of a type: CPK by its element, else a distinct palette colour by its rank among the types. */
export const defaultTypeColor = (atomTypes: Readonly<Record<number, AtomTypeInfo>>, typeId: number): string => {
  const z = typeAtomicNumbers({ [typeId]: atomTypes[typeId] })[typeId];
  if (z !== undefined) return ATOM_COLORS[z] ?? DEFAULT_ATOM_COLOR;
  const rank = Object.keys(atomTypes).map(Number).sort((a, b) => a - b).indexOf(typeId);
  return UNKNOWN_TYPE_COLORS[Math.max(0, rank) % UNKNOWN_TYPE_COLORS.length];
};

/**
 * The structure with one type's element set (a symbol, or 'X' for unknown): the type's mass follows the element unless
 * the file gave its own, and bonds the parser guessed from covalent radii are guessed again with the new element.
 */
export const withTypeElement = (data: MoleculeData, typeId: number, symbol: string): MoleculeData =>
  withTypeInfo(data, { [typeId]: { element: symbol } });

/**
 * The structure with new element, label or mass for some types at once. An element's mass follows it unless the
 * update or the file gives one; bonds the parser guessed from covalent radii are guessed again (once) when an element
 * changed.
 */
export const withTypeInfo = (
  data: MoleculeData,
  updates: Readonly<Record<number, { element?: string; label?: string; mass?: number }>>,
): MoleculeData => {
  const atomTypes = { ...data.atomTypes };
  let elementChanged = false;
  for (const [key, u] of Object.entries(updates)) {
    const typeId = Number(key);
    const info = atomTypes[typeId];
    if (!info) continue;
    let next = info;
    if (u.element !== undefined) {
      const z = u.element === 'X' ? undefined : getAtomicNumberFromSymbol(u.element);
      const el = z !== undefined ? ELEMENT_DATA[z - 1] : undefined;
      const old = ELEMENT_DATA.find((e) => e.symbol === info.element);
      // a mass that is the old element's own came from the element, not from the file
      const fileMass = info.mass > 0 && !(old && Math.abs(info.mass - old.mass) < 1e-9) ? info.mass : 0;
      next = { ...next, element: el?.symbol ?? 'X', mass: fileMass || el?.mass || 0 };
      if (next.element !== info.element) elementChanged = true;
    }
    if (u.label !== undefined) next = { ...next, label: u.label };
    if (u.mass !== undefined && u.mass > 0) next = { ...next, mass: u.mass };
    atomTypes[typeId] = next;
  }
  const bonds = elementChanged && data.bondsInferred && data.atoms.length <= 30000
    ? inferBonds(data.atoms, { atomicNumberOf: atomicNumberByType(atomTypes) })
    : data.bonds;
  return { ...data, atomTypes, bonds };
};

/** The generic name the parsers give a type the file says nothing about. */
const genericLabel = (id: number): string => `Type ${id}`;

/**
 * The text that names a type on screen (atom labels): its element symbol, else the file's own type label ("OW"), else
 * its number.
 */
export const typeDisplayName = (atomTypes: Readonly<Record<number, AtomTypeInfo>>, typeId: number): string => {
  const info = atomTypes[typeId];
  if (!info) return String(typeId);
  const z = info.element && info.element !== 'X' ? getAtomicNumberFromSymbol(info.element) : undefined;
  if (z !== undefined) return ELEMENT_DATA[z - 1].symbol;
  return info.label && info.label !== genericLabel(typeId) ? info.label : String(typeId);
};

/**
 * A trajectory that numbers its types without naming them (a dump with only a type column) takes each type's element,
 * label and mass from the data file loaded before it, when it is the same system: the same number of atoms, and every
 * type of the trajectory a type of the data file. Types the trajectory names itself keep its own information.
 */
export const inheritTypeInfo = (
  data: MoleculeData,
  from: { atomTypes: Readonly<Record<number, AtomTypeInfo>>; natoms: number } | null,
): MoleculeData => {
  if (!from || from.natoms !== data.atoms.length) return data;
  const types = Object.values(data.atomTypes);
  if (!types.every((t) => from.atomTypes[t.id] !== undefined)) return data;
  const updates: Record<number, { element?: string; label?: string; mass?: number }> = {};
  for (const t of types) {
    if (t.element !== 'X' || t.label !== genericLabel(t.id)) continue; // the trajectory named this type
    const p = from.atomTypes[t.id];
    updates[t.id] = { element: p.element, label: p.label, mass: p.mass };
  }
  return Object.keys(updates).length ? withTypeInfo(data, updates) : data;
};

/**
 * Elements the user gave earlier to types the file names no element for, keyed by the type's label ("Type 2", or a
 * type label such as "OW"): applied to such types of a newly loaded file, so a trajectory loaded after its data file
 * or a second dump of the same run keeps the user's choice.
 */
export const applyTypeElements = (data: MoleculeData, byLabel: ReadonlyMap<string, string>): MoleculeData => {
  let out = data;
  for (const info of Object.values(data.atomTypes)) {
    const sym = info.element === 'X' ? byLabel.get(info.label) : undefined;
    if (sym) out = withTypeElement(out, info.id, sym);
  }
  return out;
};
