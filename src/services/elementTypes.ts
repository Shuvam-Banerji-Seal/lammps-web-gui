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
export const withTypeElement = (data: MoleculeData, typeId: number, symbol: string): MoleculeData => {
  const info = data.atomTypes[typeId];
  if (!info) return data;
  const z = symbol === 'X' ? undefined : getAtomicNumberFromSymbol(symbol);
  const el = z !== undefined ? ELEMENT_DATA[z - 1] : undefined;
  const old = ELEMENT_DATA.find((e) => e.symbol === info.element);
  // a mass that is the old element's own came from the element, not from the file
  const fileMass = info.mass > 0 && !(old && Math.abs(info.mass - old.mass) < 1e-9) ? info.mass : 0;
  const atomTypes = { ...data.atomTypes, [typeId]: { ...info, element: el?.symbol ?? 'X', mass: fileMass || el?.mass || 0 } };
  const bonds = data.bondsInferred && data.atoms.length <= 30000
    ? inferBonds(data.atoms, { atomicNumberOf: atomicNumberByType(atomTypes) })
    : data.bonds;
  return { ...data, atomTypes, bonds };
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
