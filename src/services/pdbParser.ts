import { Atom, Bond, MoleculeData, AtomTypeInfo, BoxBounds, TrajectoryFrame } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';

/**
 * Parses PDB (Protein Data Bank) format — including multi-model entries.
 *
 * Supported records:
 *  - ATOM / HETATM : coordinates + element
 *  - MODEL / ENDMDL: one TrajectoryFrame per model (NMR entries carry many)
 *  - CONECT        : explicit bonds (deduplicated, fixed-width fields)
 *  - CRYST1        : unit cell -> simulation box (a b c alpha beta gamma)
 *
 * Multi-model: every MODEL/ENDMDL block becomes a TrajectoryFrame; the FIRST
 * model defines `atoms`, `bonds` (CONECT) and the element->type mapping, so
 * colors stay consistent during playback. `frames` is set only when there are
 * at least 2 models; atoms are matched across models by order, and if a later
 * model has a different atom count the trajectory simply stops at the last
 * model whose count matches model 1 (no error). min/max/center span ALL kept
 * models so the camera framing never jumps.
 *
 * Alternate locations (column 17, altLoc — wwPDB format 3.3, sect9:
 * "AltLoc is the place holder to indicate alternate conformation"): an atom
 * is kept when its altLoc is blank or equals the first non-blank altLoc
 * letter seen in the current model — in practice 'A', since conformer A is
 * listed first. Every other letter is skipped, including letters on atoms
 * that exist ONLY as a later conformer (e.g. 3NIR side-chain atoms written
 * only as 'B'), so alternates do not render as duplicate atoms.
 *
 * Element resolution order per atom:
 *  1. Columns 77-78 (element, right-justified) — the authoritative field.
 *  2. The RAW 4-character atom-name field (columns 13-16, NOT trimmed).
 *     wwPDB format 3.3, sect9: "Alignment of one-letter atom name such as C
 *     starts at column 14, while two-letter atom name such as FE starts at
 *     column 13" — i.e. the element symbol is right-justified in the name
 *     field, so the column-13 character tells them apart:
 *     - Column 13 blank or a digit: one-letter element, taken as the first
 *       letter in columns 14-16 (" CA " -> C, "1HG1" -> H, " OXT" -> O).
 *     - Column 13 a letter: try the two-letter element at columns 13-14
 *       ("FE  " -> Fe, "CL  " -> Cl, "CA  " -> Ca), else the single letter
 *       at column 13.
 *     - ATOM (not HETATM) hydrogens are the exception: their names fill all
 *       four columns ("HG21", "HD11"), so H followed by a letter/digit at
 *       column 13 is hydrogen (H), never Hg/He.
 *
 * CONECT (wwPDB format 3.3, sect10) uses fixed 5-character fields: the serial
 * in columns 7-11 and bonded serials in 12-16, 17-21, 22-26, 27-31 —
 * whitespace splitting would fuse two touching 5-digit serials.
 */
export const parsePDBFile = (data: string): MoleculeData => {
  const lines = data.split('\n');

  // One atom list (plus a parallel serial list) per MODEL/ENDMDL block.
  const modelAtoms: Atom[][] = [];
  const modelSerials: number[][] = [];
  const modelComments: (string | undefined)[] = [];
  let current: Atom[] = [];
  let currentSerials: number[] = [];
  let pendingComment: string | undefined;

  const conectLines: string[] = [];
  let box: BoxBounds | undefined;

  // Alternate-location keeper: the first non-blank altLoc letter seen in the
  // current model (in practice 'A'). Reset per model.
  let altLocKeeper: string | undefined;

  // keys are element symbols from the file: a null-prototype map (no prototype keys)
  const elementTypeMap: Record<string, number> = Object.create(null);

  const pushCurrentModel = () => {
    if (current.length === 0) return;
    modelAtoms.push(current);
    modelSerials.push(currentSerials);
    modelComments.push(pendingComment);
    current = [];
    currentSerials = [];
    pendingComment = undefined;
    altLocKeeper = undefined;
  };

  /**
   * Element from the RAW 4-character atom-name field (columns 13-16).
   * See the header comment for the wwPDB alignment convention this encodes.
   */
  const inferElementFromAtomName = (name4: string, isAtomRecord: boolean): string => {
    const c13 = (name4[0] ?? ' ').toUpperCase();
    const c14 = name4[1] ?? ' ';

    // Column 13 blank or a digit: one-letter element whose symbol starts at
    // column 14 (" CA " -> C, "1HG1" -> H, " OXT" -> O).
    if (c13 === ' ' || /[0-9]/.test(c13)) {
      for (let i = 1; i < name4.length; i++) {
        const ch = name4[i];
        if (ch && /[A-Za-z]/.test(ch)) return ch.toUpperCase();
      }
      return '';
    }

    // ATOM-record hydrogens fill all four columns ("HG21", "HD11"); without
    // this rule "HG21" would be read as mercury via the two-letter test.
    if (isAtomRecord && c13 === 'H' && /[A-Za-z0-9]/.test(c14)) return 'H';

    // ATOM records hold standard residues (H, C, N, O, S only), so a letter
    // at column 13 is that one-letter element. Writers that left-justify
    // names put " CA "/" OG1" at "CA  "/"OG1 ": reading two letters there
    // would give calcium / oganesson. Two-letter elements are HETATM only.
    if (isAtomRecord && /[A-Za-z]/.test(c13)) return c13;

    // Letter at column 13: try the two-letter element at columns 13-14
    // ("FE  " -> Fe, "CL  " -> Cl, "CA  " -> Ca).
    if (/[A-Za-z]/.test(c14)) {
      const two = c13 + c14.toLowerCase();
      if (getAtomicNumberFromSymbol(two) !== undefined) return two;
    }
    return c13;
  };

  for (const line of lines) {
    const recordType = line.substring(0, 6).trim();

    if (recordType === 'ATOM' || recordType === 'HETATM') {
      const x = parseFloat(line.substring(30, 38).trim());
      const y = parseFloat(line.substring(38, 46).trim());
      const z = parseFloat(line.substring(46, 54).trim());
      if (![x, y, z].every(Number.isFinite)) continue;

      // altLoc — column 17. Blank is always kept; otherwise keep only the
      // first non-blank letter seen in this model.
      const altLoc = line.length > 16 ? line[16] : ' ';
      if (altLoc !== ' ') {
        if (altLocKeeper === undefined) altLocKeeper = altLoc;
        if (altLoc !== altLocKeeper) continue;
      }

      // 1. Authoritative element columns 77-78
      let symbol = line.length >= 78 ? line.substring(76, 78).trim() : '';
      let atomicNumber = getAtomicNumberFromSymbol(symbol);

      // 2. Heuristic from the raw (untrimmed) atom-name field
      if (atomicNumber === undefined) {
        symbol = inferElementFromAtomName(line.substring(12, 16), recordType === 'ATOM');
        atomicNumber = getAtomicNumberFromSymbol(symbol);
      }

      const lookupKey = (symbol || 'X').toUpperCase();
      if (!(lookupKey in elementTypeMap)) {
        elementTypeMap[lookupKey] =
          atomicNumber !== undefined ? atomicNumber : 900 + Object.keys(elementTypeMap).length;
      }
      const type = elementTypeMap[lookupKey];

      const serial = parseInt(line.substring(6, 11).trim(), 10);

      const id = current.length + 1;
      currentSerials.push(serial);
      current.push({ id, molId: 1, type, charge: 0, x, y, z });
    } else if (recordType === 'MODEL') {
      pushCurrentModel();
      // MODEL serial lives in columns 11-14 (wwPDB sect2).
      const serialText = line.substring(10, 14).trim();
      pendingComment = serialText ? `model ${serialText}` : undefined;
    } else if (recordType === 'ENDMDL') {
      pushCurrentModel();
    } else if (recordType === 'CONECT') {
      conectLines.push(line);
    } else if (recordType === 'CRYST1') {
      // CRYST1: cols 7-15 a, 16-24 b, 25-33 c (Angstroms); angles ignored for box render
      const a = parseFloat(line.substring(6, 15).trim());
      const b = parseFloat(line.substring(15, 24).trim());
      const c = parseFloat(line.substring(24, 33).trim());
      if ([a, b, c].every(Number.isFinite) && a > 0 && b > 0 && c > 0) {
        box = { xlo: 0, xhi: a, ylo: 0, yhi: b, zlo: 0, zhi: c };
      }
    }
  }
  pushCurrentModel();

  // Keep only the leading run of models whose atom count matches model 1
  // (atoms are matched across models by order; a truncated or heterogeneous
  // tail is dropped instead of throwing).
  const refCount = modelAtoms.length > 0 ? modelAtoms[0].length : 0;
  let lastKept = 0;
  while (lastKept < modelAtoms.length && modelAtoms[lastKept].length === refCount) lastKept++;

  const frames: TrajectoryFrame[] = [];
  for (let i = 0; i < lastKept; i++) {
    frames.push({
      comment: modelComments[i],
      atoms: modelAtoms[i],
      // CRYST1 applies to every model of the entry.
      ...(box ? { box } : {}),
    });
  }

  // Model 1 is the reference structure.
  const atoms = frames.length > 0 ? frames[0].atoms : [];

  // --- Bonds: CONECT only, resolved against model 1's serials ---
  const serialToId: Record<number, number> = {};
  if (modelSerials.length > 0) {
    const refSerials = modelSerials[0];
    for (let i = 0; i < refSerials.length; i++) {
      const s = refSerials[i];
      if (Number.isFinite(s)) serialToId[s] = i + 1;
    }
  }

  const bonds: Bond[] = [];
  const bondSet = new Set<string>();
  let bondId = 1;
  for (const line of conectLines) {
    // Fixed 5-character fields (sect10): serial 7-11, bonded 12-16, 17-21,
    // 22-26, 27-31. Splitting on whitespace would fuse touching serials.
    const nums: number[] = [];
    for (let start = 6; start <= 26; start += 5) {
      const field = line.substring(start, start + 5).trim();
      if (!field) continue;
      const n = Number(field);
      if (Number.isFinite(n)) nums.push(n);
    }
    if (nums.length < 2) continue;
    const fromId = serialToId[nums[0]];
    if (!fromId) continue;
    for (let i = 1; i < nums.length; i++) {
      const toId = serialToId[nums[i]];
      if (toId && toId !== fromId) {
        const key = fromId < toId ? `${fromId}-${toId}` : `${toId}-${fromId}`;
        if (!bondSet.has(key)) {
          bondSet.add(key);
          bonds.push({ id: bondId++, type: 1, atom1Id: fromId, atom2Id: toId });
        }
      }
    }
  }

  // --- Type metadata from model 1 ---
  const atomTypes: Record<number, AtomTypeInfo> = {};
  const usedTypes = Array.from(new Set(atoms.map(a => a.type)));

  for (const type of usedTypes) {
    const elem = ELEMENT_DATA.find(e => e.number === type);
    let count = 0;
    for (const a of atoms) if (a.type === type) count++;
    atomTypes[type] = {
      id: type,
      mass: elem?.mass ?? 0,
      element: elem?.symbol ?? 'X',
      label: elem ? `${elem.name} (${elem.symbol})` : `Type ${type}`,
      count,
    };
  }

  // --- Extents across ALL kept models (consistent with the other parsers —
  // the canvas factors the box in separately for framing) ---
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const f of frames) {
    for (const a of f.atoms) {
      minX = Math.min(minX, a.x); maxX = Math.max(maxX, a.x);
      minY = Math.min(minY, a.y); maxY = Math.max(maxY, a.y);
      minZ = Math.min(minZ, a.z); maxZ = Math.max(maxZ, a.z);
    }
  }

  const safeCenter = (() => {
    if (box) return { x: box.xhi / 2, y: box.yhi / 2, z: box.zhi / 2 };
    if (atoms.length > 0) {
      return { x: (minX + maxX) / 2, y: (minY + maxY) / 2, z: (minZ + maxZ) / 2 };
    }
    return { x: 0, y: 0, z: 0 };
  })();

  return {
    atoms,
    bonds,
    atomTypes,
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center: safeCenter,
    ...(box ? { box } : {}),
    ...(frames.length > 1 ? { frames } : {}),
  };
};
