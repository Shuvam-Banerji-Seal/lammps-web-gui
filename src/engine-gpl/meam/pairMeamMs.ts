/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM/pair_meam_ms.cpp / pair_meam_ms.h
 * (PairMEAMMS: pair_style meam/ms), the coeff, read_global_meam_file and read_user_meam_file routines of
 * src/MEAM/pair_meam.cpp, and the compute() stage order. LAMMPS is Copyright (2003) Sandia Corporation and distributed
 * under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style meam/ms (docs.lammps.org/pair_meam_ms.html): the multi-state MEAM model of the library and parameter
 * files (the msmeam fields b1m..b3m, t1m..t3m in the library file) with the same pair_coeff form as pair meam
 * ("pair_coeff * * library.msmeam elements... params.msmeam elements..."). The engine's pair meam (src/engine) is a
 * separate implementation and is not changed.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { MeamModel, MAXELT, LibEntry, meamKeywords, toLattice, latticeCode } from './meamModel';
import { getZij } from './meamMath';
import { newWork, densInit, foldGhosts, densFinal, forwardGhosts, type ComputeInput } from './meamMsCompute';
import { forceAll } from './meamMsForce';

/** Lattices whose density and screening terms are ported (meamMsCompute.ts / meamModel.ts getDensref). */
const SUPPORTED_LATTICES = ['fcc', 'bcc', 'hcp', 'dim', 'dia', 'dia3', 'b1', 'c11', 'l12', 'b2', 'sc'];
const SUPPORTED_IBAR = [0, 1, 3, 4, -5];

/** Records of a potential file: comments after '#' removed, split on the given separators. */
const fileLines = (text: string): string[] => text.split(/\r?\n/).map((l) => {
  const h = l.indexOf('#');
  return h >= 0 ? l.slice(0, h) : l;
});

/** Library records (PotentialFileReader::next_line(19)): consecutive lines until at least 19 words. */
const libraryRecords = (text: string): string[][] => {
  const out: string[][] = [];
  let cur: string[] = [];
  for (const line of fileLines(text)) {
    const words = line.replace(/'/g, ' ').split(/[ \t\r\f]+/).filter((w) => w.length > 0);
    if (words.length === 0 && cur.length === 0) continue;
    cur.push(...words);
    if (cur.length >= 19) {
      out.push(cur);
      cur = [];
    }
  }
  if (cur.length > 0) throw new StyleError('MS-MEAM library file: incomplete last entry');
  return out;
};

const toNum = (s: string | undefined, what: string): number => {
  const v = Number(s);
  if (s === undefined || !Number.isFinite(v)) throw new StyleError(`MS-MEAM library file: expected a number for ${what}, got '${s ?? ''}'`);
  return v;
};

/** read_global_meam_file for the element list (msmeam fields). Throws StyleError for a bad entry. */
export const parseLibrary = (text: string, elements: string[], name: string): LibEntry[] => {
  const found = new Map<string, LibEntry>();
  for (const rec of libraryRecords(text)) {
    const element = rec[0];
    if (!elements.includes(element)) continue;
    if (found.has(element)) continue;
    let p = 1;
    const latName = rec[p++];
    const lat = toLattice(latName, true);
    if (lat === null) throw new StyleError(`Unrecognized lattice type in MEAM library file ${name}: ${latName}`);
    const next = (what: string): number => toNum(rec[p++], `${element} ${what} (${name})`);
    const e: LibEntry = {
      lat, z: 0, ielement: 0, atwt: 0, alpha: 0, b0: 0, b1: 0, b2: 0, b3: 0, b1m: 0, b2m: 0, b3m: 0,
      alat: 0, esub: 0, asub: 0, t0: 0, t1: 0, t2: 0, t3: 0, t1m: 0, t2m: 0, t3m: 0, rozero: 0, ibar: 0,
    };
    e.z = next('z');
    e.ielement = Math.trunc(next('ielement'));
    e.atwt = next('atwt');
    e.alpha = next('alpha');
    e.b0 = next('b0');
    e.b1 = next('b1');
    e.b2 = next('b2');
    e.b3 = next('b3');
    e.b1m = next('b1m');
    e.b2m = next('b2m');
    e.b3m = next('b3m');
    e.alat = next('alat');
    e.esub = next('esub');
    e.asub = next('asub');
    e.t0 = next('t0');
    e.t1 = next('t1');
    e.t2 = next('t2');
    e.t3 = next('t3');
    e.t1m = next('t1m');
    e.t2m = next('t2m');
    e.t3m = next('t3m');
    e.rozero = next('rozero');
    e.ibar = Math.trunc(next('ibar'));
    if (Math.abs(e.t0 - 1) >= 1e-20) throw new StyleError(`Unsupported parameter in MEAM library file: t0 != 1 (${element})`);
    if (e.rozero <= 0.0) throw new StyleError(`Invalid parameter in MEAM library file: rho0 must be > 0 (${element})`);
    const zWant = getZij(lat);
    if (e.z !== zWant) throw new StyleError(`Mismatched parameter in MEAM library file: z != lat (${element})`);
    found.set(element, e);
  }
  const missing = elements.filter((el) => !found.has(el));
  if (missing.length > 0) throw new StyleError(`Did not find all elements in MEAM library file ${name}, missing: ${missing.join(' ')}`);
  return elements.map((el) => found.get(el)!);
};

interface UserSetting { which: number; value: number; idx: number[] }

/** read_user_meam_file: "keyword(i,j,...) = value" lines. */
export const parseUserFile = (text: string, name: string): UserSetting[] => {
  const out: UserSetting[] = [];
  for (const line of fileLines(text)) {
    const tok = line.split(/[=(), '\t\r\f]+/).filter((w) => w.length > 0);
    if (tok.length === 0) continue;
    const keyword = tok[0];
    const which = meamKeywords.indexOf(keyword);
    if (which < 0) throw new StyleError(`Keyword ${keyword} in MEAM parameter file ${name} not recognized`);
    const nparams = tok.length;
    const nindex = nparams - 2;
    const idx: number[] = [];
    for (let i = 0; i < nindex; i++) {
      const v = Number(tok[1 + i]);
      if (!Number.isInteger(v)) throw new StyleError(`Error parsing MEAM parameter file ${name}: keyword ${keyword}`);
      idx.push(v - 1);
    }
    let value: number;
    if (which === 4) {
      const latName = tok[1 + nindex];
      const lat = toLattice(latName ?? '', false);
      if (lat === null) throw new StyleError(`Unrecognized lattice type ${latName} in MEAM parameter file ${name}`);
      value = latticeCode(lat);
    } else {
      value = Number(tok[1 + nindex]);
      if (tok[1 + nindex] === undefined || !Number.isFinite(value)) throw new StyleError(`Error parsing MEAM parameter file ${name}: keyword ${keyword}`);
    }
    if (which === 2 && value <= 0.0) throw new StyleError(`Error in MEAM parameter file ${name}: rho0 must be > 0`);
    out.push({ which, value, idx });
  }
  return out;
};

export class PairMeamMs extends Pair {
  readonly name = 'meam/ms';
  manybody = true;
  needsFull = true;
  needsHalf = true;
  virialFdotr = true;

  private model: MeamModel | null = null;
  /** LAMMPS type (index) -> element index, -1 if unmapped (index 0 unused) */
  private typeElem: number[] = [];

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('usage: pair_style meam/ms (no arguments)');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 6) throw new StyleError('Illegal pair_coeff command for pair style meam/ms: missing arguments');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style meam/ms must be * *');
    const libFile = args[2];
    // the parameter file is the first word after the element names that is a file or NULL (pair_meam.cpp coeff)
    let paridx = -1;
    let parFile = '';
    for (let i = 4; i < args.length; i++) {
      if (args[i] === 'NULL') {
        parFile = 'NULL';
        paridx = i;
        break;
      }
      try {
        parFile = ctx.readFile(args[i]);
        paridx = i;
        break;
      } catch {
        // not a file: an element name
      }
    }
    if (paridx < 0) throw new StyleError('No MS-MEAM parameter file in pair coefficients');
    if (args.length - paridx - 1 !== this.ntypes) {
      throw new StyleError(`Expected ${1 + paridx + this.ntypes} but found ${args.length} args for pair style meam/ms coefficients`);
    }
    const elements = args.slice(3, paridx);
    if (elements.length < 1) throw new StyleError('Incorrect args for pair coefficients');
    if (elements.length > MAXELT) throw new StyleError(`Too many elements extracted from MEAM library (current limit: ${MAXELT})`);
    for (let i = 0; i < elements.length; i++) {
      if (elements.indexOf(elements[i]) !== i) throw new StyleError('Must not extract the same element from MEAM library twice');
    }
    const lib = parseLibrary(ctx.readFile(libFile), elements, libFile);
    const M = new MeamModel();
    M.setupGlobal(lib);
    if (parFile !== 'NULL') {
      for (const s of parseUserFile(parFile, args[paridx])) {
        try {
          M.setupParam(s.which, s.value, s.idx);
        } catch (e) {
          throw new StyleError(`Error in MEAM parameter file ${args[paridx]}: keyword ${meamKeywords[s.which]} ${(e as Error).message}`);
        }
      }
    }
    for (let i = 0; i < M.neltypes; i++) {
      const lat = M.lattce[i][i];
      if (!SUPPORTED_LATTICES.includes(lat)) throw new StyleError(`MS-MEAM reference lattice '${lat}' of ${elements[i]} is not supported by the engine port`);
      if (!SUPPORTED_IBAR.includes(M.ibar[i])) throw new StyleError(`MS-MEAM ibar = ${M.ibar[i]} of ${elements[i]} is not supported (only 0, 1, 3, 4 and -5)`);
    }
    for (let a = 0; a < M.neltypes; a++) {
      for (let b = a; b < M.neltypes; b++) {
        if (!SUPPORTED_LATTICES.includes(M.lattce[a][b])) {
          throw new StyleError(`MS-MEAM lattce(${a + 1},${b + 1}) = ${M.lattce[a][b]} is not supported by the engine port`);
        }
      }
    }
    if (M.ialloy !== 1) throw new StyleError(`MS-MEAM ialloy = ${M.ialloy} is not supported (only ialloy = 1, the MS-MEAM default)`);
    M.setupDone();
    this.model = M;
    const typeElem: number[] = [-1];
    for (let t = 1; t <= this.ntypes; t++) {
      const m = args[paridx + t];
      const j = elements.indexOf(m);
      if (j >= 0) {
        typeElem[t] = j;
        if (ctx.s && !ctx.s.rmass) ctx.s.massByType[t] = lib[j].atwt;
      } else if (m === 'NULL') typeElem[t] = -1;
      else throw new StyleError(`Incorrect element mapping for pair style meam/ms coefficients: ${m}`);
    }
    this.typeElem = typeElem;
  }

  override initOne(_i: number, _j: number): number {
    if (!this.model) throw new StyleError('pair_coeff for style meam/ms must be set before the run');
    return this.model.cutforce;
  }

  override compute(pc: PairCompute): void {
    const M = this.model;
    if (!M) throw new StyleError('pair_coeff for style meam/ms must be set before the run');
    const full = pc.full, half = pc.half;
    if (!full || !half) throw new Error('pair style meam/ms needs half and full neighbor lists');
    const nall = pc.nall;
    const w = newWork(nall, half.neighbors.length);
    const c: ComputeInput = {
      model: M, nlocal: pc.nlocal, nall, x: pc.x, f: pc.f, type: pc.type, typeElem: this.typeElem,
      owner: pc.nb.owner, half, full, mask: NEIGHMASK, eatom: pc.eatom,
    };
    densInit(c, w);
    foldGhosts(c, w);
    const eDens = densFinal(c, w);
    forwardGhosts(c, w);
    const ePair = forceAll(c, w, 1.0 / M.dr);
    pc.acc.evdwl += eDens + ePair;
  }
}
