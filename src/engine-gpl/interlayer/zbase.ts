/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_kolmogorov_crespi_z.cpp, pair_kolmogorov_crespi_z.h,
 * pair_lebedeva_z.cpp and pair_lebedeva_z.h, at commit b332f28 (shared coefficient and
 * potential-file handling: the element mapping, the parameter table and the energy offset).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext, pairCoeffArgs } from '../../engine/force/types';
import { joinPotentialEntries, numeric } from '../../engine/force/util';
import { pow as libmPow } from '../libm/pow';

/** One potential-file entry: element indices (into the style's element list) and its numbers. */
interface FileEntry { ie: number; je: number; nums: number[] }

/** 1 eV in kcal/mol: the energy conversion between metal and real units (the physical constant). */
const EV_TO_KCAL = 23.060549;

/** Energy scale of a potential file with a UNITS: tag, against the simulation's unit style (metal and real only). */
export const energyScale = (tag: string | null, ctx: StyleContext, style: string, fileName: string): number => {
  if (!tag) return 1;
  const units = ctx.s?.units.style;
  if (tag === units) return 1;
  if (tag === 'metal' && units === 'real') return EV_TO_KCAL;
  if (tag === 'real' && units === 'metal') return 1 / EV_TO_KCAL;
  throw new StyleError(`${style} potential file ${fileName} requires ${tag} units but the simulation uses ${units ?? 'an unknown'} unit style; automatic conversion is only supported between metal and real`);
};

/** Text of a potential-file line without its comment. */
export const stripComment = (line: string): string => {
  const h = line.indexOf('#');
  return (h >= 0 ? line.slice(0, h) : line).trim();
};

/**
 * Base of the Kolmogorov-Crespi-z and Lebedeva-z pair styles: pair_style takes one cutoff, pair_coeff takes a
 * potential file and one element per atom type. Each pair_coeff re-reads the file (as LAMMPS does) and keeps the
 * element list it has seen so far. Newton's third law must be on (the style needs a half list with ghosts folded).
 */
export abstract class InterlayerZBase<P> extends Pair {
  /** Words per potential-file entry (the two element names and the numbers). */
  protected abstract readonly nwords: number;
  /** Builds a parameter entry from the numbers after the two element names (energies already scaled). */
  protected abstract makeParam(nums: number[], ie: number, je: number, scale: number): P;
  /** Energy offset of a pair at the cutoff (pair_modify shift yes). */
  protected abstract offsetOf(p: P, cut: number): number;
  /** The pair-style cutoff (settings). */
  protected cutGlobal = 0;
  protected elements: string[] = [];
  /** Element index of each atom type (-1 for NULL). */
  protected map = new Int32Array(0);
  protected setflag = new Uint8Array(0);
  protected params: P[] = [];
  /** Parameter entry of each type pair (nt*nt, symmetric), set in initOne. */
  protected pairP: (P | null)[] = [];
  protected offset = new Float64Array(0);
  protected elem2param: Int32Array = new Int32Array(0);

  constructor() {
    super();
    this.shift = true;
    this.virialFdotr = true;
    this.needsHalf = true;
    this.needsFull = false;
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.map = new Int32Array(nt).fill(-1);
    this.setflag = new Uint8Array(nt * nt);
    this.offset = new Float64Array(nt * nt);
    this.pairP = new Array<P | null>(nt * nt).fill(null);
  }

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError(`Pair style ${this.name} requires exactly one argument`);
    // the pair_style of the command must be hybrid/overlay (strmatch on force->pair_style)
    if (!this.hybridOverlay) throw new StyleError(`Pair style ${this.name} must be used as hybrid/overlay sub-style`);
    this.cutGlobal = numeric(args[0]);
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 4) throw pairCoeffArgs(`usage: pair_coeff I J filename elem1 ... elemN (style ${this.name})`);
    const nt = this.ntypes + 1;
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style ${this.name} needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    }
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') {
        this.map[t] = -1;
        continue;
      }
      let idx = this.elements.indexOf(name);
      if (idx < 0) {
        this.elements.push(name);
        idx = this.elements.length - 1;
      }
      this.map[t] = idx;
    }
    this.readFile(args[2], ctx);
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        if (this.map[i] >= 0 && this.map[j] >= 0) {
          this.setflag[i * nt + j] = 1;
          this.setflag[j * nt + i] = 1;
          count++;
        }
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** Reads the potential file (entries whose two element names are both in the element list). */
  protected readFile(fileName: string, ctx: StyleContext): void {
    const text = ctx.readFile(fileName);
    const lines = text.split('\n');
    const m = /UNITS:\s*(\S+)/.exec(lines[0] ?? '');
    const unitTag = m ? m[1] : null;
    const scale = energyScale(unitTag, ctx, this.name, fileName);
    const joined = joinPotentialEntries(lines, this.nwords);
    const maxEntries = this.entryLimit();
    const entries: FileEntry[] = [];
    const out: P[] = [];
    for (const raw of joined) {
      const words = stripComment(raw).split(/\s+/).filter((w) => w !== '');
      if (words.length === 0) continue;
      if (words.length < this.nwords) throw new StyleError(`${this.name} potential file ${fileName}: expected ${this.nwords} words per entry, got ${words.length}`);
      const ie = this.elements.indexOf(words[0]);
      const je = this.elements.indexOf(words[1]);
      if (ie < 0 || je < 0) continue;
      const nums = words.slice(2, this.nwords).map((w) => {
        const v = Number(w);
        if (!Number.isFinite(v)) throw new StyleError(`${this.name} potential file ${fileName}: expected a number, got '${w}'`);
        return v;
      });
      entries.push({ ie, je, nums });
      out.push(this.makeParam(nums, ie, je, scale));
      if (entries.length >= maxEntries) break;
    }
    const ne = this.elements.length;
    const elem2 = new Int32Array(ne * ne);
    for (let i = 0; i < ne; i++) {
      for (let j = 0; j < ne; j++) {
        let n = -1;
        for (let k = 0; k < entries.length; k++) {
          if (entries[k].ie === i && entries[k].je === j) {
            if (n >= 0) throw new StyleError(`Potential file ${fileName} has a duplicate entry for: ${this.elements[i]} ${this.elements[j]}`);
            n = k;
          }
        }
        if (n < 0) throw new StyleError(`Potential file ${fileName} is missing an entry for: ${this.elements[i]} ${this.elements[j]}`);
        elem2[i * ne + j] = n;
      }
    }
    this.params = out;
    this.elem2param = elem2;
  }

  /** The reader stops after this many matching entries (LAMMPS reads ntypes^3 of them for the z styles). */
  protected entryLimit(): number {
    return libmPow(this.ntypes, 3);
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError(`Pair style ${this.name} requires newton pair on`);
  }

  override initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.setflag[i * nt + j]) throw new StyleError(`All pair coeffs are not set for pair style ${this.name} (types ${i} ${j})`);
    if (!this.shift) throw new StyleError(`Must use 'pair_modify shift yes' with this pair style`);
    const ne = this.elements.length;
    const p = this.params[this.elem2param[this.map[i] * ne + this.map[j]]];
    this.pairP[i * nt + j] = p;
    this.pairP[j * nt + i] = p;
    const off = this.cutGlobal > 0 ? this.offsetOf(p, this.cutGlobal) : 0;
    this.offset[i * nt + j] = off;
    this.offset[j * nt + i] = off;
    return this.cutGlobal;
  }
}

/**
 * Per-atom tallies for a central-plus-extra pair force (LAMMPS Pair::ev_tally with the extra
 * v_tally2_newton term). Only the per-atom parts are written: the global virial comes from the force field's
 * sum x.f (virialFdotr).
 */
export const tallyZPair = (
  pc: PairCompute, i: number, j: number, evdwl: number, fpair: number, dx: number, dy: number, dz: number,
  fi: readonly number[] | null,
): void => {
  if (pc.eatom) {
    pc.eatom[i] += 0.5 * evdwl;
    pc.eatom[j] += 0.5 * evdwl;
  }
  if (!pc.vatom) return;
  const va = pc.vatom;
  const w = [dx * dx * fpair, dy * dy * fpair, dz * dz * fpair, dx * dy * fpair, dx * dz * fpair, dy * dz * fpair];
  for (let c = 0; c < 6; c++) {
    va[6 * i + c] += 0.5 * w[c];
    va[6 * j + c] += 0.5 * w[c];
  }
  if (fi) {
    // v_tally2_newton: x_i (x) f_i for the extra force on i, and -f_i on j (the in-plane part of the force)
    const xi = pc.x[3 * i], yi = pc.x[3 * i + 1], zi = pc.x[3 * i + 2];
    const xj = pc.x[3 * j], yj = pc.x[3 * j + 1], zj = pc.x[3 * j + 2];
    const fx = fi[0], fy = fi[1], fz = fi[2];
    const addAt = (a: number, px: number, py: number, pz: number, gx: number, gy: number, gz: number) => {
      va[6 * a] += px * gx;
      va[6 * a + 1] += py * gy;
      va[6 * a + 2] += pz * gz;
      va[6 * a + 3] += px * gy;
      va[6 * a + 4] += px * gz;
      va[6 * a + 5] += py * gz;
    };
    addAt(i, xi, yi, zi, fx, fy, fz);
    addAt(j, xj, yj, zj, -fx, -fy, -fz);
  }
};
