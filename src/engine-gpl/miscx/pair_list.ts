/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MISC/pair_list.cpp and
 * pair_list.h (the LAMMPS developers; Copyright (2003) Sandia Corporation, GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style list (docs.lammps.org/pair_list.html): pair interactions read from a file, "pair_style list listfile
 * cutoff [keyword]". Each line is "ID1 ID2 style coeffs [cutoff]" with style harmonic, morse, lj126 or quartic (other
 * styles are skipped with a warning). The style does not use a neighbor list: it looks up the two atoms by ID and
 * takes the raw coordinate difference x_1 - x_2 (native: atom->map(id), no minimum image). pair_coeff takes no
 * parameters; the file gives them.
 */
import { Pair, StyleError, evTally, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { exp } from '../libm/exp';
import { buildAtomMap } from '../../engine/atoms';

const NONE = 0, HARM = 1, MORSE = 2, LJ126 = 3, QUARTIC = 4;

/** pair_list.cpp stylename map: an unknown name maps to NONE (the entry is skipped with a warning). */
const STYLE_NAMES: Record<string, number> = { none: NONE, harmonic: HARM, morse: MORSE, lj126: LJ126, quartic: QUARTIC };

/** One line of the list file (list_param). */
interface ListParam {
  id1: number;
  id2: number;
  style: number;
  k: number;
  r0: number;
  d0: number;
  alpha: number;
  epsilon: number;
  sigma: number;
  b1: number;
  b2: number;
  offset: number;
  cutsq: number;
}

/** ValueTokenizer::next_tagint: an integer atom ID. */
const nextTagint = (w: string | undefined): number => {
  if (w === undefined || !/^[-+]?\d+$/.test(w)) throw new Error(`expected an integer, got '${w ?? ''}'`);
  return Number(w);
};

/** MathSpecial-style integer power by squaring (pair_list.cpp mypow): 0 for x == 0. */
const mypow = (x: number, n: number): number => {
  if (x === 0.0) return 0.0;
  let yy = 1.0;
  let nn = n;
  let xx = x;
  for (; nn !== 0; nn >>= 1, xx *= xx) if (nn & 1) yy *= xx;
  return yy;
};

export class PairList extends Pair {
  readonly name = 'list';
  /** No neighbor list: the style looks its pairs up by atom ID. */
  needsHalf = false;
  /** The style tallies its virial pair by pair (evTally), not through the fdotr sum. */
  explicitVirial = true;
  private cutGlobal = 0;
  private checkFlag = true;
  private params: ListParam[] = [];
  private setflag = new Uint8Array(0);

  /** pair_list.cpp settings(): "listfile cutoff [nocheck|check]". */
  settings(args: string[], ctx: StyleContext): void {
    if (args.length < 2) throw new StyleError('Illegal pair_style list command');
    const file = args[0];
    this.cutGlobal = parseNum(args[1], 'cutoff');
    let iarg = 2;
    while (iarg < args.length) {
      // native compares arg[2] (not arg[iarg]) with "check"; the same test is kept here
      if (args[iarg] === 'nocheck') {
        this.checkFlag = false;
        iarg++;
      } else if (args[2] === 'check') {
        this.checkFlag = true;
        iarg++;
      } else {
        throw new StyleError(`Unknown pair style list keyword: ${args[iarg]}`);
      }
    }
    const text = ctx.readFile(file, (n, reason) => `Error opening pair list coeffs file ${n}: ${reason}`);
    const out: ListParam[] = [];
    let nharm = 0, nmorse = 0, nlj126 = 0, nquartic = 0, nskipped = 0;
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.replace(/#.*$/, '');
      const words = line.trim().split(/\s+/).filter((w) => w.length > 0);
      if (words.length === 0) continue;
      const p: ListParam = {
        id1: 0, id2: 0, style: NONE, k: 0, r0: 0, d0: 0, alpha: 0, epsilon: 0, sigma: 0, b1: 0, b2: 0, offset: 0, cutsq: 0,
      };
      let at = 0;
      const next = (): string | undefined => words[at++];
      try {
        p.id1 = nextTagint(next());
        p.id2 = nextTagint(next());
        const name = next();
        p.style = name === undefined ? NONE : (STYLE_NAMES[name] ?? NONE);
        const num = (what: string) => parseNum(next(), what);
        switch (p.style) {
          case HARM:
            p.k = num('k'); p.r0 = num('r0'); nharm++;
            break;
          case MORSE:
            p.d0 = num('d0'); p.alpha = num('alpha'); p.r0 = num('r0'); nmorse++;
            break;
          case LJ126:
            p.epsilon = num('epsilon'); p.sigma = num('sigma'); nlj126++;
            break;
          case QUARTIC:
            p.k = num('k'); p.r0 = num('r0'); p.b1 = num('b1'); p.b2 = num('b2'); nquartic++;
            break;
          default:
            ctx.log(`WARNING: Skipping unrecognized pair list potential entry: ${line.trim()}`);
            nskipped++;
            break;
        }
      } catch (e) {
        throw new StyleError(`Error reading pair list coeffs file: ${(e as Error).message}\n${line}`);
      }
      // an optional cutoff of the pair; the global cutoff otherwise
      if (at < words.length) {
        const c = parseNum(words[at], 'cutoff');
        p.cutsq = c * c;
      } else p.cutsq = this.cutGlobal * this.cutGlobal;
      out.push(p);
    }
    ctx.log(`Read ${out.length} (${nharm}/${nmorse}/${nlj126}/${nquartic}) interacting pair lines from ${file}. ${nskipped} skipped entries.`);
    this.params = out;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.setflag = new Uint8Array((ntypes + 1) * (ntypes + 1));
  }

  /** pair_list.cpp coeff(): no parameters, only the set flags for the I <= J range. */
  coeff(args: string[]): void {
    if (args.length < 2) throw new StyleError('Illegal pair_coeff list command');
    const iR = typeBounds(args[0], this.ntypes), jR = typeBounds(args[1], this.ntypes);
    let count = 0;
    const nt = this.ntypes + 1;
    for (let i = iR[0]; i <= iR[1]; i++) {
      for (let j = Math.max(jR[0], i); j <= jR[1]; j++) {
        this.setflag[i * nt + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  setsPair(i: number, j: number): boolean {
    return this.setflag[i * (this.ntypes + 1) + j] === 1;
  }

  /** pair_list.cpp init_style(): the energy offsets at each pair's own cutoff when pair_modify shift yes. */
  initStyle(_ctx: StyleContext): void {
    if (!this.shift) return;
    for (const par of this.params) {
      if (par.style === HARM) {
        const dr = Math.sqrt(par.cutsq) - par.r0;
        par.offset = par.k * dr * dr;
      } else if (par.style === MORSE) {
        const dr = par.r0 - Math.sqrt(par.cutsq);
        const dexp = exp(par.alpha * dr);
        par.offset = par.d0 * (dexp * dexp - 2.0 * dexp - 1.0);
      } else if (par.style === LJ126) {
        const r6inv = par.cutsq * par.cutsq * par.cutsq;
        const sig6 = mypow(par.sigma, 6);
        par.offset = 4.0 * par.epsilon * r6inv * (sig6 * sig6 * r6inv - sig6);
      } else if (par.style === QUARTIC) {
        par.offset = 0.0;
      }
    }
  }

  /** pair_list.cpp init_one(): the global cutoff. */
  initOne(_i: number, _j: number): number {
    return this.cutGlobal;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const nlocal = pc.nlocal;
    const ids = s.id;
    let maxtag = 0;
    for (let i = 0; i < nlocal; i++) if (ids[i] > maxtag) maxtag = ids[i];
    // atom->map(tag) for the owned atoms (serial: every atom is local)
    const map = buildAtomMap(s);
    const x = pc.x, f = pc.f;
    let pcount = 0;
    for (const par of this.params) {
      if (par.id1 < 1 || par.id1 > maxtag) {
        if (this.checkFlag) throw new StyleError(`Invalid pair list atom ID ${par.id1}`);
        continue;
      }
      if (par.id2 < 1 || par.id2 > maxtag) {
        if (this.checkFlag) throw new StyleError(`Invalid pair list atom ID ${par.id2}`);
        continue;
      }
      const i = map[par.id1] ?? -1;
      const j = map[par.id2] ?? -1;
      if (i < 0 || j < 0) continue;
      if (i >= nlocal && j >= nlocal) continue;
      const dx = x[3 * i] - x[3 * j];
      const dy = x[3 * i + 1] - x[3 * j + 1];
      const dz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      let fpair = 0.0, epair = 0.0;
      // both atoms are owned (newton_pair on): two pairs counted per interaction
      if (this.checkFlag) pcount += 2;
      if (rsq < par.cutsq) {
        const r2inv = 1.0 / rsq;
        if (par.style === HARM) {
          const r = Math.sqrt(rsq);
          const dr = par.r0 - r;
          fpair = 2.0 * par.k * dr / r;
          epair = par.k * dr * dr - par.offset;
        } else if (par.style === MORSE) {
          const r = Math.sqrt(rsq);
          const dr = r - par.r0;
          const dexp = exp(-par.alpha * dr);
          fpair = 2.0 * par.d0 * par.alpha * (dexp * dexp - dexp) / r;
          epair = par.d0 * (dexp * dexp - 2.0 * dexp + 1.0) - par.offset;
        } else if (par.style === LJ126) {
          const r6inv = r2inv * r2inv * r2inv;
          const sig6 = mypow(par.sigma, 6);
          fpair = 24.0 * par.epsilon * r6inv * (2.0 * sig6 * sig6 * r6inv - sig6) * r2inv;
          epair = 4.0 * par.epsilon * r6inv * (sig6 * sig6 * r6inv - sig6) - par.offset;
        } else if (par.style === QUARTIC) {
          const r = Math.sqrt(rsq);
          const dr = r - par.r0;
          const ra = dr - par.b1;
          const rb = dr - par.b2;
          const r2 = dr * dr;
          fpair = -par.k / r * (r2 * (ra + rb) + 2.0 * dr * ra * rb);
          epair = par.k * r2 * ra * rb;
        }
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        evTally(pc, i, j, epair, 0.0, fpair, dx, dy, dz);
      }
    }
    if (this.checkFlag && pcount !== 2 * this.params.length) {
      throw new StyleError(`Not all pairs processed in pair_style list: ${pcount} vs ${2 * this.params.length}`);
    }
  }
}
