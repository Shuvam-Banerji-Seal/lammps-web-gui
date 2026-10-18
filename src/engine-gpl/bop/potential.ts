/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: PairBOP::read_table in
 * src/MANYBODY/pair_bop.cpp, with the PotentialFileReader and TextFileReader calls it makes
 * (src/potential_file_reader.cpp, src/text_file_reader.cpp: next_line, next_dvector, next_values). Contributing
 * authors of pair_bop.cpp: D.K. Ward and X.W. Zhou (Sandia). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_bop.html, "The parameters/coefficients format for the BOP potentials input file": line 1
 * "# elements N", N lines "atomic number, mass, symbol", then the global variables (nr, nBOt or nr, ntheta, nBOt), seven
 * small values, the pi_p values, per pair the rcut and the coefficient lines, then the tables.
 */
import { StyleError } from '../../engine/force/types';
import { nextDouble, potentialUnitFactor } from '../manybody2/common';
import type { StyleContext } from '../../engine/force/types';
import { TabularFunction } from './tabular';

const FILETYPE = 'BOP potential';
const INT_RE = /^[+-]?\d+$/;

const nextInteger = (word: string | undefined): number => {
  if (word === undefined) throw new StyleError('No more tokens');
  if (!INT_RE.test(word)) throw new StyleError(`Not a valid integer number: '${word}'`);
  return Number(word);
};

/** One pair (PairBOP::PairParameters) of the file: cutoffs and the tables. */
export interface BopPair {
  cutB: number;
  cutBsq: number;
  cutL: number;
  cutLsq: number;
  betaS: TabularFunction;
  betaP: TabularFunction;
  rep: TabularFunction;
  /** Only when the file has a cutL (cphi) table with a nonzero cutoff. */
  cphi: TabularFunction | null;
  bo: TabularFunction;
}

/** The parsed potential file (the state read_table leaves in PairBOP). */
export interface BopPotential {
  nb: number;
  names: string[];
  masses: number[];
  /** Global values: nr, ntheta, nBOt, npower, the seven small values and the largest cutB. */
  nr: number;
  nBOt: number;
  npower: number;
  small: number[];
  cutmax: number;
  piP: Float64Array;
  /** Per pair index (native order): the coefficient arrays. */
  sigmaC: Float64Array;
  sigmaA: Float64Array;
  piC: Float64Array;
  piA: Float64Array;
  sigmaDelta: Float64Array;
  piDelta: Float64Array;
  sigmaF: Float64Array;
  sigmaK: Float64Array;
  small3: Float64Array;
  pairs: BopPair[];
  /** elem2param[i * nb + j]: pair index. */
  elem2param: Int32Array;
  /** elem3param[(j * nb + i) * nb + k]: triplet table index for the angle centred on i. */
  elem3param: Int32Array;
  triplets: TabularFunction[];
}

/** Line reader with TextFileReader's semantics: '#' comments removed, blank lines skipped, lines joined to nparams words. */
class BopFileReader {
  private pos = 0;
  constructor(private readonly lines: readonly string[]) {}

  private words(line: string): string[] {
    const h = line.indexOf('#');
    const t = h >= 0 ? line.slice(0, h) : line;
    return t.split(/[ \t\r\n\f]+/).filter((w) => w.length > 0);
  }

  /** TextFileReader::next_line(nparams): null at the end of the file. */
  nextLine(nparams: number): string[] | null {
    let w: string[] = [];
    while (w.length === 0 || w.length < nparams) {
      if (this.pos >= this.lines.length) {
        if (w.length > 0 && w.length < nparams) {
          throw new StyleError(`Incorrect format in ${FILETYPE} file! ${w.length}/${nparams} parameters`);
        }
        return null;
      }
      w = w.concat(this.words(this.lines[this.pos++]));
    }
    return w;
  }

  private required(nparams: number): string[] {
    const w = this.nextLine(nparams);
    if (w === null) throw new StyleError(`Missing line in ${FILETYPE} file!`);
    return w;
  }

  /** next_values(n): the words of the joined lines (ValueTokenizer over them). */
  nextValues(n: number): string[] { return this.required(n); }

  /** next_double(): the first word of the next non-blank line. */
  nextDouble(): number { return nextDouble(this.required(1)[0]); }

  /** next_int(): the first word of the next non-blank line. */
  nextInt(): number { return nextInteger(this.required(1)[0]); }

  /** next_dvector(list, n): n numbers; a line's words beyond the needed count are dropped. */
  nextDvector(n: number): Float64Array {
    const out = new Float64Array(n);
    let i = 0;
    while (i < n) {
      const w = this.nextLine(0);
      if (w === null) {
        if (i === 0) throw new StyleError('EOF reached');
        throw new StyleError(`Incorrect format in ${FILETYPE} file! ${i}/${n} values`);
      }
      for (let k = 0; k < w.length && i < n; k++) out[i++] = nextDouble(w[k]);
    }
    return out;
  }
}

/** Parses the text of a BOP potential file as PairBOP::read_table does (metal units only, no conversion). */
export const parseBopPotential = (text: string, fileName: string, ctx: StyleContext): BopPotential => {
  potentialUnitFactor(text, FILETYPE, fileName, ctx, false);
  const r = new BopFileReader(text.split(/\r?\n/));
  const nb = r.nextInt();
  if (nb <= 0) throw new StyleError(`BOP potential file with ${nb} elements`);
  const names: string[] = [];
  const masses: number[] = [];
  for (let i = 0; i < nb; i++) {
    const v = r.nextValues(3);
    nextInteger(v[0]);
    masses.push(nextDouble(v[1]));
    if (v[2] === undefined) throw new StyleError('No more tokens');
    names.push(v[2]);
  }
  const npairs = (nb * (nb + 1)) / 2;

  // the header: 2 words (nr, nBOt: parabolic g) or 3 words (nr, ntheta, nBOt)
  const head = r.nextValues(2);
  const format = head.length;
  let nr: number, ntheta: number, nBOt: number, npower: number;
  if (format === 3) {
    nr = nextInteger(head[0]);
    ntheta = nextInteger(head[1]);
    nBOt = nextInteger(head[2]);
    if (ntheta > 10) {
      npower = 2;
    } else {
      npower = ntheta;
      ntheta = 2000;
    }
  } else if (format === 2) {
    nr = nextInteger(head[0]);
    ntheta = 2000;
    nBOt = nextInteger(head[1]);
    npower = 2;
  } else {
    throw new StyleError('Unsupported BOP potential file format');
  }

  const sv = r.nextValues(7);
  const small = [0, 1, 2, 3, 4, 5, 6].map((k) => nextDouble(sv[k]));

  const piP = new Float64Array(nb);
  for (let i = 0; i < nb; i++) piP[i] = r.nextDouble();

  const rcut = new Float64Array(npairs);
  const sigmaC = new Float64Array(npairs), sigmaA = new Float64Array(npairs);
  const piC = new Float64Array(npairs), piA = new Float64Array(npairs);
  const sigmaDelta = new Float64Array(npairs), piDelta = new Float64Array(npairs);
  const sigmaF = new Float64Array(npairs), sigmaK = new Float64Array(npairs);
  const small3 = new Float64Array(npairs);
  let cutmax = 0.0;
  for (let i = 0; i < npairs; i++) {
    rcut[i] = r.nextDouble();
    cutmax = rcut[i] > cutmax ? rcut[i] : cutmax;
    let v = r.nextValues(4);
    sigmaC[i] = nextDouble(v[0]); sigmaA[i] = nextDouble(v[1]); piC[i] = nextDouble(v[2]); piA[i] = nextDouble(v[3]);
    v = r.nextValues(2);
    sigmaDelta[i] = nextDouble(v[0]); piDelta[i] = nextDouble(v[1]);
    v = r.nextValues(3);
    sigmaF[i] = nextDouble(v[0]); sigmaK[i] = nextDouble(v[1]); small3[i] = nextDouble(v[2]);
  }

  // the angular table: 3 words per header means a table with npower = ntheta (<= 10) reads gpara polynomials
  if (format === 3 && npower <= 2) {
    throw new StyleError(
      'BOP potential file with a 3-value header and npower <= 2 (a tabulated angular function) is not supported: ' +
      'native LAMMPS reads its polynomial buffer before it is initialised',
    );
  }
  const elem3param = new Int32Array(nb * nb * nb);
  const triplets: TabularFunction[] = [];
  const singletable = new Float64Array(ntheta);
  for (let i = 0; i < nb; i++) {
    for (let j = 0; j < nb; j++) {
      for (let k = j; k < nb; k++) {
        const gpara = r.nextDvector(npower + 1);
        for (let n = 0; n < ntheta; n++) {
          const arg = -1.0 + 2.0 * n / (ntheta - 1.0);
          singletable[n] = gpara[npower];
          for (let m = npower; m > 0; m--) singletable[n] = arg * singletable[n] + gpara[m - 1];
        }
        const tab = new TabularFunction();
        tab.setValues(ntheta, -1.0, 1.0, singletable);
        const nbuf = triplets.length;
        triplets.push(tab);
        elem3param[(j * nb + i) * nb + k] = nbuf;
        if (k !== j) elem3param[(k * nb + i) * nb + j] = nbuf;
      }
    }
  }

  const pairs: BopPair[] = [];
  for (let i = 0; i < npairs; i++) {
    const rep = new TabularFunction();
    rep.setValues(nr, 0.0, rcut[i], r.nextDvector(nr));
    pairs.push({
      cutB: rcut[i], cutBsq: rcut[i] * rcut[i], cutL: 0, cutLsq: 0,
      betaS: new TabularFunction(), betaP: new TabularFunction(), rep, cphi: null, bo: new TabularFunction(),
    });
  }
  for (let i = 0; i < npairs; i++) {
    pairs[i].betaS.setValues(nr, 0.0, rcut[i], r.nextDvector(nr));
  }
  for (let i = 0; i < npairs; i++) {
    pairs[i].betaP.setValues(nr, 0.0, rcut[i], r.nextDvector(nr));
  }
  for (let i = 0; i < npairs; i++) {
    pairs[i].bo.setValues(nBOt, 0.0, 1.0, r.nextDvector(nBOt));
  }

  const elem2param = new Int32Array(nb * nb);
  let nbuf = 0;
  for (let i = 0; i < nb; i++) {
    elem2param[i * nb + i] = nbuf;
    nbuf++;
  }
  for (let i = 0; i < nb - 1; i++) {
    for (let j = i + 1; j < nb; j++) {
      elem2param[i * nb + j] = nbuf;
      elem2param[j * nb + i] = nbuf;
      nbuf++;
    }
  }

  r.nextDvector(nb); // pro_delta (not used by the force evaluation)
  r.nextDvector(nb); // pro
  const cutL = r.nextDvector(npairs);
  let addCut = cutL[0];
  for (let i = 0; i < npairs; i++) {
    if (addCut < cutL[i]) addCut = cutL[i];
    pairs[i].cutL = cutL[i];
    pairs[i].cutLsq = cutL[i] * cutL[i];
  }
  if (addCut > 0.0) {
    for (let i = 0; i < npairs; i++) {
      const table = r.nextDvector(nr);
      let any = 0;
      for (let j = 0; j < nr; j++) if (table[j] !== 0) any = 1;
      if (any === 0) {
        cutL[i] = 0.0;
        pairs[i].cutL = cutL[i];
        pairs[i].cutLsq = cutL[i] * cutL[i];
      }
      if (cutL[i] !== 0.0) {
        const cphi = new TabularFunction();
        cphi.setValues(nr, 0.0, cutL[i], table);
        pairs[i].cphi = cphi;
      }
    }
  }

  return {
    nb, names, masses, nr, nBOt, npower, small, cutmax, piP, sigmaC, sigmaA, piC, piA, sigmaDelta, piDelta,
    sigmaF, sigmaK, small3, pairs, elem2param, elem3param, triplets,
  };
};
