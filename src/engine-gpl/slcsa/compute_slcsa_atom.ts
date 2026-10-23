/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_slcsa_atom.cpp and compute_slcsa_atom.h (class ComputeSLCSAAtom, compute slcsa/atom) at
 * commit b332f28, by Paul Lafourcade (CEA-DAM-DIF, Arpajon, France) and the LAMMPS developers. Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE). The method is described in
 * https://doi.org/10.1016/j.commatsci.2023.112534 (Lafourcade et al., 2023).
 *
 * docs.lammps.org/compute_slcsa_atom.html: "compute ID group-ID slcsa/atom jmax nclasses database_mean_descriptor_file
 * lda_scalings_file lr_decision_file lr_bias_file maha_file c_ID[N]"; "The compute takes the bispectrum components of
 * compute sna/atom ... and classifies each atom with a linear discriminant analysis projection, a logistic regression
 * and a Mahalanobis distance check"; the output is nclasses+1 columns: the Mahalanobis distance to each class, then the
 * class index (-1 when the distance to that class is above its threshold).
 *
 * Native behaviour kept here:
 *  - the model is read as sequences of numbers (the native PotentialFileReader reads them as rows of the sizes given):
 *    ncomps mean values, ncomps x (nclasses-1) LDA scalings, nclasses x (nclasses-1) decision values, nclasses bias values,
 *    and nclasses rows of (threshold, mean projection, icov) for the Mahalanobis check.
 *  - the descriptor is the first ncomps columns of the per-atom array of the sna/atom compute named by c_ID[N] (native
 *    ignores N in compute_peratom).
 *  - the Mahalanobis product uses the projection index k for the difference inside the l loop
 *    (prodright[k] += icov[k][l] * (proj[k] - mean[k])), as the native source does.
 *  - the native reader and the Mahalanobis arrays are fixed for four classes (nclasses-1 = 3 values per row), so any other
 *    nclasses is refused.
 */
import { errorUrl } from '../../engine/formula';
import { Compute } from '../../engine/compute/compute';
import { StyleError, potentialFile } from '../../engine/force/types';
import { inumeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { exp as libmExp } from '../libm/exp';

const USAGE = 'usage: compute ID group-ID slcsa/atom jmax nclasses database_mean_descriptor_file lda_scalings_file lr_decision_file lr_bias_file maha_file c_ID[N]';

/** Number of bispectrum components for a twojmax (native compute_ncomps). */
const computeNcomps = (twojmax: number): number => {
  let count = 0;
  for (let j1 = 0; j1 <= twojmax; j1++) {
    for (let j2 = 0; j2 <= j1; j2++) {
      for (let j = j1 - j2; j <= Math.min(twojmax, j1 + j2); j += 2) if (j >= j1) count++;
    }
  }
  return count;
};

/** The numbers of a model file in order, comments and blank lines skipped. */
const numbersOf = (text: string, file: string): number[] => {
  const out: number[] = [];
  for (const line of text.split('\n')) {
    const body = line.replace(/#.*/, '').trim();
    if (!body) continue;
    for (const w of body.split(/\s+/)) {
      const v = Number(w);
      if (!Number.isFinite(v)) throw new StyleError(`compute slcsa/atom: expected a number in ${file}, got '${w}'`);
      out.push(v);
    }
  }
  return out;
};

/** Takes the next n numbers of a model file, or throws (native's reader reports a short file). */
const take = (vals: number[], pos: { i: number }, n: number, file: string): number[] => {
  if (pos.i + n > vals.length) throw new StyleError(`compute slcsa/atom: ${file} has fewer values than the model needs`);
  const out = vals.slice(pos.i, pos.i + n);
  pos.i += n;
  return out;
};

const NCLASSES = 4;
const NRED = NCLASSES - 1;

export class ComputeSLCSAAtom extends Compute {
  readonly style = 'slcsa/atom';
  private readonly ncomps: number;
  private readonly mean: Float64Array;
  /** LDA scalings, ncomps x NRED row-major */
  private readonly lda: Float64Array;
  /** logistic regression decision, NCLASSES x NRED row-major */
  private readonly decision: Float64Array;
  private readonly bias: Float64Array;
  private readonly threshold: Float64Array;
  /** mean projected descriptor per class, NCLASSES x NRED */
  private readonly meanProj: Float64Array;
  /** inverse covariance per class, NCLASSES x NRED x NRED */
  private readonly icov: Float64Array;
  private readonly source: Compute;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 8) throw new StyleError(`Illegal compute slcsa/atom command: expected 8 arguments (${USAGE})`);
    // As native LAMMPS does (src/EXTRA-COMPUTE/compute_slcsa_atom.cpp ComputeSLCSAAtom::ComputeSLCSAAtom at b332f28): utils::inumeric
    // reads twojmax and nclasses, then their range checks
    const twojmax = inumeric(args[0]);
    if (twojmax < 0) throw new StyleError('Illegal compute slcsa/atom command: twojmax must be >= 0');
    this.ncomps = computeNcomps(twojmax);
    const nclasses = inumeric(args[1]);
    if (nclasses < 2) throw new StyleError('Illegal compute slcsa/atom command: nclasses must be greater than 1');
    if (nclasses !== NCLASSES) throw new StyleError(`compute slcsa/atom: only nclasses ${NCLASSES} is supported (the native model reader is written for four classes)`);

    // the descriptor, checked before the model files are read: a compute reference with at most one index, which
    // exists, is per-atom, and is a vector (no index) or an array with that column
    const m = /^c_([^[\]]+)(?:\[(\d+)\])?$/.exec(args[7]);
    if (!m) throw new StyleError(`Invalid compute slcsa/atom argument: ${args[7]}`);
    const src = sys.compute(m[1], `Compute ID ${m[1]} for compute slcsa/atom does not exist`);
    if (!src.peratomFlag) throw new StyleError(`Compute slcsa/atom compute ${m[1]} does not calculate per-atom values`);
    const col = m[2] === undefined ? 0 : Number(m[2]);
    if (col === 0 && src.sizePeratomCols !== 0) throw new StyleError(`Compute slcsa/atom compute ${m[1]} does not calculate a per-atom vector`);
    if (col > 0 && src.sizePeratomCols === 0) throw new StyleError(`Compute slcsa/atom compute ${m[1]} does not calculate a per-atom array`);
    if (col > 0 && col > src.sizePeratomCols) throw new StyleError(`Compute slcsa/atom compute ${m[1]} array is accessed out-of-range${errorUrl(20)}`);
    // the engine reads the descriptor columns 1..ncomps of the array
    if (src.sizePeratomCols < this.ncomps) throw new StyleError(`compute slcsa/atom: compute ${m[1]} has ${src.sizePeratomCols} columns, the model needs ${this.ncomps}`);
    this.source = src;

    const ncomps = this.ncomps;
    const meanVals = numbersOf(sys.readFile(args[2], potentialFile('database mean descriptor file')), 'the database mean descriptor file');
    const ldaVals = numbersOf(sys.readFile(args[3], potentialFile('lda scalings file')), 'the lda scalings file');
    const decVals = numbersOf(sys.readFile(args[4], potentialFile('lr decision file')), 'the lr decision file');
    const biasVals = numbersOf(sys.readFile(args[5], potentialFile('lr bias file')), 'the lr bias file');
    const mahaVals = numbersOf(sys.readFile(args[6], potentialFile('mahalanobis stats file')), 'the mahalanobis stats file');
    this.mean = Float64Array.from(take(meanVals, { i: 0 }, ncomps, 'the database mean descriptor file'));
    this.lda = Float64Array.from(take(ldaVals, { i: 0 }, ncomps * NRED, 'the lda scalings file'));
    this.decision = Float64Array.from(take(decVals, { i: 0 }, NCLASSES * NRED, 'the lr decision file'));
    this.bias = Float64Array.from(take(biasVals, { i: 0 }, NCLASSES, 'the lr bias file'));
    this.threshold = new Float64Array(NCLASSES);
    this.meanProj = new Float64Array(NCLASSES * NRED);
    this.icov = new Float64Array(NCLASSES * NRED * NRED);
    const mp = { i: 0 };
    for (let c = 0; c < NCLASSES; c++) {
      this.threshold[c] = take(mahaVals, mp, 1, 'the mahalanobis stats file')[0];
      for (let j = 0; j < NRED; j++) this.meanProj[c * NRED + j] = take(mahaVals, mp, 1, 'the mahalanobis stats file')[0];
      for (let k = 0; k < NRED * NRED; k++) this.icov[c * NRED * NRED + k] = take(mahaVals, mp, 1, 'the mahalanobis stats file')[0];
    }


    this.peratomFlag = true;
    this.sizePeratomCols = NCLASSES + 1;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const n = s.n;
    const out = new Float64Array(n * (NCLASSES + 1));
    this.arrayAtom = out;
    const cols = this.source.sizePeratomCols;
    const desc = this.source.peratomValues();
    const ncomps = this.ncomps;
    const full = new Float64Array(ncomps);
    const proj = new Float64Array(NRED);
    const scores = new Float64Array(NCLASSES);
    const probas = new Float64Array(NCLASSES);
    const prodright = new Float64Array(NRED);
    const W = NCLASSES + 1;
    for (let i = 0; i < n; i++) {
      const o = i * W;
      if (!(s.mask[i] & this.groupBit)) {
        for (let j = 0; j < W; j++) out[o + j] = -1;
        continue;
      }
      // 1st step: centre the descriptor on the database mean
      for (let j = 0; j < ncomps; j++) full[j] = desc[i * cols + j] - this.mean[j];
      // 2nd step: LDA projection to NRED dimensions
      for (let j = 0; j < NRED; j++) {
        proj[j] = 0.0;
        for (let k = 0; k < ncomps; k++) proj[j] += full[k] * this.lda[k * NRED + j];
      }
      // 3rd step: logistic regression scores
      for (let j = 0; j < NCLASSES; j++) {
        scores[j] = this.bias[j];
        for (let k = 0; k < NRED; k++) scores[j] += this.decision[j * NRED + k] * proj[k];
      }
      // 4th step: class probabilities; the class is the argmax
      let sumexp = 0.0;
      for (let j = 0; j < NCLASSES; j++) sumexp += libmExp(scores[j]);
      for (let j = 0; j < NCLASSES; j++) probas[j] = libmExp(scores[j]) / sumexp;
      let best = 0;
      for (let j = 1; j < NCLASSES; j++) if (probas[j] > probas[best]) best = j;
      out[o + NCLASSES] = best;
      // 5th step: Mahalanobis distance to each class (the native index k is used inside the l loop)
      for (let j = 0; j < NCLASSES; j++) {
        prodright[0] = 0.0;
        prodright[1] = 0.0;
        prodright[2] = 0.0;
        for (let k = 0; k < NRED; k++) {
          for (let l = 0; l < NRED; l++) {
            prodright[k] += this.icov[j * NRED * NRED + k * NRED + l] * (proj[k] - this.meanProj[j * NRED + k]);
          }
        }
        let prodleft = 0.0;
        for (let k = 0; k < NRED; k++) prodleft += prodright[k] * (proj[k] - this.meanProj[j * NRED + k]);
        out[o + j] = Math.sqrt(prodleft);
      }
      // 6th step: the class is rejected when its distance exceeds the threshold of that class
      if (out[o + out[o + NCLASSES]] > this.threshold[out[o + NCLASSES]]) out[o + NCLASSES] = -1.0;
    }
  }
}
