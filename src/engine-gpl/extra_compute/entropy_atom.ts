/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EXTRA-COMPUTE/compute_entropy_atom.cpp and
 * compute_entropy_atom.h (ComputeEntropyAtom: compute entropy/atom, the pair entropy of each atom from its radial
 * distribution function, with the optional neighbor average "avg" and the local-density normalization "local").
 * Contributing author: Pablo Piaggi (EPFL Lausanne). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_entropy_atom.html: "compute ID group-ID entropy/atom sigma cutoff keyword value ..."; "sigma = width
 * of Gaussians used in the g(r) smoothing"; "cutoff = cutoff for the g(r) calculation"; "avg args = neigh cutoff2" with
 * "neigh value = yes or no = whether to average the pair entropy over neighbors" and "cutoff2 = cutoff for the averaging over
 * neighbors"; "local arg = yes or no = use the local density around each atom to normalize the g(r)".
 * Neighbours are taken from the engine's ghost atoms (every periodic image within the ghost cutoff), in the role of the
 * full neighbor list the native command requests; ghost atoms get their own entropy only for the averaging, as native
 * does (the neighbor average uses the entropy of each neighbor, ghosts included).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { CellList } from '../../engine/cellList';
import { log as libmLog } from '../libm/log';
import { exp as libmExp } from '../libm/exp';

/** utils::logical of the input parser: yes/on/true/1 or no/off/false/0. */
const logical = (w: string, what: string): boolean => {
  const v = w.toLowerCase();
  if (v === 'yes' || v === 'on' || v === 'true' || v === '1') return true;
  if (v === 'no' || v === 'off' || v === 'false' || v === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w}' in ${what}`);
};

const numeric = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in ${what}`);
  return v;
};

export class ComputeEntropyAtom extends Compute {
  readonly style = 'entropy/atom';
  private readonly sigma: number;
  private readonly cutoff: number;
  private readonly cutoff2: number;
  private readonly averaging: boolean;
  private readonly localDensity: boolean;
  private readonly cutsq: number;
  private readonly cutsq2: number;
  private readonly nbin: number;
  private readonly deltar: number;
  private readonly deltabin = 3;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = 'compute entropy/atom';
    if (args.length < 2 || args.length > 7) throw new StyleError(`Illegal compute entropy/atom command; wrong number of arguments`);
    this.sigma = numeric(args[0], what);
    if (this.sigma <= 0) throw new StyleError('Illegal compute entropy/atom command; sigma must be positive');
    this.cutoff = numeric(args[1], what);
    if (this.cutoff <= 0) throw new StyleError('Illegal compute entropy/atom command; cutoff must be positive');
    this.cutoff2 = 0;
    this.averaging = false;
    this.localDensity = false;
    let iarg = 2;
    while (iarg < args.length) {
      if (args[iarg] === 'avg') {
        if (iarg + 3 > args.length) throw new StyleError('Illegal compute entropy/atom command');
        this.averaging = logical(args[iarg + 1], what);
        this.cutoff2 = numeric(args[iarg + 2], what);
        if (this.cutoff2 < 0) throw new StyleError('Illegal compute entropy/atom command; negative cutoff2');
        iarg += 3;
      } else if (args[iarg] === 'local') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal compute entropy/atom command');
        this.localDensity = logical(args[iarg + 1], what);
        iarg += 2;
      } else throw new StyleError('Illegal compute entropy/atom command');
    }
    this.cutsq = this.cutoff * this.cutoff;
    this.cutsq2 = this.cutoff2 * this.cutoff2;
    this.nbin = Math.trunc(this.cutoff / this.sigma) + 1;
    this.deltar = this.sigma;
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  /**
   * The native init() checks (the pair cutoff is known only once the force field is set up, so they run before the
   * first evaluation): a pair style is needed and cutoff + cutoff2 must fit inside the pair cutoff plus the skin.
   */
  /** native init(): the pair style must exist (the cutoffs are checked at the first evaluation). */
  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute entropy/atom requires a pair style be defined');
  }

  private checkCutoffs(): void {
    const pair = this.sys.ff.pair;
    if (!pair) throw new StyleError('Compute entropy/atom requires a pair style be defined');
    const cutforce = maxCut(pair.cut);
    if (this.cutoff + this.cutoff2 > cutforce + this.sys.nb.skin) {
      throw new StyleError('Compute entropy/atom cutoff is longer than the pairwise cutoff. Increase the neighbor list skin distance.');
    }
  }

  protected computePeratom(): void {
    this.checkCutoffs();
    const pair = this.sys.ff.pair!;
    this.sys.forces();
    const s = this.sys.state;
    const nb = this.sys.nb;
    const n = s.n;
    const nall = nb.nall;
    const xa = nb.xall;
    const cutneigh = maxCut(pair.cut) + nb.skin;
    const cutneigh2 = cutneigh * cutneigh;
    // every test below accepts a distance under cutneigh, cutoff or cutoff2: cells at least that wide
    const cells = new CellList(xa, nall, Math.max(cutneigh, this.cutoff, this.cutoff2));
    const sigma = this.sigma, nbin = this.nbin, deltar = this.deltar, deltabin = this.deltabin;
    const sigmasq2 = 2 * sigma * sigma;
    const box = s.box;
    const volume = (box.hi[0] - box.lo[0]) * (box.hi[1] - box.lo[1]) * (box.hi[2] - box.lo[2]);
    const globalDensity = n / volume;
    const rbin = new Float64Array(nbin), rbinsq = new Float64Array(nbin);
    for (let k = 0; k < nbin; k++) { rbin[k] = k * deltar; rbinsq[k] = rbin[k] * rbin[k]; }
    const PI = Math.PI;
    const gofr = new Float64Array(nbin), integrand = new Float64Array(nbin);
    // with avg every atom and ghost gets an entropy (its neighbours need it); otherwise only the owned atoms in the group
    const pe = new Float64Array(this.averaging ? nall : n);
    const nlim = this.averaging ? nall : n;
    for (let i = 0; i < nlim; i++) {
      if (!this.averaging && !(s.mask[i] & this.groupBit)) { pe[i] = 0; continue; }
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      // neighbours of i within the list cutoff (the native list holds the pairs within cutforce + skin)
      let jnum = 0;
      if (this.localDensity) {
        const cand = cells.candidates(xi, yi, zi);
        for (let t = 0; t < cand.length; t++) {
          const j = cand[t];
          if (j === i) continue;
          const dx = xi - xa[3 * j], dy = yi - xa[3 * j + 1], dz = zi - xa[3 * j + 2];
          if (dx * dx + dy * dy + dz * dz < cutneigh2) jnum++;
        }
      }
      const density = this.localDensity ? jnum / ((4 / 3) * PI * cutneigh * cutneigh * cutneigh) : globalDensity;
      const normConstantBase = 4 * PI * density * (Math.sqrt(2 * PI) * sigma);
      const invNormConstantBase = 1 / normConstantBase;
      gofr.fill(0);
      const cand = cells.candidates(xi, yi, zi);
      for (let t = 0; t < cand.length; t++) {
        const j = cand[t];
        if (j === i) continue;
        const dx = xi - xa[3 * j], dy = yi - xa[3 * j + 1], dz = zi - xa[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < this.cutsq) {
          const r = Math.sqrt(rsq);
          const bin = Math.floor(r / deltar);
          let minbin = bin - deltabin;
          if (minbin < 0) minbin = 0;
          if (minbin > nbin - 1) minbin = nbin - 1;
          let maxbin = bin + deltabin;
          if (maxbin > nbin - 1) maxbin = nbin - 1;
          for (let k = minbin; k < maxbin + 1; k++) {
            const invNormKernel = invNormConstantBase / rbinsq[k];
            const distance = r - rbin[k];
            gofr[k] += invNormKernel * libmExp(-distance * distance / sigmasq2);
          }
        }
      }
      for (let k = 0; k < nbin; k++) {
        integrand[k] = gofr[k] < 1e-10 ? rbinsq[k] : (gofr[k] * libmLog(gofr[k]) - gofr[k] + 1) * rbinsq[k];
      }
      // trapezoid rule
      let value = 0;
      for (let k = 1; k < nbin - 1; k++) value += integrand[k];
      value += 0.5 * integrand[0];
      value += 0.5 * integrand[nbin - 1];
      value *= deltar;
      pe[i] = -2 * PI * density * value;
    }
    const out = new Float64Array(n);
    if (!this.averaging) {
      out.set(pe.subarray(0, n));
    } else {
      for (let i = 0; i < n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
        let sum = pe[i];
        let counter = 1;
        const cand = cells.candidates(xi, yi, zi);
        for (let t = 0; t < cand.length; t++) {
          const j = cand[t];
          if (j === i) continue;
          const dx = xi - xa[3 * j], dy = yi - xa[3 * j + 1], dz = zi - xa[3 * j + 2];
          if (dx * dx + dy * dy + dz * dz < this.cutsq2) {
            sum += pe[j];
            counter += 1;
          }
        }
        out[i] = sum / counter;
      }
    }
    this.vectorAtom = out;
  }
}

const maxCut = (cut: Float64Array): number => {
  let m = 0;
  for (let k = 0; k < cut.length; k++) if (cut[k] > m) m = cut[k];
  return m;
};
