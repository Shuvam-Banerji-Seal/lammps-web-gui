/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ORIENT/fix_orient_fcc.cpp and
 * fix_orient_fcc.h (fix orient/fcc), and src/ORIENT/fix_orient_bcc.cpp and fix_orient_bcc.h (fix orient/bcc), whose
 * constructor, post_force, find_best_ref, compare and forward-communication data are shared apart from the neighbour
 * count (12 or 8), the reference vector count (6 or 4) and the cutoff factor (0.5 or 0.75). Contributing authors:
 * Koenraad Janssens and David Olmsted (SNL); the bcc modification by Tegar Wicaksono (UBC). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The grain-boundary driving force of Janssens et al. (Nature Materials 5, 124, 2006; docs.lammps.org/fix_orient.html):
 * the order parameter xi of each owned atom is the sum over its nearest neighbours of the distance from each bond to the
 * closest reference vector of the favoured grain, scaled by 1/n; the energy added to the atom and the force on it are
 * the interpolation between 0 and Vxi between the cutoffs xi0 and xi1. Neighbour data of a ghost atom is the data of
 * its owner (the source's forward communication); the ghost's neighbour IDs are the global IDs, the owned ones local.
 *
 * Not ported: the use_xismooth branch (the source sets use_xismooth = false and never changes it) and the respa and
 * minimisation hooks (the source's setmask has POST_FORCE and POST_FORCE_RESPA only, no MIN_POST_FORCE).
 */
import { Fix } from '../../engine/fix/fix';
import { potentialFile, StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { NativeFullOrder } from './native_full_order';

/** math_const.h: MY_PI and MY_PI2 (pi/2) as written there. */
const MY_PI = 3.14159265358979323846;
const MY_PI2 = 1.57079632679489661923;
/** fix_orient_fcc.cpp and fix_orient_bcc.cpp: "double xicutoff = 1.57;" */
const XICUTOFF = 1.57;
/** fix_orient_fcc.cpp: "static constexpr int BIG = 1000000000;" (initial minimum neighbour count). */
const BIG = 1000000000;

export interface OrientSpec {
  style: 'orient/fcc' | 'orient/bcc';
  /** Nearest neighbours summed for the order parameter (12 fcc, 8 bcc). */
  nNb: number;
  /** Reference vectors per crystal, one from each pair of opposite neighbours (6 fcc, 4 bcc). */
  halfNn: number;
  /** cutsq = cutFactor * a * a * xicutoffsq (0.5 fcc, 0.75 bcc). */
  cutFactor: number;
}

/** The source's Nbr struct for one owned atom: its neighbours (up to nNb), their dxi and the potential derivative. */
interface Sort {
  id: number;
  rsq: number;
  d: [number, number, number];
}

export class FixOrientLattice extends Fix {
  readonly style: string;
  /** fix_orient_fcc.cpp and fix_orient_bcc.cpp init: neighbor->add_request(this, NeighConst::REQ_FULL), a perpetual full list (Fix.nativeFullList). */
  nativeFullList = true;
  private readonly spec: OrientSpec;
  private readonly nstats: number;
  private readonly a: number;
  private readonly Vxi: number;
  private readonly xi0: number;
  private readonly xi1: number;
  private readonly cutsq: number;
  private readonly xicutoffsq: number;
  /** half_xi_chi_vec[0]: the reference vectors of the favoured crystal (file0 or file1 by dir). */
  private readonly refXi: Float64Array;
  private addedEnergy = 0;
  /** Per owned atom: neighbour count, local neighbour index, global ID, dxi (3 per slot), duxi. */
  private nbrN = new Int32Array(0);
  private nbrId = new Int32Array(0);
  private nbrTag = new Int32Array(0);
  private nbrDxi = new Float64Array(0);
  private nbrDuxi = new Float64Array(0);
  private readonly fullOrder = new NativeFullOrder();

  constructor(sys: System, id: string, group: string, args: string[], spec: OrientSpec) {
    super(sys, id, group, args);
    this.spec = spec;
    this.style = spec.style;
    const style = spec.style;
    // fix_orient_fcc.cpp: if (narg != 11) error->all(FLERR,"Illegal fix orient/fcc command");
    // (the engine's args exclude the ID, the group and the style name)
    if (args.length !== 8) throw new StyleError(`Illegal fix ${style} command`);
    this.scalarFlag = true;
    this.extscalar = 1;
    this.energyGlobal = true;
    this.peratomFlag = true;
    this.sizePeratomCols = 2;

    this.nstats = intArg(args[0]);
    const dir = intArg(args[1]);
    this.a = numArg(args[2]);
    this.Vxi = numArg(args[3]);
    const uxifLow = numArg(args[4]);
    const uxifHigh = numArg(args[5]);

    let chiFile: string, xiFile: string;
    if (dir === 0) { chiFile = args[6]; xiFile = args[7]; }
    else if (dir === 1) { xiFile = args[6]; chiFile = args[7]; }
    else throw new StyleError(`Illegal fix ${style} command`);

    // initializations (the source's cutsq: 0.5 * a*a*xicutoffsq, or 0.75 * a*a*xicutoffsq for bcc)
    this.xicutoffsq = XICUTOFF * XICUTOFF;
    this.cutsq = spec.cutFactor * this.a * this.a * this.xicutoffsq;

    // the reference orientations: halfNn vectors of three numbers each, read as PotentialFileReader::next_dvector
    const xiRef = readOrientFile(sys, xiFile, spec.halfNn * 3, style);
    const chiRef = readOrientFile(sys, chiFile, spec.halfNn * 3, style);
    this.refXi = xiRef;

    // xiid: the mean over all 2*halfNn neighbours of the chi crystal of the distance to the xi crystal
    // (the source's find_best_ref with which_crystal 0; the sign flip gives the opposite neighbour)
    let xiid = 0.0;
    const rchi = [0, 0, 0], dxi = [0, 0, 0];
    for (let i = 0; i < spec.halfNn; i++) {
      rchi[0] = chiRef[3 * i]; rchi[1] = chiRef[3 * i + 1]; rchi[2] = chiRef[3 * i + 2];
      xiid += Math.sqrt(this.findBestRef(rchi, dxi));
      for (let j = 0; j < 3; j++) rchi[j] = -rchi[j];
      xiid += Math.sqrt(this.findBestRef(rchi, dxi));
    }
    xiid /= spec.nNb;
    this.xi0 = uxifLow * xiid;
    this.xi1 = uxifHigh * xiid;
  }

  /** find_best_ref: the reference vector nearest in direction to displs (sign included); returns xi_sq, sets dxi. */
  private findBestRef(displs: number[], dxi: number[]): number {
    const ref = this.refXi;
    let bestDot = -1.0;
    let bestI = -1;
    let bestSign = 0;
    for (let i = 0; i < this.spec.halfNn; i++) {
      const dot = displs[0] * ref[3 * i] + displs[1] * ref[3 * i + 1] + displs[2] * ref[3 * i + 2];
      if (Math.abs(dot) > bestDot) {
        bestDot = Math.abs(dot);
        bestI = i;
        bestSign = dot < 0.0 ? -1 : 1;
      }
    }
    let xiSq = 0.0;
    for (let c = 0; c < 3; c++) {
      const tmp = displs[c] - bestSign * ref[3 * bestI + c];
      xiSq += tmp * tmp;
    }
    if (xiSq > 0.0) {
      const xi = Math.sqrt(xiSq);
      for (let c = 0; c < 3; c++) dxi[c] = (bestSign * ref[3 * bestI + c] - displs[c]) / xi;
    } else {
      dxi[0] = dxi[1] = dxi[2] = 0.0;
    }
    return xiSq;
  }

  /** The POST_FORCE hook (setup runs it too, as the source's setup does for verlet). */
  postForce(): void {
    const s = this.sys.state;
    const nb = this.sys.nb;
    const nNb = this.spec.nNb;
    const nlocal = s.n;
    const xa = nb.xall, owner = nb.owner, tag = s.id, mask = s.mask, f = s.f;
    // the source's full neighbour list, requested at init; the cutoff is the neighbour cutoff sqrt(cutsq) (only pairs
    // within cutsq are used, so this selects the same pairs as the source's list at its force cutoff plus skin)
    const list = nb.occasionalList(s, Math.sqrt(this.cutsq), true, s.ntypes);
    // the neighbours of each atom in native's full-list order (the order decides ties in the 12 or 8 nearest)
    const ord = this.fullOrder.rows(s, nb, this.sys.geom, list);

    if (this.nbrN.length !== nlocal) {
      this.nbrN = new Int32Array(nlocal);
      this.nbrId = new Int32Array(nlocal * nNb);
      this.nbrTag = new Int32Array(nlocal * nNb);
      this.nbrDxi = new Float64Array(nlocal * nNb * 3);
      this.nbrDuxi = new Float64Array(nlocal);
    }
    if (this.arrayAtom.length !== 2 * nlocal) this.arrayAtom = new Float64Array(2 * nlocal);

    this.addedEnergy = 0.0;
    let count = 0;
    let mincount = BIG;
    let maxcount = 0;
    const inum = list.inum;

    // loop over owned atoms: the neighbours within cutsq, sorted by distance when there are more than nNb of them
    for (let i = 0; i < inum; i++) {
      const k0 = ord.ptr[i];
      const jnum = ord.ptr[i + 1] - k0;
      if (jnum < mincount) mincount = jnum;
      if (jnum > maxcount) maxcount = jnum;

      const sort: Sort[] = [];
      for (let kk = k0; kk < k0 + jnum; kk++) {
        const j = ord.idx[kk];
        count++;
        const dx = xa[3 * i] - xa[3 * j];
        const dy = xa[3 * i + 1] - xa[3 * j + 1];
        const dz = xa[3 * i + 2] - xa[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < this.cutsq) sort.push({ id: j, rsq, d: [dx, dy, dz] });
      }
      // the source's qsort with compare() on rsq (not called when nsort <= nNb); ties keep the list order
      if (sort.length > nNb) sort.sort((p, q) => (p.rsq < q.rsq ? -1 : p.rsq > q.rsq ? 1 : 0));

      const n = Math.min(nNb, sort.length);
      this.nbrN[i] = n;
      if (n === 0) continue;

      const dxi = [0, 0, 0];
      let xiTotal = 0.0;
      for (let j = 0; j < n; j++) {
        const xiSq = this.findBestRef(sort[j].d, dxi);
        xiTotal += Math.sqrt(xiSq);
        const slot = i * nNb + j;
        this.nbrId[slot] = sort[j].id;
        this.nbrTag[slot] = s.id[owner[sort[j].id]];
        this.nbrDxi[3 * slot] = dxi[0] / n;
        this.nbrDxi[3 * slot + 1] = dxi[1] / n;
        this.nbrDxi[3 * slot + 2] = dxi[2] / n;
      }
      xiTotal /= n;
      this.arrayAtom[2 * i] = xiTotal;

      // the potential derivative to xi and the energy added to the atom
      let edelta: number;
      if (xiTotal < this.xi0) {
        this.nbrDuxi[i] = 0.0;
        edelta = 0.0;
        this.arrayAtom[2 * i + 1] = 0.0;
      } else if (xiTotal > this.xi1) {
        this.nbrDuxi[i] = 0.0;
        edelta = this.Vxi;
        this.arrayAtom[2 * i + 1] = 1.0;
      } else {
        const omega = MY_PI2 * (xiTotal - this.xi0) / (this.xi1 - this.xi0);
        this.nbrDuxi[i] = MY_PI * this.Vxi * libmSin(2.0 * omega) / (2.0 * (this.xi1 - this.xi0));
        edelta = this.Vxi * (1 - libmCos(2.0 * omega)) / 2.0;
        this.arrayAtom[2 * i + 1] = omega / MY_PI2;
      }
      this.addedEnergy += edelta;
    }

    // the force on each owned atom of the group: the gradient of the added energy with respect to its own neighbours'
    // order parameters and with respect to the neighbours' own derivative terms (the source's two loops)
    const groupBit = this.groupBit;
    for (let i = 0; i < inum; i++) {
      if (!(mask[i] & groupBit)) continue;
      const n = this.nbrN[i];
      const duxi = this.nbrDuxi[i];
      for (let j = 0; j < n; j++) {
        const slot = i * nNb + j;
        f[3 * i] += duxi * this.nbrDxi[3 * slot];
        f[3 * i + 1] += duxi * this.nbrDxi[3 * slot + 1];
        f[3 * i + 2] += duxi * this.nbrDxi[3 * slot + 2];

        // m = local index of the neighbour; the self ID is the local index for an owned neighbour, the global ID for a
        // ghost (the neighbour's stored IDs are local or global to match, as the source's forward communication does)
        const m = this.nbrId[slot];
        const owned = m < nlocal;
        const src = owned ? m : owner[m];
        const idSelf = owned ? i : tag[i];
        let foundMyself = false;
        const nn = this.nbrN[src];
        for (let k = 0; k < nn; k++) {
          const sk = src * nNb + k;
          const matches = owned ? this.nbrId[sk] === idSelf : this.nbrTag[sk] === idSelf;
          if (matches) {
            if (foundMyself) throw new StyleError(`Fix ${this.spec.style} found self twice`);
            foundMyself = true;
            const duxiOther = this.nbrDuxi[src];
            f[3 * i] -= duxiOther * this.nbrDxi[3 * sk];
            f[3 * i + 1] -= duxiOther * this.nbrDxi[3 * sk + 1];
            f[3 * i + 2] -= duxiOther * this.nbrDxi[3 * sk + 2];
          }
        }
      }
    }

    // the statistics line every nstats steps (utils::logmesg in the source)
    const step = s.step;
    if (this.nstats && step % this.nstats === 0) {
      const ave = count / s.n;
      this.sys.log(`orient step ${step}: ${s.n} atoms have ${count} neighbors`);
      this.sys.log(`  neighs: min = ${mincount}, max =${maxcount}, ave = ${ave}`);
    }
  }

  /** compute_scalar: the energy added by the last post_force (fix_modify energy yes adds it to pe). */
  computeScalar(): number { return this.addedEnergy; }
  energy(): number { return this.addedEnergy; }
}

/** utils::inumeric (utils.cpp): an integer argument, else the source's message. */
const intArg = (w: string): number => {
  if (w === '') throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  return Number(w);
};

/** utils::numeric (utils.cpp): a floating-point argument, else the source's message. */
const numArg = (w: string): number => {
  if (w === '') throw new StyleError('Expected floating point parameter instead of NULL or empty string in input script or data file');
  const v = Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  return v;
};

/**
 * PotentialFileReader::next_dvector (filetype "fix <style>"): the first n numbers of the file, read across lines (text after
 * '#' and blank lines are skipped). Any failure is the source's "Fix orient/... file read failed: <what>" error. A UNITS:
 * tag on the first line must name the units in use (potential_file_reader.cpp).
 */
const readOrientFile = (sys: System, name: string, n: number, style: string): Float64Array => {
  const filetype = `fix ${style}`;
  const failed = (what: string): never => {
    throw new StyleError(`Fix ${style} file read failed: ${what}`);
  };
  // a file that cannot be opened is PotentialFileReader's own error (the source's catch takes only the read errors)
  const text = sys.readFile(name, potentialFile(filetype));
  const lines = text.split(/\r?\n/);
  const tag = /UNITS:\s*(\S+)/.exec(lines[0] ?? '');
  const units = sys.state.units.style;
  if (tag && tag[1] !== units) failed(`${filetype} file ${name} requires ${tag[1]} units but ${units} units are in use`);
  const out = new Float64Array(n);
  let i = 0;
  let lineNo = 0;
  while (i < n) {
    while (lineNo < lines.length && lines[lineNo].replace(/#.*/, '').trim() === '') lineNo++;
    if (lineNo >= lines.length) {
      if (i === 0) failed('EOF reached');
      failed(`Incorrect format in ${filetype} file! ${i}/${n} values`);
    }
    const words = lines[lineNo].replace(/#.*/, '').trim().split(/\s+/);
    lineNo++;
    for (const w of words) {
      if (i >= n) break;
      const v = Number(w);
      if (!Number.isFinite(v)) failed(`Expected floating point number instead of '${w}' in ${filetype} file ${name}`);
      out[i++] = v;
    }
  }
  return out;
};
