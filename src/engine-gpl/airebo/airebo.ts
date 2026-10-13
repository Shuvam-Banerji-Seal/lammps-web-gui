/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.cpp, pair_airebo.h (PairAIREBO: settings, coeff,
 * init_style, init_one, REBO_neigh, FREBO, compute), src/MANYBODY/pair_rebo.cpp and
 * pair_rebo.h (PairREBO), src/MANYBODY/pair_airebo_morse.cpp and pair_airebo_morse.h
 * (PairAIREBOMorse), at commit b332f28.
 * Neighbor lists: built here from owned and ghost positions with a binned search (the
 * LAMMPS full neighbor list with REQ_GHOST is not used). The owned-atom half-pair rule of
 * FREBO uses the atom ids through the periodic image owners (nb.owner).
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { newRef, TOL, type ReboCore } from './rebo';
import { bondorder } from './bondorder';
import { sp } from './splines';
import { readAireboFile, newParams, SplineTables, type AireboParams, type AireboVariant } from './tables';
import { newKnots, splineInit, splineInitRebo, type Knots } from './splineInit';
import { flj, type LJCore, type LJParams } from './flj';
import { torsion } from './torsion';
import { rowNeighbors } from './neigh';
import { powint } from './splines';

/** Scratch derivative of the coordination cutoff in buildRebo. */
const SCRATCH = newRef();

/**
 * Builds the REBO neighbor lists of all owned and ghost atoms (PairAIREBO::REBO_neigh) and
 * the coordination sums nC, nH. Neighbors are the atoms within rcmax of the element pair.
 */
const buildRebo = (core: ReboCore, nall: number, rcmaxsq: Float64Array, rcmaxMax: number, rcmin: Float64Array,
  rcmax: Float64Array, x: Float64Array): { rstart: Int32Array; rlist: Int32Array } => {
  const cs = rcmaxMax > 0 ? rcmaxMax : 1;
  let xmin = Infinity, ymin = Infinity, zmin = Infinity;
  let xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
  for (let a = 0; a < nall; a++) {
    const px = x[3 * a], py = x[3 * a + 1], pz = x[3 * a + 2];
    if (px < xmin) xmin = px;
    if (px > xmax) xmax = px;
    if (py < ymin) ymin = py;
    if (py > ymax) ymax = py;
    if (pz < zmin) zmin = pz;
    if (pz > zmax) zmax = pz;
  }
  if (nall === 0) return { rstart: new Int32Array(1), rlist: new Int32Array(0) };
  const nx = Math.floor((xmax - xmin) / cs) + 1;
  const ny = Math.floor((ymax - ymin) / cs) + 1;
  const nz = Math.floor((zmax - zmin) / cs) + 1;
  const cellOf = new Int32Array(nall);
  const ncell = nx * ny * nz;
  const cstart = new Int32Array(ncell + 1);
  for (let a = 0; a < nall; a++) {
    const cx = Math.floor((x[3 * a] - xmin) / cs);
    const cy = Math.floor((x[3 * a + 1] - ymin) / cs);
    const cz = Math.floor((x[3 * a + 2] - zmin) / cs);
    const c = cx + nx * (cy + ny * cz);
    cellOf[a] = c;
    cstart[c + 1]++;
  }
  for (let c = 0; c < ncell; c++) cstart[c + 1] += cstart[c];
  const fill = cstart.slice(0, ncell);
  const cellAtoms = new Int32Array(nall);
  for (let a = 0; a < nall; a++) cellAtoms[fill[cellOf[a]]++] = a;
  const rstart = new Int32Array(nall + 1);
  const out: number[] = [];
  const mt = core.mt;
  for (let a = 0; a < nall; a++) {
    rstart[a] = out.length;
    core.nC[a] = 0.0;
    core.nH[a] = 0.0;
    if (mt[a] < 0) continue;
    const itype = mt[a];
    const cx = Math.floor((x[3 * a] - xmin) / cs);
    const cy = Math.floor((x[3 * a + 1] - ymin) / cs);
    const cz = Math.floor((x[3 * a + 2] - zmin) / cs);
    const xa = x[3 * a], ya = x[3 * a + 1], za = x[3 * a + 2];
    for (let dz = -1; dz <= 1; dz++) {
      const z = cz + dz;
      if (z < 0 || z >= nz) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const y = cy + dy;
        if (y < 0 || y >= ny) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = cx + dx;
          if (xx < 0 || xx >= nx) continue;
          const c = xx + nx * (y + ny * z);
          for (let q = cstart[c]; q < cstart[c + 1]; q++) {
            const b = cellAtoms[q];
            if (b === a || mt[b] < 0) continue;
            const jtype = mt[b];
            const delx = xa - x[3 * b], dely = ya - x[3 * b + 1], delz = za - x[3 * b + 2];
            const rsq = delx * delx + dely * dely + delz * delz;
            if (rsq < rcmaxsq[itype * 2 + jtype]) {
              out.push(b);
              const r = Math.sqrt(rsq);
              const w = sp(r, rcmin[itype * 2 + jtype], rcmax[itype * 2 + jtype], SCRATCH);
              if (jtype === 0) core.nC[a] += w;
              else core.nH[a] += w;
            }
          }
        }
      }
    }
  }
  rstart[nall] = out.length;
  return { rstart, rlist: Int32Array.from(out) };
};

/** PairAIREBO (variant airebo, rebo or airebo/morse). */
export class PairAIREBO extends Pair {
  readonly name: string;
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected variant: AireboVariant;
  protected ljflag = 1;
  protected torflag = 1;
  protected morseflag = 0;
  protected cutlj = 0;
  protected sigcut = 3.0;
  protected sigwid = 0.84;
  protected sigmin = 3.0 - 0.84;
  protected fileName = '';
  protected fileRead = false;
  protected P: AireboParams = newParams();
  protected T: SplineTables = new SplineTables();
  protected K: Knots = newKnots();
  /** element index (0 = C, 1 = H) per atom type, -1 for NULL. */
  protected map = new Int32Array(0);
  protected cut3rebo = 0;
  protected cutljrebo = 0;
  protected cutljrebosq = 0;
  protected lj1 = new Float64Array(4);
  protected lj2 = new Float64Array(4);
  protected lj3 = new Float64Array(4);
  protected lj4 = new Float64Array(4);
  protected cutljsq = new Float64Array(4);
  protected dwij = newRef();
  protected dS = newRef();
  protected cutljrebosq0 = 0;

  constructor(variant: AireboVariant) {
    super();
    this.variant = variant;
    this.name = variant === 'rebo' ? 'rebo' : variant === 'airebo/morse' ? 'airebo/morse' : 'airebo';
    this.morseflag = variant === 'airebo/morse' ? 1 : 0;
    if (variant === 'rebo') {
      this.ljflag = 0;
      this.torflag = 0;
    }
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.map = new Int32Array(ntypes + 1).fill(-1);
  }

  override settings(args: string[], _ctx: StyleContext): void {
    if (this.variant === 'rebo') {
      if (args.length !== 0) throw new StyleError('Illegal pair_style rebo command');
      this.cutlj = 0.0;
      this.ljflag = 0;
      this.torflag = 0;
      return;
    }
    if (args.length !== 1 && args.length !== 3 && args.length !== 4) {
      throw new StyleError(`Illegal pair_style ${this.name} command`);
    }
    this.cutlj = parseNum(args[0], 'LJ cutoff');
    if (args.length >= 3) {
      this.ljflag = parseInt(args[1], 10);
      this.torflag = parseInt(args[2], 10);
      if (String(this.ljflag) !== args[1] || String(this.torflag) !== args[2]) {
        throw new StyleError(`Illegal pair_style ${this.name} flag: ljflag and torflag must be 0 or 1`);
      }
    }
    if (args.length === 4) {
      this.sigcut = this.cutlj;
      this.sigmin = parseNum(args[3], 'sigmin');
      this.sigwid = this.sigcut - this.sigmin;
    }
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 3 + this.ntypes) {
      throw new StyleError(`Incorrect number of args for pair coefficient (style ${this.name} needs * * file and one element per type)`);
    }
    if (args[0] !== '*' || args[1] !== '*') {
      throw new StyleError(`the first 2 arguments of pair_coeff for style ${this.name} must be * *`);
    }
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let i = 3; i < args.length; i++) {
      const w = args[i];
      // docs.lammps.org/pair_airebo.html: "If a mapping value is specified as NULL, the mapping is not performed. This can
      // be used when a airebo potential is used as part of the hybrid pair style." The NULL atoms take no part in the
      // neighbor lists (the builders skip map < 0) and in no term.
      if (w === 'NULL') {
        map[i - 2] = -1;
      } else if (w === 'C') {
        map[i - 2] = 0;
      } else if (w === 'H') {
        map[i - 2] = 1;
      } else {
        throw new StyleError(`Element ${w} not supported by pair style ${this.name}`);
      }
    }
    const fileName = args[2];
    const { params, tables } = readAireboFile(ctx.readFile(fileName), fileName, this.variant, this.morseflag === 1);
    const K = newKnots();
    splineInit(K, tables);
    if (this.variant === 'rebo') splineInitRebo(K, tables);
    this.P = params;
    this.T = tables;
    this.K = K;
    this.map = map;
    this.fileName = fileName;
    this.fileRead = true;
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.fileRead) throw new StyleError(`pair_style ${this.name} needs a pair_coeff command with a potential file`);
    if (this.shift || this.tail) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify shift and tail options`);
    }
    if (this.table !== 12) throw new StyleError(`pair_style ${this.name} does not support the pair_modify table option`);
    this.setTables();
  }

  /** PairAIREBO::init_one tables that do not depend on the type pair: REBO cutoffs and the LJ / Morse coefficients. */
  protected setTables(): void {
    const P = this.P;
    this.cut3rebo = 3.0 * P.rcmax[0];
    this.cutljrebo = P.rcLJmax[0] + P.rcmax[0];
    this.cutljrebosq = this.cutljrebo * this.cutljrebo;
    for (let ii = 0; ii < 2; ii++) {
      for (let jj = 0; jj < 2; jj++) {
        const q = ii * 2 + jj;
        this.cutljsq[q] = this.cutlj * P.sigma[q] * this.cutlj * P.sigma[q];
        if (this.morseflag) {
          // Morse parameters are stored in the LJ arrays (pair_airebo_morse.cpp)
          this.lj1[q] = P.epsilonM[q] * Math.exp(P.alphaM[q] * P.reqM[q]);
          this.lj2[q] = Math.exp(P.alphaM[q] * P.reqM[q]);
          this.lj3[q] = 2 * P.epsilonM[q] * P.alphaM[q] * Math.exp(P.alphaM[q] * P.reqM[q]);
          this.lj4[q] = P.alphaM[q];
        } else {
          this.lj1[q] = 48.0 * P.epsilon[q] * powint(P.sigma[q], 12);
          this.lj2[q] = 24.0 * P.epsilon[q] * powint(P.sigma[q], 6);
          this.lj3[q] = 4.0 * P.epsilon[q] * powint(P.sigma[q], 12);
          this.lj4[q] = 4.0 * P.epsilon[q] * powint(P.sigma[q], 6);
        }
      }
    }
  }

  /** PairAIREBO::init_one: the cutoff the ghost shell must cover (the largest of the terms). */
  override initOne(i: number, j: number): number {
    if (this.map[i] < 0 || this.map[j] < 0) return 0;
    const P = this.P;
    let cutmax = this.cut3rebo;
    if (this.ljflag) {
      cutmax = Math.max(cutmax, P.rcLJmax[0] + 2.0 * P.rcmax[0]);
      cutmax = Math.max(cutmax, this.cutlj * P.sigma[0]);
    }
    return cutmax;
  }

  override compute(pc: PairCompute): void {
    if (!this.fileRead) throw new StyleError(`pair_style ${this.name} has no potential file`);
    if (pc.vatom) throw new StyleError(`compute stress/atom is not supported by pair style ${this.name}`);
    const nall = pc.nall;
    const nlocal = pc.nlocal;
    const mt = new Int32Array(nall);
    for (let a = 0; a < nall; a++) {
      mt[a] = this.map[pc.type[a]];
    }
    const tag = new Int32Array(nall);
    const owner = pc.nb.owner;
    for (let a = 0; a < nall; a++) tag[a] = pc.s.id[owner[a]];
    const nC = new Float64Array(nall), nH = new Float64Array(nall);
    const P = this.P;
    const rcmaxMax = Math.max(P.rcmax[0], P.rcmax[1], P.rcmax[3]);
    const core: ReboCore = {
      P, T: this.T, K: this.K, mt, tag, x: pc.x, f: pc.f, rstart: new Int32Array(0), rlist: new Int32Array(0),
      nC, nH, acc: pc.acc, eatom: pc.eatom, dwij: this.dwij, dS: this.dS,
    };
    // REBO neighbor lists of owned and ghost atoms, and the coordination sums nC, nH
    const built = buildRebo(core, nall, P.rcmaxsq, rcmaxMax, P.rcmin, P.rcmax, pc.x);
    core.rstart = built.rstart;
    core.rlist = built.rlist;
    this.frebo(core, nlocal);
    if (this.ljflag || this.torflag) {
      const ljp: LJParams = {
        cutljsq: this.cutljsq, lj1: this.lj1, lj2: this.lj2, lj3: this.lj3, lj4: this.lj4,
        sigcut: this.sigcut, sigmin: this.sigmin, morseflag: this.morseflag === 1, cut3rebo: this.cut3rebo,
      };
      let ljStart = new Int32Array(nlocal + 1);
      let ljList = new Int32Array(0);
      if (this.ljflag) {
        let cutmaxsq = 0;
        for (let q = 0; q < 4; q++) if (this.cutljsq[q] > cutmaxsq) cutmaxsq = this.cutljsq[q];
        const built = rowNeighbors(pc.x, nlocal, nall, mt, this.cutljsq, Math.sqrt(cutmaxsq));
        ljStart = built.start;
        ljList = built.list;
      }
      const lcore: LJCore = { ...core, ljp, ljStart, ljList, epsilonT: P.epsilonT };
      if (this.ljflag) flj(lcore, nlocal);
      if (this.torflag) torsion(lcore, nlocal);
    }
  }

  /** PairAIREBO::FREBO: the REBO two-body and bond-order energy and forces of owned atoms. */
  protected frebo(core: ReboCore, nlocal: number): void {
    const { P, mt, tag, x, f, rstart, rlist } = core;
    const eatom = core.eatom;
    for (let i = 0; i < nlocal; i++) {
      const itag = tag[i];
      const itype = mt[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
        const j = rlist[kk];
        const jtag = tag[j];
        // each physical pair once: parity of the ids (the same rule on both sides of the bond)
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          // self-image pairs: keep the one with the larger coordinate
          if (x[3 * j + 2] < ztmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }
        const jtype = mt[j];
        const delx = x[3 * i] - x[3 * j];
        const dely = x[3 * i + 1] - x[3 * j + 1];
        const delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const rij = Math.sqrt(rsq);
        const wij = sp(rij, P.rcmin[itype * 2 + jtype], P.rcmax[itype * 2 + jtype], this.dwij);
        const dwij = this.dwij.v;
        if (wij <= TOL) continue;
        const Qij = P.Q[itype * 2 + jtype];
        const Aij = P.A[itype * 2 + jtype];
        const alphaij = P.alpha[itype * 2 + jtype];
        const VR = wij * (1.0 + (Qij / rij)) * Aij * Math.exp(-alphaij * rij);
        const pre = wij * Aij * Math.exp(-alphaij * rij);
        let dVRdi = pre * ((-alphaij) - (Qij / rsq) - (Qij * alphaij / rij));
        dVRdi += VR / wij * dwij;
        let VA = 0.0, dVA = 0.0;
        for (let m = 0; m < 3; m++) {
          const term = -wij * P.BIJc[(itype * 2 + jtype) * 3 + m] * Math.exp(-P.Beta[(itype * 2 + jtype) * 3 + m] * rij);
          VA += term;
          dVA += -P.Beta[(itype * 2 + jtype) * 3 + m] * term;
        }
        dVA += VA / wij * dwij;
        const del = BOND_DEL;
        del[0] = delx;
        del[1] = dely;
        del[2] = delz;
        const bij = bondorder(core, i, j, del, rij, VA);
        const dVAdi = bij * dVA;
        const fpair = -(dVRdi + dVAdi) / rij;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        const evdwl = VR + bij * VA;
        core.acc.evdwl += evdwl;
        if (eatom) {
          const half = 0.5 * evdwl;
          eatom[i] += half;
          eatom[j] += half;
        }
      }
    }
  }
}

/** The bond vector handed to bondorder (x_i - x_j). */
const BOND_DEL = new Float64Array(3);
