/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_drip.cpp and src/INTERLAYER/pair_drip.h (and Pair::v_tally2_newton of src/pair.cpp for
 * the per-atom virial), at commit b332f28 (the DRIP registry-dependent
 * interlayer potential with the dihedral-angle correction). Contributing author: Mingjian Wen (University of
 * Minnesota). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 */
import { hasMolecule } from '../../engine/atoms';
import { NEIGHMASK } from '../../engine/neighbor';
import { Pair, StyleError, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { joinPotentialEntries, parseNum } from '../../engine/force/util';
import { energyScale, stripComment } from './zbase';
import { CellList } from '../../engine/cellList';
import { exp as libmExp } from '../libm/exp';

/** One potential-file entry of DRIP (the parameters of one element pair). */
interface DripParam {
  ie: number; je: number;
  C0: number; C2: number; C4: number; C: number; delta: number; lambda: number; A: number; z0: number; B: number; eta: number;
  rhocut: number; rcut: number; ncut: number; rhocutsq: number; rcutsq: number; ncutsq: number;
}

/** A 3 x 3 matrix as [row][column]. */
type M3 = number[][];
const HALF = 0.5;
/** Distance sentinel of an empty neighbor slot (native 1.0e10 in find_nearest3neigh). */
const NO_NEIGHBOR = 1.0e10;
const zero3 = (): M3 => [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
const dot = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
/** Pair::v_tally2_newton (src/pair.cpp): x_a (x) f_a, the whole term to atom a (owned or ghost). */
const vTally2Newton = (va: Float64Array, a: number, fa: readonly number[], x: Float64Array): void => {
  const o = 6 * a, xa = x[3 * a], ya = x[3 * a + 1], za = x[3 * a + 2];
  va[o] += xa * fa[0]; va[o + 1] += ya * fa[1]; va[o + 2] += za * fa[2];
  va[o + 3] += xa * fa[1]; va[o + 4] += xa * fa[2]; va[o + 5] += ya * fa[2];
};
const matDotVec = (X: M3, y: readonly number[]): number[] => [0, 1, 2].map((k) => X[k][0] * y[0] + X[k][1] * y[1] + X[k][2] * y[2]);

/** Normalized cross product of (rl - rk) and (rm - rk) with its derivatives (deriv_cross): [normal, d/drk, d/drl, d/drm]. */
const derivCross = (rk: readonly number[], rl: readonly number[], rm: readonly number[]): { cross: number[]; dk: M3; dl: M3; dm: M3 } => {
  const x = [rl[0] - rk[0], rl[1] - rk[1], rl[2] - rk[2]];
  const y = [rm[0] - rk[0], rm[1] - rk[1], rm[2] - rk[2]];
  const p = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  const q = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  const cross = [p[0] / q, p[1] / q, p[2] / q];
  const qc = q * q * q;
  const dx0 = (+p[1] * y[2] - p[2] * y[1]) / qc;
  const dx1 = (-p[0] * y[2] + p[2] * y[0]) / qc;
  const dx2 = (p[0] * y[1] - p[1] * y[0]) / qc;
  const dy0 = (-p[1] * x[2] + p[2] * x[1]) / qc;
  const dy1 = (p[0] * x[2] - p[2] * x[0]) / qc;
  const dy2 = (-p[0] * x[1] + p[1] * x[0]) / qc;
  const dl: M3 = [
    [p[0] * dx0, -y[2] / q + p[1] * dx0, y[1] / q + p[2] * dx0],
    [y[2] / q + p[0] * dx1, p[1] * dx1, -y[0] / q + p[2] * dx1],
    [-y[1] / q + p[0] * dx2, y[0] / q + p[1] * dx2, p[2] * dx2],
  ];
  const dm: M3 = [
    [p[0] * dy0, x[2] / q + p[1] * dy0, -x[1] / q + p[2] * dy0],
    [-x[2] / q + p[0] * dy1, p[1] * dy1, x[0] / q + p[2] * dy1],
    [x[1] / q + p[0] * dy2, -x[0] / q + p[1] * dy2, p[2] * dy2],
  ];
  const dk: M3 = zero3();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) dk[i][j] = -(dl[i][j] + dm[i][j]);
  return { cross, dk, dl, dm };
};

/** Taper of the attractive part, 1 at r = 0 and 0 at the cutoff, with its derivative (tap). */
const tap = (r: number, cutoff: number): { t: number; dtap: number } => {
  if (r <= 0) return { t: 1, dtap: 0 };
  const roc = r / cutoff;
  const rocSq = roc * roc;
  return {
    t: rocSq * rocSq * (-35.0 + 84.0 * roc + rocSq * (-70.0 + 20.0 * roc)) + 1,
    dtap: rocSq * roc / cutoff * (-140.0 + 420.0 * roc + rocSq * (-420.0 + 140.0 * roc)),
  };
};

/** Taper in rho^2 of the dihedral term, with its derivative wrt rho^2 (tap_rho). */
const tapRho = (rhosq: number, cutRhosq: number): { t: number; drhosq: number } => {
  const rocSq = rhosq / cutRhosq;
  const roc = Math.sqrt(rocSq);
  return {
    t: rocSq * rocSq * (-35.0 + 84.0 * roc + rocSq * (-70.0 + 20.0 * roc)) + 1,
    drhosq: rocSq / cutRhosq * (-70.0 + 210.0 * roc + rocSq * (-210.0 + 70.0 * roc)),
  };
};

/**
 * pair_style drip (docs.lammps.org/pair_drip.html): the dihedral-angle-corrected registry-dependent interlayer
 * potential for multilayer graphene. Usage: pair_style drip inside pair_style hybrid/overlay (no arguments); pair_coeff
 * * * file element... with the C.drip file. Each atom's normal is built from its three nearest neighbors of the same
 * molecule within ncut; the neighbors and the normal's derivatives are the ones of native (calc_normal, td, dihedral).
 */
export class PairDRIP extends Pair {
  readonly name: string = 'drip';
  /** manybody_flag of native: the element mapping of "* *" skips the NULL types (see hybrid). */
  override manybody = true;
  private elements: string[] = [];
  /** Element index of each atom type (-1 for NULL). */
  private map = new Int32Array(0);
  private setflag = new Uint8Array(0);
  private params: DripParam[] = [];
  private elem2param = new Int32Array(0);
  /** Three nearest same-molecule neighbors per atom (owned and ghost), -1 when a ghost has fewer: 3 per atom. */
  private nearest = new Int32Array(0);

  constructor() {
    super();
    this.virialFdotr = true;
    this.needsFull = true;
    this.needsHalf = false;
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.setflag = new Uint8Array((ntypes + 1) * (ntypes + 1));
  }

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
    if (!this.hybridOverlay) throw new StyleError('Pair style drip must be used as sub-style with hybrid/overlay');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    // pair_drip.cpp coeff(): Pair::map_element2type(narg - 3, ...) checks the element count before anything is read
    const nt = this.ntypes + 1;
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    this.map = new Int32Array(nt).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') continue;
      let idx = this.elements.indexOf(name);
      if (idx < 0) {
        this.elements.push(name);
        idx = this.elements.length - 1;
      }
      this.map[t] = idx;
    }
    this.readFile(args[2], ctx);
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        if (this.map[i] >= 0 && this.map[j] >= 0) {
          this.setflag[i * nt + j] = 1;
          this.setflag[j * nt + i] = 1;
        }
      }
    }
  }

  /** Reads the DRIP file: 15 words per entry (two element names, 13 numbers); entries of other elements are skipped. */
  private readFile(fileName: string, ctx: StyleContext): void {
    const lines = ctx.readFile(fileName, potentialFile('drip')).split('\n');
    const m = /UNITS:\s*(\S+)/.exec(lines[0] ?? '');
    const scale = energyScale(m ? m[1] : null, ctx, this.name, fileName);
    const out: DripParam[] = [];
    for (const raw of joinPotentialEntries(lines, 15)) {
      const words = stripComment(raw).split(/\s+/).filter((w) => w !== '');
      if (words.length === 0) continue;
      if (words.length < 15) throw new StyleError(`drip potential file ${fileName}: expected 15 words per entry, got ${words.length}`);
      const ie = this.elements.indexOf(words[0]);
      const je = this.elements.indexOf(words[1]);
      if (ie < 0 || je < 0) continue;
      const v = words.slice(2).map((w, k) => parseNum(w, `drip parameter ${k + 1}`));
      // the energies C0, C2, C4, C, A and B are converted between metal and real units, as native does
      const [C0, C2, C4, C, delta, lambda, A, z0, B, eta, rhocut, rcut, ncut] = v;
      out.push({
        ie, je,
        C0: C0 * scale, C2: C2 * scale, C4: C4 * scale, C: C * scale, delta, lambda, A: A * scale, z0, B: B * scale, eta,
        rhocut, rcut, ncut, rhocutsq: rhocut * rhocut, rcutsq: rcut * rcut, ncutsq: ncut * ncut,
      });
    }
    const ne = this.elements.length;
    const elem2 = new Int32Array(ne * ne);
    for (let i = 0; i < ne; i++) {
      for (let j = 0; j < ne; j++) {
        let n = -1;
        for (let k = 0; k < out.length; k++) {
          if (out[k].ie === i && out[k].je === j) {
            if (n >= 0) throw new StyleError(`DRIP potential file has a duplicate entry for: ${this.elements[i]} ${this.elements[j]}`);
            n = k;
          }
        }
        if (n < 0) throw new StyleError(`Potential file is missing an entry for: ${this.elements[i]} ${this.elements[j]}`);
        elem2[i * ne + j] = n;
      }
    }
    this.params = out;
    this.elem2param = elem2;
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError('Pair style drip requires newton pair on');
    if (!ctx.s || !hasMolecule(ctx.s)) throw new StyleError('Pair style drip requires atom attribute molecule');
  }

  override initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.setflag[i * nt + j]) throw new StyleError(`All pair coeffs are not set for pair style drip (types ${i} ${j})`);
    const ne = this.elements.length;
    const p = this.params[this.elem2param[this.map[i] * ne + this.map[j]]];
    // the neighbor list reaches the main cutoff plus the normal cutoff (init_one)
    return p.rcut + p.ncut;
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style drip needs a full neighbor list');
    const { x, f, type } = pc;
    const ne = this.elements.length;
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    this.findNearest3(pc);
    const nst = this.nearest;
    for (let i = 0; i < list.inum; i++) {
      if (this.map[type[i]] < 0 || nst[3 * i] === -1) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = this.map[type[i]];
      const nrm = derivCross([x[3 * nst[3 * i]], x[3 * nst[3 * i] + 1], x[3 * nst[3 * i] + 2]],
        [x[3 * nst[3 * i + 1]], x[3 * nst[3 * i + 1] + 1], x[3 * nst[3 * i + 1] + 2]],
        [x[3 * nst[3 * i + 2]], x[3 * nst[3 * i + 2] + 1], x[3 * nst[3 * i + 2] + 2]]);
      const ni = nrm.cross;
      const fi = [0, 0, 0];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        if (this.map[type[j]] < 0 || nst[3 * j] === -1) continue;
        const jtype = this.map[type[j]];
        const rvec = [x[3 * j] - xi, x[3 * j + 1] - yi, x[3 * j + 2] - zi];
        const rsq = dot(rvec, rvec);
        const p = this.params[this.elem2param[itype * ne + jtype]];
        if (rsq < p.rcutsq && mol[owner[i]] !== mol[owner[j]]) {
          const fj = [0, 0, 0];
          const phiAttr = this.calcAttractive(p, rsq, rvec, fi, fj);
          const phiRepul = this.calcRepulsive(pc, i, j, p, rsq, rvec, ni, nrm, fi, fj);
          const evdwl = HALF * (phiRepul + phiAttr);
          pc.acc.evdwl += evdwl;
          if (pc.eatom) {
            pc.eatom[i] += 0.5 * evdwl;
            pc.eatom[j] += 0.5 * evdwl;
          }
          f[3 * j] += fj[0];
          f[3 * j + 1] += fj[1];
          f[3 * j + 2] += fj[2];
          if (pc.vatom) vTally2Newton(pc.vatom, j, fj, x);
        }
      }
      f[3 * i] += fi[0];
      f[3 * i + 1] += fi[1];
      f[3 * i + 2] += fi[2];
      if (pc.vatom) vTally2Newton(pc.vatom, i, fi, x);
    }
  }

  /**
   * The three nearest neighbors of every atom (owned and ghost) of a mapped element within ncut and the same molecule
   * (find_nearest3neigh). Native takes them from its neighbor lists; here every atom of the box is searched. A ghost
   * with fewer than three gives -1; an owned atom with fewer is an error.
   */
  private findNearest3(pc: PairCompute): void {
    const { x, type, nall, nlocal } = pc;
    const ne = this.elements.length;
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    const nearest = new Int32Array(3 * nall).fill(-1);
    // every test accepts rsq < ncutsq of some element pair, so cells as wide as the largest ncut hold all candidates
    let ncutMax = 0;
    for (const p of this.params) if (p.ncut > ncutMax) ncutMax = p.ncut;
    const cells = new CellList(x, nall, ncutMax);
    for (let i = 0; i < nall; i++) {
      const itype = this.map[type[i]];
      if (itype < 0) continue;
      let nb1 = -1, nb2 = -1, nb3 = -1;
      let nb1rsq = NO_NEIGHBOR + 1, nb2rsq = 2.0 * NO_NEIGHBOR, nb3rsq = 3.0 * NO_NEIGHBOR;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      // candidates: the atoms of the cells around i, ascending (the same order as the all-atom loop)
      const cand = cells.candidates(xi, yi, zi);
      for (let t = 0; t < cand.length; t++) {
        const j = cand[t];
        if (j === i) continue;
        const jtype = this.map[type[j]];
        if (jtype < 0) continue;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        const ncutsq = this.params[this.elem2param[itype * ne + jtype]].ncutsq;
        if (!(rsq < ncutsq && mol[owner[i]] === mol[owner[j]])) continue;
        if (rsq < nb1rsq) {
          nb3 = nb2; nb2 = nb1; nb1 = j;
          nb3rsq = nb2rsq; nb2rsq = nb1rsq; nb1rsq = rsq;
        } else if (rsq < nb2rsq) {
          nb3 = nb2; nb2 = j;
          nb3rsq = nb2rsq; nb2rsq = rsq;
        } else if (rsq < nb3rsq) {
          nb3 = j;
          nb3rsq = rsq;
        }
      }
      if (nb3rsq >= NO_NEIGHBOR) {
        if (i < nlocal) throw new StyleError('No enough neighbors to construct normal. Check the configuration to see whether atoms fly away.');
        continue;
      }
      nearest[3 * i] = nb1;
      nearest[3 * i + 1] = nb2;
      nearest[3 * i + 2] = nb3;
    }
    this.nearest = nearest;
  }

  /**
   * Attractive part, the r^-6 term with the taper (calc_attractive). Adds the forces on i and j to fi and fj and returns
   * the energy.
   */
  private calcAttractive(p: DripParam, rsq: number, rvec: number[], fi: number[], fj: number[]): number {
    const r = Math.sqrt(rsq);
    const roz0sq = rsq / (p.z0 * p.z0);
    const { t: tp, dtap } = tap(r, p.rcut);
    const r6 = p.A / (roz0sq * roz0sq * roz0sq);
    const dr6 = -6 * r6 / r;
    const phi = -r6 * tp;
    const fpair = -HALF * (r6 * dtap + dr6 * tp);
    for (let k = 0; k < 3; k++) {
      fi[k] += rvec[k] * fpair / r;
      fj[k] -= rvec[k] * fpair / r;
    }
    return phi;
  }

  /**
   * Repulsive part, depending on the transverse distance and the dihedral angle (calc_repulsive). The forces on the
   * neighbors of i and j go straight to the force array; those on i and j to fi and fj. Returns the energy.
   */
  private calcRepulsive(
    pc: PairCompute, i: number, j: number, p: DripParam, rsq: number, rvec: number[], ni: number[],
    nrm: { dk: M3; dl: M3; dm: M3 }, fi: number[], fj: number[],
  ): number {
    const { f } = pc;
    const nst = this.nearest;
    const { C0, C2, C4, C, delta, lambda, z0 } = p;
    const cutoff = p.rcut;
    const nbi = [nst[3 * i], nst[3 * i + 1], nst[3 * i + 2]];
    const nbj = [nst[3 * j], nst[3 * j + 1], nst[3 * j + 2]];
    const r = Math.sqrt(rsq);
    // the normal's derivatives wrt atom i are zero (the normal is built from the neighbors only)
    const dniDri = zero3();
    const { drhosqDri, drhosqDrj, drhosqDrn1, drhosqDrn2, drhosqDrn3 } = this.drhosq(rvec, ni, dniDri, nrm.dk, nrm.dl, nrm.dm);
    const { tdij, rhosqij, dtdij } = this.td(C0, C2, C4, delta, rvec, r, ni);
    const dih = this.dihedral(pc, i, j, p, rhosqij);
    const V2 = C + tdij + dih.gij;
    const { t: tp, dtap: dtp } = tap(r, cutoff);
    const V1 = libmExp(-lambda * (r - z0));
    const dV1 = -V1 * lambda;
    const phi = tp * V1 * V2;
    const fnbi: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const fnbj: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let k = 0; k < 3; k++) {
      const tmp = HALF * (dtp * V1 + tp * dV1) * V2 * rvec[k] / r;
      fi[k] += tmp;
      fj[k] -= tmp;
      fi[k] -= HALF * tp * V1 * ((dtdij + dih.dDrhosq) * drhosqDri[k] + dih.dDi[k]);
      fj[k] -= HALF * tp * V1 * ((dtdij + dih.dDrhosq) * drhosqDrj[k] + dih.dDj[k]);
      fnbi[0][k] = -HALF * tp * V1 * ((dtdij + dih.dDrhosq) * drhosqDrn1[k] + dih.dDk[0][k]);
      fnbi[1][k] = -HALF * tp * V1 * ((dtdij + dih.dDrhosq) * drhosqDrn2[k] + dih.dDk[1][k]);
      fnbi[2][k] = -HALF * tp * V1 * ((dtdij + dih.dDrhosq) * drhosqDrn3[k] + dih.dDk[2][k]);
      fnbj[0][k] = -HALF * tp * V1 * dih.dDl[0][k];
      fnbj[1][k] = -HALF * tp * V1 * dih.dDl[1][k];
      fnbj[2][k] = -HALF * tp * V1 * dih.dDl[2][k];
    }
    for (let k = 0; k < 3; k++) {
      for (let m = 0; m < 3; m++) {
        f[3 * nbi[m] + k] += fnbi[m][k];
        f[3 * nbj[m] + k] += fnbj[m][k];
      }
    }
    if (pc.vatom) {
      for (let m = 0; m < 3; m++) vTally2Newton(pc.vatom, nbi[m], fnbi[m], pc.x);
      for (let m = 0; m < 3; m++) vTally2Newton(pc.vatom, nbj[m], fnbj[m], pc.x);
    }
    return phi;
  }

  /** Derivatives of rho^2 of the pair wrt its atoms and the normal's neighbors (get_drhosqij). */
  private drhosq(rij: number[], ni: number[], dniDri: M3, dn1: M3, dn2: M3, dn3: M3) {
    const niDotRij = dot(ni, rij);
    const a = matDotVec(dniDri, rij), b1 = matDotVec(dn1, rij), b2 = matDotVec(dn2, rij), b3 = matDotVec(dn3, rij);
    const drhosqDri: number[] = [], drhosqDrj: number[] = [], drhosqDrn1: number[] = [], drhosqDrn2: number[] = [], drhosqDrn3: number[] = [];
    for (let k = 0; k < 3; k++) {
      drhosqDri[k] = -2 * rij[k] - 2 * niDotRij * (-ni[k] + a[k]);
      drhosqDrj[k] = 2 * rij[k] - 2 * niDotRij * ni[k];
      drhosqDrn1[k] = -2 * niDotRij * b1[k];
      drhosqDrn2[k] = -2 * niDotRij * b2[k];
      drhosqDrn3[k] = -2 * niDotRij * b3[k];
    }
    return { drhosqDri, drhosqDrj, drhosqDrn1, drhosqDrn2, drhosqDrn3 };
  }

  /** Transverse decay f(rho) and its rho^2 derivative (td); rho^2 is clamped at zero. */
  private td(C0: number, C2: number, C4: number, delta: number, rvec: number[], r: number, n: number[]) {
    const nDotR = dot(n, rvec);
    let rhosqij = r * r - nDotR * nDotR;
    if (rhosqij < 0) rhosqij = 0;
    const delSq = delta * delta;
    const rodSq = rhosqij / delSq;
    const tdij = libmExp(-rodSq) * (C0 + rodSq * (C2 + rodSq * C4));
    const dtdij = -tdij / delSq + libmExp(-rodSq) * (C2 + 2 * C4 * rodSq) / delSq;
    return { tdij, rhosqij, dtdij };
  }

  /**
   * Dihedral-angle correction gij (dihedral): the sum over the three neighbors k of i and l of j of the exp(-eta * product
   * of cos omega) terms, times the rho^2 taper B. Returns the value and its derivatives wrt rho^2 and the 8 atoms.
   */
  private dihedral(pc: PairCompute, i: number, j: number, p: DripParam, rhosq: number) {
    const { x } = pc;
    const nst = this.nearest;
    const zeroOut = { gij: 0, dDrhosq: 0, dDi: [0, 0, 0], dDj: [0, 0, 0], dDk: zero3(), dDl: zero3() };
    if (rhosq >= p.rhocutsq) return zeroOut;
    const k = [nst[3 * i], nst[3 * i + 1], nst[3 * i + 2]];
    const l = [nst[3 * j], nst[3 * j + 1], nst[3 * j + 2]];
    const pos = (a: number): number[] => [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
    // cos[m][n] = cos omega_{k_m i j l_n}, with its derivatives wrt k_m, i, j and l_n
    const cos: number[][] = zero3();
    const dcos: { dk: number[]; di: number[]; dj: number[]; dl: number[] }[][] = [[], [], []];
    for (let m = 0; m < 3; m++) {
      for (let n = 0; n < 3; n++) {
        const c = cosOmega(pos(k[m]), pos(i), pos(j), pos(l[n]));
        cos[m][n] = c.cos;
        dcos[m][n] = c;
      }
    }
    const eta = p.eta;
    const epart = [0, 1, 2].map((m) => libmExp(-eta * cos[m][0] * cos[m][1] * cos[m][2]));
    const D2 = epart[0] + epart[1] + epart[2];
    const tr = tapRho(rhosq, p.rhocutsq);
    const D0 = p.B * tr.t;
    const gij = D0 * D2;
    const dDrhosq = p.B * tr.drhosq * D2;
    // d gij / d cos[m][n]
    const dcosKl = zero3();
    dcosKl[0][0] = -D0 * epart[0] * eta * cos[0][1] * cos[0][2];
    dcosKl[0][1] = -D0 * epart[0] * eta * cos[0][0] * cos[0][2];
    dcosKl[0][2] = -D0 * epart[0] * eta * cos[0][0] * cos[0][1];
    dcosKl[1][0] = -D0 * epart[1] * eta * cos[1][1] * cos[1][2];
    dcosKl[1][1] = -D0 * epart[1] * eta * cos[1][0] * cos[1][2];
    dcosKl[1][2] = -D0 * epart[1] * eta * cos[1][0] * cos[1][1];
    dcosKl[2][0] = -D0 * epart[2] * eta * cos[2][1] * cos[2][2];
    dcosKl[2][1] = -D0 * epart[2] * eta * cos[2][0] * cos[2][2];
    dcosKl[2][2] = -D0 * epart[2] * eta * cos[2][0] * cos[2][1];
    const dDk = zero3(), dDl = zero3();
    const dDi = [0, 0, 0], dDj = [0, 0, 0];
    for (let m = 0; m < 3; m++) {
      for (let d = 0; d < 3; d++) {
        dDk[0][d] += dcosKl[0][m] * dcos[0][m].dk[d];
        dDk[1][d] += dcosKl[1][m] * dcos[1][m].dk[d];
        dDk[2][d] += dcosKl[2][m] * dcos[2][m].dk[d];
        dDl[0][d] += dcosKl[m][0] * dcos[m][0].dl[d];
        dDl[1][d] += dcosKl[m][1] * dcos[m][1].dl[d];
        dDl[2][d] += dcosKl[m][2] * dcos[m][2].dl[d];
      }
      for (let n = 0; n < 3; n++) {
        for (let d = 0; d < 3; d++) {
          dDi[d] += dcosKl[m][n] * dcos[m][n].di[d];
          dDj[d] += dcosKl[m][n] * dcos[m][n].dj[d];
        }
      }
    }
    return { gij, dDrhosq, dDi, dDj, dDk, dDl };
  }
}

/**
 * cos omega of the dihedral k-i-j-l and its derivatives wrt k, i, j and l (deriv_cos_omega). The two unit normals are
 * the cross products of (i, j, k) and (j, i, l), with the signs native uses.
 */
const cosOmega = (rk: number[], ri: number[], rj: number[], rl: number[]) => {
  const a = derivCross(ri, rj, rk);
  const ejik = a.cross.map((v) => -v);
  const dejikDri = a.dk.map((row) => row.map((v) => -v));
  const dejikDrj = a.dl.map((row) => row.map((v) => -v));
  const dejikDrk = a.dm.map((row) => row.map((v) => -v));
  const b = derivCross(rj, ri, rl);
  const eijl = b.cross.map((v) => -v);
  const deijlDrj = b.dk.map((row) => row.map((v) => -v));
  const deijlDri = b.dl.map((row) => row.map((v) => -v));
  const deijlDrl = b.dm.map((row) => row.map((v) => -v));
  const dk = matDotVec(dejikDrk, eijl);
  const t1 = matDotVec(dejikDri, eijl), t2 = matDotVec(deijlDri, ejik);
  const di = [0, 1, 2].map((m) => t1[m] + t2[m]);
  const u1 = matDotVec(dejikDrj, eijl), u2 = matDotVec(deijlDrj, ejik);
  const dj = [0, 1, 2].map((m) => u1[m] + u2[m]);
  const dl = matDotVec(deijlDrl, ejik);
  return { cos: dot(ejik, eijl), dk, di, dj, dl };
};
