/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MGPT/pair_mgpt.cpp, src/MGPT/pair_mgpt.h and src/MGPT/mgpt_linalg.cpp (the generic, unoptimized linear
 * algebra that LAMMPS compiles when __SSE3__ is not defined, as in the reference binary) at commit b332f28, and the
 * ev_tally, ev_tally3, ev_tally4 and virial_fdotr_compute functions of src/pair.cpp that pair mgpt calls.
 * Contributing authors (pair_mgpt.cpp): Tomas Oppelstrup, LLNL (oppelstrup2@llnl.gov) and John Moriarty, LLNL
 * (moriarty2@llnl.gov). Fast MGPT algorithm developed by Tomas Oppelstrup (2015) based on the matrix MGPT v4.4 FORTRAN
 * routine of John Moriarty (2006) as converted to C++ for LAMMPS application by Jamie Marian and Alexander Stukowski
 * (2011). Copyright (2015) Lawrence Livermore National Security, LLC, LLNL-CODE-674031, distributed under the GNU
 * General Public License version 2; the LLNL notice is kept in NOTICE.llnl next to this file. LAMMPS is Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 *
 * pair_style mgpt: the quantum-based MGPT multi-ion potential (docs.lammps.org/pair_mgpt.html) with its volume-dependent
 * pair, three-ion and four-ion terms evaluated through d (or f) state bond matrices. Every owned and ghost atom takes part
 * (the full neighbor list with the neighbors of ghosts), each cluster weighted by where its centre lies relative to the
 * subdomain. The computation runs on native LAMMPS's atom numbering (./neigh.ts), so the cluster enumeration, the
 * signs that follow the atom order and every sum are native's.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { Pair } from '../../engine/force/types';
import type { Geometry } from '../../engine/domain';
import { exp } from '../libm/exp';
import { log } from '../libm/log';
import { pow } from '../libm/pow';
import type { SplineValue } from './splinetab';
import { evalPot, evalVir, readpot, type PotData } from './readpot';
import { buildLayout, positions, type MgptLayout } from './neigh';

/** Matrix stride: the C++ Matrix is double m[8][8], element m[r][c] at 8 r + c. */
const MS = 64;
/** bond_data: H, Hx, Hy, Hz. */
const BS = 4 * MS;
/** triplet_data: H1H2, H1xH2, H1yH2, H1zH2, H1H2x, H1H2y, H1H2z. */
const TS = 7 * MS;
const H1H2 = 0, H1XH2 = MS, H1YH2 = 2 * MS, H1ZH2 = 3 * MS, H1H2X = 4 * MS, H1H2Y = 5 * MS, H1H2Z = 6 * MS;
/** MathConst::THIRD */
const THIRD = 1.0 / 3.0;

/**
 * The bond hash of one compute_x call (pair_mgpt.h Hash<bond_data,Doublet>): the bond matrices of each atom pair, made on
 * first use. Matrices live in one pool (BS doubles per bond).
 */
class BondStore {
  pool = new Float64Array(BS * 256);
  fl = new Float64Array(256);
  count = 0;
  private index = new Map<number, number>();
  constructor(private nall: number) {}
  reset(nall: number): void {
    this.nall = nall;
    this.count = 0;
    this.index.clear();
  }
  lookup(i: number, j: number): number {
    const k = i < j ? i * this.nall + j : j * this.nall + i;
    const b = this.index.get(k);
    return b === undefined ? -1 : b;
  }
  insert(i: number, j: number): number {
    const k = i < j ? i * this.nall + j : j * this.nall + i;
    if (this.count === this.fl.length) {
      const p = new Float64Array(this.pool.length * 2);
      p.set(this.pool);
      this.pool = p;
      const f = new Float64Array(this.fl.length * 2);
      f.set(this.fl);
      this.fl = f;
    }
    const b = this.count++;
    this.index.set(k, b);
    return b;
  }
}

export class PairMGPT extends Pair {
  readonly name = 'mgpt';
  manybody = true;
  needsFull = false;
  needsHalf = false;
  virialFdotr = true;

  private P: PotData | null = null;
  private cutoff = 0;
  private rmax = 0;
  private rcrit = 0;
  private lmax = 5;
  private lang = 3;
  private anorm3 = 0;
  private anorm4 = 0;
  private ddl = new Float64Array(5);
  private volpresFlag = 1;
  private nbodyFlag = 15;
  private layout: MgptLayout | null = null;
  private bonds = new BondStore(0);
  /** triplet_data work areas T12, T23, T31, T45, T56, T64 */
  private T = new Float64Array(6 * TS);
  /** fl_deriv_new outputs: fl_0, fl_x, fl_y, fl_z, fl_rp, fl_p1, fl_r0, fl_al */
  private FL = new Float64Array(8);
  /** tr_trace3 outputs */
  private TR = new Float64Array(3);
  private sv: SplineValue = { y: 0, dy: 0, d2y: 0 };

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 5) throw new StyleError('Not enough arguments for mgpt (MGPT) pair coefficients.');
    // Make sure I,J args are * *
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Incorrect args for pair coefficients');
    const vol = Number(args[4]);
    if (args[4].trim() === '' || !Number.isFinite(vol)) {
      throw new StyleError(`Expected floating point parameter instead of '${args[4]}' in input script or data file`);
    }
    if (vol <= 0.0) throw new StyleError('Invalid volume in mgpt (MGPT) pair coefficients.');

    let volpres = 1;
    let nbody = 0;
    let nbodyTag = false;
    for (let iarg = 5; iarg < args.length;) {
      const w = args[iarg];
      if (w === 'volpress') { // Volumetric pressure flag
        if (iarg + 2 > args.length) throw new StyleError('Incorrect args for pair coefficients');
        if (args[iarg + 1] === 'yes') volpres = 1;
        else if (args[iarg + 1] === 'no') volpres = 0;
        else {
          throw new StyleError(`Invalid value for volumetric pressure argument. It should be "volpress yes" or "volpress no". The value is "${args[iarg + 1]}".`);
        }
        iarg += 2;
      } else if (w === 'nbody') {
        if (iarg + 2 > args.length) throw new StyleError('Incorrect args for pair coefficients');
        const v = args[iarg + 1];
        // strspn(arg, "1234") == strlen(arg): only the digits 1-4 (an empty word also passes, natively too)
        if (/^[1234]*$/.test(v)) {
          nbody = 0;
          for (let i = 0; i < 4; i++) if (v.includes(String.fromCharCode(49 + i))) nbody += 1 << i;
        } else {
          throw new StyleError(`Invalid value for nbody flag. It should be e.g. "nbody=1234" (for single, pair, triple, and quad forces/energies). `
            + `For e.g. only pair and triple forces/energies, use "nbody=23". The default is "nbody=1234". The current value is "${v}".`);
        }
        nbodyTag = true;
        iarg += 2;
      } else if (w === 'precision') {
        if (iarg + 2 > args.length) throw new StyleError('Incorrect args for pair coefficients');
        // "precision single" selects single-precision matrix kernels only in a build with __SSE3__; the reference build
        // (default CMake flags) uses the generic double-precision routines either way (mgpt_linalg.cpp)
        if (args[iarg + 1] !== 'single' && args[iarg + 1] !== 'double') {
          throw new StyleError(`Invalid value for precision argument. It should be "precision single" or "precision double". The value is "${args[iarg + 1]}".`);
        }
        iarg += 2;
      } else {
        throw new StyleError('Invalid argument. Allowed arguments are: volpress {yes|no} , default = yes; precision {single|double} , default = double; '
          + `nbody {[1234,]*} , default = whichever terms potential require. The invalid argument is "${w}".`);
      }
    }

    let parmin: string, potin: string;
    try {
      parmin = ctx.readFile(args[2]);
      potin = ctx.readFile(args[3]);
    } catch {
      throw new StyleError(`Cannot open MGPT potential files ${args[2]} ${args[3]}`);
    }
    const P = readpot(parmin, args[2], potin, args[3], vol);

    // Set up default and requested nbody forces to include
    let nbodyDefault = (1 << 0) + (1 << 1) + (1 << 2) + (1 << 3);
    if (P.vd === 0.0 && P.dvd === 0.0) nbodyDefault -= 1 << 2; // No 3-body contributions
    if (P.ve === 0.0 && P.dve === 0.0) nbodyDefault -= 1 << 3; // No 4-body contributions
    if (!nbodyTag) nbody = nbodyDefault;
    if (nbody !== nbodyDefault) {
      ctx.log(`Warning: nbody=${nbody} (suggested=${nbodyDefault}) set to disregard multibody-forces in potential.`);
    }

    this.P = P;
    this.volpresFlag = volpres;
    this.nbodyFlag = nbody;
    this.anorm3 = P.anorm3;
    this.anorm4 = P.anorm4;
    this.lmax = P.lmax;
    this.lang = P.lang;
    this.ddl = Float64Array.from(P.ddl);
    if (this.lmax !== 5 && this.lmax !== 7) {
      throw new StyleError(`pair_style mgpt: the parmin file ${args[2]} gives lmax = ${this.lmax}; the bond matrices exist for d (lmax 5) and f (lmax 7) states only`);
    }

    // Set matrix param, cutoff, LAMMPS param
    this.rcrit = P.rcrit;
    this.rmax = P.rmax;
    this.cutoff = P.rmax;
    if (this.rcrit > this.rmax) this.cutoff = this.rcrit;

    // Set atomic mass (Atom::set_mass for every type).
    const s = ctx.s;
    if (s) {
      if (s.rmass) throw new StyleError(`Cannot set mass for atom style ${s.atomStyle}`);
      if (!(P.mass > 0.0)) throw new StyleError(`Invalid mass value ${P.mass} for atom type 1`);
      for (let t = 1; t <= this.ntypes; t++) s.massByType[t] = P.mass;
    }
  }

  override initStyle(ctx: StyleContext): void {
    if (!this.P) throw new StyleError('All pair coeffs are not set');
    if (ctx.newtonPair === false) throw new StyleError('Pair style mgpt requires newton pair on.');
    if (ctx.s?.box.triclinic) {
      throw new StyleError('pair_style mgpt with a triclinic box is not supported by the browser engine (orthogonal boxes only)');
    }
  }

  override initOne(_i: number, _j: number): number {
    return this.cutoff;
  }

  override compute(pc: PairCompute): void {
    const P = this.P;
    if (!P) throw new StyleError('All pair coeffs are not set');
    const s = pc.s, nb = pc.nb, g = pc.geom;
    if (g.triclinic) {
      throw new StyleError('pair_style mgpt with a triclinic box is not supported by the browser engine (orthogonal boxes only)');
    }
    // mgpt reads the neighbor indices without masking the special bits (pair_mgpt.cpp NIDX with NEIGHMASK defined), so
    // native LAMMPS has no meaningful result with special neighbors; neighbor exclusions are not reproduced here
    if (s.topo.bonds.n > 0) throw new StyleError('pair_style mgpt with bonded atoms (special neighbors) is not supported');
    if (nb.excludes.length) throw new StyleError('pair_style mgpt with neigh_modify exclude is not supported by the browser engine');
    if (nb.includeBit) throw new StyleError('pair_style mgpt with neigh_modify include is not supported by the browser engine');

    let L = this.layout;
    if (!L || L.nbuild !== nb.nbuild || L.stamp !== (s.order ?? null) || L.n !== s.n) {
      L = buildLayout(s, nb, g, this.cutoff);
      this.layout = L;
    }
    positions(L, s, g);
    const nall = L.nall, n = L.n;
    const F = new Float64Array(3 * nall);
    const eatom = pc.eatom ? new Float64Array(nall) : null;
    const vatom = pc.vatom ? new Float64Array(6 * nall) : null;
    const vir = new Float64Array(6);
    const eng = this.computeX(L, g, s.n, F, eatom, vatom, vir);
    const X = L.X;

    if (!this.hybridSub) {
      // Pair::virial_fdotr_compute (owned atoms, then ghosts, in native order) after the volume terms of compute_x,
      // then CommBrick::reverse_comm: the ghost forces are added to the atoms they were sent from, last swap first
      for (let k = 0; k < nall; k++) {
        const fx = F[3 * k], fy = F[3 * k + 1], fz = F[3 * k + 2];
        const x = X[3 * k], y = X[3 * k + 1], z = X[3 * k + 2];
        vir[0] += fx * x;
        vir[1] += fy * y;
        vir[2] += fz * z;
        vir[3] += fy * x;
        vir[4] += fz * x;
        vir[5] += fz * y;
      }
      for (let c = 0; c < 6; c++) pc.acc.virial[c] += vir[c];
      fold(L, F, 3);
      const f = pc.f;
      for (let k = 0; k < n; k++) {
        const e = 3 * L.order[k];
        f[e] += F[3 * k]; f[e + 1] += F[3 * k + 1]; f[e + 2] += F[3 * k + 2];
      }
      if (eatom && pc.eatom) {
        fold(L, eatom, 1);
        for (let k = 0; k < n; k++) pc.eatom[L.order[k]] += eatom[k];
      }
      if (vatom && pc.vatom) {
        fold(L, vatom, 6);
        for (let k = 0; k < n; k++) for (let c = 0; c < 6; c++) pc.vatom[6 * L.order[k] + c] += vatom[6 * k + c];
      }
      pc.fdotrDone = true;
    } else {
      // a sub-style of pair_style hybrid: the forces go onto the engine's own atoms (owned and ghost images), whose
      // fdotr sum and fold the hybrid force field takes; native ghosts the engine does not hold go to their source atom
      const eng2 = engineIndex(L, pc);
      for (let k = nall - 1; k >= n; k--) {
        if (eng2[k] >= 0) continue;
        const t = L.from[k];
        for (let c = 0; c < 3; c++) F[3 * t + c] += F[3 * k + c];
        if (eatom) eatom[t] += eatom[k];
        if (vatom) for (let c = 0; c < 6; c++) vatom[6 * t + c] += vatom[6 * k + c];
      }
      for (let k = 0; k < nall; k++) {
        const e = eng2[k];
        if (e < 0) continue;
        for (let c = 0; c < 3; c++) pc.f[3 * e + c] += F[3 * k + c];
        if (eatom && pc.eatom) pc.eatom[e] += eatom[k];
        if (vatom && pc.vatom) for (let c = 0; c < 6; c++) pc.vatom[6 * e + c] += vatom[6 * k + c];
      }
      // As native LAMMPS does (src/pair_hybrid.cpp PairHybrid::compute at b332f28): the sub-styles run with
      // "vflag_substyle = vflag & ~VIRIAL_FDOTR", so mgpt's vflag_global is 0 and the volume terms that compute_x puts
      // into its own virial are never added to the hybrid's ("if (vflag_global) ... virial[n] += styles[m]->virial[n]"
      // is off as well, the hybrid summing F dot r instead): under pair hybrid the pressure has no mgpt volume terms.
    }
    pc.acc.evdwl += eng;
  }

  /* ---------------------------------------------------------------------- */

  /** pair_mgpt.h c1_outside (orthogonal box): the atom lies too far outside the subdomain for a pair term. */
  private c1Outside(X: Float64Array, a: number, g: Geometry): number {
    const stol = 1e-5;
    const rmax = this.rmax;
    let rout = 0.0;
    for (let p = 0; p < 3; p++) {
      const cog = X[3 * a + p];
      const lo = g.lo[p], hi = g.hi[p];
      if (cog < lo - 0.5 * rmax - stol) return 1;
      if (cog > hi + 0.5 * rmax + stol) return 1;
      if (cog < lo - stol) {
        const t = cog - (lo - stol);
        rout = rout + t * t;
      } else if (cog > hi + stol) {
        const t = cog - (hi + stol);
        rout = rout + t * t;
      }
    }
    if (rout > 0.25 * rmax * rmax) return 1;
    return 0;
  }

  /** pair_mgpt.h c2_outside (orthogonal box): the bond's midpoint lies too far outside the subdomain. */
  private c2Outside(X: Float64Array, a: number, b: number, g: Geometry): number {
    const stol = 1e-5;
    const rcrit = this.rcrit;
    let rout = 0.0;
    for (let p = 0; p < 3; p++) {
      const cog = 0.5 * (X[3 * a + p] + X[3 * b + p]);
      const lo = g.lo[p], hi = g.hi[p];
      if (cog < lo - 0.5 * rcrit - stol) return 1;
      if (cog > hi + 0.5 * rcrit + stol) return 1;
      if (cog < lo - stol) {
        const t = cog - (lo - stol);
        rout = rout + t * t;
      } else if (cog > hi + stol) {
        const t = cog - (hi + stol);
        rout = rout + t * t;
      }
    }
    if (rout > 0.25 * rcrit * rcrit) return 1;
    return 0;
  }

  /**
   * pair_mgpt.h get_weight (orthogonal box): the weight of a cluster of nat atoms (2..4) by the distance of its centre
   * from the nearest subdomain face, ramping from 0 to 1 over 2 stol across the face.
   */
  private getWeight(X: Float64Array, g: Geometry, nat: number, a: number, b: number, c: number, d: number): number {
    let weight = 1.0;
    const stol = 1e-5;
    for (let p = 0; p < 3; p++) {
      let cog = 0.0;
      cog = cog + X[3 * a + p];
      cog = cog + X[3 * b + p];
      if (nat > 2) cog = cog + X[3 * c + p];
      if (nat > 3) cog = cog + X[3 * d + p];
      cog = cog * (1.0 / nat);
      const s0 = g.lo[p], s1 = g.hi[p];
      const q = cog < 0.5 * (s0 + s1) ? cog - s0 : s1 - cog;
      let w = q * (0.5 / stol) + 0.5;
      if (w > 1.0) w = 1.0;
      if (w < 0.0) w = 0.0;
      weight = weight * w;
    }
    return weight;
  }

  /** mgpt_linalg.cpp transprod_generic: C = A B^T over rows and columns 1..lmax (C[c + 8 i + j]). */
  private trMul(A: Float64Array, a: number, B: Float64Array, b: number, C: Float64Array, c: number): void {
    const n = this.lmax;
    for (let i = 1; i <= n; i++) {
      const ai = a + 8 * i;
      for (let j = 1; j <= n; j++) {
        const bj = b + 8 * j;
        let s = 0.0;
        for (let k = 1; k <= n; k++) s = s + A[ai + k] * B[bj + k];
        C[c + 8 * i + j] = s;
      }
    }
  }

  /** mgpt_linalg.cpp transtrace3_generic: the traces of A B0^T, A B1^T, A B2^T into TR. */
  private trTrace3(A: Float64Array, a: number, B0: Float64Array, b0: number, B1: Float64Array, b1: number, B2: Float64Array, b2: number): void {
    const n = this.lmax;
    let t0 = 0.0, t1 = 0.0, t2 = 0.0;
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= n; j++) {
        const idx = 8 * i + j;
        const atmp = A[a + idx];
        t0 = t0 + atmp * B0[b0 + idx];
        t1 = t1 + atmp * B1[b1 + idx];
        t2 = t2 + atmp * B2[b2 + idx];
      }
    }
    const TR = this.TR;
    TR[0] = t0; TR[1] = t1; TR[2] = t2;
  }

  /** pair_mgpt.h transtrace: the trace of a b^T over rows and columns 1..lmax. */
  private transtrace(A: Float64Array, a: number, B: Float64Array, b: number): number {
    const n = this.lmax;
    let s = 0.0;
    for (let i = 1; i <= n; i++) {
      for (let k = 1; k <= n; k++) s = s + A[a + 8 * i + k] * B[b + 8 * i + k];
    }
    return s;
  }

  /** PairMGPT::make_bond: the bond matrices of atoms i < j (rrij = x_i - x_j) into bond b of the store. */
  private makeBond(X: Float64Array, i: number, j: number, b: number): void {
    const B = this.bonds;
    const M = B.pool;
    const base = BS * b;
    const rr0 = X[3 * i] - X[3 * j], rr1 = X[3 * i + 1] - X[3 * j + 1], rr2 = X[3 * i + 2] - X[3 * j + 2];
    let rij = 0.0;
    rij = rij + rr0 * rr0;
    rij = rij + rr1 * rr1;
    rij = rij + rr2 * rr2;
    // Zero all matrix elements
    M.fill(0.0, base, base + BS);
    if (rij <= this.rcrit * this.rcrit) {
      if (this.lang === 3) this.hamltn5(rr0, rr1, rr2, M, base, b);
      else this.hamltn7(rr0, rr1, rr2, M, base, b);
    } else {
      B.fl[b] = 0.0;
    }
  }

  /** PairMGPT::get_triplet: the bond matrices of (i, j) and (i, k), made on first use, multiplied into triplet t. */
  private getTriplet(X: Float64Array, i: number, j: number, k: number, t: number, dv: Float64Array, a: number, c: number): number {
    const B = this.bonds;
    let bij = B.lookup(i, j);
    let bik = B.lookup(i, k);
    if (bij < 0) {
      bij = B.insert(i, j);
      if (i < j) this.makeBond(X, i, j, bij);
      else this.makeBond(X, j, i, bij);
    }
    if (bik < 0) {
      bik = B.insert(i, k);
      if (i < k) this.makeBond(X, i, k, bik);
      else this.makeBond(X, k, i, bik);
    }
    // make_triplet
    const P = B.pool, T = this.T;
    const tb = TS * t, ij = BS * bij, ik = BS * bik;
    this.trMul(P, ij, P, ik, T, tb + H1H2);
    this.trMul(P, ij + MS, P, ik, T, tb + H1XH2);
    this.trMul(P, ij + 2 * MS, P, ik, T, tb + H1YH2);
    this.trMul(P, ij + 3 * MS, P, ik, T, tb + H1ZH2);
    this.trMul(P, ij, P, ik + MS, T, tb + H1H2X);
    this.trMul(P, ij, P, ik + 2 * MS, T, tb + H1H2Y);
    this.trMul(P, ij, P, ik + 3 * MS, T, tb + H1H2Z);
    dv[a] = B.fl[bij];
    dv[c] = B.fl[bik];
    return tb;
  }

  /**
   * PairMGPT::compute_x: forces (F, native atoms), per-atom energy and virial, and the volume terms of the global
   * virial (vir[0..2]); returns the energy tallied into eng_vdwl. evflag is set at every evaluation (the engine has
   * energies and the virial at every step); natively a step without energy or per-atom virial takes the same forces
   * with another summation order and leaves the three- and four-ion volume terms out of the virial.
   */
  private computeX(L: MgptLayout, g: Geometry, natoms: number, ff: Float64Array, eatom: Float64Array | null,
    vatom: Float64Array | null, virial: Float64Array): number {
    const P = this.P!;
    const xx = L.X;
    const ntot = L.nall, nloc = L.n;
    const nnF = L.first, nnL = L.list;
    const bonds = this.bonds;
    bonds.reset(ntot);
    this.T.fill(0.0);
    const TR = this.TR, T = this.T;
    const sv = this.sv;

    const nbody = this.nbodyFlag;
    const pairForces = (nbody >> 1) % 2, threeBodyForces = (nbody >> 2) % 2, fourBodyForces = (nbody >> 3) % 2;
    const pairEnergies = (nbody >> 1) % 2, threeBodyEnergies = (nbody >> 2) % 2, fourBodyEnergies = (nbody >> 3) % 2;
    const singleEnergies = nbody % 2;
    const volpres = this.volpresFlag;

    // Energy and force scale factor for unit conversion.
    const eScale = 0.5;
    const anorm3 = this.anorm3, anorm4 = this.anorm4;
    const vc = P.vc, vd = P.vd, ve = P.ve, dvc = P.dvc, dvd = P.dvd, dve = P.dve;

    let engVdwl = 0.0;
    let volvir2 = 0.0;
    let vir3t = 0.0, vir3v = 0.0, vir4 = 0.0;
    const rcut2Pair = this.rmax * this.rmax, rcut2Bond = this.rcrit * this.rcrit;

    let rhoinv: number;
    {
      let vtot = 1.0;
      for (let i = 0; i < 3; i++) vtot = vtot * (g.hi[i] - g.lo[i]);
      rhoinv = vtot / natoms;
    }

    let fix = 0.0, fjx = 0.0, fkx = 0.0, fmx = 0.0, dfix = 0.0, dfjx = 0.0, dfkx = 0.0, dfmx = 0.0;
    let fiy = 0.0, fjy = 0.0, fky = 0.0, fmy = 0.0, dfiy = 0.0, dfjy = 0.0, dfky = 0.0, dfmy = 0.0;
    let fiz = 0.0, fjz = 0.0, fkz = 0.0, fmz = 0.0, dfiz = 0.0, dfjz = 0.0, dfkz = 0.0, dfmz = 0.0;
    const fsave = new Float64Array(12);

    const first = new Int32Array(ntot + 1);
    const nlistShort = new Int32Array(nnL.length + 1);

    // per-atom virial helpers (Pair::ev_tally / ev_tally3 / ev_tally4 with newton on)
    const vadd = (a: number, v0: number, v1: number, v2: number, v3: number, v4: number, v5: number): void => {
      const va = vatom!;
      va[6 * a] += v0; va[6 * a + 1] += v1; va[6 * a + 2] += v2;
      va[6 * a + 3] += v3; va[6 * a + 4] += v4; va[6 * a + 5] += v5;
    };

    first[0] = 0;
    for (let i = 0; i < ntot; i++) {
      fix = fiy = fiz = 0.0;
      first[i + 1] = first[i];
      const c1 = this.c1Outside(xx, i, g);
      for (let jx = nnF[i]; jx < nnF[i + 1]; jx++) {
        fjx = fjy = fjz = 0.0;
        const j = nnL[jx];
        let rij2 = 0.0;
        for (let p = 0; p < 3; p++) {
          const t = xx[3 * i + p] - xx[3 * j + p];
          rij2 = rij2 + t * t;
        }

        if (c1 === 0 && rij2 < rcut2Pair) {
          if (j < i) {
            const w2 = this.getWeight(xx, g, 2, i, j, 0, 0);
            if (w2 > 0.0) {
              // Compute pair energy/force
              const rij = Math.sqrt(rij2);
              evalPot(P, rij, sv);
              let dePair = sv.y, df = sv.dy;
              dePair = dePair * eScale * w2;
              df = df / rij * w2;

              if (pairEnergies === 0) dePair = 0.0;
              if (pairForces === 0) df = 0.0;

              if (volpres && pairEnergies) {
                evalVir(P, rij, sv);
                const dvir = sv.y;
                volvir2 = volvir2 - dvir * w2;
                // Per-atom virial contribution of volumetric energy term
                if (vatom) {
                  for (let pp = 0; pp < 3; pp++) {
                    vatom[6 * i + pp] -= 0.5 * rhoinv * eScale * dvir * w2;
                    vatom[6 * j + pp] -= 0.5 * rhoinv * eScale * dvir * w2;
                  }
                }
              }

              const drijx = xx[3 * j] - xx[3 * i];
              const drijy = xx[3 * j + 1] - xx[3 * i + 1];
              const drijz = xx[3 * j + 2] - xx[3 * i + 2];

              fix = fix + df * drijx;
              fjx = fjx - df * drijx;
              fiy = fiy + df * drijy;
              fjy = fjy - df * drijy;
              fiz = fiz + df * drijz;
              fjz = fjz - df * drijz;

              // ev_tally(i,j,nloc,newton_pair,de_pair,0.0,-df * e_scale,-drijx,-drijy,-drijz)
              engVdwl += dePair;
              if (eatom) {
                const half = 0.5 * (dePair + 0.0);
                eatom[i] += half;
                eatom[j] += half;
              }
              if (vatom) {
                const fpair = -df * eScale;
                const dx = -drijx, dy = -drijy, dz = -drijz;
                const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
                const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
                vadd(i, 0.5 * v0, 0.5 * v1, 0.5 * v2, 0.5 * v3, 0.5 * v4, 0.5 * v5);
                vadd(j, 0.5 * v0, 0.5 * v1, 0.5 * v2, 0.5 * v3, 0.5 * v4, 0.5 * v5);
              }

              ff[3 * j] += fjx * eScale;
              ff[3 * j + 1] += fjy * eScale;
              ff[3 * j + 2] += fjz * eScale;
            }
          }
        }

        if (rij2 < rcut2Bond && this.c2Outside(xx, i, j, g) === 0) {
          // Add j to short neighbor list for i. Insert j to keep list sorted.
          let p = first[i + 1] - 1;
          while (p >= first[i] && nlistShort[p] > j) {
            nlistShort[p + 1] = nlistShort[p];
            p = p - 1;
          }
          nlistShort[p + 1] = j;
          first[i + 1] = first[i + 1] + 1;
        }
      }

      ff[3 * i] += fix * eScale;
      ff[3 * i + 1] += fiy * eScale;
      ff[3 * i + 2] += fiz * eScale;
    }

    const dv = new Float64Array(6); // dvir_ij, dvir_jk, dvir_ki, dvir_im, dvir_jm, dvir_km
    const DIJ = 0, DJK = 1, DKI = 2, DIM = 3, DJM = 4, DKM = 5;
    const T12 = 0, T23 = 1, T31 = 2, T45 = 3, T56 = 4, T64 = 5;
    const bp = bonds;

    for (let i = 0; i < ntot; i++) {
      fix = fiy = fiz = 0.0;

      if (threeBodyEnergies || threeBodyForces || fourBodyEnergies || fourBodyForces) {
        for (let jx = first[i]; jx < first[i + 1]; jx++) {
          fjx = fjy = fjz = 0.0;
          const j = nlistShort[jx];

          for (let kx = first[i]; kx < jx; kx++) {
            fkx = fky = fkz = 0.0;
            const k = nlistShort[kx];

            // Search lists of j and k, and see if 1) j is in k-list (closed triplet) 2) j and k have a common neighbor
            const cij = 1, cki = 1;
            const sij = i < j ? 1 : -1;
            const sjk = j < k ? 1 : -1;
            const ski = k < i ? 1 : -1;

            let t12 = -1, t23 = -1, t31 = -1;

            let mj = first[j];
            while (mj < first[j + 1] && nlistShort[mj] < k) mj = mj + 1;
            let cjk: number;
            if (mj < first[j + 1] && nlistShort[mj] === k) {
              // Closed triplet
              cjk = 1;
              if (j > i) continue; // Require k<j<i for closed triplets
            } else {
              // Open triplet
              cjk = 0;
            }

            const w3 = this.getWeight(xx, g, 3, i, j, k, 0);

            let tripletDefer: number;
            if (w3 > 0.0) {
              tripletDefer = 0;

              dv[DIJ] = dv[DJK] = dv[DKI] = 0.0;
              if (cij && cjk) t12 = this.getTriplet(xx, j, i, k, T12, dv, DIJ, DJK);
              if (cki && cjk) t23 = this.getTriplet(xx, k, i, j, T23, dv, DKI, DJK);
              if (cij && cki) t31 = this.getTriplet(xx, i, j, k, T31, dv, DIJ, DKI);

              fsave[0] = fix; fsave[1] = fiy; fsave[2] = fiz;
              fsave[3] = fjx; fsave[4] = fjy; fsave[5] = fjz;
              fsave[6] = fkx; fsave[7] = fky; fsave[8] = fkz;
              fix = fiy = fiz = 0.0;
              fjx = fjy = fjz = 0.0;
              fkx = fky = fkz = 0.0;

              let tr0 = 0.0, tr1 = 0.0, tr2 = 0.0, tr3 = 0.0;
              let xvir3t = 0.0, xvir3v = 0.0;

              if (t12 >= 0 && t23 >= 0) {
                const bki = bp.lookup(k, i);
                const bk = BS * bki;
                const Pm = bp.pool;
                if (threeBodyEnergies) {
                  tr0 = this.transtrace(T, t12 + H1H2, Pm, bk);
                  const dvir = ((dv[DIJ] + dv[DJK] + bp.fl[bki]) * vc + dvc) * tr0 * w3 / anorm3;
                  vir3t = vir3t + dvir;
                  xvir3t = xvir3t + dvir;
                }
                this.trTrace3(Pm, bk, T, t12 + H1XH2, T, t12 + H1YH2, T, t12 + H1ZH2);
                const trd1x = TR[0], trd1y = TR[1], trd1z = TR[2];
                this.trTrace3(Pm, bk, T, t12 + H1H2X, T, t12 + H1H2Y, T, t12 + H1H2Z);
                const trd2x = TR[0], trd2y = TR[1], trd2z = TR[2];
                this.trTrace3(T, t12 + H1H2, Pm, bk + MS, Pm, bk + 2 * MS, Pm, bk + 3 * MS);
                const trd3x = TR[0], trd3y = TR[1], trd3z = TR[2];

                dfix = ((-sij) * trd1x + (ski) * trd3x) * (vc / anorm3);
                dfjx = ((sij) * trd1x + (-sjk) * trd2x) * (vc / anorm3);
                dfkx = ((sjk) * trd2x + (-ski) * trd3x) * (vc / anorm3);

                dfiy = ((-sij) * trd1y + (ski) * trd3y) * (vc / anorm3);
                dfjy = ((sij) * trd1y + (-sjk) * trd2y) * (vc / anorm3);
                dfky = ((sjk) * trd2y + (-ski) * trd3y) * (vc / anorm3);

                dfiz = ((-sij) * trd1z + (ski) * trd3z) * (vc / anorm3);
                dfjz = ((sij) * trd1z + (-sjk) * trd2z) * (vc / anorm3);
                dfkz = ((sjk) * trd2z + (-ski) * trd3z) * (vc / anorm3);

                if (threeBodyForces) {
                  fix = fix + dfix * w3; fiy = fiy + dfiy * w3; fiz = fiz + dfiz * w3;
                  fjx = fjx + dfjx * w3; fjy = fjy + dfjy * w3; fjz = fjz + dfjz * w3;
                  fkx = fkx + dfkx * w3; fky = fky + dfky * w3; fkz = fkz + dfkz * w3;
                }
              }

              if (t12 >= 0) {
                if (threeBodyEnergies) {
                  tr1 = this.transtrace(T, t12 + H1H2, T, t12 + H1H2);
                  const dvir = (2.0 * (dv[DIJ] + dv[DJK]) * vd + dvd) * tr1 * w3 / anorm4;
                  vir3v = vir3v + dvir;
                  xvir3v = xvir3v + dvir;
                }
                this.trTrace3(T, t12 + H1H2, T, t12 + H1XH2, T, t12 + H1YH2, T, t12 + H1ZH2);
                const trd1x = TR[0], trd1y = TR[1], trd1z = TR[2];
                this.trTrace3(T, t12 + H1H2, T, t12 + H1H2X, T, t12 + H1H2Y, T, t12 + H1H2Z);
                const trd2x = TR[0], trd2y = TR[1], trd2z = TR[2];

                dfix = 2.0 * (-sij) * trd1x * (vd / anorm4);
                dfkx = 2.0 * (sjk) * trd2x * (vd / anorm4);
                dfjx = -(dfix + dfkx);

                dfiy = 2.0 * (-sij) * trd1y * (vd / anorm4);
                dfky = 2.0 * (sjk) * trd2y * (vd / anorm4);
                dfjy = -(dfiy + dfky);

                dfiz = 2.0 * (-sij) * trd1z * (vd / anorm4);
                dfkz = 2.0 * (sjk) * trd2z * (vd / anorm4);
                dfjz = -(dfiz + dfkz);

                if (threeBodyForces) {
                  fix = fix + dfix * w3; fiy = fiy + dfiy * w3; fiz = fiz + dfiz * w3;
                  fjx = fjx + dfjx * w3; fjy = fjy + dfjy * w3; fjz = fjz + dfjz * w3;
                  fkx = fkx + dfkx * w3; fky = fky + dfky * w3; fkz = fkz + dfkz * w3;
                }
              }

              if (t23 >= 0) {
                if (threeBodyEnergies) {
                  tr2 = this.transtrace(T, t23 + H1H2, T, t23 + H1H2);
                  const dvir = (2.0 * (dv[DJK] + dv[DKI]) * vd + dvd) * tr2 * w3 / anorm4;
                  vir3v = vir3v + dvir;
                  xvir3v = xvir3v + dvir;
                }
                this.trTrace3(T, t23 + H1H2, T, t23 + H1XH2, T, t23 + H1YH2, T, t23 + H1ZH2);
                const trd1x = TR[0], trd1y = TR[1], trd1z = TR[2];
                this.trTrace3(T, t23 + H1H2, T, t23 + H1H2X, T, t23 + H1H2Y, T, t23 + H1H2Z);
                const trd2x = TR[0], trd2y = TR[1], trd2z = TR[2];

                dfix = 2.0 * (ski) * trd1x * (vd / anorm4);
                dfjx = 2.0 * (-sjk) * trd2x * (vd / anorm4);
                dfkx = -(dfix + dfjx);

                dfiy = 2.0 * (ski) * trd1y * (vd / anorm4);
                dfjy = 2.0 * (-sjk) * trd2y * (vd / anorm4);
                dfky = -(dfiy + dfjy);

                dfiz = 2.0 * (ski) * trd1z * (vd / anorm4);
                dfjz = 2.0 * (-sjk) * trd2z * (vd / anorm4);
                dfkz = -(dfiz + dfjz);

                if (threeBodyForces) {
                  fix = fix + dfix * w3; fiy = fiy + dfiy * w3; fiz = fiz + dfiz * w3;
                  fjx = fjx + dfjx * w3; fjy = fjy + dfjy * w3; fjz = fjz + dfjz * w3;
                  fkx = fkx + dfkx * w3; fky = fky + dfky * w3; fkz = fkz + dfkz * w3;
                }
              }

              if (t31 >= 0) {
                if (threeBodyEnergies) {
                  tr3 = this.transtrace(T, t31 + H1H2, T, t31 + H1H2);
                  const dvir = (2.0 * (dv[DKI] + dv[DIJ]) * vd + dvd) * tr3 * w3 / anorm4;
                  vir3v = vir3v + dvir;
                  xvir3v = xvir3v + dvir;
                }
                this.trTrace3(T, t31 + H1H2, T, t31 + H1XH2, T, t31 + H1YH2, T, t31 + H1ZH2);
                const trd1x = TR[0], trd1y = TR[1], trd1z = TR[2];
                this.trTrace3(T, t31 + H1H2, T, t31 + H1H2X, T, t31 + H1H2Y, T, t31 + H1H2Z);
                const trd2x = TR[0], trd2y = TR[1], trd2z = TR[2];

                dfjx = 2.0 * (sij) * trd1x * (vd / anorm4);
                dfkx = 2.0 * (-ski) * trd2x * (vd / anorm4);
                dfix = -(dfjx + dfkx);

                dfjy = 2.0 * (sij) * trd1y * (vd / anorm4);
                dfky = 2.0 * (-ski) * trd2y * (vd / anorm4);
                dfiy = -(dfjy + dfky);

                dfjz = 2.0 * (sij) * trd1z * (vd / anorm4);
                dfkz = 2.0 * (-ski) * trd2z * (vd / anorm4);
                dfiz = -(dfjz + dfkz);

                if (threeBodyForces) {
                  fix = fix + dfix * w3; fiy = fiy + dfiy * w3; fiz = fiz + dfiz * w3;
                  fjx = fjx + dfjx * w3; fjy = fjy + dfjy * w3; fjz = fjz + dfjz * w3;
                  fkx = fkx + dfkx * w3; fky = fky + dfky * w3; fkz = fkz + dfkz * w3;
                }
              }

              const v33 = tr0 / anorm3;
              const v43 = (tr1 + tr2 + tr3) / anorm4;
              const deTriplet = (vc * v33 + vd * v43) * eScale * w3;

              {
                // ev_tally3(i,j,k,de_triplet,0.0,fj,fk,drji,drki) with fj, fk scaled by e_scale
                engVdwl += deTriplet;
                if (eatom) {
                  const third = THIRD * (deTriplet + 0.0);
                  eatom[i] += third;
                  eatom[j] += third;
                  eatom[k] += third;
                }
                if (vatom) {
                  const fj0 = fjx * eScale, fj1 = fjy * eScale, fj2 = fjz * eScale;
                  const fk0 = fkx * eScale, fk1 = fky * eScale, fk2 = fkz * eScale;
                  const ji0 = xx[3 * j] - xx[3 * i], ji1 = xx[3 * j + 1] - xx[3 * i + 1], ji2 = xx[3 * j + 2] - xx[3 * i + 2];
                  const ki0 = xx[3 * k] - xx[3 * i], ki1 = xx[3 * k + 1] - xx[3 * i + 1], ki2 = xx[3 * k + 2] - xx[3 * i + 2];
                  const v0 = ji0 * fj0 + ki0 * fk0;
                  const v1 = ji1 * fj1 + ki1 * fk1;
                  const v2 = ji2 * fj2 + ki2 * fk2;
                  const v3 = ji0 * fj1 + ki0 * fk1;
                  const v4 = ji0 * fj2 + ki0 * fk2;
                  const v5 = ji1 * fj2 + ki1 * fk2;
                  for (const a of [i, j, k]) vadd(a, THIRD * v0, THIRD * v1, THIRD * v2, THIRD * v3, THIRD * v4, THIRD * v5);
                  if (volpres) {
                    const dvir = -(xvir3v + xvir3t) * rhoinv * eScale * (1.0 / 3.0);
                    for (let pp = 0; pp < 3; pp++) {
                      vatom[6 * i + pp] += dvir;
                      vatom[6 * j + pp] += dvir;
                      vatom[6 * k + pp] += dvir;
                    }
                  }
                }

                fix = fix + fsave[0]; fiy = fiy + fsave[1]; fiz = fiz + fsave[2];
                fjx = fjx + fsave[3]; fjy = fjy + fsave[4]; fjz = fjz + fsave[5];
                fkx = fkx + fsave[6]; fky = fky + fsave[7]; fkz = fkz + fsave[8];
              }
            } else {
              tripletDefer = 1;
            }

            if (fourBodyEnergies || fourBodyForces) {
              if (j < i) { // Search for quadruplet
                let mj2 = first[j];
                let mk = first[k];
                // i is in both the j-list and the k-list, and i > k, and lists are sorted, so the loop terminates.
                while (nlistShort[mj2] < i && nlistShort[mk] < i) {
                  if (mj2 >= first[j + 1] || mk >= first[k + 1]) {
                    throw new StyleError(`pair_style mgpt: broken quadruplet loop (j=${j} first[j]=${first[j]} first[j+1]=${first[j + 1]} mj=${mj2}, k=${k} first[k]=${first[k]} first[k+1]=${first[k + 1]} mk=${mk})`);
                  }

                  if (nlistShort[mj2] === nlistShort[mk]) {
                    // Closed quadruplet
                    const m = nlistShort[mj2];
                    const cjm = 1, ckm = 1;

                    const sim = i < m ? 1 : -1;
                    const sjm = j < m ? 1 : -1;
                    const skm = k < m ? 1 : -1;

                    const w4 = this.getWeight(xx, g, 4, i, j, k, m);

                    if (w4 > 0.0) {
                      // Already know ij,jk,ki,jm,km bonds. Look for im bond.
                      let mi = first[i];
                      while (mi < first[i + 1] && nlistShort[mi] < m) mi = mi + 1;
                      const cim = mi < first[i + 1] && nlistShort[mi] === m ? 1 : 0;

                      if (cim === 0 || cjk === 0 || (cjk && cim && m < k)) {
                        if (tripletDefer) {
                          dv[DIJ] = dv[DJK] = dv[DKI] = 0.0;
                          if (cij && cjk) t12 = this.getTriplet(xx, j, i, k, T12, dv, DIJ, DJK);
                          if (cki && cjk) t23 = this.getTriplet(xx, k, i, j, T23, dv, DKI, DJK);
                          if (cij && cki) t31 = this.getTriplet(xx, i, j, k, T31, dv, DIJ, DKI);
                          tripletDefer = 0;
                        }

                        fmx = fmy = fmz = 0.0;
                        let xvir4 = 0.0;

                        fsave[0] = fix; fsave[1] = fiy; fsave[2] = fiz;
                        fsave[3] = fjx; fsave[4] = fjy; fsave[5] = fjz;
                        fsave[6] = fkx; fsave[7] = fky; fsave[8] = fkz;
                        fsave[9] = fmx; fsave[10] = fmy; fsave[11] = fmz;
                        fix = fiy = fiz = 0.0;
                        fjx = fjy = fjz = 0.0;
                        fkx = fky = fkz = 0.0;
                        fmx = fmy = fmz = 0.0;

                        let tr1 = 0.0, tr2 = 0.0, tr3 = 0.0;

                        dv[DIM] = dv[DJM] = dv[DKM] = 0.0;
                        let t45 = -1, t56 = -1, t64 = -1;
                        if (t12 >= 0 && ckm && cim) t45 = this.getTriplet(xx, m, i, k, T45, dv, DIM, DKM);
                        if (t23 >= 0 && cim && cjm) t56 = this.getTriplet(xx, m, i, j, T56, dv, DIM, DJM);
                        if (t31 >= 0 && cjm && ckm) t64 = this.getTriplet(xx, m, j, k, T64, dv, DJM, DKM);

                        const quad = (ta: number, tb: number): void => {
                          // trd_update_4(ta, tb)
                          this.trTrace3(T, tb + H1H2, T, ta + H1XH2, T, ta + H1YH2, T, ta + H1ZH2);
                          q1x = TR[0]; q1y = TR[1]; q1z = TR[2];
                          this.trTrace3(T, tb + H1H2, T, ta + H1H2X, T, ta + H1H2Y, T, ta + H1H2Z);
                          q2x = TR[0]; q2y = TR[1]; q2z = TR[2];
                          this.trTrace3(T, ta + H1H2, T, tb + H1XH2, T, tb + H1YH2, T, tb + H1ZH2);
                          q3x = TR[0]; q3y = TR[1]; q3z = TR[2];
                          this.trTrace3(T, ta + H1H2, T, tb + H1H2X, T, tb + H1H2Y, T, tb + H1H2Z);
                          q4x = TR[0]; q4y = TR[1]; q4z = TR[2];
                        };
                        let q1x = 0.0, q2x = 0.0, q3x = 0.0, q4x = 0.0;
                        let q1y = 0.0, q2y = 0.0, q3y = 0.0, q4y = 0.0;
                        let q1z = 0.0, q2z = 0.0, q3z = 0.0, q4z = 0.0;
                        const acc4 = (w: number): void => {
                          fix = fix + dfix * w; fiy = fiy + dfiy * w; fiz = fiz + dfiz * w;
                          fjx = fjx + dfjx * w; fjy = fjy + dfjy * w; fjz = fjz + dfjz * w;
                          fkx = fkx + dfkx * w; fky = fky + dfky * w; fkz = fkz + dfkz * w;
                          fmx = fmx + dfmx * w; fmy = fmy + dfmy * w; fmz = fmz + dfmz * w;
                        };

                        if (t12 >= 0 && t45 >= 0) {
                          if (fourBodyEnergies) {
                            tr1 = this.transtrace(T, t12 + H1H2, T, t45 + H1H2);
                            const dvir = ((dv[DIJ] + dv[DJK] + dv[DIM] + dv[DKM]) * ve + dve) * tr1 * w4 / anorm4;
                            vir4 = vir4 + dvir;
                            xvir4 = xvir4 + dvir;
                          }
                          quad(t12, t45);
                          // dfix_update_4a
                          dfix = ((-sij) * q1x + (-sim) * q3x) * (ve / anorm4);
                          dfjx = ((sij) * q1x + (-sjk) * q2x) * (ve / anorm4);
                          dfkx = ((sjk) * q2x + (-skm) * q4x) * (ve / anorm4);
                          dfmx = ((sim) * q3x + (skm) * q4x) * (ve / anorm4);
                          dfiy = ((-sij) * q1y + (-sim) * q3y) * (ve / anorm4);
                          dfjy = ((sij) * q1y + (-sjk) * q2y) * (ve / anorm4);
                          dfky = ((sjk) * q2y + (-skm) * q4y) * (ve / anorm4);
                          dfmy = ((sim) * q3y + (skm) * q4y) * (ve / anorm4);
                          dfiz = ((-sij) * q1z + (-sim) * q3z) * (ve / anorm4);
                          dfjz = ((sij) * q1z + (-sjk) * q2z) * (ve / anorm4);
                          dfkz = ((sjk) * q2z + (-skm) * q4z) * (ve / anorm4);
                          dfmz = ((sim) * q3z + (skm) * q4z) * (ve / anorm4);
                          if (fourBodyForces) acc4(w4);
                        }

                        if (t23 >= 0 && t56 >= 0) {
                          if (fourBodyEnergies) {
                            tr2 = this.transtrace(T, t23 + H1H2, T, t56 + H1H2);
                            const dvir = ((dv[DKI] + dv[DJK] + dv[DIM] + dv[DJM]) * ve + dve) * tr2 * w4 / anorm4;
                            vir4 = vir4 + dvir;
                            xvir4 = xvir4 + dvir;
                          }
                          quad(t23, t56);
                          // dfix_update_4b
                          dfix = ((ski) * q1x + (-sim) * q3x) * (ve / anorm4);
                          dfjx = ((-sjk) * q2x + (-sjm) * q4x) * (ve / anorm4);
                          dfkx = ((-ski) * q1x + (sjk) * q2x) * (ve / anorm4);
                          dfmx = ((sim) * q3x + (sjm) * q4x) * (ve / anorm4);
                          dfiy = ((ski) * q1y + (-sim) * q3y) * (ve / anorm4);
                          dfjy = ((-sjk) * q2y + (-sjm) * q4y) * (ve / anorm4);
                          dfky = ((-ski) * q1y + (sjk) * q2y) * (ve / anorm4);
                          dfmy = ((sim) * q3y + (sjm) * q4y) * (ve / anorm4);
                          dfiz = ((ski) * q1z + (-sim) * q3z) * (ve / anorm4);
                          dfjz = ((-sjk) * q2z + (-sjm) * q4z) * (ve / anorm4);
                          dfkz = ((-ski) * q1z + (sjk) * q2z) * (ve / anorm4);
                          dfmz = ((sim) * q3z + (sjm) * q4z) * (ve / anorm4);
                          if (fourBodyForces) acc4(w4);
                        }

                        if (t31 >= 0 && t64 >= 0) {
                          if (fourBodyEnergies) {
                            tr3 = this.transtrace(T, t31 + H1H2, T, t64 + H1H2);
                            const dvir = ((dv[DKI] + dv[DIJ] + dv[DJM] + dv[DKM]) * ve + dve) * tr3 * w4 / anorm4;
                            vir4 = vir4 + dvir;
                            xvir4 = xvir4 + dvir;
                          }
                          quad(t31, t64);
                          // dfix_update_4c
                          dfix = ((-sij) * q1x + (ski) * q2x) * (ve / anorm4);
                          dfjx = ((sij) * q1x + (-sjm) * q3x) * (ve / anorm4);
                          dfkx = ((-ski) * q2x + (-skm) * q4x) * (ve / anorm4);
                          dfmx = ((sjm) * q3x + (skm) * q4x) * (ve / anorm4);
                          dfiy = ((-sij) * q1y + (ski) * q2y) * (ve / anorm4);
                          dfjy = ((sij) * q1y + (-sjm) * q3y) * (ve / anorm4);
                          dfky = ((-ski) * q2y + (-skm) * q4y) * (ve / anorm4);
                          dfmy = ((sjm) * q3y + (skm) * q4y) * (ve / anorm4);
                          dfiz = ((-sij) * q1z + (ski) * q2z) * (ve / anorm4);
                          dfjz = ((sij) * q1z + (-sjm) * q3z) * (ve / anorm4);
                          dfkz = ((-ski) * q2z + (-skm) * q4z) * (ve / anorm4);
                          dfmz = ((sjm) * q3z + (skm) * q4z) * (ve / anorm4);
                          if (fourBodyForces) acc4(w4);
                        }

                        const deQuad = ve * (tr1 + tr2 + tr3) / anorm4 * eScale * w4;

                        // ev_tally4(i,j,k,m,de_quad,fi,fj,fk,drim,drjm,drkm) with fi, fj, fk scaled by e_scale
                        engVdwl += deQuad;
                        if (eatom) {
                          const fourth = 0.25 * deQuad;
                          eatom[i] += fourth;
                          eatom[j] += fourth;
                          eatom[k] += fourth;
                          eatom[m] += fourth;
                        }
                        if (vatom) {
                          const fi0 = fix * eScale, fi1 = fiy * eScale, fi2 = fiz * eScale;
                          const fj0 = fjx * eScale, fj1 = fjy * eScale, fj2 = fjz * eScale;
                          const fk0 = fkx * eScale, fk1 = fky * eScale, fk2 = fkz * eScale;
                          const im0 = xx[3 * i] - xx[3 * m], im1 = xx[3 * i + 1] - xx[3 * m + 1], im2 = xx[3 * i + 2] - xx[3 * m + 2];
                          const jm0 = xx[3 * j] - xx[3 * m], jm1 = xx[3 * j + 1] - xx[3 * m + 1], jm2 = xx[3 * j + 2] - xx[3 * m + 2];
                          const km0 = xx[3 * k] - xx[3 * m], km1 = xx[3 * k + 1] - xx[3 * m + 1], km2 = xx[3 * k + 2] - xx[3 * m + 2];
                          let v0 = im0 * fi0 + jm0 * fj0 + km0 * fk0;
                          let v1 = im1 * fi1 + jm1 * fj1 + km1 * fk1;
                          let v2 = im2 * fi2 + jm2 * fj2 + km2 * fk2;
                          let v3 = im0 * fi1 + jm0 * fj1 + km0 * fk1;
                          let v4 = im0 * fi2 + jm0 * fj2 + km0 * fk2;
                          let v5 = im1 * fi2 + jm1 * fj2 + km1 * fk2;
                          v0 *= 0.25; v1 *= 0.25; v2 *= 0.25; v3 *= 0.25; v4 *= 0.25; v5 *= 0.25;
                          for (const a of [i, j, k, m]) vadd(a, v0, v1, v2, v3, v4, v5);
                          if (volpres) {
                            const dvir = -xvir4 * rhoinv * eScale * (1.0 / 4.0);
                            for (let pp = 0; pp < 3; pp++) {
                              vatom[6 * i + pp] += dvir;
                              vatom[6 * j + pp] += dvir;
                              vatom[6 * k + pp] += dvir;
                              vatom[6 * m + pp] += dvir;
                            }
                          }
                        }

                        fix = fix + fsave[0]; fiy = fiy + fsave[1]; fiz = fiz + fsave[2];
                        fjx = fjx + fsave[3]; fjy = fjy + fsave[4]; fjz = fjz + fsave[5];
                        fkx = fkx + fsave[6]; fky = fky + fsave[7]; fkz = fkz + fsave[8];
                        fmx = fmx + fsave[9]; fmy = fmy + fsave[10]; fmz = fmz + fsave[11];

                        ff[3 * m] += fmx * eScale;
                        ff[3 * m + 1] += fmy * eScale;
                        ff[3 * m + 2] += fmz * eScale;
                      }
                    }
                    mj2 = mj2 + 1;
                    mk = mk + 1;
                  } else if (nlistShort[mj2] < nlistShort[mk]) {
                    mj2 = mj2 + 1;
                  } else {
                    mk = mk + 1;
                  }
                }
              }
            }

            ff[3 * k] += fkx * eScale;
            ff[3 * k + 1] += fky * eScale;
            ff[3 * k + 2] += fkz * eScale;
          }

          ff[3 * j] += fjx * eScale;
          ff[3 * j + 1] += fjy * eScale;
          ff[3 * j + 2] += fjz * eScale;
        }
      }

      ff[3 * i] += fix * eScale;
      ff[3 * i + 1] += fiy * eScale;
      ff[3 * i + 2] += fiz * eScale;

      if (singleEnergies === 1 && i < nloc) {
        const evol0 = P.evol0;
        engVdwl = engVdwl + evol0 * eScale;
        if (eatom) eatom[i] = eatom[i] + evol0 * eScale;
        if (volpres && vatom) {
          for (let pp = 0; pp < 3; pp++) vatom[6 * i + pp] = vatom[6 * i + pp] - rhoinv * P.devol0 * eScale;
        }
      }
    }

    if (volpres) {
      // Include contributions to the pressure due to derivatives of the energy with respect to the potential input volume.
      if (singleEnergies) for (let i = 0; i < 3; i++) virial[i] = virial[i] - nloc * rhoinv * P.devol0 * eScale;
      if (pairEnergies) for (let i = 0; i < 3; i++) virial[i] = virial[i] + rhoinv * eScale * volvir2;
      if (threeBodyEnergies) for (let i = 0; i < 3; i++) virial[i] = virial[i] - (vir3v + vir3t) * rhoinv * eScale;
      if (fourBodyEnergies) for (let i = 0; i < 3; i++) virial[i] = virial[i] - vir4 * rhoinv * eScale;
    }
    return engVdwl;
  }

  /* ----------------------------------------------------------------------
     Reimplementation of bond length potential, including derivatives with respect to x, y, and z.
  ---------------------------------------------------------------------- */
  private flDerivNew(r: number, ri: number, xhat: number, yhat: number, zhat: number): void {
    const P = this.P!;
    const rp = P.rp, p1 = P.p1, r0 = P.r00, al = P.al;
    const mode = P.mode;
    const pn = P.pn;
    const FL = this.FL;

    let t: number, s: number, tRpTi: number, tP1Ti: number;
    const rpi = 1.0 / rp;
    if (mode <= 4) {
      t = pow(rp * ri, p1);
      s = -p1 * t * ri;
      tRpTi = p1 * rpi;
      tP1Ti = log(rp * ri);
    } else if (pn === 1.0) {
      const p1Rpi = -p1 * rpi;
      t = exp(p1 + r * p1Rpi);
      s = p1Rpi * t;
      tRpTi = -r * p1Rpi * rpi;
      tP1Ti = 1.0 - r * rpi;
    } else {
      const pni = 1.0 / pn;
      const rprpn = pow(r * rpi, pn);
      t = exp(-p1 * pni * (rprpn - 1.0));
      s = -p1 * rprpn * ri * t;
      tRpTi = p1 * rprpn * rpi;
      tP1Ti = pni - pni * rprpn;
    }
    const tx = s * xhat;
    const ty = s * yhat;
    const tz = s * zhat;

    FL[4] = tRpTi;
    FL[5] = tP1Ti;

    if (r <= r0) {
      FL[0] = t;
      FL[1] = tx;
      FL[2] = ty;
      FL[3] = tz;
      FL[6] = 0.0;
      FL[7] = 0.0;
    } else {
      const r0i = 1.0 / r0;
      const u = r * r0i - 1.0;
      const q = al * u * u;
      const sq = 2 * al * u * r0i;
      const qx = sq * xhat;
      const qy = sq * yhat;
      const qz = sq * zhat;
      const qR0 = -2.0 * al * u * r * r0i * r0i;
      const qAl = u * u;

      const expQ = exp(-q);
      if (mode <= 2) {
        FL[0] = expQ * t;
        FL[1] = expQ * (tx - t * qx);
        FL[2] = expQ * (ty - t * qy);
        FL[3] = expQ * (tz - t * qz);
        FL[6] = -qR0;
        FL[7] = -qAl;
      } else {
        FL[0] = expQ * (1.0 + q) * t;
        FL[1] = expQ * (tx + q * (tx - t * qx));
        FL[2] = expQ * (ty + q * (ty - t * qy));
        FL[3] = expQ * (tz + q * (tz - t * qz));
        FL[6] = -qR0 * q / (1.0 + q);
        FL[7] = -qAl * q / (1.0 + q);
      }
    }
  }

  /**
   * Bond matrix element (i, j) and its x, y, z derivatives (MAKE_ELEMENT_5 / MAKE_ELEMENT_7): tmp is the angular part,
   * tx, ty, tz its derivatives.
   */
  private static element(M: number[] | Float64Array, base: number, i: number, j: number, fl: number, flx: number,
    fly: number, flz: number, flRi: number, x: number, y: number, z: number, tmp: number, tmpx: number, tmpy: number, tmpz: number): void {
    const A = M as Float64Array;
    const tmpsum = tmpx * x + tmpy * y + tmpz * z;
    A[base + 8 * j + i] = A[base + 8 * i + j] = fl * tmp;
    A[base + MS + 8 * j + i] = A[base + MS + 8 * i + j] = flx * tmp + flRi * (tmpx - x * tmpsum);
    A[base + 2 * MS + 8 * j + i] = A[base + 2 * MS + 8 * i + j] = fly * tmp + flRi * (tmpy - y * tmpsum);
    A[base + 3 * MS + 8 * j + i] = A[base + 3 * MS + 8 * i + j] = flz * tmp + flRi * (tmpz - z * tmpsum);
  }

  /** PairMGPT::hamltn_5_raw: the d-state bond matrix of bond vector (xin, yin, zin) and its derivatives. */
  private hamltn5(xin: number, yin: number, zin: number, M: Float64Array, base: number, b: number): void {
    const P = this.P!;
    const r = Math.sqrt(xin * xin + yin * yin + zin * zin), ri = 1.0 / r;
    const x = xin * ri, y = yin * ri, z = zin * ri;

    const x2 = x * x, y2 = y * y, z2 = z * z;
    const xy = x * y, xz = x * z, yz = y * z;

    const sr3 = Math.sqrt(3.0), sr3i = 1.0 / sr3;
    const frac13 = 1.0 / 3.0, frac23 = 2.0 / 3.0, frac43 = 4.0 / 3.0;

    const ddl1 = this.ddl[1], ddl2 = this.ddl[2], ddl3 = this.ddl[3];
    const w4 = ddl1 - frac43 * ddl2 + frac13 * ddl3;
    const w2 = ddl2 - ddl3;
    const w0 = ddl2;

    // del4: gsl[n] and its x, y, z derivatives
    const g = [0, 0.5 * (3.0 * z2 - 1.0), sr3 * xz, sr3 * yz, sr3 * (x2 - y2) * 0.5, sr3 * xy];
    const gx = [0, 0.0, sr3 * z, 0.0, sr3 * x, sr3 * y];
    const gy = [0, 0.0, 0.0, sr3 * z, -sr3 * y, sr3 * x];
    const gz = [0, 3.0 * z, sr3 * x, sr3 * y, 0.0, 0.0];

    // Compute bond length potential
    this.flDerivNew(r, ri, x, y, z);
    const FL = this.FL;
    const fl = FL[0], flx = FL[1], fly = FL[2], flz = FL[3];
    const flRi = fl * ri;
    this.bonds.fl[b] = FL[4] * P.drp + FL[5] * P.dp1 + FL[6] * P.dr00 + FL[7] * P.dal;

    const el = (i: number, j: number, dl2: number, dl2x: number, dl2y: number, dl2z: number): void => {
      const dl0 = i === j ? 1.0 : 0.0;
      const dl4 = g[i] * g[j];
      const dl4x = gx[i] * g[j] + g[i] * gx[j];
      const dl4y = gy[i] * g[j] + g[i] * gy[j];
      const dl4z = gz[i] * g[j] + g[i] * gz[j];
      const tmp = w4 * dl4 + w2 * dl2 + w0 * dl0;
      const tmpx = w4 * dl4x + w2 * dl2x;
      const tmpy = w4 * dl4y + w2 * dl2y;
      const tmpz = w4 * dl4z + w2 * dl2z;
      PairMGPT.element(M, base, i, j, fl, flx, fly, flz, flRi, x, y, z, tmp, tmpx, tmpy, tmpz);
    };

    // del2
    el(1, 1, z2 - frac23, 0.0, 0.0, 2 * z);
    el(1, 2, xz * sr3i, z * sr3i, 0.0, x * sr3i);
    el(1, 3, yz * sr3i, 0.0, z * sr3i, y * sr3i);
    el(1, 4, -(x2 - y2) * sr3i, -2.0 * sr3i * x, 2.0 * sr3i * y, 0.0);
    el(1, 5, -2.0 * xy * sr3i, -2.0 * y * sr3i, -2.0 * x * sr3i, 0.0);
    el(2, 2, -y2, 0.0, -2.0 * y, 0.0);
    el(2, 3, xy, y, x, 0.0);
    el(2, 4, xz, z, 0.0, x);
    el(2, 5, yz, 0.0, z, y);
    el(3, 3, -x2, -2.0 * x, 0.0, 0.0);
    el(3, 4, -yz, 0.0, -z, -y);
    el(3, 5, xz, z, 0.0, x);
    el(4, 4, -z2, 0.0, 0.0, -2.0 * z);
    el(4, 5, 0.0, 0.0, 0.0, 0.0);
    el(5, 5, -z2, 0.0, 0.0, -2.0 * z);
  }

  /** PairMGPT::hamltn_7_raw: the f-state bond matrix of bond vector (xin, yin, zin) and its derivatives. */
  private hamltn7(xin: number, yin: number, zin: number, M: Float64Array, base: number, b: number): void {
    const P = this.P!;
    const r = Math.sqrt(xin * xin + yin * yin + zin * zin), ri = 1.0 / r;
    const x = xin * ri, y = yin * ri, z = zin * ri;

    const x2 = x * x, y2 = y * y, z2 = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const x4 = x2 * x2, y4 = y2 * y2;

    const sr01 = Math.sqrt(0.1);
    const sr015 = Math.sqrt(0.15);
    const sr024 = Math.sqrt(0.24);
    const sr0375 = Math.sqrt(0.375);
    const sr06 = Math.sqrt(0.6);
    const sr0625 = Math.sqrt(0.625);
    const sr09 = Math.sqrt(0.9);
    const sr15 = Math.sqrt(1.5);
    const sr24 = Math.sqrt(2.4);
    const sr36 = Math.sqrt(3.6);
    const sr375 = Math.sqrt(3.75);
    const sr96 = Math.sqrt(9.6);
    const sr150 = Math.sqrt(15.0);

    const ddl1 = this.ddl[1], ddl2 = this.ddl[2], ddl3 = this.ddl[3], ddl4 = this.ddl[4];
    const w6 = ddl1 - 1.5 * ddl2 + 0.6 * ddl3 - 0.1 * ddl4;
    const w4 = 0.625 * ddl2 - ddl3 + 0.375 * ddl4;
    const w2 = 0.625 * (ddl2 - ddl4);
    const w0 = 0.625 * ddl2 + 0.375 * ddl4;

    // del6: gslf[n] and its x, y, z derivatives
    const g = [0,
      0.5 * (5.0 * z2 - 3.0) * z,
      sr0375 * (5.0 * z2 - 1.0) * x,
      sr0375 * (5.0 * z2 - 1.0) * y,
      sr375 * (x2 - y2) * z,
      sr150 * xy * z,
      sr0625 * (x2 - 3.0 * y2) * x,
      sr0625 * (3.0 * x2 - y2) * y];
    const gx = [0, 0.0, sr0375 * (5.0 * z2 - 1.0), 0.0, 2.0 * sr375 * xz, sr150 * yz, 3.0 * sr0625 * (x2 - y2), 6.0 * sr0625 * xy];
    const gy = [0, 0.0, 0.0, sr0375 * (5.0 * z2 - 1.0), -2.0 * sr375 * yz, sr150 * xz, -6.0 * sr0625 * xy, 3.0 * sr0625 * (x2 - y2)];
    const gz = [0, 7.5 * z2 - 1.5, sr0375 * 10.0 * xz, sr0375 * 10.0 * yz, sr375 * (x2 - y2), sr150 * xy, 0.0, 0.0];

    // Compute bond length potential
    this.flDerivNew(r, ri, x, y, z);
    const FL = this.FL;
    const fl = FL[0], flx = FL[1], fly = FL[2], flz = FL[3];
    const flRi = fl * ri;
    this.bonds.fl[b] = FL[4] * P.drp + FL[5] * P.dp1 + FL[6] * P.dr00 + FL[7] * P.dal;

    const el = (i: number, j: number, dl2: number, dl2x: number, dl2y: number, dl2z: number,
      dl4: number, dl4x: number, dl4y: number, dl4z: number): void => {
      const dl0 = i === j ? 1.0 : 0.0;
      const dl6 = g[i] * g[j];
      const dl6x = gx[i] * g[j] + g[i] * gx[j];
      const dl6y = gy[i] * g[j] + g[i] * gy[j];
      const dl6z = gz[i] * g[j] + g[i] * gz[j];
      const tmp = w6 * dl6 + w4 * dl4 + w2 * dl2 + w0 * dl0;
      const tmpx = w6 * dl6x + w4 * dl4x + w2 * dl2x;
      const tmpy = w6 * dl6y + w4 * dl4y + w2 * dl2y;
      const tmpz = w6 * dl6z + w4 * dl4z + w2 * dl2z;
      PairMGPT.element(M, base, i, j, fl, flx, fly, flz, flRi, x, y, z, tmp, tmpx, tmpy, tmpz);
    };

    el(1, 1,
      0.4 * (3.0 * z2 - 1.0), 0.0, 0.0, 2.4 * z,
      0.60 * (5.0 * z2 - 4.0) * z2, 0.0, 0.0, 0.60 * (20.0 * z2 - 8.0) * z);
    el(1, 2,
      sr024 * xz, sr024 * z, 0.0, sr024 * x,
      sr024 * (5.0 * z2 - 2.0) * xz, sr024 * (5.0 * z2 - 2.0) * z, 0.0, sr024 * (15.0 * z2 - 2.0) * x);
    el(1, 3,
      sr024 * yz, 0.0, sr024 * z, sr024 * y,
      sr024 * (5.0 * z2 - 2.0) * yz, 0.0, sr024 * (5.0 * z2 - 2.0) * z, sr024 * (15.0 * z2 - 2.0) * y);
    el(1, 4,
      -sr06 * (x2 - y2), -2.0 * sr06 * x, 2.0 * sr06 * y, 0.0,
      -sr06 * (x2 - y2) * z2, -2.0 * sr06 * x * z2, 2.0 * sr06 * y * z2, -2.0 * sr06 * (x2 - y2) * z);
    el(1, 5,
      -sr24 * xy, -sr24 * y, -sr24 * x, 0.0,
      -sr24 * xy * z2, -sr24 * y * z2, -sr24 * x * z2, -2.0 * sr24 * xy * z);
    el(1, 6,
      0.0, 0.0, 0.0, 0.0,
      sr36 * (3.0 * y2 - x2) * xz, 3.0 * sr36 * (y2 - x2) * z, 6.0 * sr36 * y * xz, sr36 * (3.0 * y2 - x2) * x);
    el(1, 7,
      0.0, 0.0, 0.0, 0.0,
      -sr36 * (3.0 * x2 - y2) * yz, -6.0 * sr36 * x * yz, -3.0 * sr36 * (x2 - y2) * z, -sr36 * (3.0 * x2 - y2) * y);
    el(2, 2,
      0.3 - 1.2 * y2 + 0.3 * z2, 0.0, -2.4 * y, 0.6 * z,
      -0.4 * x2 - 2.5 * (y2 - 0.6 * x2) * z2, -0.8 * x + 3.0 * x * z2, -5.0 * y * z2, -5.0 * (y2 - 0.6 * x2) * z);
    el(2, 3,
      1.2 * xy, 1.2 * y, 1.2 * x, 0.0,
      (4.0 * z2 - 0.4) * xy, (4.0 * z2 - 0.4) * y, (4.0 * z2 - 0.4) * x, 8.0 * z * xy);
    el(2, 4,
      sr09 * xz, sr09 * z, 0.0, sr09 * x,
      sr01 * (6.0 * z2 - 8.0 * y2 - 1.0) * xz, sr01 * (6.0 * z2 - 8.0 * y2 - 1.0) * z, -16.0 * sr01 * y * xz,
      sr01 * (18.0 * z2 - 8.0 * y2 - 1.0) * x);
    el(2, 5,
      sr09 * yz, 0.0, sr09 * z, sr09 * y,
      sr01 * (2.0 * z2 - 8.0 * y2 + 3.0) * yz, 0.0, sr01 * (2.0 * z2 - 24.0 * y2 + 3.0) * z, sr01 * (6.0 * z2 - 8.0 * y2 + 3.0) * y);
    el(2, 6,
      -sr015 * (x2 - y2), -2.0 * sr015 * x, 2.0 * sr015 * y, 0.0,
      sr375 * (x2 - y2 - 1.4 * x4 + 1.2 * x2 * y2 + y4), sr375 * (2.0 - 5.6 * x2 + 2.4 * y2) * x,
      sr375 * (-2.0 + 2.4 * x2 + 4.0 * y2) * y, 0.0);
    el(2, 7,
      -sr06 * xy, -sr06 * y, -sr06 * x, 0.0,
      sr96 * (z2 - x2 + 0.25) * xy, sr96 * (z2 - 3.0 * x2 + 0.25) * y, sr96 * (z2 - x2 + 0.25) * x, 2.0 * sr96 * z * xy);
    el(3, 3,
      0.3 - 1.2 * x2 + 0.3 * z2, -2.4 * x, 0.0, 0.6 * z,
      -0.4 * y2 - 2.5 * (x2 - 0.6 * y2) * z2, -5.0 * x * z2, y * (3.0 * z2 - 0.8), -5.0 * (x2 - 0.6 * y2) * z);
    el(3, 4,
      -sr09 * yz, 0.0, -sr09 * z, -sr09 * y,
      -sr01 * (6.0 * z2 - 8.0 * x2 - 1.0) * yz, 16.0 * sr01 * x * yz, -sr01 * (6.0 * z2 - 8.0 * x2 - 1.0) * z,
      -sr01 * (18.0 * z2 - 8.0 * x2 - 1.0) * y);
    el(3, 5,
      sr09 * xz, sr09 * z, 0.0, sr09 * x,
      sr01 * (2.0 * z2 - 8.0 * x2 + 3.0) * xz, sr01 * (2.0 * z2 - 24.0 * x2 + 3.0) * z, 0.0, sr01 * (6.0 * z2 - 8.0 * x2 + 3.0) * x);
    el(3, 6,
      sr06 * xy, sr06 * y, sr06 * x, 0.0,
      sr96 * (y2 - z2 - 0.25) * xy, sr96 * (y2 - z2 - 0.25) * y, sr96 * (3.0 * y2 - z2 - 0.25) * x, -2.0 * sr96 * z * xy);
    el(3, 7,
      -sr015 * (x2 - y2), -2.0 * sr015 * x, 2.0 * sr015 * y, 0.0,
      sr375 * (x2 - y2 + 1.4 * y4 - 1.2 * x2 * y2 - x4), sr375 * (2.0 - 2.4 * y2 - 4.0 * x2) * x,
      sr375 * (-2.0 + 5.6 * y2 - 2.4 * x2) * y, 0.0);
    el(4, 4,
      0.0, 0.0, 0.0, 0.0,
      (2.0 - 3.0 * z2) * z2 - 4.0 * x2 * y2, -8.0 * x * y2, -8.0 * x2 * y, (4.0 - 12.0 * z2) * z);
    el(4, 5,
      0.0, 0.0, 0.0, 0.0,
      2.0 * (x2 - y2) * xy, 2.0 * (3.0 * x2 - y2) * y, 2.0 * (x2 - 3.0 * y2) * x, 0.0);
    el(4, 6,
      sr15 * xz, sr15 * z, 0.0, sr15 * x,
      -sr15 * (2.0 * z2 - 1.0) * xz, -sr15 * (2.0 * z2 - 1.0) * z, 0.0, -sr15 * (6.0 * z2 - 1.0) * x);
    el(4, 7,
      sr15 * yz, 0.0, sr15 * z, sr15 * y,
      -sr15 * (2.0 * z2 - 1.0) * yz, 0.0, -sr15 * (2.0 * z2 - 1.0) * z, -sr15 * (6.0 * z2 - 1.0) * y);
    const t1 = 2.0 * z2 - 1.0;
    el(5, 5,
      0.0, 0.0, 0.0, 0.0,
      -t1 * t1 + 4.0 * x2 * y2, 8.0 * x * y2, 8.0 * x2 * y, -8.0 * t1 * z);
    el(5, 6,
      -sr15 * yz, 0.0, -sr15 * z, -sr15 * y,
      sr15 * (2.0 * z2 - 1.0) * yz, 0.0, sr15 * (2.0 * z2 - 1.0) * z, sr15 * (6.0 * z2 - 1.0) * y);
    el(5, 7,
      sr15 * xz, sr15 * z, 0.0, sr15 * x,
      -sr15 * (2.0 * z2 - 1.0) * xz, -sr15 * (2.0 * z2 - 1.0) * z, 0.0, -sr15 * (6.0 * z2 - 1.0) * x);
    el(6, 6,
      0.5 - 1.5 * z2, 0.0, 0.0, -3.0 * z,
      (1.5 * z2 - 1.5) * z2, 0.0, 0.0, (6.0 * z2 - 3.0) * z);
    el(6, 7,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0);
    el(7, 7,
      0.5 - 1.5 * z2, 0.0, 0.0, -3.0 * z,
      (1.5 * z2 - 1.5) * z2, 0.0, 0.0, (6.0 * z2 - 3.0) * z);
  }
}

/**
 * CommBrick::reverse_comm on one processor: the values of each ghost (w per atom) are added to the atom its
 * coordinates were sent from, from the last ghost to the first (inside one swap every target receives one value).
 */
const fold = (L: MgptLayout, a: Float64Array, w: number): void => {
  for (let k = L.nall - 1; k >= L.n; k--) {
    const t = L.from[k];
    for (let c = 0; c < w; c++) a[w * t + c] += a[w * k + c];
  }
};

/** The engine atom (owned or ghost) of each native atom with the same owner and periodic image, -1 when none. */
const engineIndex = (L: MgptLayout, pc: PairCompute): Int32Array => {
  const nb = pc.nb;
  const gi = nb.gimage;
  const key = (owner: number, ix: number, iy: number, iz: number): number =>
    ((owner * 64 + (ix + 32)) * 64 + (iy + 32)) * 64 + (iz + 32);
  const byKey = new Map<number, number>();
  for (let k = 0; k < nb.nall; k++) byKey.set(key(nb.owner[k], gi[3 * k], gi[3 * k + 1], gi[3 * k + 2]), k);
  const out = new Int32Array(L.nall);
  for (let k = 0; k < L.nall; k++) {
    const e = L.order[L.root[k]];
    const hit = k < L.n ? e : byKey.get(key(e, L.img[3 * k], L.img[3 * k + 1], L.img[3 * k + 2]));
    out[k] = hit === undefined ? -1 : hit;
  }
  return out;
};
