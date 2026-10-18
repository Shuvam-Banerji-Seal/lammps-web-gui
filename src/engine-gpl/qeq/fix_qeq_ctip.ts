/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QEQ/fix_qeq_ctip.cpp and src/QEQ/fix_qeq_ctip.h (FixQEqCTIP: constructor, init, pre_force, init_matvec,
 * compute_H, sparse_matvec, calculate_check_Q, extract_ctip), at commit b332f28. Author credited in the C++ file
 * headers: Gabriel Plummer (NASA). The CG and the charge sums it shares with the other qeq fixes come from
 * src/QEQ/fix_qeq.cpp (CG, calculate_Q), reached through ./native_solver.ts. Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix qeq/ctip Nevery cutoff tolerance maxiter coul/ctip [cdamp value] [maxrepeat value] [warn yes|no]
 * (docs.lammps.org/fix_qeq.html, section qeq/ctip). "Keyword *cdamp* can be used to change the damping parameter used to
 * calculate Coulomb interactions. Keyword *maxrepeat* can be used to adjust the number of equilibration cycles allowed to
 * ensure no atoms have crossed the charge bounds." The per-type parameters come from the pair style coul/ctip
 * (src/engine-gpl/ctip/pair_coul_ctip.ts), as native extract_ctip does; qfile words other than coul/ctip are refused.
 *
 * Per solve (pre_force): the matrix elements of compute_H (the CTIP shielded interaction, its shift constants and the
 * self-image terms) are built once; then up to maxrepeat passes of { init_matvec (the diagonal and right-hand side of
 * each atom's charge regime: below qmin, between, above qmax), CG for s and t, calculate_check_Q }. A pass repeats while
 * any group atom whose charge crosses qmin or qmax moved by more than the tolerance. Passes that all repeat are an error.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NativeQeqSolver, type QeqLinearSystem, type QeqSolveResult } from './native_solver';
import { A1, A2, A3, A4, A5, EWALD_P, MY_PIS, PairCoulCtip, type CtipParams } from '../ctip/pair_coul_ctip';
import { exp as libmExp } from '../libm/exp';

/** Per type: s2d_self (the self-energy term of the diagonal) and the shielded shift constants per type pair (nt * nt). */
interface CtipTables {
  nt: number;
  s2dSelf: Float64Array;
  shieldCu: Float64Array;
  reffc: Float64Array;
  reffc4: Float64Array;
  reffc7: Float64Array;
  s2dShift: Float64Array;
  fShift: Float64Array;
  eShift: Float64Array;
}

/** The q-independent matrix of one solve: off-diagonal pairs, the self-image diagonal, the fixed (non-group) term. */
interface CtipMatrix {
  pairA: number[];
  pairB: number[];
  pairH: number[];
  selfDiag: Float64Array;
  fixedTerm: Float64Array;
}

const numArg = (w: string, what: string): number => {
  const v = Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`fix qeq/ctip: expected a number for ${what}, got '${w}'`);
  return v;
};

const intArg = (w: string, what: string): number => {
  if (!/^\d+$/.test(w)) throw new StyleError(`fix qeq/ctip: expected an integer for ${what}, got '${w}'`);
  return Number(w);
};

/** utils::logical: yes/no, on/off, true/false, 1/0. */
const logicalArg = (w: string, what: string): boolean => {
  if (['yes', 'on', 'true', '1'].includes(w)) return true;
  if (['no', 'off', 'false', '0'].includes(w)) return false;
  throw new StyleError(`fix qeq/ctip: expected yes or no for ${what}, got '${w}'`);
};

/** Receives the distance vector of one periodic image. */
type VisitFn = (dx: number, dy: number, dz: number) => void;

export class FixQeqCtip extends Fix {
  readonly style = 'qeq/ctip';
  scalarFlag = true;
  extscalar = 0;

  private readonly nEvery: number;
  private readonly cutoff: number;
  private readonly tol: number;
  private readonly maxiter: number;
  private cdamp = 0.3;
  private maxrepeat = 10;
  private warn = true;
  /** f_ID: the CG count of the last solve (native matvecs, an int: (CG(s) + CG(t)) / 2). */
  private matvecs = 0;
  private readonly solver: NativeQeqSolver;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 5) {
      throw new StyleError('Illegal fix qeq/ctip command: expected "Nevery cutoff tolerance maxiter coul/ctip [keyword value ...]"');
    }
    this.nEvery = intArg(args[0], 'Nevery');
    this.cutoff = numArg(args[1], 'cutoff');
    this.tol = numArg(args[2], 'tolerance');
    this.maxiter = intArg(args[3], 'maxiter');
    if (!(this.nEvery > 0) || !(this.cutoff > 0) || !(this.tol > 0) || !(this.maxiter > 0)) {
      throw new StyleError('Illegal fix qeq command: Nevery, cutoff, tolerance and maxiter must be positive');
    }
    if (args[4] !== 'coul/ctip') {
      throw new StyleError(`fix qeq/ctip: qfile '${args[4]}' is not supported (the parameters come from pair_style coul/ctip: use coul/ctip)`);
    }
    for (let k = 5; k < args.length; k += 2) {
      const key = args[k];
      if (k + 1 >= args.length) throw new StyleError(`fix qeq/ctip: keyword '${key}' needs a value`);
      const val = args[k + 1];
      if (key === 'cdamp') this.cdamp = numArg(val, 'cdamp');
      else if (key === 'maxrepeat') this.maxrepeat = intArg(val, 'maxrepeat');
      else if (key === 'warn') this.warn = logicalArg(val, 'warn');
      else throw new StyleError(`Unknown fix qeq/ctip keyword: ${key}`);
    }
    if (!(this.maxrepeat > 0)) throw new StyleError('fix qeq/ctip: maxrepeat must be > 0');
    // extract_ctip: the pair style must exist when the fix is created
    this.pair();
    this.solver = new NativeQeqSolver({
      kind: 'point', maxiter: this.maxiter, tol: this.tol, warn: this.warn,
      warnCg: (ratio, loop) => this.sys.warn(`Fix qeq CG convergence failed (${ratio}) after ${loop} iterations at step ${this.sys.state.step}`),
    });
    // FixQEq constructor: every atom's history starts at its charge
    this.solver.seed(sys.state.id, sys.state.q, sys.state.n);
  }

  /** The pair style coul/ctip (pair_match "^coul/ctip": directly or among the sub-styles of pair hybrid). */
  private pair(): PairCoulCtip {
    const pair = this.sys.ff.pair;
    if (pair instanceof PairCoulCtip) return pair;
    const subs = (pair as unknown as { subs?: { style: unknown }[] } | null)?.subs ?? [];
    for (const sub of subs) if (sub.style instanceof PairCoulCtip) return sub.style;
    throw new StyleError('No pair style coul/ctip for fix qeq/ctip');
  }

  /** Charges are equilibrated before the first force computation of a run (native setup_pre_force -> pre_force). */
  setupPreForce(): void {
    if (this.sys.state.step % this.nEvery === 0) this.solve();
  }

  /** fix_qeq.html: Nevery = "perform charge equilibration every this many steps". */
  preForce(): void {
    if (this.sys.state.step % this.nEvery === 0) this.solve();
  }

  computeScalar(): number { return this.matvecs; }

  /** Native init(): s2d_self per type and the shielded shift constants (shieldcu, reffc, s2d/f/e_shift) per type pair. */
  private tablesFor(pair: PairCoulCtip): CtipTables {
    const nt = this.sys.state.ntypes + 1;
    const r = this.cutoff, rsq = r * r, cd = this.cdamp;
    const qqr2e = this.sys.units.qqr2e;
    const cutoffcu = rsq * r, cutoff4 = rsq * rsq, cdampcu = cd * cd * cd;
    const erfcdCut = libmExp(-cd * cd * rsq);
    const tCut = 1.0 / (1.0 + EWALD_P * cd * r);
    const erfccCut = (tCut * (A1 + tCut * (A2 + tCut * (A3 + tCut * (A4 + tCut * A5))))) * erfcdCut / r;
    const tb: CtipTables = {
      nt, s2dSelf: new Float64Array(nt), shieldCu: new Float64Array(nt * nt), reffc: new Float64Array(nt * nt),
      reffc4: new Float64Array(nt * nt), reffc7: new Float64Array(nt * nt), s2dShift: new Float64Array(nt * nt),
      fShift: new Float64Array(nt * nt), eShift: new Float64Array(nt * nt),
    };
    const gam = (t: number): number => pair.typeParams(t).gamma;
    for (let t = 1; t < nt; t++) {
      const reff = Math.cbrt(rsq * r + 1.0 / (gam(t) * gam(t) * gam(t)));
      const reffsq = reff * reff;
      const reff4 = reffsq * reffsq;
      const reff7 = reff4 * reffsq * reff;
      tb.s2dSelf[t] = 2.0 * qqr2e *
        (1.5 * erfccCut + 2.0 * cd / MY_PIS * erfcdCut + cd * cd * cd / MY_PIS * rsq * erfcdCut + 0.5 / reff - 1.5 / r +
          rsq * rsq * rsq / reff7 + cd / MY_PIS);
    }
    for (let t1 = 1; t1 < nt; t1++) {
      for (let t2 = 1; t2 < nt; t2++) {
        const k = t1 * nt + t2;
        const shield = Math.sqrt(gam(t1) * gam(t2));
        const shieldcu = shield * shield * shield;
        tb.shieldCu[k] = shieldcu;
        tb.reffc[k] = Math.cbrt(cutoffcu + 1.0 / shieldcu);
        const reffcsq = tb.reffc[k] * tb.reffc[k];
        tb.reffc4[k] = reffcsq * reffcsq;
        tb.reffc7[k] = tb.reffc4[k] * reffcsq * tb.reffc[k];
        tb.s2dShift[k] = 2.0 * erfccCut / cutoffcu + 4.0 * cd / MY_PIS * erfcdCut / rsq + 4.0 * cdampcu / MY_PIS * erfcdCut -
          2 / cutoffcu + 4 * cutoff4 / tb.reffc7[k] - 2 * r / tb.reffc4[k];
        tb.fShift[k] = erfccCut / rsq + 2.0 * cd / MY_PIS * erfcdCut / r - 1 / rsq + rsq / tb.reffc4[k];
        tb.eShift[k] = erfccCut / r + 1 / tb.reffc[k] - 1 / r;
      }
    }
    return tb;
  }

  /** The image visitor of one atom pair: the distance vectors dv - n L within the cutoff range (ortho or triclinic). */
  private imageVisitor(): (dv: number[], visit: VisitFn) => void {
    const geom = this.sys.geom as unknown as { lx: number; ly: number; lz: number; xy: number; xz: number; yz: number; periodic: boolean[] };
    const rc = this.cutoff;
    const Ld = [geom.lx, geom.ly, geom.lz];
    const ortho = geom.xy === 0 && geom.xz === 0 && geom.yz === 0;
    const nmax = [0, 1, 2].map((dd) => (geom.periodic[dd] ? Math.ceil(rc / Ld[dd]) + 1 : 0));
    const shifts: number[][] = [];
    if (!ortho) {
      const ax = [geom.lx, 0, 0], bx = [geom.xy, geom.ly, 0], cx = [geom.xz, geom.yz, geom.lz];
      for (let nx = -nmax[0]; nx <= nmax[0]; nx++)
        for (let ny = -nmax[1]; ny <= nmax[1]; ny++)
          for (let nz = -nmax[2]; nz <= nmax[2]; nz++)
            shifts.push([nx * ax[0] + ny * bx[0] + nz * cx[0], nx * ax[1] + ny * bx[1] + nz * cx[1], nx * ax[2] + ny * bx[2] + nz * cx[2]]);
    }
    return (dv, visit) => {
      if (ortho) {
        const lo = [0, 0, 0], hi = [0, 0, 0];
        for (let dd = 0; dd < 3; dd++) {
          if (!geom.periodic[dd]) continue;
          lo[dd] = Math.floor((dv[dd] - rc) / Ld[dd]);
          hi[dd] = Math.ceil((dv[dd] + rc) / Ld[dd]);
        }
        for (let nx = lo[0]; nx <= hi[0]; nx++)
          for (let ny = lo[1]; ny <= hi[1]; ny++)
            for (let nz = lo[2]; nz <= hi[2]; nz++) visit(dv[0] - nx * Ld[0], dv[1] - ny * Ld[1], dv[2] - nz * Ld[2]);
      } else {
        for (const sh of shifts) visit(dv[0] - sh[0], dv[1] - sh[1], dv[2] - sh[2]);
      }
    };
  }

  /** The CTIP pair value f(r) of one image: native compute_H's H.val (0.5 f per list entry; both entries of a pair sum to f). */
  private fValue(tb: CtipTables, ti: number, tj: number, rsq: number): number {
    const r = Math.sqrt(rsq);
    const k = ti * tb.nt + tj;
    const reff = Math.cbrt(rsq * r + 1 / tb.shieldCu[k]);
    const erfcd = libmExp(-this.cdamp * this.cdamp * rsq);
    const t = 1 / (1 + EWALD_P * this.cdamp * r);
    const erfcc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * erfcd;
    return this.sys.units.qqr2e * (erfcc / r + 1 / reff - 1 / r - tb.eShift[k] + tb.fShift[k] * (r - this.cutoff) -
      tb.s2dShift[k] * 0.5 * (r - this.cutoff) * (r - this.cutoff));
  }

  /** Native compute_H: H over every periodic image within the cutoff, as off-diagonal pairs, self-image diagonal and the fixed-atom term. */
  private matrix(tb: CtipTables, idx: number[], row: Int32Array): CtipMatrix {
    const s = this.sys.state;
    const ng = idx.length;
    const rc = this.cutoff, cut2 = rc * rc;
    const bit = this.groupBit;
    const pairA: number[] = [], pairB: number[] = [], pairH: number[] = [];
    const selfDiag = new Float64Array(ng);
    const fixedTerm = new Float64Array(ng);
    const forEachImage = this.imageVisitor();
    const dv = [0, 0, 0];
    for (let a = 0; a < ng; a++) {
      const i = idx[a];
      for (let v = 0; v < s.n; v++) {
        const vg = (s.mask[v] & bit) !== 0;
        if (vg && row[v] < a) continue; // a group pair is one term: counted from its lower row
        dv[0] = s.x[3 * i] - s.x[3 * v];
        dv[1] = s.x[3 * i + 1] - s.x[3 * v + 1];
        dv[2] = s.x[3 * i + 2] - s.x[3 * v + 2];
        forEachImage(dv, (dx, dy, dz) => {
          const rsq = dx * dx + dy * dy + dz * dz;
          if (!(rsq > 0) || rsq > cut2) return;
          const f = this.fValue(tb, s.type[i], s.type[v], rsq);
          if (vg) {
            if (v === i) selfDiag[a] += f;
            else { pairA.push(a); pairB.push(row[v]); pairH.push(f); }
          } else {
            // a neighbour outside the group: half of the entry (its own list is not visited), with its stored charge
            fixedTerm[a] += 0.5 * f * s.q[v];
          }
        });
      }
    }
    return { pairA, pairB, pairH, selfDiag, fixedTerm };
  }

  /** The solve: native pre_force (up to maxrepeat passes of init_matvec, CG on s and t, calculate_check_Q). */
  private solve(): void {
    const s = this.sys.state;
    const bit = this.groupBit;
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) idx.push(i);
    const ng = idx.length;
    if (ng === 0) throw new StyleError('Fix qeq/ctip group has no atoms');
    const pair = this.pair();
    const tb = this.tablesFor(pair);
    const row = new Int32Array(s.n).fill(-1);
    for (let a = 0; a < ng; a++) row[idx[a]] = a;
    const mt = this.matrix(tb, idx, row);
    const ids = idx.map((i) => s.id[i]);
    let pass = 1;
    let matvecs = 0;
    for (; pass <= this.maxrepeat; pass++) {
      const sys = this.linearSystem(pair, tb, idx, mt);
      const res: QeqSolveResult = this.solver.solve({ sys, idx, ids, n: s.n, q: s.q });
      matvecs = Math.trunc(res.iterations);
      // calculate_check_Q: a charge that crosses qmin or qmax and moved by more than the tolerance repeats the pass
      let nout = 0;
      for (let a = 0; a < ng; a++) {
        const i = idx[a];
        const p = pair.typeParams(s.type[i]);
        const qOld = s.q[i], qNew = res.q[a];
        const check1 = (qNew - p.qmin) * (qOld - p.qmin);
        const check2 = (qNew - p.qmax) * (qOld - p.qmax);
        if ((check1 < 0.0 || check2 < 0.0) && Math.abs(qNew - qOld) > this.tol) nout++;
      }
      for (let a = 0; a < ng; a++) s.q[idx[a]] = res.q[a];
      if (nout === 0) break;
    }
    if (pass > this.maxrepeat) throw new StyleError('Fix qeq some charges not bound within the domain');
    this.matvecs = matvecs;
    // kspace refresh (native qsum_qsq) or the charges of the neighbour data, as the other qeq fixes do
    if (this.sys.ff.kspace) this.sys.setupNeighbors();
    else this.sys.nb.refreshCharges(s);
    this.sys.bump();
  }

  /** init_matvec: the diagonal, the Jacobi preconditioner and the right-hand side of each atom's charge regime. */
  private linearSystem(pair: PairCoulCtip, tb: CtipTables, idx: number[], mt: CtipMatrix): QeqLinearSystem {
    const s = this.sys.state;
    const ng = idx.length;
    const diag = new Float64Array(ng), precond = new Float64Array(ng), bS = new Float64Array(ng);
    for (let a = 0; a < ng; a++) {
      const i = idx[a];
      const t = s.type[i];
      const p: CtipParams = pair.typeParams(t);
      const qi = s.q[i];
      const sd = tb.s2dSelf[t];
      let dia: number;
      if (qi < p.qmin) {
        dia = p.eta + 2 * p.omega - sd;
        bS[a] = -(p.chi - 2 * p.qmin * p.omega);
      } else if (qi < p.qmax) {
        dia = p.eta - sd;
        bS[a] = -p.chi;
      } else {
        dia = p.eta + 2 * p.omega - sd;
        bS[a] = -(p.chi - 2 * p.qmax * p.omega);
      }
      precond[a] = 1 / dia;
      diag[a] = dia + mt.selfDiag[a];
    }
    return {
      ng, diag, precond, bS, fixed: mt.fixedTerm,
      pairA: Int32Array.from(mt.pairA), pairB: Int32Array.from(mt.pairB), pairH: Float64Array.from(mt.pairH),
    };
  }
}
