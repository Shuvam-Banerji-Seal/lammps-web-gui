/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-PAIR/pair_bondval.cpp and pair_bondval.h at commit b332f28 (the bond-valence pair styles by the
 * group of Andrew M. Rappe and their LAMMPS authors). Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/pair_bondval.html (verbatim): "pair_style bondval power cutoff"; "power = exponent for the bond-valence
 * force (typically 2.0)"; "cutoff = global cutoff for bond-valence interactions (distance units)". The bond valence is
 * "V_{ij} = ( r0_{ij} / r_{ij} )^{\alpha_{ij}}" and the energy term "E_{BV} = \sum_i S_i ( V_i - V_{0,i} )^2", with the
 * bond-valence sum "V_i = \sum_{j \neq i} V_{ij}". The doc says these styles "do not support parameter mixing" and
 * "do not support the pair_modify shift, table, or tail options".
 *
 * Coefficients (docs.lammps.org/pair_bondval.html, the usage examples): "pair_coeff I J r0 alpha S V0 [cutoff]" with the
 * cutoff optional; the doc says "Thus only same-species values (I = J) are nonzero in input file" for S and V0.
 *
 * Shared base for pair_style bondval and bondval/vec (pair_bondval_vec.ts). The native compute() sums the bond-valence
 * sum s_i over the neighbours of each owned atom (both directions of every pair, as native's reverse communication does),
 * copies fp_i = S_i * power * (s_i - V0_i) to the ghosts, and runs the force loop over the half list. The energy
 * S_i (s_i - V0_i)^2 is added per owned atom (native: eng_vdwl += phi); the virial is the dot product of coordinates and
 * forces (virialFdotr), as native virial_fdotr_compute() does. Both styles have single_enable = 0 in native.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { pow as libmPow } from '../libm/pow';

/** Shared settings, coefficients and bookkeeping of the two bond-valence styles. */
export abstract class PairBondValBase extends Pair {
  manybody = true;
  needsFull = true;
  needsHalf = true;
  virialFdotr = true;
  /** pair_style bondval power: the exponent of the bond-valence force. */
  power = 0;
  /** pair_style bondval cutoff: the default cutoff of pair_coeff (used when no cutoff is given). */
  cutGlobal = 0;
  p!: PairParams;
  /** Per type pair (nt x nt, symmetric after initOne): r0, alpha (Nij), S, V0. */
  protected r0 = new Float64Array(0);
  protected alpha = new Float64Array(0);
  protected sp = new Float64Array(0);
  protected v0 = new Float64Array(0);
  protected nt = 0;

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['r0', 'alpha', 's', 'v0', 'cut']);
  }

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 2) throw new StyleError('Illegal pair_style command');
    this.power = parseNum(args[0], 'power');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    // native settings(): the cutoff of every set type pair with I < J becomes the new global cutoff
    if (this.p) {
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i + 1; j <= this.ntypes; j++) {
          if (this.p.isSet(i, j)) {
            this.p.p('cut')[this.p.idx(i, j)] = this.cutGlobal;
            this.p.p('cut')[this.p.idx(j, i)] = this.cutGlobal;
          }
        }
      }
    }
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 6 || args.length > 7) throw new StyleError('Incorrect args for pair coefficients (pair_coeff I J r0 alpha S V0 [cutoff])');
    const vals = [
      parseNum(args[2], 'r0'), parseNum(args[3], 'alpha'), parseNum(args[4], 'S'), parseNum(args[5], 'V0'),
      args.length === 7 ? parseNum(args[6], 'cutoff') : this.cutGlobal,
    ];
    this.p.setRange(args[0], args[1], vals);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    const nt = this.ntypes + 1;
    this.nt = nt;
    this.r0 = new Float64Array(nt * nt);
    this.alpha = new Float64Array(nt * nt);
    this.sp = new Float64Array(nt * nt);
    this.v0 = new Float64Array(nt * nt);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j}): ${this.name} does not mix`);
    const k1 = i * this.nt + j, k2 = j * this.nt + i;
    this.r0[k1] = this.r0[k2] = this.p.get('r0', i, j);
    this.alpha[k1] = this.alpha[k2] = this.p.get('alpha', i, j);
    this.sp[k1] = this.sp[k2] = this.p.get('s', i, j);
    this.v0[k1] = this.v0[k2] = this.p.get('v0', i, j);
    return this.p.get('cut', i, j);
  }

  /** Adds an energy and the per-atom energy of one owned atom. */
  protected tallyEnergy(pc: PairCompute, i: number, phi: number): void {
    pc.acc.evdwl += phi;
    if (pc.eatom) pc.eatom[i] += phi;
  }
}

/** pair_style bondval: the bond-valence energy E_BV = sum_i S_i (V_i - V0_i)^2. */
export class PairBondVal extends PairBondValBase {
  readonly name = 'bondval';
  private sBuf = new Float64Array(0);
  private fpBuf = new Float64Array(0);

  compute(pc: PairCompute): void {
    const full = pc.full;
    const half = pc.half;
    if (!full || !half) throw new Error(`pair style ${this.name} needs full and half neighbor lists`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal, nall = pc.nall;
    const nt = this.nt;
    const cutsq = this.cutsq;
    const { r0, alpha, sp, v0 } = this;
    const power = this.power;
    if (this.sBuf.length !== nall) {
      this.sBuf = new Float64Array(nall);
      this.fpBuf = new Float64Array(nall);
    }
    const s0 = this.sBuf;
    const fp = this.fpBuf;
    // bond-valence sum of every owned atom over its full neighbours (native: both directions of each pair)
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let s = 0;
      const k0: number = full.firstneigh[i], k1: number = k0 + full.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j: number = full.neighbors[k] & NEIGHMASK;
        const t = ti + type[j];
        if (alpha[t] === 0) continue;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        s += libmPow(r0[t] / r, alpha[t]);
      }
      s0[i] = s;
    }
    // fp_i = S_i * power * (s_i - V0_i); the energy S_i (s_i - V0_i)^2 per owned atom
    for (let i = 0; i < nlocal; i++) {
      const ti = type[i] * nt + type[i];
      const s = s0[i] - v0[ti];
      fp[i] = sp[ti] * power * s;
      this.tallyEnergy(pc, i, sp[ti] * s * s);
    }
    pc.nb.forwardCopy(fp, 1);
    // forces over the half list: psi' = (fp_i + fp_j) A_ij with A_ij = alpha (r0/r)^alpha / r
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const fpi = fp[i];
      const k0: number = half.firstneigh[i], k1: number = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j: number = half.neighbors[k] & NEIGHMASK;
        const t = ti + type[j];
        if (alpha[t] === 0) continue;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        const recip = 1 / r;
        const aij = alpha[t] * libmPow(r0[t] * recip, alpha[t]) * recip;
        const fpair = (fpi + fp[j]) * aij * recip;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        if (pc.vatom) this.tallyVatom(pc, i, j, fx, fy, fz, dx, dy, dz);
      }
    }
  }

  /** Per-atom virial: half of the pair term to each atom (del (x) force on i). */
  protected tallyVatom(pc: PairCompute, i: number, j: number, fx: number, fy: number, fz: number, dx: number, dy: number, dz: number): void {
    const va = pc.vatom!;
    const v = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
    for (const k of [i, j]) for (let c = 0; c < 6; c++) va[6 * k + c] += 0.5 * v[c];
  }
}
