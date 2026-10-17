/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MISC/pair_srp.cpp and src/MISC/pair_srp.h at commit b332f28 (pair_style srp: the segmental repulsive
 * potential of Sirk et al., J. Chem. Phys. 136, 134903 (2012); Timothy Sirk, ARL, and Pieter in 't Veld, BASF). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Native structure kept: the style makes its fix SRP when it is created (the bond particles, see fix_srp.ts), runs only
 * as a sub-style of pair hybrid (bond particles interact with each other only), and walks the half list of bond
 * particles in native order (NativePairOrder). At every neighbor build each bond particle (owned or ghost) maps its two
 * atoms to their closest images (remapBonds), and with 'exclude yes' (the default) a pair of bonds that share an atom is
 * left out (onetwoexclude). The force between the two bonds acts on the four atoms: by the midpoint distance ('mid',
 * half on each atom) or by the minimum distance of the segments ('min', split by the lever rule); the energy
 * 0.5 a0 rc (1 - d/rc)^2 is tallied as two half pairs (ev_tally of i0-i1 and j0-j1, newton on).
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { buildAtomMap } from '../../engine/atoms';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { FixSRP, SRP_TAGS, closestImage, imageChains } from './fix_srp';

const SMALL = 1.0e-10;
const BIG = 1.0e10;

/** utils::numeric. */
const numeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  return v;
};

/**
 * ev_tally (newton on) without the global virial: inside pair hybrid native takes the virial of the summed forces
 * (virial_fdotr_compute of the hybrid), so a sub-style adds its energy and per-atom terms only.
 */
const tally = (pc: PairCompute, i: number, j: number, evdwl: number, fpair: number, dx: number, dy: number, dz: number): void => {
  pc.acc.evdwl += evdwl;
  if (pc.eatom) {
    const e = 0.5 * evdwl;
    pc.eatom[i] += e; pc.eatom[j] += e;
  }
  if (pc.vatom) {
    const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
    const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
    for (const k of [i, j]) {
      const va = pc.vatom;
      va[6 * k] += 0.5 * v0; va[6 * k + 1] += 0.5 * v1; va[6 * k + 2] += 0.5 * v2;
      va[6 * k + 3] += 0.5 * v3; va[6 * k + 4] += 0.5 * v4; va[6 * k + 5] += 0.5 * v5;
    }
  }
};

/** native srp_instance: each pair srp names its fix with the next number. */
let srpInstance = 0;
export const nextSrpInstance = (): number => srpInstance++;

export class PairSRP extends Pair {
  readonly name: string = 'srp';
  /** the global virial is the sum x.f over owned and ghost atoms (native: the hybrid's virial_fdotr_compute) */
  virialFdotr = true;
  protected cutGlobal = 0;
  protected btypeStr = '';
  protected btype = 0;
  protected midpoint = false;
  protected min = false;
  protected exclude = true;
  protected bptypeStr = '';
  protected bptype = 0;
  private a0 = new Float64Array(0);
  protected cutT = new Float64Array(0);
  protected setflag = new Uint8Array(0);
  protected fixId = '';
  protected fix: FixSRP | null = null;
  /** the style of the fix this pair makes (SRP; SRPREACT for srp/react) */
  protected fixStyle = 'SRP';
  private readonly order = new NativePairOrder(true);
  /** segment[2 e], segment[2 e + 1]: the engine indices of the two atoms of bond particle e (closest images), per build */
  private segment = new Int32Array(0);
  private segBuild = -1;

  settings(args: string[], ctx: StyleContext): void {
    if (args.length < 3 || args.length > 7) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = numeric(args[0]);
    if (args[1] === '*') { this.btype = 0; this.btypeStr = ''; } else this.btypeStr = args[1];
    this.midpoint = false;
    this.min = false;
    if (args[2] === 'min') this.min = true;
    else if (args[2] === 'mid') this.midpoint = true;
    else throw new StyleError('Illegal pair_style command');
    this.exclude = true;
    const s = ctx.s;
    this.bptype = s ? s.ntypes : 0;
    let iarg = 3;
    while (iarg < args.length) {
      if (args[iarg] === 'exclude') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal pair srp command');
        const v = args[iarg + 1].toLowerCase();
        if (['yes', 'on', 'true', '1'].includes(v)) this.exclude = true;
        else if (['no', 'off', 'false', '0'].includes(v)) this.exclude = false;
        else throw new StyleError(`Expected boolean parameter instead of '${args[iarg + 1]}' in input script or data file`);
        if (this.min && !this.exclude) throw new StyleError('Illegal exclude option in pair srp command');
        iarg += 2;
      } else if (args[iarg] === 'bptype') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal pair srp command');
        this.bptypeStr = args[iarg + 1];
        iarg += 2;
      } else throw new StyleError('Illegal pair srp command');
    }
    this.createFix(ctx);
  }

  /** The fix SRP is made with the style (native constructor: "{:02d}_FIX_SRP" + "_FIX_SRP all SRP"). */
  protected createFix(ctx: StyleContext): void {
    const fixId = `${String(nextSrpInstance()).padStart(2, '0')}_FIX_SRP_FIX_SRP`;
    this.fixId = fixId;
    if (!ctx.addFix) throw new StyleError('pair srp needs to create its fix SRP');
    ctx.addFix((sys) => new FixSRP(sys, fixId, 'all', []));
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.a0 = new Float64Array(nt * nt);
    this.cutT = new Float64Array(nt * nt);
    this.setflag = new Uint8Array(nt * nt);
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 3 || args.length > 4) throw new StyleError('Incorrect args for pair coefficients');
    const s = ctx.s;
    const ntypes = this.ntypes;
    if (this.btypeStr) {
      const b = Number(this.btypeStr);
      const nbondtypes = s ? s.topo.nbondtypes : 0;
      if (!Number.isInteger(b) || b > nbondtypes || b <= 0) throw new StyleError(`Invalid bond type ${this.btypeStr} for pair style srp`);
      this.btype = b;
    }
    if (this.bptypeStr) this.bptype = Number(this.bptypeStr);
    if (!Number.isInteger(this.bptype) || this.bptype < 1 || this.bptype > ntypes) throw new StyleError(`Invalid bond particle type ${this.bptype} for pair style srp`);
    const nt = ntypes + 1;
    // reset cutoffs if explicitly set
    for (let i = 1; i <= this.bptype; i++) for (let j = i; j <= this.bptype; j++) if (this.setflag[i * nt + j]) this.cutT[i * nt + j] = this.cutGlobal;
    const [ilo, ihi] = typeBounds(args[0], this.bptype);
    const [jlo, jhi] = typeBounds(args[1], this.bptype);
    const a0One = numeric(args[2]);
    const cutOne = args.length === 4 ? numeric(args[3]) : this.cutGlobal;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.a0[i * nt + j] = a0One;
        this.cutT[i * nt + j] = cutOne;
        this.setflag[i * nt + j] = 1;
        count++;
      }
    }
    if (count === 0) ctx.log?.('WARNING: PairSRP: No pair coefficients were set');
    // the fix gets its types here too: the engine inits the fixes before the pair styles (native: the pair's
    // init_style hands them over before the fixes' init, which needs bptype for the neighbor exclusions)
    (ctx.fixOf?.(this.fixStyle) as FixSRP | null)?.setTypes(this.btype, this.bptype);
    this.version++;
  }

  initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError(`Pair ${this.name} requires newton pair on`);
    const fix = ctx.fixOf?.(this.fixStyle) as FixSRP | null;
    if (!fix || fix.id !== this.fixId) throw new StyleError(`Fix ${this.fixStyle} has been changed unexpectedly`);
    this.fix = fix;
    ctx.log?.(`Using type ${this.bptype} for bond particles`);
    fix.setTypes(this.btype, this.bptype);
    this.segBuild = -1;
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.setflag[i * nt + j]) throw new StyleError('All pair coeffs are not set');
    this.cutT[j * nt + i] = this.cutT[i * nt + j];
    this.a0[j * nt + i] = this.a0[i * nt + j];
    return this.cutT[i * nt + j];
  }

  /** remapBonds: the closest images of the two atoms of every bond particle, owned or ghost (at each neighbor build). */
  private remapBonds(pc: PairCompute): void {
    const s = pc.s;
    const nb = pc.nb;
    const nall = pc.nall;
    if (this.segment.length < 2 * nall) this.segment = new Int32Array(2 * nall);
    const seg = this.segment;
    seg.fill(-1);
    const tags = s.custom.get(SRP_TAGS)!.data;
    const map = buildAtomMap(s);
    const { head, next } = imageChains(nb.owner, nall, nb.nlocal);
    const x = pc.x;
    for (let e = 0; e < nall; e++) {
      const o = nb.owner[e];
      if (s.type[o] !== this.bptype) continue;
      const a = map[tags[2 * o]], b = map[tags[2 * o + 1]];
      seg[2 * e] = closestImage(x, head, next, a ?? -1, x[3 * e], x[3 * e + 1], x[3 * e + 2]);
      seg[2 * e + 1] = closestImage(x, head, next, b ?? -1, x[3 * e], x[3 * e + 1], x[3 * e + 2]);
    }
  }

  compute(pc: PairCompute): void {
    if (this.segBuild !== pc.nb.nbuild) {
      this.remapBonds(pc);
      this.segBuild = pc.nb.nbuild;
    }
    const s = pc.s;
    const x = pc.x, f = pc.f;
    const nt = this.ntypes + 1;
    const bp = this.bptype * nt + this.bptype;
    const cut = this.cutT[bp], cutsq = cut * cut, a0 = this.a0[bp];
    const seg = this.segment;
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const j = pairs[p + 1] & NEIGHMASK;
      const i0 = seg[2 * i], j0 = seg[2 * i + 1];
      const i1 = seg[2 * j], j1 = seg[2 * j + 1];
      // enforce 1-2 exclusions: the two bonds share an atom
      if (this.exclude && (i0 === i1 || i0 === j1 || i1 === j0 || j0 === j1)) continue;
      let dx: number, dy: number, dz: number;
      let ti = 0, tj = 0;
      if (this.midpoint) {
        dx = 0.5 * (x[3 * i0] - x[3 * i1] + x[3 * j0] - x[3 * j1]);
        dy = 0.5 * (x[3 * i0 + 1] - x[3 * i1 + 1] + x[3 * j0 + 1] - x[3 * j1 + 1]);
        dz = 0.5 * (x[3 * i0 + 2] - x[3 * i1 + 2] + x[3 * j0 + 2] - x[3 * j1 + 2]);
      } else {
        // getMinDist
        const diffx0 = x[3 * j0] - x[3 * i0], diffy0 = x[3 * j0 + 1] - x[3 * i0 + 1], diffz0 = x[3 * j0 + 2] - x[3 * i0 + 2];
        const diffx1 = x[3 * j1] - x[3 * i1], diffy1 = x[3 * j1 + 1] - x[3 * i1 + 1], diffz1 = x[3 * j1 + 2] - x[3 * i1 + 2];
        const dPx = 0.5 * (diffx0 - diffx1) + x[3 * i0] - x[3 * i1];
        const dPy = 0.5 * (diffy0 - diffy1) + x[3 * i0 + 1] - x[3 * i1 + 1];
        const dPz = 0.5 * (diffz0 - diffz1) + x[3 * i0 + 2] - x[3 * i1 + 2];
        const RiRi = diffx0 * diffx0 + diffy0 * diffy0 + diffz0 * diffz0;
        const RiRj = diffx0 * diffx1 + diffy0 * diffy1 + diffz0 * diffz1;
        const RjRj = diffx1 * diffx1 + diffy1 * diffy1 + diffz1 * diffz1;
        let denom = RiRj * RiRj - RiRi * RjRj;
        if (Math.abs(denom) < SMALL) denom = denom < 0 ? -BIG : BIG;
        const termx0 = RiRj * diffx1 - RjRj * diffx0, termy0 = RiRj * diffy1 - RjRj * diffy0, termz0 = RiRj * diffz1 - RjRj * diffz0;
        const num0 = dPx * termx0 + dPy * termy0 + dPz * termz0;
        ti = num0 / denom;
        if (ti > 0.5) ti = 0.5;
        if (ti < -0.5) ti = -0.5;
        const termx1 = RiRj * diffx0 - RiRi * diffx1, termy1 = RiRj * diffy0 - RiRi * diffy1, termz1 = RiRj * diffz0 - RiRi * diffz1;
        const num1 = dPx * termx1 + dPy * termy1 + dPz * termz1;
        tj = -num1 / denom;
        if (tj > 0.5) tj = 0.5;
        if (tj < -0.5) tj = -0.5;
        dx = dPx - ti * diffx0 + tj * diffx1;
        dy = dPy - ti * diffy0 + tj * diffy1;
        dz = dPz - ti * diffz0 + tj * diffz1;
      }
      const dijsq = dx * dx + dy * dy + dz * dz;
      if (!(dijsq < cutsq)) continue;
      const dij = Math.sqrt(dijsq);
      if (dij < SMALL) continue;
      const wd = 1.0 - dij / cut;
      const evdwl = 0.5 * a0 * cut * wd * wd;
      if (this.midpoint) {
        const fpair = 0.5 * a0 * wd / dij; // 0.5 factor for lever rule
        const fx = fpair * dx, fy = fpair * dy, fz = fpair * dz;
        f[3 * i0] += fx; f[3 * i0 + 1] += fy; f[3 * i0 + 2] += fz;
        f[3 * j0] += fx; f[3 * j0 + 1] += fy; f[3 * j0 + 2] += fz;
        f[3 * i1] -= fx; f[3 * i1 + 1] -= fy; f[3 * i1 + 2] -= fz;
        f[3 * j1] -= fx; f[3 * j1 + 1] -= fy; f[3 * j1 + 2] -= fz;
        tally(pc, i0, i1, 0.5 * evdwl, fpair, dx, dy, dz);
        tally(pc, j0, j1, 0.5 * evdwl, fpair, dx, dy, dz);
      } else {
        const fpair = a0 * wd / dij;
        const lever0 = 0.5 + ti, lever1 = 0.5 + tj;
        const fx = fpair * dx, fy = fpair * dy, fz = fpair * dz;
        const fxl0 = fx * lever0, fyl0 = fy * lever0, fzl0 = fz * lever0;
        const fxl1 = fx * lever1, fyl1 = fy * lever1, fzl1 = fz * lever1;
        f[3 * i0] += fxl0; f[3 * i0 + 1] += fyl0; f[3 * i0 + 2] += fzl0;
        f[3 * j0] += (fx - fxl0); f[3 * j0 + 1] += (fy - fyl0); f[3 * j0 + 2] += (fz - fzl0);
        f[3 * i1] -= fxl1; f[3 * i1 + 1] -= fyl1; f[3 * i1 + 2] -= fzl1;
        f[3 * j1] -= (fx - fxl1); f[3 * j1 + 1] -= (fy - fyl1); f[3 * j1 + 2] -= (fz - fzl1);
        tally(pc, i0, i1, 0.5 * evdwl, 0.5 * fpair, dx, dy, dz);
        tally(pc, j0, j1, 0.5 * evdwl, 0.5 * fpair, dx, dy, dz);
      }
    }
  }
}
