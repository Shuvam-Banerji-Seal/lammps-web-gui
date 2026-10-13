/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_coul_shield.cpp and src/INTERLAYER/pair_coul_shield.h, at commit b332f28. Contributing
 * author in the C++ header: Wengen Ouyang (Wuhan University); the potential is described in Maaravi et al., J. Phys.
 * Chem. C 121, 22826-22835 (2017). Copyright (2003) Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { hasChargeStyle, hasMolecule } from '../../engine/atoms';
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { calcDTap, calcTap } from './taper';

/**
 * pair_style coul/shield (docs.lammps.org/pair_coul_shield.html): a shielded Coulomb interaction between atoms of
 * different molecules, cut off with the taper function. Usage: pair_style coul/shield cutoff [tap_flag], and
 * pair_coeff I J sigmae [cutoff] for each type pair (set explicitly for every pair).
 */
export class PairCoulShield extends Pair {
  readonly name: string = 'coul/shield';
  manybody = false;
  needsHalf = true;
  needsFull = false;
  virialFdotr = true;
  private tapFlag = 1;
  private cutGlobal = 0;
  /** Per type pair (nt*nt, symmetric). */
  private cutc = new Float64Array(0);
  private sigmae = new Float64Array(0);
  private offset = new Float64Array(0);
  private setflag = new Uint8Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.cutc = new Float64Array(nt * nt);
    this.sigmae = new Float64Array(nt * nt);
    this.offset = new Float64Array(nt * nt);
    this.setflag = new Uint8Array(nt * nt);
  }

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length < 1 || args.length > 2) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (args.length === 2) {
      const t = Number(args[1]);
      if (!Number.isInteger(t)) throw new StyleError(`expected an integer for tap_flag, got '${args[1]}'`);
      this.tapFlag = t;
    }
    // LAMMPS sets the cutoff of the explicitly set pairs (I < J) when the global cutoff changes
    if (this.ntypes > 0) {
      const nt = this.ntypes + 1;
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i + 1; j <= this.ntypes; j++) {
          if (this.setflag[i * nt + j]) this.cutc[i * nt + j] = this.cutGlobal;
        }
      }
    }
  }

  override coeff(args: string[], _ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3 || args.length > 4) throw new StyleError('Incorrect args for pair coefficients');
    const nt = this.ntypes + 1;
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const sigmaeOne = parseNum(args[2], 'sigmae');
    const cutOne = args.length === 4 ? parseNum(args[3], 'cutoff') : this.cutGlobal;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.sigmae[i * nt + j] = sigmaeOne;
        this.cutc[i * nt + j] = cutOne;
        this.setflag[i * nt + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  override initStyle(ctx: StyleContext): void {
    if (!ctx.s || !hasChargeStyle(ctx.s.atomStyle)) throw new StyleError('Pair style coul/shield requires atom attribute q');
    if (!hasMolecule(ctx.s)) throw new StyleError('Pair style coul/shield requires atom attribute molecule');
  }

  override initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.setflag[i * nt + j]) {
      throw new StyleError('for pair style coul/shield, parameters need to be set explicitly for all pairs.');
    }
    // the offset of pair_modify shift yes is only written for the untapered form; that form is refused here
    if (this.shift && this.tapFlag === 0) {
      throw new StyleError('pair_modify shift yes with pair style coul/shield tap_flag 0 is not supported by the browser engine');
    }
    this.sigmae[j * nt + i] = this.sigmae[i * nt + j];
    this.offset[i * nt + j] = 0;
    this.offset[j * nt + i] = 0;
    this.cutc[j * nt + i] = this.cutc[i * nt + j];
    return this.cutc[i * nt + j];
  }

  override compute(pc: PairCompute): void {
    const list = pc.half;
    if (!list) throw new Error(`pair style ${this.name} needs a half neighbor list`);
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    const qqrd2e = pc.qqrd2e;
    const tapOn = this.tapFlag !== 0;
    const nb = list.neighbors;
    let ecoulSum = 0;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const factor = pc.specialCoul[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq < cutsq[t] && mol[owner[i]] !== mol[owner[j]])) continue;
        const r = Math.sqrt(rsq);
        const r3 = rsq * r;
        const rarg = 1.0 / this.sigmae[t];
        const th = r3 + rarg * rarg * rarg;
        const epsr = 1.0 / Math.pow(th, 1.0 / 3.0);
        let depsdr = epsr * epsr;
        depsdr *= depsdr;
        const vc = qqrd2e * qi * q[j] * epsr;
        const tap = tapOn ? calcTap(r, this.cutc[t]) : 1.0;
        const dtap = tapOn ? calcDTap(r, this.cutc[t]) : 0.0;
        const forcecoul = qqrd2e * qi * q[j] * r * depsdr;
        const fvc = forcecoul * tap - vc * dtap / r;
        const fpair = factor * fvc;
        f[3 * i] += dx * fpair;
        f[3 * i + 1] += dy * fpair;
        f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair;
        f[3 * j + 1] -= dy * fpair;
        f[3 * j + 2] -= dz * fpair;
        const ecoul = (tapOn ? vc * tap : vc - this.offset[t]) * factor;
        ecoulSum += ecoul;
        tallyAtom(pc, i, j, ecoul, fpair, dx, dy, dz);
      }
    }
    pc.acc.ecoul += ecoulSum;
  }
}
