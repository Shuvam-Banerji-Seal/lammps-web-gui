/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/COLLOID/pair_lubricateU_poly.cpp and
 * pair_lubricateU_poly.h (Amit Kumar and Michael Bybee (UIUC)). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lubricateU/poly (docs.lammps.org/pair_lubricateU.html): the polydisperse variant of lubricateU. It needs
 * newton pair off and a full neighbor list: every atom gets only its own forces and torques, from each of its neighbors
 * (no ghost reverse sums), and its compute() also runs at setup (native has no setupflag check in this style). The
 * near-field resistances use the ratio beta = radj/radi of the two radii (native pair_lubricateU_poly.cpp).
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, type NeighList } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { FixWallBase } from '../../engine/fix/walls';
import type { Fix } from '../../engine/fix/fix';
import { log } from '../libm/log';
import { pow } from '../libm/pow';
import { PairLubricateU, evTallyXYZ, findWallFixes, intArg, type WallView } from './pair_lubricateU';

export class PairLubricateUPoly extends PairLubricateU {
  override readonly name: string = 'lubricateU/poly';
  protected override tol = 1e-3;
  protected override skipAtSetup = false;
  protected override newton = false;
  protected override useHalfOrientation(): boolean { return false; }
  override needsFull = true;
  override needsHalf = false;

  protected override list(): NeighList {
    const l = this.pc.full;
    if (!l) throw new StyleError(`pair ${this.name} needs a full neighbor list`);
    return l;
  }

  settings(args: string[], ctx: StyleContext): void {
    if (args.length < 5 || args.length > 7) throw new StyleError('Illegal pair_style command');
    this.mu = parseNum(args[0], 'mu');
    this.flaglog = intArg(args[1], 'flaglog');
    this.cutInnerGlobal = parseNum(args[2], 'cutinner');
    this.cutGlobal = parseNum(args[3], 'cutoff');
    this.gdot = parseNum(args[4], 'gdot');
    this.flagHI = 1;
    this.flagVF = 1;
    if (args.length >= 6) this.flagHI = intArg(args[5], 'flagHI');
    if (args.length === 7) this.flagVF = intArg(args[6], 'flagVF');
    if (this.flaglog === 1 && this.flagHI === 0) {
      ctx.log('WARNING: Cannot include log terms without 1/r terms; setting flagHI to 1');
      this.flagHI = 1;
    }
    const nt = this.ntypes + 1;
    if (this.setflag.length === nt * nt) {
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) {
          if (this.setflag[i * nt + j]) {
            this.cutInnerCoeff[i * nt + j] = this.cutInnerGlobal;
            this.cutCoeff[i * nt + j] = this.cutGlobal;
          }
        }
      }
    }
    this.Ef.fill(0);
    this.Ef[1] = 0.5 * this.gdot;
    this.Ef[3] = 0.5 * this.gdot;
  }

  /** Native init_style of lubricateU/poly: its own requirements, the volume of the particles summed over the owned atoms. */
  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) throw new StyleError(`pair ${this.name} needs the atoms (run setup first)`);
    if (ctx.newtonPair !== false) throw new StyleError('Pair lubricateU/poly requires newton pair off');
    if (!ctx.ghostVelocity) throw new StyleError('Pair lubricateU/poly requires ghost atoms store velocity');
    if (!s.omega) throw new StyleError('Pair lubricateU/poly requires atom attribute omega');
    if (!s.radius) throw new StyleError('Pair lubricateU/poly requires atom attribute radius');
    for (let i = 0; i < s.n; i++) {
      if (s.radius[i] === 0.0) throw new StyleError('Pair lubricateU/poly requires extended particles');
    }
    this.flagdeform = ctx.fixOf?.('deform') ? 1 : 0;
    const walls = findWallFixes(ctx);
    this.flagwall = 0;
    this.wallFix = null;
    if (walls.length > 1) throw new StyleError('Cannot use multiple fix wall commands with pair lubricateU/poly');
    if (walls.length === 1) {
      const f = walls[0];
      if (!(f instanceof FixWallBase)) throw new StyleError(`Fix ${f.style} is not compatible with pair lubricateU/poly`);
      this.wallFix = f as unknown as Fix & WallView;
      this.flagwall = 1;
      if (this.wallFix.walls.some((w) => !w.coord.edge && typeof w.coord.v === 'object')) this.flagwall = 2;
    }
    const prd = this.prdOf(s.box.lo, s.box.hi);
    const volT = !this.flagwall ? prd[0] * prd[1] * prd[2] : (() => {
      const d = this.wallDims(prd);
      return d[0] * d[1] * d[2];
    })();
    // assuming polydisperse spheres, the volume of the particles is summed over the owned atoms
    let volP = 0.0;
    for (let i = 0; i < s.n; i++) volP += (4.0 / 3.0) * Math.PI * pow(s.radius[i], 3.0);
    this.volP = volP;
    let volf = this.volP / volT;
    if (!this.flagVF) volf = 0;
    if (ctx.log) ctx.log(`lubricateU: vol_f = ${volf}, vol_p = ${this.volP}, vol_T = ${volT}`);
    this.setIsotropic(volf);
  }

  /** Native init_style constants of lubricateU/poly (no radius factor: R0 = 6 pi mu (...), RT0 = 8 pi mu, RS0 = 20/3 pi mu (...)). */
  protected override setIsotropic(volf: number): void {
    const mu = this.mu;
    if (this.flaglog === 0) {
      this.R0 = 6 * Math.PI * mu * (1.0 + 2.16 * volf);
      this.RT0 = 8 * Math.PI * mu;
      this.RS0 = 20.0 / 3.0 * Math.PI * mu * (1.0 + 3.33 * volf + 2.80 * volf * volf);
    } else {
      this.R0 = 6 * Math.PI * mu * (1.0 + 2.725 * volf - 6.583 * volf * volf);
      this.RT0 = 8 * Math.PI * mu * (1.0 + 0.749 * volf - 2.469 * volf * volf);
      this.RS0 = 20.0 / 3.0 * Math.PI * mu * (1.0 + 3.64 * volf - 6.95 * volf * volf);
    }
  }

  /** Native compute_RE of lubricateU/poly: the strain-rate part, the owned atom only (full list). */
  protected override computeRE(x: Float64Array): void {
    if (!this.flagHI) return;
    const pc = this.pc, list = this.list();
    const type = pc.type, cutsq = this.cutsq, nt = this.ntypes + 1;
    const cutInner = this.cutInnerCoeff, Ef = this.Ef, F = this.F, T = this.T;
    const mu = this.mu, flaglog = this.flaglog, vxmu2f = this.vxmu2f;
    for (let i = 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      const radi = this.radAll[i];
      let pre0 = Math.PI * mu * radi;
      pre0 *= 6.0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const radj = this.radAll[j];
        const r = Math.sqrt(rsq);
        const xl0 = -delx / r * radi, xl1 = -dely / r * radi, xl2 = -delz / r * radi;
        let hSep = r - radi - radj;
        if (r < cutInner[itype * nt + jtype]) hSep = cutInner[itype * nt + jtype] - radi - radj;
        hSep = hSep / radi;
        const b01 = radj / radi;
        const b11 = 1.0 + b01;
        let aSq: number, aSh = 0;
        if (flaglog) {
          const b02 = b01 * b01, b03 = b02 * b01, b04 = b03 * b01;
          const b12 = b11 * b11, b13 = b12 * b11;
          const logB13 = log(1.0 / hSep) / b13;
          const hB11 = hSep / b11;
          aSq = pre0 * (b02 / b12 / hSep + ((0.2 + 1.4 * b01 + 0.2 * b02) + (1.0 + 18.0 * (b01 + b03) - 29.0 * b02 + b04) * hB11 / 21.0) * logB13);
          aSh = pre0 * ((8.0 * (b01 + b03) + 4.0 * b02) / 15.0 + (64.0 - 180.0 * (b01 + b03) + 232.0 * b02 + 64.0 * b04) * hB11 / 375.0) * logB13;
        } else {
          aSq = pre0 * (b01 * b01 / (b11 * b11 * hSep));
        }
        // relative velocity at the point of closest approach due to Ef only
        const vr1 = -2.0 * (Ef[0] * xl0 + Ef[1] * xl1 + Ef[2] * xl2);
        const vr2 = -2.0 * (Ef[3] * xl0 + Ef[4] * xl1 + Ef[5] * xl2);
        const vr3 = -2.0 * (Ef[6] * xl0 + Ef[7] * xl1 + Ef[8] * xl2);
        const vnnr = (vr1 * delx + vr2 * dely + vr3 * delz) / r;
        const vn1 = vnnr * delx / r, vn2 = vnnr * dely / r, vn3 = vnnr * delz / r;
        const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
        let fx = aSq * vn1, fy = aSq * vn2, fz = aSq * vn3;
        if (flaglog) {
          fx = fx + aSh * vt1;
          fy = fy + aSh * vt2;
          fz = fz + aSh * vt3;
        }
        fx = vxmu2f * fx; fy = vxmu2f * fy; fz = vxmu2f * fz;
        F[3 * i] -= fx; F[3 * i + 1] -= fy; F[3 * i + 2] -= fz;
        if (flaglog) {
          const tx = xl1 * fz - xl2 * fy, ty = xl2 * fx - xl0 * fz, tz = xl0 * fy - xl1 * fx;
          T[3 * i] -= vxmu2f * tx; T[3 * i + 1] -= vxmu2f * ty; T[3 * i + 2] -= vxmu2f * tz;
        }
      }
    }
  }

  /** Native compute_RU of lubricateU/poly: R_FU U of the owned atom from each neighbor (full list, no ghost forces). */
  protected override computeRU(x: Float64Array): void {
    const pc = this.pc, list = this.list();
    const type = pc.type, cutsq = this.cutsq, nt = this.ntypes + 1;
    const cutInner = this.cutInnerCoeff, F = this.F, T = this.T, V = this.V, W = this.W;
    const mu = this.mu, flaglog = this.flaglog, vxmu2f = this.vxmu2f;
    const vT = this.volTCompute(pc);
    if (vT !== null) {
      const volf = this.volP / vT;
      if (flaglog === 0) {
        this.R0 = 6 * Math.PI * mu * (1.0 + 2.16 * volf);
        this.RT0 = 8 * Math.PI * mu;
      } else {
        this.R0 = 6 * Math.PI * mu * (1.0 + 2.725 * volf - 6.583 * volf * volf);
        this.RT0 = 8 * Math.PI * mu * (1.0 + 0.749 * volf - 2.469 * volf * volf);
      }
    }
    const R0 = this.R0, RT0 = this.RT0;
    F.fill(0);
    T.fill(0);
    for (let i = 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      const radi = this.radAll[i];
      let pre0 = Math.PI * mu * radi;
      const pre1 = 8.0 * pre0 * radi * radi;
      pre0 *= 6.0;
      const wi0 = W[3 * i], wi1 = W[3 * i + 1], wi2 = W[3 * i + 2];
      // contribution due to the isotropic terms
      F[3 * i] += -vxmu2f * R0 * radi * V[3 * i];
      F[3 * i + 1] += -vxmu2f * R0 * radi * V[3 * i + 1];
      F[3 * i + 2] += -vxmu2f * R0 * radi * V[3 * i + 2];
      const radi3 = radi * radi * radi;
      T[3 * i] += -vxmu2f * RT0 * radi3 * wi0;
      T[3 * i + 1] += -vxmu2f * RT0 * radi3 * wi1;
      T[3 * i + 2] += -vxmu2f * RT0 * radi3 * wi2;
      if (!this.flagHI) continue;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const radj = this.radAll[j];
        const r = Math.sqrt(rsq);
        const wj0 = W[3 * j], wj1 = W[3 * j + 1], wj2 = W[3 * j + 2];
        const xl0 = -delx / r * radi, xl1 = -dely / r * radi, xl2 = -delz / r * radi;
        const jl0 = delx / r * radj, jl1 = dely / r * radj, jl2 = delz / r * radj;
        // velocity at the point of closest approach on both particles, v = v + omega x l
        const vi0 = V[3 * i] + (wi1 * xl2 - wi2 * xl1);
        const vi1 = V[3 * i + 1] + (wi2 * xl0 - wi0 * xl2);
        const vi2 = V[3 * i + 2] + (wi0 * xl1 - wi1 * xl0);
        const vj0 = V[3 * j] + (wj1 * jl2 - wj2 * jl1);
        const vj1 = V[3 * j + 1] + (wj2 * jl0 - wj0 * jl2);
        const vj2 = V[3 * j + 2] + (wj0 * jl1 - wj1 * jl0);
        let hSep = r - radi - radj;
        if (r < cutInner[itype * nt + jtype]) hSep = cutInner[itype * nt + jtype] - radi - radj;
        hSep = hSep / radi;
        const b01 = radj / radi;
        const b11 = 1.0 + b01;
        let aSq: number, aSh = 0, aPu = 0;
        if (flaglog) {
          const b02 = b01 * b01, b03 = b02 * b01, b04 = b03 * b01;
          const b12 = b11 * b11, b13 = b12 * b11;
          const logH = log(1.0 / hSep);
          const logB13 = log(1.0 / hSep) / b13;
          const hB11 = hSep / b11;
          aSq = pre0 * (b02 / b12 / hSep + ((0.2 + 1.4 * b01 + 0.2 * b02) + (1.0 + 18.0 * (b01 + b03) - 29.0 * b02 + b04) * hB11 / 21.0) * logB13);
          aSh = pre0 * ((8.0 * (b01 + b03) + 4.0 * b02) / 15.0 + (64.0 - 180.0 * (b01 + b03) + 232.0 * b02 + 64.0 * b04) * hB11 / 375.0) * logB13;
          aPu = pre1 * (0.4 * b01 * b11 + (0.128 + 0.096 * b01 + 0.528 * b02) * b12 * hSep) * logH;
        } else {
          aSq = pre0 * (b01 * b01 / (b11 * b11 * hSep));
        }
        const vr1 = vi0 - vj0, vr2 = vi1 - vj1, vr3 = vi2 - vj2;
        const vnnr = (vr1 * delx + vr2 * dely + vr3 * delz) / r;
        const vn1 = vnnr * delx / r, vn2 = vnnr * dely / r, vn3 = vnnr * delz / r;
        const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
        let fx = aSq * vn1, fy = aSq * vn2, fz = aSq * vn3;
        if (flaglog) {
          fx = fx + aSh * vt1;
          fy = fy + aSh * vt2;
          fz = fz + aSh * vt3;
        }
        fx = vxmu2f * fx; fy = vxmu2f * fy; fz = vxmu2f * fz;
        F[3 * i] -= fx; F[3 * i + 1] -= fy; F[3 * i + 2] -= fz;
        if (flaglog) {
          let tx = xl1 * fz - xl2 * fy, ty = xl2 * fx - xl0 * fz, tz = xl0 * fy - xl1 * fx;
          T[3 * i] -= vxmu2f * tx; T[3 * i + 1] -= vxmu2f * ty; T[3 * i + 2] -= vxmu2f * tz;
          // torque due to a_pu
          const wdotn = ((wi0 - wj0) * delx + (wi1 - wj1) * dely + (wi2 - wj2) * delz) / r;
          const wt1 = (wi0 - wj0) - wdotn * delx / r;
          const wt2 = (wi1 - wj1) - wdotn * dely / r;
          const wt3 = (wi2 - wj2) - wdotn * delz / r;
          tx = aPu * wt1;
          ty = aPu * wt2;
          tz = aPu * wt3;
          T[3 * i] -= vxmu2f * tx; T[3 * i + 1] -= vxmu2f * ty; T[3 * i + 2] -= vxmu2f * tz;
        }
      }
    }
  }

  /** Native compute_Fh of lubricateU/poly: the stresslet virial only (no flagHI test here, as native); newton off tallies. */
  protected override computeFh(x: Float64Array): void {
    const pc = this.pc, list = this.list();
    const type = pc.type, cutsq = this.cutsq, nt = this.ntypes + 1;
    const cutInner = this.cutInnerCoeff, Ef = this.Ef, V = this.V, W = this.W;
    const mu = this.mu, flaglog = this.flaglog, vxmu2f = this.vxmu2f, nlocal = this.nlocal;
    const gdot = this.gdot;
    const vT = this.volTCompute(pc);
    if (vT !== null) {
      const volf = this.volP / vT;
      if (flaglog === 0) {
        this.RS0 = 20.0 / 3.0 * Math.PI * mu * (1.0 + 3.33 * volf + 2.80 * volf * volf);
      } else {
        this.RS0 = 20.0 / 3.0 * Math.PI * mu * (1.0 + 3.64 * volf - 6.95 * volf * volf);
      }
    }
    const RS0 = this.RS0;
    this.F.fill(0);
    this.T.fill(0);
    for (let i = 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      const radi = this.radAll[i];
      // isotropic RS0 stress: pseudo force (delx = radi and fy only)
      const fyS = vxmu2f * RS0 * pow(radi, 3.0) * gdot / 2.0 / radi;
      if (this.evflag) evTallyXYZ(pc, i, i, nlocal, false, -0.0, -fyS, -0.0, radi, 0.0, 0.0);
      const wi0 = W[3 * i], wi1 = W[3 * i + 1], wi2 = W[3 * i + 2];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = xtmp - x[3 * j], dy = ytmp - x[3 * j + 1], dz = ztmp - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const radj = this.radAll[j];
        const r = Math.sqrt(rsq);
        const wj0 = W[3 * j], wj1 = W[3 * j + 1], wj2 = W[3 * j + 2];
        const xl0 = -dx / r * radi, xl1 = -dy / r * radi, xl2 = -dz / r * radi;
        const jl0 = dx / r * radj, jl1 = dy / r * radj, jl2 = dz / r * radj;
        const vi0 = V[3 * i] + (wi1 * xl2 - wi2 * xl1);
        const vi1 = V[3 * i + 1] + (wi2 * xl0 - wi0 * xl2);
        const vi2 = V[3 * i + 2] + (wi0 * xl1 - wi1 * xl0);
        const vj0 = V[3 * j] + (wj1 * jl2 - wj2 * jl1);
        const vj1 = V[3 * j + 1] + (wj2 * jl0 - wj0 * jl2);
        const vj2 = V[3 * j + 2] + (wj0 * jl1 - wj1 * jl0);
        const vr1 = vi0 - vj0 - 2.0 * (Ef[0] * xl0 + Ef[1] * xl1 + Ef[2] * xl2);
        const vr2 = vi1 - vj1 - 2.0 * (Ef[3] * xl0 + Ef[4] * xl1 + Ef[5] * xl2);
        const vr3 = vi2 - vj2 - 2.0 * (Ef[6] * xl0 + Ef[7] * xl1 + Ef[8] * xl2);
        const vnnr = (vr1 * dx + vr2 * dy + vr3 * dz) / r;
        const vn1 = vnnr * dx / r, vn2 = vnnr * dy / r, vn3 = vnnr * dz / r;
        const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
        let hSep = r - radi - radj;
        if (r < cutInner[itype * nt + jtype]) hSep = cutInner[itype * nt + jtype] - radi - radj;
        hSep = hSep / radi;
        const pre0 = 6.0 * (Math.PI * mu * radi);
        const b01 = radj / radi;
        const b11 = 1.0 + b01;
        let aSq: number, aSh = 0;
        if (flaglog) {
          const b02 = b01 * b01, b03 = b02 * b01, b04 = b03 * b01;
          const b12 = b11 * b11, b13 = b12 * b11;
          const logB13 = log(1.0 / hSep) / b13;
          const hB11 = hSep / b11;
          aSq = pre0 * (b02 / b12 / hSep + ((0.2 + 1.4 * b01 + 0.2 * b02) + (1.0 + 18.0 * (b01 + b03) - 29.0 * b02 + b04) * hB11 / 21.0) * logB13);
          aSh = pre0 * ((8.0 * (b01 + b03) + 4.0 * b02) / 15.0 + (64.0 - 180.0 * (b01 + b03) + 232.0 * b02 + 64.0 * b04) * hB11 / 375.0) * logB13;
        } else {
          aSq = pre0 * (b01 * b01 / (b11 * b11 * hSep));
        }
        let fx = aSq * vn1, fy = aSq * vn2, fz = aSq * vn3;
        if (flaglog) {
          fx = fx + aSh * vt1;
          fy = fy + aSh * vt2;
          fz = fz + aSh * vt3;
        }
        fx = vxmu2f * fx; fy = vxmu2f * fy; fz = vxmu2f * fz;
        // set j = nlocal so that only I gets tallied (native ev_tally_xyz(i, nlocal, nlocal, 0, ...))
        if (this.evflag) evTallyXYZ(pc, i, nlocal, nlocal, false, -fx, -fy, -fz, dx, dy, dz);
      }
    }
  }
}
