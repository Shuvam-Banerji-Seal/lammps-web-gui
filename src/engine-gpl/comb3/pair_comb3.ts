/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb3.cpp and src/MANYBODY/pair_comb3.h (PairComb3: settings, coeff, init_style,
 * init_one, Short_neigh, compute, with the two-body, bond-order and three-body loops), at commit b332f28.
 * Authors credited in the C++ file headers: Tzu-Ray Shan (Sandia) and the COMB3 authors. Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Stages A, B and C: the radical and torsion terms (radtor.ts), the Coulomb and charge-force terms (coulomb.ts)
 * and the charge-force routine chargeForces (fix qeq/comb) are included. polar_on (dipole polarization) is
 * refused with a StyleError.
 *
 * Ghost atoms: the C++ builds short neighbor lists for ghost atoms too (sht_first[j] for a ghost j). Here a
 * ghost's short list is the translate of its owner's list: the neighbor l of the owner becomes the atom with the
 * same owner and the image offset of l plus that of the ghost (atoms outside the ghost shell are skipped, as the
 * C++ ghost neighbor list skips them).
 */
import { Pair, StyleError, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { readComb3Entries, readComb3Lib, setupComb3Params, type Comb3Lib, type Comb3Param } from './params';
import { comb3Tables, triPoint, vdwaals, type Comb3Tables, type TriPoint } from './tables';
import { Comb3Kernel } from './terms';
import { Comb3RadTor } from './radtor';
import { Comb3Charge } from './coulomb';
import { potalFactors } from './tables';
import { vTally2, vTally3 } from '../airebo/tally';

const v3 = (): Float64Array => new Float64Array(3);

/** PairComb3: charge-optimized many-body potential with the COMB3 bond order (no polarization). */
export class PairComb3 extends Pair {
  readonly name = 'comb3';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private fileRead = false;
  private polar = false;
  private elements: string[] = [];
  /** element index per atom type (1-based); -1 for NULL. */
  private map = new Int32Array(0);
  private params: Comb3Param[] = [];
  private elem3param = new Int32Array(0);
  private nel = 0;
  private cutmin = 0;
  private cutmax = 0;
  private lib: Comb3Lib | null = null;
  private tables: Comb3Tables | null = null;
  private readonly kernel = new Comb3Kernel();
  private readonly rt = new Comb3RadTor(this.kernel);
  private readonly coul = new Comb3Charge(this.kernel);
  /** fac11e of potal_calc (Coulomb self-energy shift), set by compute and read by the charge forces. */
  private fac11e = 0;
  /** bbij[i][nj] of the last compute (bond order of the pair, read by the charge forces), stride bbStride. */
  private bbij = new Float64Array(0);
  private bbStride = 0;
  /** The context of the last compute: the charge forces read its neighbor lists and positions. */
  private lastPc: PairCompute | null = null;
  private readonly tp: TriPoint = { mr1: 0, mr2: 0, mr3: 0, sr1: 0, sr2: 0, sr3: 0 };
  private readonly vdwOut = { eng: 0, fforce: 0 };
  private readonly repOut = { fforce: 0, eng: 0 };
  private readonly faOut = { eng: 0, force: 0 };
  private readonly p6Out = { eng: 0, force: 0 };
  private readonly bijOut = new Float64Array(6);
  private readonly bijOut2 = new Float64Array(6);
  private readonly prefIJ = new Float64Array(5);
  private readonly prefJI = new Float64Array(5);
  private readonly fi = v3();
  private readonly fj = v3();
  private readonly fk = v3();
  private readonly fl = v3();
  private readonly fip = v3();
  private readonly fjp = v3();
  private readonly fkp = v3();
  private readonly flp = v3();
  private readonly delrj = v3();
  private readonly delrk = v3();
  private readonly delrl = v3();
  private readonly delrm = v3();
  private readonly delrp = v3();
  private readonly vecJI = v3();
  /** Short neighbor lists of the owned atoms (Short_neigh): sht[shtStart[i] .. shtStart[i+1]). */
  private shtStart = new Int32Array(0);
  private shtList = new Int32Array(0);
  private nco = new Float64Array(0);
  private xcc = new Float64Array(0);
  private xch = new Float64Array(0);
  private xco = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    if (args[0] !== 'polar_on' && args[0] !== 'polar_off') throw new StyleError('Illegal pair_style command');
    if (args[0] === 'polar_on') {
      throw new StyleError('pair_style comb3 polar_on (the polarization dipole terms) is not supported by the browser engine yet');
    }
    this.polar = false;
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 4) throw new StyleError('usage: pair_coeff * * filename elem1 ... elemN (style comb3)');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style comb3 must be * *');
    const filename = args[2];
    const names = args.slice(3);
    if (names.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style comb3 needs one element name per atom type (${this.ntypes}), got ${names.length}`);
    }
    const elements: string[] = [];
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const name = names[t - 1];
      if (name === 'NULL') throw new StyleError('pair_style comb3 does not support NULL in pair_coeff (every atom type needs an element)');
      let e = elements.indexOf(name);
      if (e < 0) { e = elements.length; elements.push(name); }
      map[t] = e;
    }
    // pair_comb3.cpp coeff: the library file is read when an element is C (cflag)
    this.lib = names.includes('C') ? readComb3Lib(ctx.readFile('lib.comb3', potentialFile('comb3'))) : null;
    const entries = readComb3Entries(ctx.readFile(filename, potentialFile('COMB3')), elements);
    const sp = setupComb3Params(entries, elements.length);
    this.fileRead = true;
    this.elements = elements;
    this.nel = elements.length;
    this.map = map;
    this.params = entries;
    this.elem3param = sp.elem3param;
    this.cutmin = sp.cutmin;
    this.cutmax = sp.cutmax;
    this.tables = comb3Tables(entries, sp.elem3param, this.nel, map, this.cutmax);
    ctx.log('Pair COMB3:\n  generating Coulomb integral lookup table ...');
  }

  initStyle(): void {
    if (!this.fileRead) throw new StyleError('pair_style comb3 needs a pair_coeff command with a COMB3 potential file');
    if (this.shift || this.tail) throw new StyleError('pair_style comb3 does not support the pair_modify shift and tail options');
  }

  initOne(): number {
    return this.cutmax;
  }

  /** Element triplet parameter index (elem3param[i][j][k]). */
  private p3(i: number, j: number, k: number): number {
    return this.elem3param[(i * this.nel + j) * this.nel + k];
  }

  /** Short_neigh: owned atoms' neighbors within cutmin, and their partial coordinations (NCo, xcctmp, ...). */
  private buildShortNeigh(pc: PairCompute, inum: number): void {
    const list = pc.full!;
    const x = pc.x, type = pc.type, owner = pc.nb.owner;
    const nlocal = pc.nlocal, nall = pc.nall;
    if (this.nco.length < nall) {
      this.nco = new Float64Array(nall);
      this.xcc = new Float64Array(nall);
      this.xch = new Float64Array(nall);
      this.xco = new Float64Array(nall);
    }
    if (this.shtStart.length < nlocal + 1) this.shtStart = new Int32Array(nlocal + 1);
    const out: number[] = [];
    const ownNco = new Float64Array(nlocal), ownCc = new Float64Array(nlocal), ownCh = new Float64Array(nlocal), ownCo = new Float64Array(nlocal);
    for (let i = 0; i < nlocal; i++) {
      this.shtStart[i] = out.length;
      if (i >= inum) continue;
      const itype = this.map[type[i]];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & NEIGHMASK;
        const dx = xtmp - x[3 * j], dy = ytmp - x[3 * j + 1], dz = ztmp - x[3 * j + 2];
        const rsq1 = dx * dx + dy * dy + dz * dz;
        const jtype = this.map[type[j]];
        const p = this.params[this.p3(itype, jtype, jtype)];
        if (rsq1 > this.cutmin) continue;
        out.push(j);
        const rr1 = Math.sqrt(rsq1);
        const c = this.kernel.fc(rr1, p) * p.pcross;
        ownNco[i] += c;
        if (p.jelementgp === 1) ownCc[i] += c;
        if (p.jelementgp === 2) ownCh[i] += c;
        if (p.jelementgp === 3) ownCo[i] += c;
      }
    }
    this.shtStart[nlocal] = out.length;
    this.shtList = Int32Array.from(out);
    // per-atom values of ghosts are their owners' (comm->forward_comm of pack_flag 2)
    for (let j = 0; j < nall; j++) {
      const o = owner[j];
      this.nco[j] = ownNco[o];
      this.xcc[j] = ownCc[o];
      this.xch[j] = ownCh[o];
      this.xco[j] = ownCo[o];
    }
  }

  /** The short neighbor list of atom j (owned, or a ghost translated from its owner's list). */
  private shortOf(pc: PairCompute, j: number, keyOf: Map<number, number>): Int32Array {
    if (j < pc.nlocal) return this.shtList.subarray(this.shtStart[j], this.shtStart[j + 1]);
    const cache = this.ghostCache.get(j);
    if (cache) return cache;
    const owner = pc.nb.owner, gimage = pc.nb.gimage;
    const o = owner[j];
    const ox = gimage[3 * j], oy = gimage[3 * j + 1], oz = gimage[3 * j + 2];
    const own = this.shtList.subarray(this.shtStart[o], this.shtStart[o + 1]);
    const out: number[] = [];
    for (let k = 0; k < own.length; k++) {
      const l = own[k];
      const key = atomKey(owner[l], gimage[3 * l] + ox, gimage[3 * l + 1] + oy, gimage[3 * l + 2] + oz);
      const m = keyOf.get(key);
      if (m !== undefined) out.push(m);
    }
    const arr = Int32Array.from(out);
    this.ghostCache.set(j, arr);
    return arr;
  }

  private ghostCache = new Map<number, Int32Array>();

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style comb3 needs a full neighbor list');
    this.lastPc = pc;
    if (!this.tables) throw new StyleError('pair_style comb3 needs a pair_coeff command first');
    const nall = pc.nall, nlocal = pc.nlocal;
    const x = pc.x, f = pc.f, q = pc.q, type = pc.type;
    const owner = pc.nb.owner, gimage = pc.nb.gimage;
    const ids = pc.s.id;
    const tb = this.tables;
    const kernel = this.kernel;
    const rt = this.rt;
    kernel.lib = this.lib;
    if (this.lib) {
      kernel.maxx = this.lib.maxx; kernel.maxy = this.lib.maxy; kernel.maxz = this.lib.maxz;
    }
    const dot = (a: Float64Array, b: Float64Array): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    this.ghostCache.clear();
    const inum = list.inum;
    this.buildShortNeigh(pc, inum);
    // bbij storage: one row per owned atom, nj = 1 .. (short-list length of the atom)
    let maxSht = 1;
    for (let i = 0; i < nlocal; i++) maxSht = Math.max(maxSht, this.shtStart[i + 1] - this.shtStart[i]);
    this.bbStride = maxSht + 1;
    if (this.bbij.length < nlocal * this.bbStride) this.bbij = new Float64Array(nlocal * this.bbStride);
    const coul = this.coul;
    coul.tb = tb;
    coul.esucon = pc.s.units.qqr2e;
    const potal = potalFactors(this.cutmax, coul.esucon);
    this.fac11e = potal.fac11e;
    const fac11 = potal.fac11;
    const fac11e = potal.fac11e;
    kernel.xcc = this.xcc; kernel.xch = this.xch; kernel.xco = this.xco;
    // key map of all owned and ghost atoms (owner and image offsets), for the ghost short lists
    const keyOf = new Map<number, number>();
    for (let j = 0; j < nall; j++) keyOf.set(atomKey(owner[j], gimage[3 * j], gimage[3 * j + 1], gimage[3 * j + 2]), j);
    const nco = this.nco;
    const acc = pc.acc;
    const eatom = pc.eatom, vatom = pc.vatom;
    const tally = (i: number, j: number, e: number): void => {
      acc.evdwl += e;
      if (eatom) { eatom[i] += 0.5 * e; eatom[j] += 0.5 * e; }
    };
    const tallyQ = (i: number, j: number, e: number): void => {
      acc.ecoul += e;
      if (eatom) { eatom[i] += 0.5 * e; eatom[j] += 0.5 * e; }
    };
    const tagOf = (j: number): number => ids[owner[j]];
    const delrj = this.delrj, delrk = this.delrk, delrl = this.delrl, vecJI = this.vecJI;
    const fi = this.fi, fj = this.fj, fk = this.fk, fl = this.fl;
    const fip = this.fip, fjp = this.fjp, fkp = this.fkp, flp = this.flp;
    const tp = this.tp;
    const bijO = this.bijOut, bijP = this.bijOut2;
    const lcutOf = (ij: number): number => this.params[ij].lcutsq;

    for (let i = 0; i < inum; i++) {
      const itag = ids[i];
      const itype = this.map[type[i]];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const iq = q[i];
      let nj = 0;
      const iparamI = this.p3(itype, itype, itype);
      // self energy, only on i atom
      const yaself = kernel.self(this.params[iparamI], iq);
      acc.ecoul += yaself;
      if (eatom) eatom[i] += yaself;

      // two-body interactions: van der Waals (long range, lcut) and repulsion (cut)
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & NEIGHMASK;
        const jtag = tagOf(j);
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < ztmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }
        const jtype = this.map[type[j]];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const iparamIJ = this.p3(itype, jtype, jtype);
        const iparamJI = this.p3(jtype, itype, itype);
        if (rsq > lcutOf(iparamIJ)) continue;
        const inty = tb.intype[itype * this.nel + jtype];
        triPoint(rsq, this.cutmax, tp);
        // Q-independent van der Waals
        const vdw = vdwaals(tb, inty, tp, Math.sqrt(rsq));
        const fpairV = vdw.fforce;
        f[3 * i] += delx * fpairV; f[3 * i + 1] += dely * fpairV; f[3 * i + 2] += delz * fpairV;
        f[3 * j] -= delx * fpairV; f[3 * j + 1] -= dely * fpairV; f[3 * j + 2] -= delz * fpairV;
        tally(i, j, vdw.eng);
        if (vatom) vTally2(vatom, i, j, fpairV, delx, dely, delz);
        // Coulomb: direct 1/r term and the field correction (polarization is off)
        const dq = coul.direct(this.params[iparamIJ], this.params[iparamJI], tp, rsq, iq, q[j], fac11, fac11e,
          kernel.xco[i], kernel.xco[j]);
        const fld = coul.field(this.params[iparamIJ], this.params[iparamJI], rsq, iq, q[j]);
        const fvionij = dq.pd + fld.f;
        f[3 * i] += delx * fvionij; f[3 * i + 1] += dely * fvionij; f[3 * i + 2] += delz * fvionij;
        f[3 * j] -= delx * fvionij; f[3 * j + 1] -= dely * fvionij; f[3 * j + 2] -= delz * fvionij;
        tallyQ(i, j, dq.pot + fld.eng);
        if (vatom) vTally2(vatom, i, j, fvionij, delx, dely, delz);
        if (rsq > this.params[iparamIJ].cutsq) continue;
        kernel.repulsive(this.params[iparamIJ], this.params[iparamJI], rsq, iq, q[j], this.repOut);
        const fpair = this.repOut.fforce, evdwl = this.repOut.eng;
        f[3 * i] += delx * fpair; f[3 * i + 1] += dely * fpair; f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair; f[3 * j + 1] -= dely * fpair; f[3 * j + 2] -= delz * fpair;
        tally(i, j, evdwl);
        if (vatom) vTally2(vatom, i, j, fpair, delx, dely, delz);
      }

      // many-body interactions over the short neighbor list of i
      const xcn = nco[i];
      const sj = this.shtList.subarray(this.shtStart[i], this.shtStart[i + 1]);
      for (let jj = 0; jj < sj.length; jj++) {
        const j = sj[jj];
        const jtag = tagOf(j);
        if (jtag <= itag) continue;
        const ycn = nco[j];
        const jtype = this.map[type[j]];
        const iparamIJ = this.p3(itype, jtype, jtype);
        const iparamJI = this.p3(jtype, itype, itype);
        delrj[0] = x[3 * j] - xtmp; delrj[1] = x[3 * j + 1] - ytmp; delrj[2] = x[3 * j + 2] - ztmp;
        const rsq1 = delrj[0] * delrj[0] + delrj[1] * delrj[1] + delrj[2] * delrj[2];
        if (rsq1 > this.params[iparamIJ].cutsq) continue;
        nj++;
        const jq = q[j];
        // bond order zeta of the i-j bond: loop over the other short neighbors k of i
        const sl = this.shortOf(pc, j, keyOf);
        let zetaIJ = 0.0;
        let kconjug = 0.0;
        let bbtor = 0.0;
        for (let kk = 0; kk < sj.length; kk++) {
          const k = sj[kk];
          if (j === k) continue;
          const ktype = this.map[type[k]];
          const iparamIJK = this.p3(itype, jtype, ktype);
          const iparamIK = this.p3(itype, ktype, ktype);
          delrk[0] = x[3 * k] - xtmp; delrk[1] = x[3 * k + 1] - ytmp; delrk[2] = x[3 * k + 2] - ztmp;
          const rsq2 = delrk[0] * delrk[0] + delrk[1] * delrk[1] + delrk[2] * delrk[2];
          if (rsq2 > this.params[iparamIK].cutsq) continue;
          zetaIJ += kernel.zeta(this.params[iparamIJK], this.params[iparamIK], rsq1, rsq2, delrj, delrk, xcn);
          // radical initialization: apply only to CC, CO and OC bonds
          if (this.params[iparamIJ].rad_flag > 0 && this.params[iparamIK].ielementgp === 1 && this.params[iparamIK].jelementgp === 1) {
            kconjug += rt.radInit(rsq2, this.params[this.p3(ktype, itype, itype)], nco[k]);
          }
          // torsion: i-j-k-l, all C-C bonds
          if (this.params[iparamIJ].tor_flag !== 0) {
            let srmu = dot(delrj, delrk) / Math.sqrt(rsq1 * rsq2);
            srmu = Math.sqrt(1.0 - srmu * srmu);
            if (srmu > 0.1) {
              for (let ll = 0; ll < sl.length; ll++) {
                const l = sl[ll];
                if (l === i || l === j || l === k) continue;
                const ltype = this.map[type[l]];
                delrl[0] = x[3 * l] - x[3 * j]; delrl[1] = x[3 * l + 1] - x[3 * j + 1]; delrl[2] = x[3 * l + 2] - x[3 * j + 2];
                const rsq3 = delrl[0] * delrl[0] + delrl[1] * delrl[1] + delrl[2] * delrl[2];
                const iparamJL = this.p3(jtype, ltype, ltype);
                if (rsq3 > this.params[iparamJL].cutsq) continue;
                const iparamIKL = this.p3(itype, ktype, ltype);
                bbtor += rt.bbtor1(this.params[iparamIJ].tor_flag, this.params[iparamIKL], this.params[iparamJL],
                  rsq1, rsq2, rsq3, delrj, delrk, delrl, srmu);
              }
            }
          }
        }
        // bond order zeta of the j-i bond: loop over the short neighbors l of j
        let zetaJI = 0.0;
        let lconjug = 0.0;
        for (let ll = 0; ll < sl.length; ll++) {
          const l = sl[ll];
          if (l === i) continue;
          const ltype = this.map[type[l]];
          const iparamJIL = this.p3(jtype, itype, ltype);
          const iparamJL = this.p3(jtype, ltype, ltype);
          delrl[0] = x[3 * l] - x[3 * j]; delrl[1] = x[3 * l + 1] - x[3 * j + 1]; delrl[2] = x[3 * l + 2] - x[3 * j + 2];
          const rsq2 = delrl[0] * delrl[0] + delrl[1] * delrl[1] + delrl[2] * delrl[2];
          if (rsq2 > this.params[iparamJL].cutsq) continue;
          vecJI[0] = -delrj[0]; vecJI[1] = -delrj[1]; vecJI[2] = -delrj[2];
          zetaJI += kernel.zeta(this.params[iparamJIL], this.params[iparamJL], rsq1, rsq2, vecJI, delrl, ycn);
          // radical initialization of the j-l bond
          if (this.params[iparamJI].rad_flag > 0 && this.params[iparamJL].ielementgp === 1 && this.params[iparamJL].jelementgp === 1) {
            lconjug += rt.radInit(rsq2, this.params[this.p3(ltype, jtype, jtype)], nco[l]);
          }
        }

        // force_zeta: pair energy, bond-order prefactors, and the radical and torsion parts
        const r = Math.sqrt(rsq1);
        const pIJ = this.params[iparamIJ], pJI = this.params[iparamJI];
        kernel.fa(r, pIJ, pJI, iq, jq, this.faOut);
        const attEng = this.faOut.eng, attForce = this.faOut.force;
        kernel.bijD(zetaIJ, pIJ, r, i, xcn, bijO);
        kernel.bijD(zetaJI, pJI, r, j, ycn, bijP);
        const boij = bijO[0], boji = bijP[0];
        let bij = 0.5 * (boij + boji);
        let pradx = 0.0, prady = 0.0;
        if (pIJ.rad_flag > 0) {
          rt.radCalc(r, pIJ, pJI, kconjug, lconjug, xcn, ycn);
          bij += rt.brad[0];
          pradx = rt.brad[1] * attEng;
          prady = rt.brad[2] * attEng;
          rt.brad[3] = 1.0 * rt.brad[3] * attEng;
        }
        if (pIJ.tor_flag !== 0) {
          rt.torCalc(r, pIJ, pJI, kconjug, lconjug, xcn, ycn);
          bij += rt.btor[0] * bbtor;
          rt.ptorr = attEng * rt.btor[0];
          pradx += 1.0 * rt.btor[1] * bbtor * attEng;
          prady += rt.btor[2] * bbtor * attEng;
          rt.brad[3] += 1.0 * rt.btor[3] * bbtor * attEng;
        }
        this.bbij[i * this.bbStride + nj] = bij;
        let fpair = 1.0 * bij * attForce / r;
        let evdwl = 1.0 * bij * attEng;
        this.prefIJ[0] = -0.5 * attEng * bijO[1];
        this.prefIJ[1] = -0.5 * attEng * bijO[2];
        this.prefIJ[2] = -0.5 * attEng * bijO[3];
        this.prefIJ[3] = -0.5 * attEng * bijO[4];
        this.prefIJ[4] = -0.5 * attEng * bijO[5];
        this.prefJI[0] = -0.5 * attEng * bijP[1];
        this.prefJI[1] = -0.5 * attEng * bijP[2];
        this.prefJI[2] = -0.5 * attEng * bijP[3];
        this.prefJI[3] = -0.5 * attEng * bijP[4];
        this.prefJI[4] = -0.5 * attEng * bijP[5];
        if (pIJ.rad_flag > 0 || pIJ.tor_flag !== 0) {
          this.prefIJ[1] -= pradx;
          this.prefJI[1] -= prady;
        }

        // two-body Legendre-polynomial (p6p) energy of the pair
        this.p6Out.force = 0.0;
        kernel.selfp6p(this.params[iparamIJ], this.params[iparamJI], rsq1, this.p6Out);
        // selfp6p adds its force to a zeroed accumulator, which is then summed into fpair as the C++ does
        fpair += this.p6Out.force;
        evdwl += this.p6Out.eng;
        f[3 * i] += delrj[0] * fpair; f[3 * i + 1] += delrj[1] * fpair; f[3 * i + 2] += delrj[2] * fpair;
        f[3 * j] -= delrj[0] * fpair; f[3 * j + 1] -= delrj[1] * fpair; f[3 * j + 2] -= delrj[2] * fpair;
        tally(i, j, evdwl);
        if (vatom) vTally2(vatom, i, j, -fpair, -delrj[0], -delrj[1], -delrj[2]);

        // attractive term via the loop over k (three-body forces i-j-k)
        for (let kk = 0; kk < sj.length; kk++) {
          const k = sj[kk];
          if (j === k) continue;
          const ktype = this.map[type[k]];
          const iparamIJK = this.p3(itype, jtype, ktype);
          const iparamIKJ = this.p3(itype, ktype, jtype);
          const iparamJIK = this.p3(jtype, itype, ktype);
          const iparamIK = this.p3(itype, ktype, ktype);
          delrk[0] = x[3 * k] - xtmp; delrk[1] = x[3 * k + 1] - ytmp; delrk[2] = x[3 * k + 2] - ztmp;
          const rsq2 = delrk[0] * delrk[0] + delrk[1] * delrk[1] + delrk[2] * delrk[2];
          if (rsq2 > this.params[iparamIK].cutsq) continue;
          kernel.attractive(this.params[iparamIJK], this.params[iparamJIK], this.params[iparamIKJ], this.prefIJ,
            rsq1, rsq2, delrj, delrk, fi, fj, fk, xcn);
          const ep6pIJ = kernel.ep6p(this.params[iparamIJK], this.params[iparamIKJ], rsq1, rsq2, delrj, delrk);
          kernel.fp6p(this.params[iparamIJK], this.params[iparamIKJ], rsq1, rsq2, delrj, delrk, fip, fjp, fkp);
          for (let c = 0; c < 3; c++) {
            fi[c] += fip[c];
            fj[c] += fjp[c];
            fk[c] += fkp[c];
          }
          for (let c = 0; c < 3; c++) {
            f[3 * i + c] += fi[c];
            f[3 * j + c] += fj[c];
            f[3 * k + c] += fk[c];
          }
          tally(i, j, ep6pIJ);
          // torsion: i-j-k-l (applied when the torsion prefactor ptorr of this pair is nonzero)
          if (this.params[iparamIJK].tor_flag !== 0 && Math.abs(rt.ptorr) > 1.0e-8) {
            let srmu = dot(delrj, delrk) / Math.sqrt(rsq1 * rsq2);
            srmu = Math.sqrt(1.0 - srmu * srmu);
            if (srmu > 0.1) {
              for (let ll = 0; ll < sl.length; ll++) {
                const l = sl[ll];
                if (l === i || l === j || l === k) continue;
                const ltype = this.map[type[l]];
                delrl[0] = x[3 * l] - x[3 * j]; delrl[1] = x[3 * l + 1] - x[3 * j + 1]; delrl[2] = x[3 * l + 2] - x[3 * j + 2];
                const rsq3 = delrl[0] * delrl[0] + delrl[1] * delrl[1] + delrl[2] * delrl[2];
                const iparamJL = this.p3(jtype, ltype, ltype);
                if (rsq3 > this.params[iparamJL].cutsq) continue;
                const iparamIKL = this.p3(itype, ktype, ltype);
                rt.torForce(this.params[iparamIJ].tor_flag, this.params[iparamIKL], this.params[iparamJL], srmu,
                  rsq1, rsq2, rsq3, delrj, delrk, delrl);
                for (let c = 0; c < 3; c++) {
                  f[3 * i + c] += rt.fiTor[c];
                  f[3 * j + c] += rt.fjTor[c];
                  f[3 * k + c] += rt.fkTor[c];
                  f[3 * l + c] += rt.flTor[c];
                }
              }
            }
          }
          // radical: k side of the i-j-k angle (CC, CO and OC bonds)
          if (this.params[iparamIJK].rad_flag >= 1 && this.params[iparamIJK].ielementgp === 1 && this.params[iparamIJK].kelementgp === 1) {
            const pKI = this.params[this.p3(ktype, itype, itype)];
            const kradtot = -kernel.fc(Math.sqrt(rsq2), pKI) * pKI.pcross + nco[k];
            rt.radForceik(pKI, rsq2, delrk, kconjug, kradtot);
            for (let c = 0; c < 3; c++) {
              f[3 * i + c] += rt.fiRad[c];
              f[3 * k + c] += rt.fkRad[c];
            }
            if (Math.abs(rt.radtmp) > 1.0e-12) {
              const sm = this.shortOf(pc, k, keyOf);
              for (let mm = 0; mm < sm.length; mm++) {
                const m = sm[mm];
                if (m === k) continue;
                const mtype = this.map[type[m]];
                const delrm = this.delrm;
                delrm[0] = x[3 * m] - x[3 * k]; delrm[1] = x[3 * m + 1] - x[3 * k + 1]; delrm[2] = x[3 * m + 2] - x[3 * k + 2];
                const rsq3 = delrm[0] * delrm[0] + delrm[1] * delrm[1] + delrm[2] * delrm[2];
                const pKM = this.params[this.p3(ktype, mtype, mtype)];
                if (rsq3 > pKM.cutsq) continue;
                rt.radForce(pKM, rsq3, delrm, rt.radtmp);
                for (let c = 0; c < 3; c++) {
                  f[3 * k + c] += rt.fjRad[c];
                  f[3 * m + c] += rt.fkRad[c];
                }
              }
            }
          }
          // as native: the i-j-k three-body virial only (the torsion and radical forces have no per-atom tally)
          if (vatom) vTally3(vatom, i, j, k, fj, fk, delrj, delrk);
        }

        // attractive term via the loop over l (three-body forces j-i-l)
        for (let ll = 0; ll < sl.length; ll++) {
          const l = sl[ll];
          if (l === i) continue;
          const ltype = this.map[type[l]];
          const iparamJIL = this.p3(jtype, itype, ltype);
          const iparamJLI = this.p3(jtype, ltype, itype);
          const iparamIJL = this.p3(itype, jtype, ltype);
          const iparamJL = this.p3(jtype, ltype, ltype);
          delrk[0] = x[3 * l] - x[3 * j]; delrk[1] = x[3 * l + 1] - x[3 * j + 1]; delrk[2] = x[3 * l + 2] - x[3 * j + 2];
          const rsq2 = delrk[0] * delrk[0] + delrk[1] * delrk[1] + delrk[2] * delrk[2];
          if (rsq2 > this.params[iparamJL].cutsq) continue;
          vecJI[0] = -delrj[0]; vecJI[1] = -delrj[1]; vecJI[2] = -delrj[2];
          // the C++ call passes (fj, fi, fl) as the force vectors of (j, i, l)
          kernel.attractive(this.params[iparamJIL], this.params[iparamIJL], this.params[iparamJLI], this.prefJI,
            rsq1, rsq2, vecJI, delrk, fj, fi, fl, ycn);
          const ep6pJI = kernel.ep6p(this.params[iparamJIL], this.params[iparamJLI], rsq1, rsq2, vecJI, delrk);
          kernel.fp6p(this.params[iparamJIL], this.params[iparamJLI], rsq1, rsq2, vecJI, delrk, fjp, fip, flp);
          tally(j, i, ep6pJI);
          for (let c = 0; c < 3; c++) {
            fj[c] += fjp[c];
            fi[c] += fip[c];
            fl[c] += flp[c];
          }
          for (let c = 0; c < 3; c++) {
            f[3 * j + c] += fj[c];
            f[3 * i + c] += fi[c];
            f[3 * l + c] += fl[c];
          }
          // radical i-j-l-p: CC, CO and OC bonds
          if (this.params[iparamJIL].rad_flag >= 1 && this.params[iparamJIL].ielementgp === 1 && this.params[iparamJIL].kelementgp === 1) {
            const pLJ = this.params[this.p3(ltype, jtype, jtype)];
            const lradtot = -kernel.fc(Math.sqrt(rsq2), pLJ) * pLJ.pcross + nco[l];
            rt.radForceik(pLJ, rsq2, delrk, lconjug, lradtot);
            for (let c = 0; c < 3; c++) {
              f[3 * j + c] += rt.fiRad[c];
              f[3 * l + c] += rt.fkRad[c];
            }
            if (Math.abs(rt.radtmp) > 1.0e-12) {
              const sp = this.shortOf(pc, l, keyOf);
              for (let pp = 0; pp < sp.length; pp++) {
                const p = sp[pp];
                if (p === l) continue;
                const ptype = this.map[type[p]];
                const delrp = this.delrp;
                delrp[0] = x[3 * p] - x[3 * l]; delrp[1] = x[3 * p + 1] - x[3 * l + 1]; delrp[2] = x[3 * p + 2] - x[3 * l + 2];
                const rsq3 = delrp[0] * delrp[0] + delrp[1] * delrp[1] + delrp[2] * delrp[2];
                const pLP = this.params[this.p3(ltype, ptype, ptype)];
                if (rsq3 > pLP.cutsq) continue;
                rt.radForce(pLP, rsq3, delrp, rt.radtmp);
                for (let c = 0; c < 3; c++) {
                  f[3 * l + c] += rt.fjRad[c];
                  f[3 * p + c] += rt.fkRad[c];
                }
              }
            }
          }
          if (vatom) vTally3(vatom, j, i, l, fi, fl, vecJI, delrk);
        }
      }
    }
  }

  /**
   * combqeq (pair_comb3.cpp): the charge forces of the atoms in the group, written into qf (length nall; ghost
   * entries are summed into their owners); returns the group's total charge force (enegtot). The pair terms use
   * the same tags as the C++ combqeq (the half rule itag >= jtag), not the energy loop's rule.
   */
  chargeForces(qf: Float64Array, isGroup: (i: number) => boolean): number {
    const pc = this.lastPc;
    if (!pc || !pc.full) throw new StyleError('fix qeq/comb: pair_style comb3 has not computed forces yet');
    const tb = this.tables;
    if (!tb) throw new StyleError('pair_style comb3 needs a pair_coeff command first');
    const nb = pc.nb;
    const x = pc.x, type = pc.type, q = nb.qall;
    const nlocal = pc.nlocal, nall = pc.nall;
    const owner = nb.owner, ids = pc.s.id;
    const list = pc.full;
    const kernel = this.kernel;
    const coul = this.coul;
    qf.fill(0);
    const tp = this.tp;
    for (let i = 0; i < nlocal; i++) {
      if (!isGroup(i)) continue;
      const itag = ids[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = this.map[type[i]];
      const iq = q[i];
      const pI = this.params[this.p3(itype, itype, itype)];
      // charge force from the self energy
      let fqi = coul.qfoSelf(pI, iq);
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      // two-body: the C++ combqeq half rule on the tags
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & NEIGHMASK;
        const jtag = ids[owner[j]];
        if (itag >= jtag) continue;
        const jtype = this.map[type[j]];
        const jq = q[j];
        const dx = xtmp - x[3 * j], dy = ytmp - x[3 * j + 1], dz = ztmp - x[3 * j + 2];
        const rsq1 = dx * dx + dy * dy + dz * dz;
        const pIJ = this.params[this.p3(itype, jtype, jtype)];
        const pJI = this.params[this.p3(jtype, itype, itype)];
        if (rsq1 > pIJ.lcutsq) continue;
        triPoint(rsq1, this.cutmax, tp);
        const d = coul.qfoDirect(pIJ, pJI, tp, rsq1, this.fac11e, iq, jq, kernel.xco[i], kernel.xco[j]);
        fqi += d.fqij;
        qf[j] += d.fqji;
        const fld = coul.qfoField(pIJ, pJI, rsq1, iq, jq);
        fqi += fld.fqij;
        qf[j] += fld.fqji;
      }
      // bond-order (short-range) charge forces over the short list of i
      const s0 = this.shtStart[i], s1 = this.shtStart[i + 1];
      let nj = 0;
      for (let jj = s0; jj < s1; jj++) {
        const j = this.shtList[jj];
        const jtag = ids[owner[j]];
        if (itag >= jtag) continue;
        const jtype = this.map[type[j]];
        const jq = q[j];
        const dx = xtmp - x[3 * j], dy = ytmp - x[3 * j + 1], dz = ztmp - x[3 * j + 2];
        const rsq1 = dx * dx + dy * dy + dz * dz;
        const pIJ = this.params[this.p3(itype, jtype, jtype)];
        const pJI = this.params[this.p3(jtype, itype, itype)];
        if (rsq1 >= pIJ.cutsq) continue;
        nj++;
        const r = Math.sqrt(rsq1);
        const sh = coul.qfoShort(pIJ, pJI, rsq1, iq, jq, this.bbij[i * this.bbStride + nj], kernel.fc(r, pIJ));
        fqi += sh.fqij;
        qf[j] += sh.fqji;
      }
      qf[i] += fqi;
    }
    // reverse communication: ghost charge forces go to their owners
    for (let k = nlocal; k < nall; k++) qf[owner[k]] += qf[k];
    let eneg = 0.0;
    for (let i = 0; i < nlocal; i++) if (isGroup(i)) eneg += qf[i];
    return eneg;
  }
}

/** Key of an atom from its owner and image offsets (images are small integers). */
const atomKey = (owner: number, ix: number, iy: number, iz: number): number =>
  ((owner * 1024 + (ix + 512)) * 1024 + (iy + 512)) * 1024 + (iz + 512);
