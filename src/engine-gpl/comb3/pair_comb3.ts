/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb3.cpp and src/MANYBODY/pair_comb3.h (PairComb3: settings, coeff, init_style,
 * init_one, Short_neigh, compute, with the two-body, bond-order and three-body loops), at commit b332f28.
 * Authors credited in the C++ file headers: Tzu-Ray Shan (Sandia) and the COMB3 authors. Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Stage A: fixed charges that are all zero, polar_off, no radical and no torsion terms (the carbon
 * bond-order corrections), no charge-equilibration terms. Anything else throws a StyleError naming the term.
 *
 * Ghost atoms: the C++ builds short neighbor lists for ghost atoms too (sht_first[j] for a ghost j). Here a
 * ghost's short list is the translate of its owner's list: the neighbor l of the owner becomes the atom with the
 * same owner and the image offset of l plus that of the ghost (atoms outside the ghost shell are skipped, as the
 * C++ ghost neighbor list skips them).
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { readComb3Entries, readComb3Lib, setupComb3Params, type Comb3Lib, type Comb3Param } from './params';
import { comb3Tables, triPoint, vdwaals, type Comb3Tables, type TriPoint } from './tables';
import { Comb3Kernel } from './terms';

const v3 = (): Float64Array => new Float64Array(3);

/** PairComb3: charge-optimized many-body potential with the COMB3 bond order (stage A). */
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
  /** Entries with radical or torsion terms; a run that uses one of their element triplets is refused (stage B). */
  private radTorEntries: Comb3Param[] = [];
  private readonly kernel = new Comb3Kernel();
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
  private readonly vecJI = v3();
  /** Short neighbor lists of the owned atoms (Short_neigh): sht[shtStart[i] .. shtStart[i+1]). */
  private shtStart = new Int32Array(0);
  private shtList = new Int32Array(0);
  private nco = new Float64Array(0);
  private xcc = new Float64Array(0);
  private xch = new Float64Array(0);
  private xco = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style comb3 command: usage pair_style comb3 polar_on|polar_off');
    if (args[0] === 'polar_on') {
      throw new StyleError('pair_style comb3 polar_on (the polarization dipole terms) is not supported by the browser engine yet');
    }
    if (args[0] !== 'polar_off') throw new StyleError(`Illegal pair_style comb3 command: unknown setting '${args[0]}'`);
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
    this.lib = names.includes('C') ? readComb3Lib(ctx.readFile('lib.comb3')) : null;
    const entries = readComb3Entries(ctx.readFile(filename), elements);
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
    this.radTorEntries = entries.filter((p) => p.rad_flag !== 0 || p.tor_flag !== 0);
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
    if (!this.tables) throw new StyleError('pair_style comb3 needs a pair_coeff command first');
    if (pc.vatom) throw new StyleError('compute stress/atom with pair_style comb3 is not supported by the browser engine yet');
    // the radical and torsion terms (carbon bond-order corrections) are not ported yet (stage B): refused when an
    // entry with them has all three element types present among the atoms of this run
    if (this.radTorEntries.length) {
      const present = new Uint8Array(this.nel);
      for (let i = 0; i < pc.nlocal; i++) present[this.map[pc.type[i]]] = 1;
      const hit = this.radTorEntries.find((p) => present[p.ielement] && present[p.jelement] && present[p.kelement]);
      if (hit) {
        throw new StyleError(`pair_style comb3: the radical and torsion terms of ${this.elements[hit.ielement]} ${this.elements[hit.jelement]} ${this.elements[hit.kelement]} (carbon bond-order corrections) are not ported yet (stage B)`);
      }
    }
    const nall = pc.nall, nlocal = pc.nlocal;
    const x = pc.x, f = pc.f, q = pc.q, type = pc.type;
    const owner = pc.nb.owner, gimage = pc.nb.gimage;
    const ids = pc.s.id;
    for (let i = 0; i < nlocal; i++) {
      if (q[i] !== 0) {
        throw new StyleError('pair_style comb3: nonzero charges need the Coulomb and charge-dependent terms, which are not ported yet (stage C)');
      }
    }
    const tb = this.tables;
    const kernel = this.kernel;
    kernel.lib = this.lib;
    if (this.lib) {
      kernel.maxx = this.lib.maxx; kernel.maxy = this.lib.maxy; kernel.maxz = this.lib.maxz;
    }
    this.ghostCache.clear();
    const inum = list.inum;
    this.buildShortNeigh(pc, inum);
    kernel.xcc = this.xcc; kernel.xch = this.xch; kernel.xco = this.xco;
    // key map of all owned and ghost atoms (owner and image offsets), for the ghost short lists
    const keyOf = new Map<number, number>();
    for (let j = 0; j < nall; j++) keyOf.set(atomKey(owner[j], gimage[3 * j], gimage[3 * j + 1], gimage[3 * j + 2]), j);
    const nco = this.nco;
    const acc = pc.acc;
    const eatom = pc.eatom;
    const tally = (i: number, j: number, e: number): void => {
      acc.evdwl += e;
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
        // the Coulomb, field and polarization terms vanish for zero charges (stage A)
        if (rsq > this.params[iparamIJ].cutsq) continue;
        kernel.repulsive(this.params[iparamIJ], this.params[iparamJI], rsq, iq, q[j], this.repOut);
        const fpair = this.repOut.fforce, evdwl = this.repOut.eng;
        f[3 * i] += delx * fpair; f[3 * i + 1] += dely * fpair; f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair; f[3 * j + 1] -= dely * fpair; f[3 * j + 2] -= delz * fpair;
        tally(i, j, evdwl);
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
        const jq = q[j];
        // bond order zeta of the i-j bond: loop over the other short neighbors k of i
        let zetaIJ = 0.0;
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
        }
        // bond order zeta of the j-i bond: loop over the short neighbors l of j
        const sl = this.shortOf(pc, j, keyOf);
        let zetaJI = 0.0;
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
        }

        // force_zeta (without the radical and torsion parts): pair energy and bond-order prefactors
        const r = Math.sqrt(rsq1);
        kernel.fa(r, this.params[iparamIJ], this.params[iparamJI], iq, jq, this.faOut);
        const attEng = this.faOut.eng, attForce = this.faOut.force;
        kernel.bijD(zetaIJ, this.params[iparamIJ], r, i, xcn, bijO);
        kernel.bijD(zetaJI, this.params[iparamJI], r, j, ycn, bijP);
        const boij = bijO[0], boji = bijP[0];
        const bij = 0.5 * (boij + boji);
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

        // two-body Legendre-polynomial (p6p) energy of the pair
        this.p6Out.force = 0.0;
        kernel.selfp6p(this.params[iparamIJ], this.params[iparamJI], rsq1, this.p6Out);
        // selfp6p adds its force to a zeroed accumulator, which is then summed into fpair as the C++ does
        fpair += this.p6Out.force;
        evdwl += this.p6Out.eng;
        f[3 * i] += delrj[0] * fpair; f[3 * i + 1] += delrj[1] * fpair; f[3 * i + 2] += delrj[2] * fpair;
        f[3 * j] -= delrj[0] * fpair; f[3 * j + 1] -= delrj[1] * fpair; f[3 * j + 2] -= delrj[2] * fpair;
        tally(i, j, evdwl);

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
        }
      }
    }
  }
}

/** Key of an atom from its owner and image offsets (images are small integers). */
const atomKey = (owner: number, ix: number, iy: number, iz: number): number =>
  ((owner * 1024 + (ix + 512)) * 1024 + (iy + 512)) * 1024 + (iz + 512);
