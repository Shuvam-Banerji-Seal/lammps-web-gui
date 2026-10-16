/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb.cpp and src/MANYBODY/pair_comb.h (PairComb: compute, coeff,
 * sm_table, potal_calc, tri_point, direct, field, self, repulsive, zeta, elp, flp, force_zeta,
 * attractive, comb_zetaterm_d, costheta_d, Over_cor, Short_neigh, and the charge-force helpers
 * qfo_self, qfo_direct, qfo_field, qfo_short used by fix qeq/comb), at commit b332f28. Authors credited
 * in the C++ file headers: Tzu-Ray Shan (U Florida, then Sandia). Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Neighbor lists: the engine's full neighbor list (owned and ghost atoms, with the owner of each
 * ghost) replaces the LAMMPS full list with ghost atoms. The half-pair rule of the two-body Coulomb
 * and repulsion terms uses the atom IDs of the owners (LAMMPS uses atom->tag of the ghosts).
 * Per-atom energy and per-atom virial (compute stress/atom) are tallied as the native ev_tally and v_tally3 calls do
 * (the global virial stays the fdotr sum, as in the engine's other styles).
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { erfcExact } from '../../engine/force/erfc';
import {
  combBijD, combBij, combBsi, combDi, combDj, combFc, combFc2, combFc2D, combFc3, combFc3D, combFcD,
  combGijk, combGijkD, cube, powint, readCombEntries, setupCombParams, square,
  type CombParam,
} from './params';
import { cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

const MY_PIS = Math.sqrt(Math.PI);
const ALF = 0.20;
const NSM = 25000;

/** Scratch vector (3 components). */
type V3 = Float64Array;
const v3 = (): V3 => new Float64Array(3);

const dot3 = (a: V3 | readonly number[], b: V3 | readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** The Streitz-Mintmire 1/r tables (sm_table). */
interface Tables {
  intype: Int32Array;
  nntypes: number;
  fafb: Float64Array;
  dfafb: Float64Array;
  erpaw: Float64Array;
}

/** Interpolation indices and weights of tri_point. */
interface TriPoint { mr1: number; mr2: number; mr3: number; sr1: number; sr2: number; sr3: number }

/** PairComb: charge-optimized many-body (COMB) potential. */
export class PairComb extends Pair {
  readonly name = 'comb';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected fileName = '';
  protected fileRead = false;
  protected elements: string[] = [];
  /** element index per atom type (1-based); -1 for NULL. */
  protected map = new Int32Array(0);
  protected params: CombParam[] = [];
  protected elem3param = new Int32Array(0);
  protected cutmax = 0;
  protected cutmin = 0;
  protected corFlag = 0;
  protected tables: Tables | null = null;
  /** cuo_flag: the over-coordination flag of the last compute (persists between computes as in LAMMPS). */
  protected cuoFlag = 0;
  /** bbij[i][nj] of the last compute, read by the charge forces of fix qeq/comb (stride bbStride). */
  protected bbij = new Float64Array(0);
  protected bbStride = 0;
  /** Coordination count NCo of the last compute (owned atoms). */
  protected NCo = new Int32Array(0);
  /** Context of the last compute: fix qeq/comb reads the charge forces of the same neighbor lists. */
  protected lastPc: PairCompute | null = null;
  /** Reused per-pair results (the pair loops run millions of times; no allocation per pair). */
  protected tp: TriPoint = { mr1: 0, mr2: 0, mr3: 0, sr1: 0, sr2: 0, sr3: 0 };
  protected dOut = { pot: 0, pd: 0 };
  protected fOut = { v: 0, f: 0 };
  protected qfOut = { fqij: 0, fqjj: 0 };
  /** Short-range (cutmin) neighbor list of the owned atoms (Short_neigh). */
  protected shtStart = new Int32Array(0);
  protected shtList = new Int32Array(0);

  settings(args: string[]): void {
    if (args.length > 0) throw new StyleError(`Illegal pair_style comb command: unexpected '${args[0]}' (pair_style comb takes no arguments)`);
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError('usage: pair_coeff * * filename elem1 ... elemN (style comb)');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style comb must be * *');
    const filename = args[2];
    const names = args.slice(3);
    if (names.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style comb needs one element name per atom type (${this.ntypes}), got ${names.length}`);
    }
    // map_element2type: unique element names in the order of the atom types
    const elements: string[] = [];
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const name = names[t - 1];
      if (name === 'NULL') throw new StyleError('pair_style comb does not support NULL in pair_coeff (every atom type needs an element)');
      let e = elements.indexOf(name);
      if (e < 0) { e = elements.length; elements.push(name); }
      map[t] = e;
    }
    const entries = readCombEntries(ctx.readFile(filename), elements);
    const sp = setupCombParams(entries, elements);
    this.fileName = filename;
    this.fileRead = true;
    this.elements = elements;
    this.map = map;
    this.params = entries;
    this.elem3param = sp.elem3param;
    this.cutmax = sp.cutmax;
    this.cutmin = sp.cutmin;
    this.corFlag = sp.corFlag;
    this.tables = this.smTable();
    ctx.log('Pair COMB:\n  generating Coulomb integral lookup table ...');
    ctx.log(this.corFlag ? '  will apply over-coordination correction ...' : '  will not apply over-coordination correction ...');
  }

  initStyle(): void {
    if (!this.fileRead) throw new StyleError('pair_style comb needs a pair_coeff command with a COMB potential file');
    if (this.shift || this.tail) throw new StyleError('pair_style comb does not support the pair_modify shift and tail options');
  }

  initOne(): number {
    return this.cutmax;
  }

  /** Element triplet parameter index (elem3param[i][j][k]). */
  protected p3(i: number, j: number, k: number): number {
    const nel = this.elements.length;
    return this.elem3param[(i * nel + j) * nel + k];
  }

  /** sm_table: direct 1/r Slater-orbital overlap tables and the erfc table. */
  protected smTable(): Tables {
    const n = this.ntypes;
    const dra = 0.001;
    const drin = 0.1;
    const rc = this.cutmax;
    const alf = ALF;
    const nntypes = Math.trunc(((n + 1) * n) / 2);
    const ncoul = Math.trunc((rc - drin) / dra) + 1;
    // intype: 0-0 = 0, 1-1 = 1, 0-1 = 1-0 = 2 (numbering as in sm_table)
    const intype = new Int32Array(n * n);
    let m = 0, k = n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (j === i) { intype[i * n + j] = m; m += 1; } else if (j > i) { intype[i * n + j] = k; k += 1; } else intype[i * n + j] = intype[j * n + i];
      }
    }
    // a few rows of slack: tri_point indices reach ncoul at the top of the cutoff
    const rows = ncoul + 4;
    const fafb = new Float64Array(rows * nntypes);
    const dfafb = new Float64Array(rows * nntypes);
    for (let i = 0; i < n; i++) {
      if (this.map[i + 1] < 0) continue;
      for (let j = 0; j < n; j++) {
        if (this.map[j + 1] < 0) continue;
        let r = drin;
        if (j === i) {
          const itype = this.elementIdx(i);
          const inty = intype[itype * n + itype];
          const z = this.params[this.p3(itype, itype, itype)].esm1;
          const zrc = z * rc;
          const exp2ersh = libmExp(-2.0 * zrc);
          const fafash = -exp2ersh * (1.0 / rc + z * (11.0 / 8.0 + 3.0 / 4.0 * zrc + zrc * zrc / 6.0));
          const dfafash = exp2ersh * (1.0 / (rc * rc) + 2.0 * z / rc + z * z * (2.0 + 7.0 / 6.0 * zrc + zrc * zrc / 3.0));
          for (let kk = 0; kk < ncoul; kk++) {
            const zr = z * r;
            const exp2er = libmExp(-2.0 * zr);
            const F1 = -exp2er * (1.0 / r + z * (11.0 / 8.0 + 3.0 / 4.0 * zr + zr * zr / 6.0));
            const dF1 = exp2er * (1.0 / (r * r) + 2.0 * z / r + z * z * (2.0 + 7.0 / 6.0 * zr + zr * zr / 3.0));
            fafb[kk * nntypes + inty] = F1 - fafash - (r - rc) * dfafash;
            dfafb[kk * nntypes + inty] = dF1 - dfafash;
            r += dra;
          }
        } else {
          const itype = this.elementIdx(i);
          const jtype = this.elementIdx(j);
          const inty = intype[itype * n + jtype];
          const ea = this.params[this.p3(itype, jtype, jtype)].esm1;
          const ea3 = ea * ea * ea;
          const eb = this.params[this.p3(jtype, itype, itype)].esm1;
          const eb3 = eb * eb * eb;
          const E1 = ea * eb3 * eb / ((ea + eb) * (ea + eb) * (ea - eb) * (ea - eb));
          const E2 = eb * ea3 * ea / ((ea + eb) * (ea + eb) * (eb - ea) * (eb - ea));
          const E3 = (3.0 * ea * ea * eb3 * eb - eb3 * eb3) /
            ((ea + eb) * (ea + eb) * (ea + eb) * (ea - eb) * (ea - eb) * (ea - eb));
          const E4 = (3.0 * eb * eb * ea3 * ea - ea3 * ea3) /
            ((ea + eb) * (ea + eb) * (ea + eb) * (eb - ea) * (eb - ea) * (eb - ea));
          const exp2earsh = libmExp(-2.0 * ea * rc);
          const exp2ebrsh = libmExp(-2.0 * eb * rc);
          const fafbsh = -exp2earsh * (E1 + E3 / rc) - exp2ebrsh * (E2 + E4 / rc);
          const dfafbsh = exp2earsh * (2.0 * ea * (E1 + E3 / rc) + E3 / (rc * rc)) +
            exp2ebrsh * (2.0 * eb * (E2 + E4 / rc) + E4 / (rc * rc));
          for (let kk = 0; kk < ncoul; kk++) {
            const exp2ear = libmExp(-2.0 * ea * r);
            const exp2ebr = libmExp(-2.0 * eb * r);
            fafb[kk * nntypes + inty] = -exp2ear * (E1 + E3 / r) - exp2ebr * (E2 + E4 / r) - fafbsh - (r - rc) * dfafbsh;
            dfafb[kk * nntypes + inty] = exp2ear * (2.0 * ea * (E1 + E3 / r) + E3 / (r * r)) +
              exp2ebr * (2.0 * eb * (E2 + E4 / r) + E4 / (r * r)) - dfafbsh;
            r += dra;
          }
        }
      }
    }
    const erpaw = new Float64Array(NSM * 2);
    for (let i = 0; i < NSM; i++) {
      const r = dra * i + drin;
      erpaw[2 * i] = erfcExact(r * alf);
      erpaw[2 * i + 1] = libmExp(-r * r * alf * alf);
    }
    return { intype, nntypes, fafb, dfafb, erpaw };
  }

  /** potal_calc: self-energy constants (potal, fac11, fac11e) for the units' qqr2e. */
  protected potalCalc(esucon: number): { potal: number; fac11: number; fac11e: number } {
    const alf = ALF;
    let rcoul = 0.0;
    for (const p of this.params) if (p.lcut > rcoul) rcoul = p.lcut;
    const fac11 = (erfcExact(rcoul * alf) / rcoul / rcoul + 2.0 * alf / MY_PIS * libmExp(-alf * alf * rcoul * rcoul) / rcoul) * esucon / rcoul;
    const fac11e = (erfcExact(rcoul * alf) / rcoul) * esucon;
    const potal = -(alf / MY_PIS * esucon + fac11e * 0.5);
    return { potal, fac11, fac11e };
  }

  /** Element index of atom type i+1 (0-based i). */
  protected elementIdx(i: number): number { return this.map[i + 1]; }

  /** qqr2e of the units in use (force->qqr2e); set per compute, used by direct() and qfo_direct(). */
  protected esucon = 1;

  /** One compute call: the LAMMPS PairComb::compute loop with the engine's full list. */
  compute(pc: PairCompute): void {
    const nmask = NEIGHMASK;
    const list = pc.full;
    if (!list) throw new Error('pair style comb needs a full neighbor list');
    if (!this.tables) throw new StyleError('pair_style comb needs a pair_coeff command first');
    const tb = this.tables;
    const nb = pc.nb;
    const x = pc.x, f = pc.f, q = pc.q, type = pc.type;
    const nlocal = pc.nlocal;
    const owner = nb.owner;
    const ids = pc.s.id;
    const tagOf = (j: number): number => ids[owner[j]];
    const acc = pc.acc;
    const eatom = pc.eatom;
    this.esucon = pc.s.units.qqr2e;
    this.cuoFlag = 0;
    this.lastPc = pc;

    // Short_neigh: the neighbors within cutmin of each owned atom (order of the full list).
    this.buildShortNeigh(pc, nlocal, list.inum);
    const sht = this.shtList, shtStart = this.shtStart;

    // Stride of bbij: the largest short list (bbij[i][nj] with nj counting from 1).
    let maxSht = 1;
    for (let i = 0; i < nlocal; i++) maxSht = Math.max(maxSht, shtStart[i + 1] - shtStart[i]);
    if (this.bbStride < maxSht + 1 || this.bbij.length < nlocal * (maxSht + 1)) {
      this.bbStride = maxSht + 1;
      this.bbij = new Float64Array(nlocal * this.bbStride);
    }
    if (this.NCo.length < nlocal) this.NCo = new Int32Array(nlocal);

    const { potal, fac11, fac11e } = this.potalCalc(this.esucon);
    const delr1 = v3(), delr2 = v3();
    const fi = v3(), fj = v3(), fk = v3(), filp = v3(), fjlp = v3(), fklp = v3();
    // Eov and Fov persist across atoms (PairComb::compute keeps them as locals).
    let Eov = 0.0, Fov = 0.0;
    const elemOf = this.map;

    for (let i = 0; i < nlocal; i++) {
      const itag = ids[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = elemOf[type[i]];
      const iq = q[i];
      this.NCo[i] = 0;
      let nj = 0;
      const iparamI = this.p3(itype, itype, itype);
      const pI = this.params[iparamI];

      // self energy, only on i atom
      const yaself = this.self(pI, iq, potal);
      acc.evdwl += yaself;
      if (eatom) eatom[i] += yaself;

      // two-body interactions (long-range Coulomb and short-range repulsion)
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & nmask;
        const jtag = tagOf(j);
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < x[3 * i + 2]) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }
        const jtype = elemOf[type[j]];
        const jq = q[j];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const iparamIJ = this.p3(itype, jtype, jtype);
        const pIJ = this.params[iparamIJ];
        // long range q-dependent
        if (rsq > pIJ.lcutsq) continue;
        const inty = tb.intype[itype * this.ntypes + jtype];
        const tp = this.triPoint(rsq);
        // 1/r energy and forces
        const d = this.direct(inty, tp, rsq, iq, jq, fac11, fac11e);
        let vionij = d.pot, fvionij = d.pd;
        // field correction to self energy
        const fld = this.field(pIJ, rsq, iq, jq);
        vionij += fld.v;
        fvionij -= fld.f;
        f[3 * i] += delx * fvionij; f[3 * i + 1] += dely * fvionij; f[3 * i + 2] += delz * fvionij;
        f[3 * j] -= delx * fvionij; f[3 * j + 1] -= dely * fvionij; f[3 * j + 2] -= delz * fvionij;
        this.tally(pc, i, j, 0.0, vionij);
        if (pc.vatom) this.vatomPair(pc, i, j, fvionij, delx, dely, delz);
        // short range q-independent
        if (rsq > pIJ.cutsq) continue;
        const rep = this.repulsive(pIJ, rsq, iq, jq);
        const fpair = rep.fforce, evdwl = rep.eng;
        f[3 * i] += delx * fpair; f[3 * i + 1] += dely * fpair; f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair; f[3 * j + 1] -= dely * fpair; f[3 * j + 2] -= delz * fpair;
        this.tally(pc, i, j, evdwl, 0.0);
        if (pc.vatom) this.vatomPair(pc, i, j, fpair, delx, dely, delz);
      }

      // accumulate coordination number information (over-coordination correction of HfO2)
      const s0 = shtStart[i], s1 = shtStart[i + 1];
      if (this.corFlag) {
        for (let jj = s0; jj < s1; jj++) {
          const j = sht[jj];
          const jtype = elemOf[type[j]];
          const iparamIJ = this.p3(itype, jtype, jtype);
          if (this.params[iparamIJ].hfocor > 0.0) {
            delr1[0] = x[3 * j] - xtmp; delr1[1] = x[3 * j + 1] - ytmp; delr1[2] = x[3 * j + 2] - ztmp;
            const rsq1 = dot3(delr1, delr1);
            if (rsq1 > this.params[iparamIJ].cutsq) continue;
            this.NCo[i] += 1;
          }
        }
      }

      // three-body interactions: half i-j loop over the short list, full k loop
      for (let jj = s0; jj < s1; jj++) {
        const j = sht[jj];
        const jtype = elemOf[type[j]];
        const iparamIJ = this.p3(itype, jtype, jtype);
        const pIJ = this.params[iparamIJ];
        const jq = q[j];
        delr1[0] = x[3 * j] - xtmp; delr1[1] = x[3 * j + 1] - ytmp; delr1[2] = x[3 * j + 2] - ztmp;
        const rsq1 = dot3(delr1, delr1);
        if (rsq1 > pIJ.cutsq) continue;
        nj++;
        // bond order zeta for each i-j interaction via the loop over k
        let zetaIJ = 0.0;
        let cuo1 = 0, cuo2 = 0;
        for (let kk = s0; kk < s1; kk++) {
          const k = sht[kk];
          if (j === k) continue;
          const ktype = elemOf[type[k]];
          const iparamIJK = this.p3(itype, jtype, ktype);
          delr2[0] = x[3 * k] - xtmp; delr2[1] = x[3 * k + 1] - ytmp; delr2[2] = x[3 * k + 2] - ztmp;
          const rsq2 = dot3(delr2, delr2);
          const pIJK = this.params[iparamIJK];
          if (rsq2 > pIJK.cutsq) continue;
          zetaIJ += this.zeta(pIJK, rsq1, rsq2, delr1, delr2);
          if (pIJK.hfocor === -2.0) cuo1 = 1;
          if (pIJK.hfocor === -1.0) cuo2 = 1;
        }
        this.cuoFlag = cuo1 && cuo2 ? 1 : 0;
        const fz = this.forceZeta(pIJ, nj, i, rsq1, zetaIJ, iq, jq);
        let fpair = fz.fforce;
        const prefactor = fz.prefactor;
        let evdwl = fz.eng;
        // over-coordination correction for HfO2
        if (this.corFlag && this.NCo[i] !== 0) {
          const oc = this.overCor(pIJ, rsq1, this.NCo[i]);
          Eov = oc.Eov; Fov = oc.Fov;
        }
        evdwl += Eov;
        fpair += Fov;
        f[3 * i] += delr1[0] * fpair; f[3 * i + 1] += delr1[1] * fpair; f[3 * i + 2] += delr1[2] * fpair;
        f[3 * j] -= delr1[0] * fpair; f[3 * j + 1] -= delr1[1] * fpair; f[3 * j + 2] -= delr1[2] * fpair;
        this.tally(pc, i, j, evdwl, 0.0);
        if (pc.vatom) this.vatomPair(pc, i, j, -fpair, -delr1[0], -delr1[1], -delr1[2]);

        // attractive term via loop over k (3-body forces)
        for (let kk = s0; kk < s1; kk++) {
          const k = sht[kk];
          if (j === k) continue;
          const ktype = elemOf[type[k]];
          const iparamIJK = this.p3(itype, jtype, ktype);
          const pIJK = this.params[iparamIJK];
          delr2[0] = x[3 * k] - xtmp; delr2[1] = x[3 * k + 1] - ytmp; delr2[2] = x[3 * k + 2] - ztmp;
          const rsq2 = dot3(delr2, delr2);
          if (rsq2 > pIJK.cutsq) continue;
          fi.fill(0); fj.fill(0); fk.fill(0);
          this.attractive(pIJK, prefactor, rsq1, rsq2, delr1, delr2, fi, fj, fk);
          // 3-body LP and BB correction and forces
          const elpIJ = this.elp(pIJK, rsq1, rsq2, delr1, delr2);
          this.flp(pIJK, rsq1, rsq2, delr1, delr2, filp, fjlp, fklp);
          for (let c = 0; c < 3; c++) {
            fi[c] += filp[c];
            fj[c] += fjlp[c];
            fk[c] += fklp[c];
          }
          for (let c = 0; c < 3; c++) {
            f[3 * i + c] += fi[c];
            f[3 * j + c] += fj[c];
            f[3 * k + c] += fk[c];
          }
          this.tally(pc, i, j, elpIJ, 0.0);
          if (pc.vatom) this.vatomTriple(pc, i, j, k, fj, fk, delr1, delr2);
        }
      }
      if (this.cuoFlag) pI.cutsq *= 0.65;
    }
    this.cuoFlag = 0;
  }

  /** ev_tally with newton on: full energies; per-atom energy split half to each atom. */
  protected tally(pc: PairCompute, i: number, j: number, evdwl: number, ecoul: number): void {
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
    if (pc.eatom) {
      const e = 0.5 * (evdwl + ecoul);
      pc.eatom[i] += e;
      pc.eatom[j] += e;
    }
  }

  /**
   * Per-atom virial of one pair, ev_tally with newton on (src/pair.cpp at b332f28): the pair virial
   * dx*dx*fpair ... dy*dz*fpair is split half to each atom. Only pc.vatom is written: the global virial is the fdotr sum.
   */
  protected vatomPair(pc: PairCompute, i: number, j: number, fpair: number, dx: number, dy: number, dz: number): void {
    const va = pc.vatom!;
    const v = [dx * dx * fpair, dy * dy * fpair, dz * dz * fpair, dx * dy * fpair, dx * dz * fpair, dy * dz * fpair];
    for (const a of [i, j]) for (let c = 0; c < 6; c++) va[6 * a + c] += 0.5 * v[c];
  }

  /**
   * Per-atom virial of one three-body term, Pair::v_tally3 (src/pair.cpp at b332f28): the sum of drik fj and drjk fk
   * (drik = delr1 from i to j, drjk = delr2 from i to k) is taken in thirds to i, j and k.
   */
  protected vatomTriple(pc: PairCompute, i: number, j: number, k: number, fj: V3, fk: V3, drik: V3, drjk: V3): void {
    const va = pc.vatom!;
    const v = [
      drik[0] * fj[0] + drjk[0] * fk[0],
      drik[1] * fj[1] + drjk[1] * fk[1],
      drik[2] * fj[2] + drjk[2] * fk[2],
      drik[0] * fj[1] + drjk[0] * fk[1],
      drik[0] * fj[2] + drjk[0] * fk[2],
      drik[1] * fj[2] + drjk[1] * fk[2],
    ];
    for (const a of [i, j, k]) for (let c = 0; c < 6; c++) va[6 * a + c] += v[c] / 3;
  }

  /** Short_neigh: owned-atom neighbors with rsq <= cutmin (kept in the full list's order). */
  protected buildShortNeigh(pc: PairCompute, nlocal: number, inum: number): void {
    const nmask = NEIGHMASK;
    const list = pc.full!;
    const x = pc.x;
    if (this.shtStart.length < nlocal + 1) this.shtStart = new Int32Array(nlocal + 1);
    const out: number[] = [];
    for (let i = 0; i < nlocal; i++) {
      this.shtStart[i] = out.length;
      if (i >= inum) continue;
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & nmask;
        const dx = xtmp - x[3 * j], dy = ytmp - x[3 * j + 1], dz = ztmp - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq > this.cutmin) continue;
        out.push(j);
      }
    }
    this.shtStart[nlocal] = out.length;
    this.shtList = Int32Array.from(out);
  }

  /** self: self energy of one ion with charge qi (self in pair_comb.cpp). */
  protected self(p: CombParam, qi: number, selfpot: number): number {
    const s1 = p.chi, s2 = p.dj, s3 = p.dk, s4 = p.dl, s5 = p.dm;
    const qmin = p.QL1 * 0.90;
    const qmax = p.QU1 * 0.90;
    const cmin = 1000.0, cmax = 1000.0;
    let selfTmp = qi * (s1 + qi * (s2 + selfpot + qi * (s3 + qi * (s4 + qi * qi * s5))));
    if (qi < qmin) selfTmp += cmin * powint(qi - qmin, 4);
    if (qi > qmax) selfTmp += cmax * powint(qi - qmax, 4);
    return selfTmp;
  }

  /** tri_point: three-point interpolation indices and weights of the 1/r table (written to this.tp). */
  protected triPoint(rsq: number): TriPoint {
    const out = this.tp;
    const rin = 0.10, dr = 0.0010;
    let r = Math.sqrt(rsq);
    if (r < rin + 2.0 * dr) r = rin + 2.0 * dr;
    if (r > this.cutmax - 2.0 * dr) r = this.cutmax - 2.0 * dr;
    let rridr = (r - rin) / dr;
    let mr1 = Math.trunc(rridr) - 1;
    const dd = rridr - mr1;
    if (dd > 0.5) mr1 += 1;
    const mr2 = mr1 + 1;
    const mr3 = mr2 + 1;
    const rr1 = mr1 * dr;
    rridr = (r - rin - rr1) / dr;
    const rridr2 = rridr * rridr;
    out.mr1 = mr1; out.mr2 = mr2; out.mr3 = mr3;
    out.sr1 = (rridr2 - rridr) * 0.50;
    out.sr2 = 1.0 - rridr2;
    out.sr3 = (rridr2 + rridr) * 0.50;
    return out;
  }

  /** direct: 1/r Coulomb energy (pot) and its radial derivative (pd) with Slater overlap. */
  protected direct(inty: number, tp: TriPoint, rsq: number, iq: number, jq: number, fac11: number, fac11e: number): { pot: number; pd: number } {
    const out = this.dOut;
    const tb = this.tables!;
    const { mr1, mr2, mr3, sr1, sr2, sr3 } = tp;
    const r = Math.sqrt(rsq);
    const r3 = r * rsq;
    const alf = ALF;
    const alfdpi = 2.0 * alf / MY_PIS;
    const esucon = this.esucon;
    const nt = tb.nntypes;
    const e = tb.erpaw;
    const fafb = tb.fafb, dfafb = tb.dfafb;
    // 1/r energy
    const erfcc = sr1 * e[2 * mr1] + sr2 * e[2 * mr2] + sr3 * e[2 * mr3];
    const fafbn1 = sr1 * fafb[mr1 * nt + inty] + sr2 * fafb[mr2 * nt + inty] + sr3 * fafb[mr3 * nt + inty];
    const potij = erfcc / r * esucon - fac11e;
    const sme2 = potij + fafbn1 * esucon;
    const pot = 1.0 * iq * jq * sme2;
    // 1/r force (wrt r)
    const erfcd = sr1 * e[2 * mr1 + 1] + sr2 * e[2 * mr2 + 1] + sr3 * e[2 * mr3 + 1];
    const dfafbn1 = sr1 * dfafb[mr1 * nt + inty] + sr2 * dfafb[mr2 * nt + inty] + sr3 * dfafb[mr3 * nt + inty];
    const dvdrr = (erfcc / r3 + alfdpi * erfcd / rsq) * esucon - fac11;
    const smf2 = dvdrr - dfafbn1 * esucon / r;
    out.pot = pot;
    out.pd = 1.0 * iq * jq * smf2;
    return out;
  }

  /** field: charge-field correction to the self energy (energy v and force f). */
  protected field(p: CombParam, rsq: number, iq: number, jq: number): { v: number; f: number } {
    const out = this.fOut;
    const r = Math.sqrt(rsq);
    const r5 = r * r * r * r * r;
    const r6 = r5 * r;
    const rc = p.lcut;
    const rc5 = rc * rc * rc * rc * rc;
    const rc6 = rc5 * rc;
    const cmi1 = p.cmn1, cmi2 = p.cmn2, cmj1 = p.cml1, cmj2 = p.cml2;
    const rf5 = 1.0 / r5 - 1.0 / rc5 + 5.0 * (r - rc) / rc6;
    const drf6 = 5.0 / rc6 - 5.0 / r6;
    // field correction energy
    const smpn = rf5 * jq * (cmi1 + jq * cmi2);
    const smpl = rf5 * iq * (cmj1 + iq * cmj2);
    const v = 1.0 * (smpn + smpl);
    // field correction force
    const rfx1 = jq * drf6 * (cmi1 + jq * cmi2) / r;
    const rfx2 = iq * drf6 * (cmj1 + iq * cmj2) / r;
    out.v = v;
    out.f = 1.0 * (rfx1 + rfx2);
    return out;
  }

  /** repulsive: short-range q-dependent repulsion; returns fforce and energy. */
  protected repulsive(p: CombParam, rsq: number, iq: number, jq: number): { fforce: number; eng: number } {
    const romi = p.addrep;
    const rrcs = p.bigr + p.bigd;
    const r = Math.sqrt(rsq);
    if (r > rrcs) return { fforce: 0, eng: 0 };
    const tmpFc = combFc(r, p);
    const tmpFcD = combFcD(r, p);
    const tmpExp = libmExp(-p.rlm1 * r);
    const arr1 = 2.22850, arr2 = 1.89350;
    const fc2j = combFc2(r);
    const fc3j = combFc3(r);
    const fcp2j = combFc2D(r);
    const fcp3j = combFc3D(r);
    const Di = combDi(p, iq);
    const Dj = combDj(p, jq);
    const Asi = p.biga1 * libmExp(p.lam11 * Di);
    const Asj = p.biga2 * libmExp(p.lam12 * Dj);
    const bigA = Asi > 0.0 && Asj > 0.0 ? Math.sqrt(Asi * Asj) * p.romiga : 0.0;
    let fforce = -bigA * tmpExp * (tmpFcD - tmpFc * p.rlm1) / r;
    // additional repulsion for TiO2 and HfO2 (switch by cor_flag)
    let vrcs = 0.0, fvrcs = 0.0;
    if (romi > 0.0) {
      if (!this.corFlag) {
        vrcs = romi * square(1.0 - r / rrcs);
        fvrcs = romi * 2.0 * (r / rrcs - 1.0) / rrcs;
      } else {
        const rslp = (arr1 - r) / (arr1 - arr2);
        const rslp2 = rslp * rslp;
        const rslp4 = rslp2 * rslp2;
        vrcs = fc2j * fc3j * romi * (50.0 * rslp4 - 30.0 * rslp2 + 4.50) / 8.0;
        fvrcs = fcp2j * fcp3j * romi * rslp * (-25.0 * rslp2 + 7.50) / (arr1 - arr2);
      }
      const fforceTmp = fforce * vrcs - (tmpFc * bigA * tmpExp * fvrcs);
      fforce += fforceTmp;
    }
    const eng = (tmpFc * bigA * tmpExp) * (1.0 + vrcs);
    return { fforce, eng };
  }

  /** zeta: bond-order sum term of one k neighbor of i-j. */
  protected zeta(p: CombParam, rsqij: number, rsqik: number, delrij: V3, delrik: V3): number {
    const rij = Math.sqrt(rsqij);
    if (rij > p.bigr + p.bigd) return 0.0;
    const rik = Math.sqrt(rsqik);
    const costheta = dot3(delrij, delrik) / (rij * rik);
    const arg = p.powermint === 3 ? cube(p.rlm2 * (rij - rik)) : p.rlm2 * (rij - rik);
    let exDelr: number;
    if (arg > 69.0776) exDelr = 1.e30;
    else if (arg < -69.0776) exDelr = 0.0;
    else exDelr = libmExp(arg);
    return combFc(rik, p) * combGijk(costheta, p) * exDelr;
  }

  /** elp: Legendre polynomial and bond-bending angle energy. */
  protected elp(p: CombParam, rsqij: number, rsqik: number, delrij: V3, delrik: V3): number {
    if (p.aconf > 1.0e-6 || p.plp1 > 1.0e-6 || p.plp3 > 1.0e-6 || p.plp6 > 1.0e-6) {
      const pplp1 = p.plp1, pplp3 = p.plp3, pplp6 = p.plp6;
      const c123 = libmCos(p.a123 * Math.PI / 180.0);
      const rij = Math.sqrt(rsqij);
      const rik = Math.sqrt(rsqik);
      const costheta = dot3(delrij, delrik) / (rij * rik);
      const fcj = combFc(rij, p);
      const fck = combFc(rik, p);
      const rmu = costheta;
      let comtt: number;
      if (p.plp1 > 1.0e-6 || p.plp3 > 1.0e-6 || p.plp6 > 1.0e-6) {
        const rmu2 = rmu * rmu;
        const lp1 = rmu;
        const lp3 = 0.5 * (5.0 * rmu2 * rmu - 3.0 * rmu);
        const lp6 = (231.0 * rmu2 * rmu2 * rmu2 - 315.0 * rmu2 * rmu2 + 105.0 * rmu2 - 5.0) / 16.0;
        comtt = pplp1 * lp1 + pplp3 * lp3 + pplp6 * lp6;
      } else comtt = 0.0;
      // bond-bending terms
      if (p.aconf > 1e-4) {
        if (p.hfocor >= 0.0) comtt += p.aconf * (rmu - c123) * (rmu - c123);
        else if (p.hfocor < 0.0) comtt += p.aconf * (4.0 - (rmu - c123) * (rmu - c123));
      }
      return 0.5 * fcj * fck * comtt;
    }
    return 0.0;
  }

  /** flp: forces of the Legendre and bond-bending angle terms on i, j and k. */
  protected flp(p: CombParam, rsqij: number, rsqik: number, delrij: V3, delrik: V3, drilp: V3, drjlp: V3, drklp: V3): void {
    let ffj1 = 0.0, ffj2 = 0.0, ffk1 = 0.0, ffk2 = 0.0;
    if (p.aconf > 1.0e-4 || p.plp1 > 1.0e-6 || p.plp3 > 1.0e-6 || p.plp6 > 1.0e-6) {
      const pplp1 = p.plp1, pplp3 = p.plp3, pplp6 = p.plp6;
      const c123 = libmCos(p.a123 * Math.PI / 180.0);
      const rij = Math.sqrt(rsqij);
      const rik = Math.sqrt(rsqik);
      const costheta = dot3(delrij, delrik) / (rij * rik);
      const fcj = combFc(rij, p);
      const fck = combFc(rik, p);
      const fckD = combFcD(rik, p);
      const rmu = costheta;
      let comtt: number, comttD: number;
      if (p.plp1 > 1.0e-6 || p.plp3 > 1.0e-6 || p.plp6 > 1.0e-6) {
        const rmu2 = rmu * rmu;
        const lp1 = rmu;
        const lp3 = (2.5 * rmu2 * rmu - 1.5 * rmu);
        const lp6 = (231.0 * rmu2 * rmu2 * rmu2 - 315.0 * rmu2 * rmu2 + 105.0 * rmu2 - 5.0) / 16.0;
        const lp1D = 1.0;
        const lp3D = (7.5 * rmu2 - 1.5);
        const lp6D = (1386.0 * rmu2 * rmu2 * rmu - 1260.0 * rmu2 * rmu + 210.0) / 16.0;
        comtt = pplp1 * lp1 + pplp3 * lp3 + pplp6 * lp6;
        comttD = pplp1 * lp1D + pplp3 * lp3D + pplp6 * lp6D;
      } else {
        comtt = 0.0;
        comttD = 0.0;
      }
      // bond-bending terms derivatives
      if (p.aconf > 1.0e-4) {
        if (p.hfocor >= 0.0) {
          comtt += p.aconf * (rmu - c123) * (rmu - c123);
          comttD += 2.0 * p.aconf * (rmu - c123);
        } else if (p.hfocor < 0.0) {
          comtt += p.aconf * (4.0 - (rmu - c123) * (rmu - c123));
          comttD += -2.0 * p.aconf * (rmu - c123);
        }
      }
      const com4k = 2.0 * fcj * fckD * comtt;
      const com5 = fcj * fck * comttD;
      ffj1 = -0.5 * (com5 / (rij * rik));
      ffj2 = 0.5 * (com5 * rmu / rsqij);
      ffk1 = ffj1;
      ffk2 = 0.5 * (-com4k / rik + com5 * rmu / rsqik);
    }
    // j-atom, k-atom, and i-atom (minus the sum of the other two)
    for (let c = 0; c < 3; c++) {
      drjlp[c] = ffj2 * delrij[c] + ffj1 * delrik[c];
      drklp[c] = ffk2 * delrik[c] + ffk1 * delrij[c];
    }
    for (let c = 0; c < 3; c++) drilp[c] = -(drjlp[c] + drklp[c]);
  }

  /** force_zeta: attractive (bond-order) energy of i-j and its force factors; stores bbij[i][nj]. */
  protected forceZeta(p: CombParam, nj: number, i: number, rsq: number, zetaIJ: number, iq: number, jq: number):
    { fforce: number; prefactor: number; eng: number } {
    const r = Math.sqrt(rsq);
    const fa = this.combFa(r, p, iq, jq);
    const faD = this.combFaD(r, p, iq, jq);
    const bij = combBij(zetaIJ, p);
    this.bbij[i * this.bbStride + nj] = bij;
    const fforce = 0.5 * bij * faD / r;
    const prefactor = -0.5 * fa * combBijD(zetaIJ, p);
    const eng = 0.5 * bij * fa;
    return { fforce, prefactor, eng };
  }

  /** comb_fa: attractive pair function (energy form). */
  protected combFa(r: number, p: CombParam, qi: number, qj: number): number {
    if (r > p.bigr + p.bigd) return 0.0;
    const Di = combDi(p, qi), Dj = combDj(p, qj);
    const { Bsi, Bsj } = combBsi(p, qi, qj, Di, Dj);
    const bigB = Bsi > 0.0 && Bsj > 0.0 ? Math.sqrt(Bsi * Bsj) * p.romigb : 0.0;
    return -bigB * libmExp(-p.rlm2 * r) * combFc(r, p);
  }

  /** comb_fa_d: derivative of combFa with respect to r. */
  protected combFaD(r: number, p: CombParam, qi: number, qj: number): number {
    if (r > p.bigr + p.bigd) return 0.0;
    const Di = combDi(p, qi), Dj = combDj(p, qj);
    const { Bsi, Bsj } = combBsi(p, qi, qj, Di, Dj);
    const bigB = Bsi > 0.0 && Bsj > 0.0 ? Math.sqrt(Bsi * Bsj) * p.romigb : 0.0;
    return bigB * libmExp(-p.rlm2 * r) * (p.rlm2 * combFc(r, p) - combFcD(r, p));
  }

  /** attractive: the zeta derivative with respect to the three atoms (writes fi, fj, fk). */
  protected attractive(p: CombParam, prefactor: number, rsqij: number, rsqik: number, delrij: V3, delrik: V3,
    fi: V3, fj: V3, fk: V3): void {
    const rij = Math.sqrt(rsqij);
    const rijinv = 1.0 / rij;
    const rijHat = [rijinv * delrij[0], rijinv * delrij[1], rijinv * delrij[2]];
    const rik = Math.sqrt(rsqik);
    const rikinv = 1.0 / rik;
    const rikHat = [rikinv * delrik[0], rikinv * delrik[1], rikinv * delrik[2]];
    this.zetaTermD(prefactor, rijHat, rij, rikHat, rik, fi, fj, fk, p);
  }

  /** comb_zetaterm_d: derivative of the zeta term with respect to Ri, Rj and Rk. */
  protected zetaTermD(prefactor: number, rijHat: number[], rij: number, rikHat: number[], rik: number,
    dri: V3, drj: V3, drk: V3, p: CombParam): void {
    const fc = combFc(rik, p);
    const dfc = combFcD(rik, p);
    const tmp = p.powermint === 3 ? cube(p.rlm2 * (rij - rik)) : p.rlm2 * (rij - rik);
    let exDelr: number;
    if (tmp > 69.0776) exDelr = 1.e30;
    else if (tmp < -69.0776) exDelr = 0.0;
    else exDelr = libmExp(tmp);
    const exDelrD = p.powermint === 3 ? 3.0 * cube(p.rlm2) * square(rij - rik) * exDelr : p.rlm2 * exDelr;
    const cosTheta = dot3(rijHat, rikHat);
    const gijk = combGijk(cosTheta, p);
    const gijkD = combGijkD(cosTheta, p);
    const dc = this.costhetaD(rijHat, rij, rikHat, rik);
    // derivative wrt Ri
    for (let c = 0; c < 3; c++) {
      let v = -dfc * gijk * exDelr * rikHat[c];
      v = fc * gijkD * exDelr * dc.dri[c] + v;
      v = fc * gijk * exDelrD * rikHat[c] + v;
      v = -fc * gijk * exDelrD * rijHat[c] + v;
      dri[c] = prefactor * v;
    }
    // derivative wrt Rj
    for (let c = 0; c < 3; c++) {
      let v = fc * gijkD * exDelr * dc.drj[c];
      v = fc * gijk * exDelrD * rijHat[c] + v;
      drj[c] = prefactor * v;
    }
    // derivative wrt Rk
    for (let c = 0; c < 3; c++) {
      let v = dfc * gijk * exDelr * rikHat[c];
      v = fc * gijkD * exDelr * dc.drk[c] + v;
      v = -fc * gijk * exDelrD * rikHat[c] + v;
      drk[c] = prefactor * v;
    }
  }

  /** costheta_d: derivative of cos(theta) with respect to the three atoms. */
  protected costhetaD(rijHat: number[], rij: number, rikHat: number[], rik: number): { dri: number[]; drj: number[]; drk: number[] } {
    const cosTheta = dot3(rijHat, rikHat);
    const drj = [0, 0, 0], drk = [0, 0, 0], dri = [0, 0, 0];
    for (let c = 0; c < 3; c++) drj[c] = (-cosTheta * rijHat[c] + rikHat[c]) * (1.0 / rij);
    for (let c = 0; c < 3; c++) drk[c] = (-cosTheta * rikHat[c] + rijHat[c]) * (1.0 / rik);
    for (let c = 0; c < 3; c++) dri[c] = -(drj[c] + drk[c]);
    return { dri, drj, drk };
  }

  /**
   * yasu_char (fix qeq/comb): the charge force qf of every atom on the charges now in the neighbor data
   * (owned and ghost), with the bond-order terms of the last compute. Returns the sum of qf over the
   * owned atoms for which isGroup is true. Ghost contributions are summed into their owners at the end
   * (LAMMPS reverse_comm); the owned and ghost entries start at zero (the forward comm of the zeroed
   * group atoms, and the non-group entries are never read).
   */
  chargeForces(qf: Float64Array, isGroup: (i: number) => boolean): number {
    const nmask = NEIGHMASK;
    const pc = this.lastPc;
    if (!pc || !pc.full) throw new StyleError('fix qeq/comb: pair style comb has not computed forces yet');
    const tb = this.tables;
    if (!tb) throw new StyleError('pair_style comb needs a pair_coeff command first');
    const nb = pc.nb;
    const x = pc.x, type = pc.type, q = nb.qall;
    const nlocal = pc.nlocal;
    const owner = nb.owner;
    const ids = pc.s.id;
    const list = pc.full;
    const elemOf = this.map;
    qf.fill(0);
    const { potal, fac11e } = this.potalCalc(this.esucon);
    for (let i = 0; i < nlocal; i++) {
      if (!isGroup(i)) continue;
      const itag = ids[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = elemOf[type[i]];
      const iq = q[i];
      const pI = this.params[this.p3(itype, itype, itype)];
      // charge force from self energy
      let fqi = this.qfoSelf(pI, iq, potal);
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      // two-body interactions (the half rule of the energy loop)
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & nmask;
        const jtag = ids[owner[j]];
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < x[3 * i + 2]) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }
        const jtype = elemOf[type[j]];
        const jq = q[j];
        const delx = x[3 * j] - xtmp, dely = x[3 * j + 1] - ytmp, delz = x[3 * j + 2] - ztmp;
        const rsq1 = delx * delx + dely * dely + delz * delz;
        const pIJ = this.params[this.p3(itype, jtype, jtype)];
        // long range q-dependent
        if (rsq1 > pIJ.lcutsq) continue;
        const inty = tb.intype[itype * this.ntypes + jtype];
        const tp = this.triPoint(rsq1);
        // 1/r charge forces
        const fqij0 = this.qfoDirect(inty, tp, rsq1, fac11e);
        fqi += jq * fqij0;
        qf[j] += iq * fqij0;
        // field correction to self energy and charge force
        const fld = this.qfoField(pIJ, rsq1, iq, jq);
        fqi += fld.fqij;
        qf[j] += fld.fqjj;
      }
      // three-body interactions (the full list, counted as in compute)
      let nj = 0;
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & nmask;
        const jtype = elemOf[type[j]];
        const jq = q[j];
        const delx = x[3 * j] - xtmp, dely = x[3 * j + 1] - ytmp, delz = x[3 * j + 2] - ztmp;
        const rsq1 = delx * delx + dely * dely + delz * delz;
        const pIJ = this.params[this.p3(itype, jtype, jtype)];
        if (rsq1 > pIJ.cutsq) continue;
        nj++;
        // charge force in Aij and Bij
        const sh = this.qfoShort(pIJ, i, nj, rsq1, iq, jq);
        fqi += sh.fqij;
        qf[j] += sh.fqjj;
      }
      qf[i] += fqi;
    }
    // reverse comm: ghost charge forces into their owners
    for (let k = nlocal; k < nb.nall; k++) qf[owner[k]] += qf[k];
    let eneg = 0.0;
    for (let i = 0; i < nlocal; i++) if (isGroup(i)) eneg += qf[i];
    return eneg;
  }

  /** qfo_self: derivative of the self energy with respect to the charge. */
  protected qfoSelf(p: CombParam, qi: number, selfpot: number): number {
    const s1 = p.chi, s2 = p.dj, s3 = p.dk, s4 = p.dl, s5 = p.dm;
    const qmin = p.QL1 * 0.90;
    const qmax = p.QU1 * 0.90;
    const cmin = 1000.0, cmax = 1000.0;
    let selfD = s1 + qi * (2.0 * (s2 + selfpot) + qi * (3.0 * s3 + qi * (4.0 * s4 + qi * qi * 6.0 * s5)));
    if (qi < qmin) selfD += 4.0 * cmin * cube(qi - qmin);
    if (qi > qmax) selfD += 4.0 * cmax * cube(qi - qmax);
    return selfD;
  }

  /** qfo_direct: charge derivative of the 1/r energy. */
  protected qfoDirect(inty: number, tp: TriPoint, rsq: number, fac11e: number): number {
    const tb = this.tables!;
    const { mr1, mr2, mr3, sr1, sr2, sr3 } = tp;
    const r = Math.sqrt(rsq);
    const esucon = this.esucon;
    const nt = tb.nntypes;
    const e = tb.erpaw;
    const erfcc = sr1 * e[2 * mr1] + sr2 * e[2 * mr2] + sr3 * e[2 * mr3];
    const fafbn1 = sr1 * tb.fafb[mr1 * nt + inty] + sr2 * tb.fafb[mr2 * nt + inty] + sr3 * tb.fafb[mr3 * nt + inty];
    const vm = erfcc / r * esucon - fac11e;
    return 1.0 * (vm + esucon * fafbn1);
  }

  /** qfo_field: charge derivatives of the field correction (qij for i, qjj for j). */
  protected qfoField(p: CombParam, rsq: number, iq: number, jq: number): { fqij: number; fqjj: number } {
    const out = this.qfOut;
    const r = Math.sqrt(rsq);
    const r5 = r * r * r * r * r;
    const rc = p.lcut;
    const rc5 = rc * rc * rc * rc * rc;
    const rc6 = rc5 * rc;
    const cmi1 = p.cmn1, cmi2 = p.cmn2, cmj1 = p.cml1, cmj2 = p.cml2;
    const rf5 = 1.0 / r5 - 1.0 / rc5 + 5.0 * (r - rc) / rc6;
    out.fqij = 1.0 * rf5 * (cmj1 + 2.0 * iq * cmj2);
    out.fqjj = 1.0 * rf5 * (cmi1 + 2.0 * jq * cmi2);
    return out;
  }

  /** qfo_short: charge derivatives of the attractive (bond-order) term; bbij[i][nj] from compute. */
  protected qfoShort(p: CombParam, i: number, nj: number, rsq: number, iq: number, jq: number): { fqij: number; fqjj: number } {
    const r = Math.sqrt(rsq);
    const romi = p.addrep, rrcs = p.bigr + p.bigd;
    const romie = p.romiga, romib = p.romigb;
    const tmpFc = combFc(r, p);
    const tmpExp1 = libmExp(-p.rlm1 * r);
    const tmpExp2 = libmExp(-p.rlm2 * r);
    const bij = this.bbij[i * this.bbStride + nj];
    const arr1 = 2.22850, arr2 = 1.89350;
    const fc2j = combFc2(r);
    const fc3j = combFc3(r);
    let vrcs = 0.0;
    if (romi > 0.0) {
      if (!this.corFlag) vrcs = romi * cube(1.0 - r / rrcs);
      else {
        const rslp = (arr1 - r) / (arr1 - arr2);
        const rslp2 = rslp * rslp;
        const rslp4 = rslp2 * rslp2;
        vrcs = fc2j * fc3j * romi * (50.0 * rslp4 - 30.0 * rslp2 + 4.50) / 8.0;
      }
    }
    const Di = combDi(p, iq), Dj = combDj(p, jq);
    const Asi = p.biga1 * libmExp(p.lam11 * Di);
    const Asj = p.biga2 * libmExp(p.lam12 * Dj);
    const { Bsi, Bsj } = combBsi(p, iq, jq, Di, Dj);
    const QUchi = (p.QU1 - iq) * p.bD1;
    const QUchj = (p.QU2 - jq) * p.bD2;
    const QOchi = (iq - p.Qo1) * p.bB1;
    const QOchj = (jq - p.Qo2) * p.bB2;
    const YYDiqp = QUchi === 0.0 ? 0.0 : -p.nD1 * QUchi * p.bD1 * Math.pow(Math.abs(QUchi), (p.nD1 - 2.0));
    const YYDjqp = QUchj === 0.0 ? 0.0 : -p.nD2 * QUchj * p.bD2 * Math.pow(Math.abs(QUchj), (p.nD2 - 2.0));
    const YYAsiqp = Asi * p.lam11 * YYDiqp;
    const YYAsjqp = Asj * p.lam12 * YYDjqp;
    const YYBsiqp = QOchi === 0.0
      ? Bsi * p.lam21 * YYDiqp
      : Bsi * p.lam21 * YYDiqp - p.bigb1 * libmExp(p.lam21 * Di) * 10.0 * QOchi * p.bB1 * powint(Math.abs(QOchi), (10 - 2));
    const YYBsjqp = QOchj === 0.0
      ? Bsj * p.lam22 * YYDjqp
      : Bsj * p.lam22 * YYDjqp - p.bigb2 * libmExp(p.lam22 * Dj) * 10.0 * QOchj * p.bB2 * powint(Math.abs(QOchj), (10 - 2));
    const caj = Asi > 0.0 && Asj > 0.0 ? 1.0 / (2.0 * Math.sqrt(Asi * Asj)) * romie : 0.0;
    const cbj = Bsi > 0.0 && Bsj > 0.0 ? 1.0 / (2.0 * Math.sqrt(Bsi * Bsj)) * romib : 0.0;
    const cfqr = 0.50 * tmpFc * (1.0 + vrcs); // 0.5 b/c full atom loop
    const cfqs = -0.50 * tmpFc * bij;
    const ca1 = Asj * caj * YYAsiqp;
    const ca2 = Bsj * cbj * YYBsiqp;
    const ca3 = Asi * caj * YYAsjqp;
    const ca4 = Bsi * cbj * YYBsjqp;
    let fqij = cfqr * tmpExp1 * ca1;
    fqij += cfqs * tmpExp2 * ca2;
    let fqjj = cfqr * tmpExp1 * ca3;
    fqjj += cfqs * tmpExp2 * ca4;
    const out = this.qfOut;
    out.fqij = fqij;
    out.fqjj = fqjj;
    return out;
  }

  /** Over_cor: over-coordination energy and force correction (HfO2 / Cu2O). */
  protected overCor(p: CombParam, rsq1: number, NCoi: number): { Eov: number; Fov: number } {
    const r = Math.sqrt(rsq1);
    const NCon = Math.trunc(NCoi - 7);
    const tmpFc = combFc(r, p);
    // the C++ source evaluates the derivative with comb_fc (not comb_fc_d); kept as in LAMMPS
    const tmpFcD = combFc(r, p);
    let Eov = 0.0, Fov = 0.0;
    const ECo = p.hfocor;
    const BCo = 0.1;
    if (NCon >= 0.20) {
      Eov = tmpFc * ECo * NCon / (1.0 + libmExp(BCo * NCon));
      Fov = -(tmpFcD * Eov + tmpFc * ECo / (1.0 + libmExp(BCo * NCon)) -
        (tmpFc * ECo * NCon * BCo * libmExp(BCo * NCon)) /
        ((1.0 + libmExp(BCo * NCon)) * (1.0 + libmExp(BCo * NCon))));
      Fov /= r;
    }
    return { Eov, Fov };
  }
}
