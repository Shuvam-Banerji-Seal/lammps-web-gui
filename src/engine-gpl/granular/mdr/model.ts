/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/fix_granular_mdr.cpp (FixGranularMDR::setup_pre_force, pre_force, calculate_contact_penalty,
 * mean_surf_disp, update_fix_gran_wall) and src/GRANULAR/fix_granular_mdr.h, and the pair side of
 * src/GRANULAR/gran_sub_mod_normal.cpp (GranSubModNormalMDR::calculate_forces) through normal.ts, at commit b332f28.
 * Copyright (LAMMPS development team, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The per-atom state of fix GRANULAR/MDR lives in custom per-atom properties named as the native fix (Ro, Vgeo, ... sigmazz,
 * history_setup_flag; compute property/atom d_Ro reads them). The contact histories (27 values per touching pair
 * and per wall contact) are kept by atom-ID key; a contact that stops touching drops its history, as native zeroes it.
 * The fix's pre_force runs at the start of the pair compute (native: PRE_FORCE, before the pair force).
 */
import type { PairCompute } from '../../../engine/force/types';
import type { SimState } from '../../../engine/types';
import { NEIGHMASK } from '../../../engine/neighbor';
import type { MdrApi, MdrNormalOut, MdrPairIn, MdrWallIn, WallContacts } from '../../../engine/force/pair/granular';
import {
  AC_0, DELTAO_0, DELTAP_0, DELTA_0, DELTA_MAX, HIST_SIZE, MDR_OVERLAP_LIMIT, PENALTY,
  mdrConsts, mdrDampPrefactor, mdrSide, type MdrAtomArrays, type MdrConsts, type MdrMaterial,
} from './normal';
import { exp as libmExp } from '../../libm/exp';
import { acos as libmAcos } from '../../libm/acos';

const EPSILON = 1e-10;

/** The MDR model of one pair style instance (pair granular with the mdr normal model). */
export class MdrModel implements MdrApi {
  private readonly pairHist = new Map<string, Float64Array>();
  private readonly wallHist = new Map<string, Float64Array>();
  private readonly walls = new Map<string, WallContacts>();

  /** The per-atom arrays of the current force evaluation (bound by beginStep). */
  private stats: MdrAtomArrays | null = null;
  private readonly K: MdrConsts;

  constructor(readonly pair: MdrMaterial) {
    this.K = mdrConsts(pair);
  }

  /** A wall fix registers its contact enumeration (it is called at each step's pre-force for the bulk update). */
  wallProvider(name: string, contacts: WallContacts): void {
    this.walls.set(name, contacts);
  }

  /** The per-atom arrays of fix GRANULAR/MDR (created on first use). */
  static arrays(s: SimState): MdrAtomArrays {
    const get = (name: string): Float64Array => {
      let c = s.custom.get(name);
      if (!c || c.data.length < s.n) {
        c = { int: false, cols: 0, data: new Float64Array(Math.max(s.n, c?.data.length ?? 0)) };
        s.custom.set(name, c);
      }
      return c.data;
    };
    return {
      Ro: get('Ro'), Vcaps: get('Vcaps'), Vgeo: get('Vgeo'), Velas: get('Velas'), eps_bar: get('eps_bar'),
      dRnumerator: get('dRnumerator'), dRdenominator: get('dRdenominator'), Acon0: get('Acon0'), Acon1: get('Acon1'),
      Atot: get('Atot'), Atot_sum: get('Atot_sum'), ddelta_bar: get('ddelta_bar'), psi: get('psi'),
      sigmaxx: get('sigmaxx'), sigmayy: get('sigmayy'), sigmazz: get('sigmazz'), dRavg: get('dRavg'),
    };
  }

  private entry(map: Map<string, Float64Array>, key: string): Float64Array {
    let h = map.get(key);
    if (!h) {
      h = new Float64Array(HIST_SIZE);
      map.set(key, h);
    }
    return h;
  }

  /** The pair key of two atom IDs (lower ID first, as the core pair loop keys its shear history). */
  static pairKey(a: number, b: number): string {
    return a > b ? `${b}:${a}` : `${a}:${b}`;
  }

  /** A pair that does not touch drops its history (native zeroes the 27 values in the touch test). */
  pairDrop(ida: number, idb: number): void {
    this.pairHist.delete(MdrModel.pairKey(ida, idb));
  }

  /** A wall contact that is gone drops its history. */
  wallDrop(key: string): void {
    this.wallHist.delete(key);
  }

  /**
   * fix GRANULAR/MDR pre_force, run at the start of the pair compute: the per-atom update (initialisation of new atoms,
   * apparent radius growth), then, on steps (not at setup), the penalty, the mean surface displacement and the wall
   * bulk update. The pair list is the core's half list (the lower index owns a pair).
   */
  beginStep(s: SimState, pc: PairCompute, hu: boolean): void {
    const st = MdrModel.arrays(s);
    this.stats = st;
    const n = s.n;
    const radius = s.radius!;
    const { Ro, Vgeo, Velas, Vcaps, eps_bar, dRnumerator, dRdenominator, Acon0, Acon1, Atot, Atot_sum, ddelta_bar, psi, sigmaxx, sigmayy, sigmazz, dRavg } = st;
    const flag = s.custom.get('history_setup_flag') ?? { int: false, cols: 0, data: new Float64Array(n) };
    s.custom.set('history_setup_flag', flag);
    const hsf = flag.data;
    const psib = this.K.psib;
    const newAtom = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (hsf[i] < EPSILON) {
        Ro[i] = radius[i];
        Acon0[i] = 0;
        Acon1[i] = 0;
        Vcaps[i] = 0;
        eps_bar[i] = 0;
        dRnumerator[i] = 0;
        dRdenominator[i] = 0;
        Atot_sum[i] = 0;
        ddelta_bar[i] = 0;
        sigmaxx[i] = 0;
        sigmayy[i] = 0;
        sigmazz[i] = 0;
        hsf[i] = 1;
        newAtom[i] = 1;
      }
    }
    for (let i = 0; i < n; i++) {
      // only updated outside of setup, unless a new atom
      if (!hu && !newAtom[i]) continue;
      const R = radius[i];
      const Rsq = R * R;
      const Vo = (4.0 / 3.0) * Math.PI * (Ro[i] * Ro[i] * Ro[i]);
      const Vgeoi = (4.0 / 3.0) * Math.PI * Rsq * R - Vcaps[i];
      Vgeo[i] = Math.min(Vgeoi, Vo);
      Velas[i] = Vo * (1.0 + eps_bar[i]);
      Atot[i] = 4.0 * Math.PI * Rsq + Atot_sum[i];
      psi[i] = (Atot[i] - Acon1[i]) / Atot[i];
      if (psib < psi[i]) {
        const w = psi[i] > 0.1 ? 1.0 / (1.0 + libmExp(-75.0 * (psi[i] - 0.2))) : 0.0;
        const dR = Math.max((dRnumerator[i] / (dRdenominator[i] - 4.0 * Math.PI * Rsq)) * w, 0.0);
        const Nwindow = 10.0;
        if (dR > 0.0) dRavg[i] += (dR - dRavg[i]) / Nwindow;
        if (radius[i] + dR < 1.5 * Ro[i] && dR > 0.0) radius[i] += dRavg[i];
      }
      Acon0[i] = Acon1[i];
    }
    // the temporary sums restart each step (not at setup)
    for (let i = 0; i < n; i++) {
      ddelta_bar[i] = 0;
      if (hu) {
        Vcaps[i] = 0;
        eps_bar[i] = 0;
        dRnumerator[i] = 0;
        dRdenominator[i] = 0;
        Acon1[i] = 0;
        Atot_sum[i] = 0;
        sigmaxx[i] = 0;
        sigmayy[i] = 0;
        sigmazz[i] = 0;
      }
    }
    const list = pc.half;
    // histories of pairs that left the neighbor list are gone (native rebuilds the list with the kept histories only)
    const seen = new Set<string>();
    if (list) {
      const owner = pc.nb.owner;
      const id = s.id;
      for (let i = 0; i < list.inum; i++) {
        const oi = owner[i];
        for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
          const j = list.neighbors[k] & NEIGHMASK;
          seen.add(MdrModel.pairKey(id[oi], id[owner[j]]));
        }
      }
      for (const key of this.pairHist.keys()) if (!seen.has(key)) this.pairHist.delete(key);
      if (hu) {
        this.penalty(s, pc);
        this.meanSurfDisp(s, pc);
      }
    }
    if (hu) this.wallUpdate(s);
  }

  /** calculate_contact_penalty: non-physical contacts through a closed triangle of touching particles. */
  private penalty(s: SimState, pc: PairCompute): void {
    const list = pc.half!;
    const owner = pc.nb.owner;
    const id = s.id;
    const radius = s.radius!;
    const x = pc.x;
    for (const h of this.pairHist.values()) h[PENALTY] = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const radi = radius[oi];
      const jstart = list.firstneigh[i], jnum = list.numneigh[i];
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[jstart + jj] & NEIGHMASK;
        const oj = owner[j];
        const radj = radius[oj];
        const delx_ij = x[3 * j] - xtmp, dely_ij = x[3 * j + 1] - ytmp, delz_ij = x[3 * j + 2] - ztmp;
        const rsq_ij = delx_ij * delx_ij + dely_ij * dely_ij + delz_ij * delz_ij;
        const r_ij = Math.sqrt(rsq_ij);
        const rinv_ij = 1.0 / r_ij;
        const deltan_ij = radi + radj - r_ij;
        if (deltan_ij < 0.0) continue;
        for (let kk = jj + 1; kk < jnum; kk++) {
          const k = list.neighbors[jstart + kk] & NEIGHMASK;
          const ok = owner[k];
          const radk = radius[ok];
          const delx_ik = x[3 * k] - xtmp, dely_ik = x[3 * k + 1] - ytmp, delz_ik = x[3 * k + 2] - ztmp;
          const rsq_ik = delx_ik * delx_ik + dely_ik * dely_ik + delz_ik * delz_ik;
          const r_ik = Math.sqrt(rsq_ik);
          const rinv_ik = 1.0 / r_ik;
          const deltan_ik = radi + radk - r_ik;
          if (deltan_ik < 0.0) continue;
          const delx_jk = x[3 * k] - x[3 * j], dely_jk = x[3 * k + 1] - x[3 * j + 1], delz_jk = x[3 * k + 2] - x[3 * j + 2];
          const rsq_jk = delx_jk * delx_jk + dely_jk * dely_jk + delz_jk * delz_jk;
          const r_jk = Math.sqrt(rsq_jk);
          const rinv_jk = 1.0 / r_jk;
          const deltan_jk = radj + radk - r_jk;
          if (deltan_jk < 0.0) continue;
          const pij = this.entry(this.pairHist, MdrModel.pairKey(id[oi], id[oj]));
          const pik = this.entry(this.pairHist, MdrModel.pairKey(id[oi], id[ok]));
          // the longest side carries the penalty, the angle is taken at the vertex opposite to it
          const r_max = Math.max(r_ij, Math.max(r_ik, r_jk));
          if (r_ij === r_max) {
            // the central particle is k
            const enx_ki = -delx_ik * rinv_ik, eny_ki = -dely_ik * rinv_ik, enz_ki = -delz_ik * rinv_ik;
            const enx_kj = -delx_jk * rinv_jk, eny_kj = -dely_jk * rinv_jk, enz_kj = -delz_jk * rinv_jk;
            const alpha = libmAcos(enx_ki * enx_kj + eny_ki * eny_kj + enz_ki * enz_kj);
            pij[PENALTY] += 1.0 / (1.0 + libmExp(-50.0 * (alpha / Math.PI - 0.5)));
          } else if (r_ik === r_max) {
            // the central particle is j
            const enx_ji = -delx_ij * rinv_ij, eny_ji = -dely_ij * rinv_ij, enz_ji = -delz_ij * rinv_ij;
            const enx_jk = delx_jk * rinv_jk, eny_jk = dely_jk * rinv_jk, enz_jk = delz_jk * rinv_jk;
            const alpha = libmAcos(enx_ji * enx_jk + eny_ji * eny_jk + enz_ji * enz_jk);
            pik[PENALTY] += 1.0 / (1.0 + libmExp(-50.0 * (alpha / Math.PI - 0.5)));
          } else {
            // the central particle is i
            const enx_ij = delx_ij * rinv_ij, eny_ij = dely_ij * rinv_ij, enz_ij = delz_ij * rinv_ij;
            const enx_ik = delx_ik * rinv_ik, eny_ik = dely_ik * rinv_ik, enz_ik = delz_ik * rinv_ik;
            const alpha = libmAcos(enx_ij * enx_ik + eny_ij * eny_ik + enz_ij * enz_ik);
            const pjk = this.entry(this.pairHist, MdrModel.pairKey(id[oj], id[ok]));
            pjk[PENALTY] += 1.0 / (1.0 + libmExp(-50.0 * (alpha / Math.PI - 0.5)));
          }
        }
      }
    }
  }

  /** mean_surf_disp: the change of the mean surface displacement of each atom, from the contacts of the last step. */
  private meanSurfDisp(s: SimState, pc: PairCompute): void {
    const list = pc.half!;
    const owner = pc.nb.owner;
    const id = s.id;
    const radius = s.radius!;
    const x = pc.x;
    const { Acon0, ddelta_bar } = this.statsOf();
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      for (let jj = 0; jj < list.numneigh[i]; jj++) {
        const j = list.neighbors[list.firstneigh[i] + jj] & NEIGHMASK;
        const oj = owner[j];
        const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const radsum = radius[oi] + radius[oj];
        const key = MdrModel.pairKey(id[oi], id[oj]);
        if (!(rsq < radsum * radsum)) {
          const h0 = this.pairHist.get(key);
          if (h0) h0.fill(0);
          continue;
        }
        const h = this.entry(this.pairHist, key);
        const delta = radsum - Math.sqrt(rsq);
        let deltamax = h[DELTA_MAX];
        const deltap0 = h[DELTAP_0], deltap1 = h[DELTAP_0 + 1];
        if (delta > deltamax) deltamax = delta;
        const delta0old = h[DELTA_0], delta1old = h[DELTA_0 + 1];
        // side 0 is the atom with the larger ID (i0)
        let i0: number, i1: number;
        if (id[oi] > id[oj]) {
          i0 = oi;
          i1 = oj;
        } else {
          i0 = oj;
          i1 = oi;
        }
        const R0 = radius[i0], R1 = radius[i1];
        const deltaOpt1 = (deltamax * (deltamax - 2.0 * R1)) / (2.0 * (deltamax - R0 - R1));
        const deltaOpt2 = (deltamax * (deltamax - 2.0 * R0)) / (2.0 * (deltamax - R0 - R1));
        let dgeo0: number, dgeo1: number;
        if (R0 < R1) {
          dgeo0 = Math.max(deltaOpt1, deltaOpt2);
          dgeo1 = Math.min(deltaOpt1, deltaOpt2);
        } else {
          dgeo0 = Math.min(deltaOpt1, deltaOpt2);
          dgeo1 = Math.max(deltaOpt1, deltaOpt2);
        }
        if (dgeo0 / R0 > MDR_OVERLAP_LIMIT) {
          dgeo0 = R0 * MDR_OVERLAP_LIMIT;
          dgeo1 = deltamax - dgeo0;
        } else if (dgeo1 / R1 > MDR_OVERLAP_LIMIT) {
          dgeo1 = R1 * MDR_OVERLAP_LIMIT;
          dgeo0 = deltamax - dgeo1;
        }
        const deltap = deltap0 + deltap1;
        const delta0 = dgeo0 + ((deltap0 - dgeo0) / (deltap - deltamax)) * (delta - deltamax);
        const delta1 = dgeo1 + ((deltap1 - dgeo1) / (deltap - deltamax)) * (delta - deltamax);
        const ddel0 = delta0 - delta0old;
        const ddel1 = delta1 - delta1old;
        if (Acon0[i0] !== 0.0) ddelta_bar[i0] += (h[AC_0] / Acon0[i0]) * ddel0;
        if (Acon0[i1] !== 0.0) ddelta_bar[i1] += (h[AC_0 + 1] / Acon0[i1]) * ddel1;
      }
    }
  }

  /** update_fix_gran_wall: the bulk response of each atom from its wall contacts of the last step. */
  private wallUpdate(s: SimState): void {
    const { Acon0, ddelta_bar } = this.statsOf();
    const radius = s.radius!;
    for (const contacts of this.walls.values()) {
      for (let i = 0; i < s.n; i++) {
        for (const c of contacts(i)) {
          const h = this.wallHist.get(c.key);
          if (!h || Acon0[i] === 0.0) continue;
          const delta = radius[i] - c.dist;
          const ddelta = delta - h[DELTA_0];
          ddelta_bar[i] += (1.0 * h[AC_0]) / Acon0[i] * ddelta;
        }
      }
    }
  }

  /**
   * One particle pair: the normal force of the contact (both sides, GranSubModNormalMDR::calculate_forces with
   * contact_type PAIR). Returns the force and the damping prefactor for the core's tangential and normal damping.
   */
  pairNormal(c: MdrPairIn): MdrNormalOut {
    const key = MdrModel.pairKey(c.ida, c.idb);
    const hist = this.entry(this.pairHist, key);
    const st = this.statsOf();
    const K = this.K;
    // the deltamax update is not guarded by the history update flag (native)
    if (c.delta >= hist[DELTA_MAX]) hist[DELTA_MAX] = c.delta;
    const deltamax = hist[DELTA_MAX];
    const deltap0 = hist[DELTAP_0], deltap1 = hist[DELTAP_0 + 1];
    const deltap = deltap0 + deltap1;
    const wij = Math.max(1.0 - hist[PENALTY], 0.0);
    const iLarger = c.ida > c.idb;
    let F0 = 0, F1 = 0, aDamp = 0, wAcSum = 0;
    for (let side = 0; side < 2; side++) {
      // side 0 is the larger-ID atom, side 1 the other (native: contactSide and the tags)
      const isI = (side === 0) === iLarger;
      const ai = isI ? c.oi : c.oj;
      const Ri = isI ? c.ri : c.rj;
      const Rj = isI ? c.rj : c.ri;
      const denom = 1.0 / (2.0 * (deltamax - Ri - Rj));
      const opt1 = deltamax * (deltamax - 2.0 * Rj) * denom;
      const opt2 = deltamax * (deltamax - 2.0 * Ri) * denom;
      let dgeo: number, dalt: number;
      if (Ri < Rj) {
        dgeo = Math.max(opt1, opt2);
        dalt = Math.min(opt1, opt2);
      } else {
        dgeo = Math.min(opt1, opt2);
        dalt = Math.max(opt1, opt2);
      }
      if (dgeo / Ri > MDR_OVERLAP_LIMIT) dgeo = Ri * MDR_OVERLAP_LIMIT;
      else if (dalt / Rj > MDR_OVERLAP_LIMIT) dgeo = deltamax - Rj * MDR_OVERLAP_LIMIT;
      const dp = side === 0 ? deltap0 : deltap1;
      const delta = dgeo + ((dp - dgeo) * (c.delta - deltamax)) / (deltap - deltamax);
      const r = mdrSide({
        hist, side: side as 0 | 1, st, ai, R: Ri, delta, nx: c.nx, ny: c.ny, nz: c.nz, c: K, hu: c.hu, wij,
      });
      if (side === 0) F0 = r.F0;
      else F1 = r.F0;
      aDamp += r.aDamp;
      wAcSum += r.wAc;
    }
    const fne = (wij * (F0 + F1)) * 0.5;
    const dampScale = Math.sqrt(c.meff * 2.0 * K.Eeff * aDamp);
    const etaN = mdrDampPrefactor({ hist, hu: c.hu, aDamp, dampScale, eta: this.pair.eta, dtype: this.pair.dtype });
    // native GranSubModNormalMDR::calculate_forces: svector = Ac_avg * 0.5, Ac_avg summed over both sides (pair/local p13)
    return { fne, a: Math.sqrt(c.delta * c.Rf), etaN, svector: 0.5 * wAcSum };
  }

  /** One wall contact (fix wall/gran/region with the mdr model): one side, the wall weight wfm. */
  wallNormal(c: MdrWallIn): MdrNormalOut {
    const hist = this.entry(this.wallHist, c.key);
    const st = this.statsOf();
    const K = mdrConsts(c.p);
    const wij = Math.max(1.0 - hist[PENALTY], 0.0);
    const r = mdrSide({ hist, side: 0, st, ai: c.ai, R: c.R, delta: c.delta, nx: c.nx, ny: c.ny, nz: c.nz, c: K, hu: c.hu, wij });
    // wall force magnifier of the native wall (fix_granular_mdr uses the same contact, wfm = exp(10.7 delta_o / R_o - 10) + 1)
    const wfm = libmExp((10.7 * hist[DELTAO_0]) / st.Ro[c.ai] - 10.0) + 1.0;
    const fne = wij * r.F0 * wfm;
    const aDamp = r.aDamp / 2.0;
    const Eeff2particle = 0.5 * K.Eeff;
    const dampScale = Math.sqrt(c.meff * 2.0 * Eeff2particle * aDamp);
    const etaN = mdrDampPrefactor({ hist, hu: c.hu, aDamp, dampScale, eta: c.p.eta, dtype: c.p.dtype });
    return { fne, a: Math.sqrt(c.delta * c.Rf), etaN, svector: 0.5 * r.wAc };
  }

  private statsOf(): MdrAtomArrays {
    if (!this.stats) throw new Error('MdrModel: beginStep must run before a contact is evaluated');
    return this.stats;
  }
}
