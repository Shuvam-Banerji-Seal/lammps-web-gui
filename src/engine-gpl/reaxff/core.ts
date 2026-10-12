/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/reaxff_bond_orders.cpp (BO, BOp, Add_dBond_to_Forces), src/REAXFF/
 * reaxff_forces.cpp (Init_Forces_noQEq, Compute_Forces, Compute_Total_Force), src/REAXFF/
 * reaxff_reset_tools.cpp (Reset, Reset_Atoms, Reset_Workspace), src/REAXFF/reaxff_init_md.cpp
 * (Init_Taper), src/REAXFF/pair_reaxff.cpp (write_reax_lists: the far neighbor list), at commit
 * b332f28. Authors credited in the C++ file headers: Hasan Metin Aktulga (Purdue University,
 * now Lawrence Berkeley National Laboratory) and Ray Shan (Sandia, per-atom and fix hooks);
 * heavily modified and adapted for LAMMPS by the LAMMPS developers. Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Data layout: the LAMMPS structs (far_neighbor_data, bond_data, bond_order_data) are
 * struct-of-arrays; a "list" is a per-atom region [start, end) of one shared array, as in
 * reaxff_list.cpp. The far neighbor list is built per evaluation with a binned search over the
 * owned and ghost positions, with the LAMMPS half newton-off rule (npair_bin_ghost.cpp: a pair
 * is stored by i when j > i, for owned and ghost i alike). Virial tallies are not made: the
 * pair style takes the global virial from x . f (virialFdotr).
 */
import { StyleError } from '../../engine/force/types';
import type { ControlParams, ReaxParam, SingleBody, TwoBody } from './ffield';

export const MIN_SINE = 1e-10;
export const DANGER_ZONE = 0.9;
/** constPI of reaxff_defs.h (not Math.PI: the C++ code uses this literal). */
export const constPI = 3.14159265;
export const DEG2RAD = (a: number): number => (a * constPI) / 180.0;
export const C_ele = 332.06371;
export const EV_to_KCALpMOL = 14.4;
export const KCALpMOL_to_EV = 23.02;
export const HB_THRESHOLD = 1e-2;
export const MIN_BONDS = 25;
export const MIN_3BODIES = 1000;

/** Per-evaluation energy totals (reaxff_types.h energy_data). */
export interface Energies {
  e_bond: number; e_ov: number; e_un: number; e_lp: number; e_ang: number; e_pen: number; e_coa: number;
  e_hb: number; e_tor: number; e_con: number; e_vdW: number; e_ele: number; e_pol: number;
}

export const newEnergies = (): Energies => ({
  e_bond: 0, e_ov: 0, e_un: 0, e_lp: 0, e_ang: 0, e_pen: 0, e_coa: 0, e_hb: 0, e_tor: 0, e_con: 0,
  e_vdW: 0, e_ele: 0, e_pol: 0,
});

/** The far neighbor list: per atom [start, end) into nbr / d / dvec (dvec = x_j - x_i). */
export interface FarList {
  start: Int32Array;
  end: Int32Array;
  nbr: Int32Array;
  d: Float64Array;
  dvec: Float64Array;
  count: number;
}

/** Bond list (bond_data + bond_order_data), one slot per bond direction. */
export interface BondList {
  start: Int32Array;
  end: Int32Array;
  nbr: Int32Array;
  sym: Int32Array;
  dbond: Int32Array;
  d: Float64Array;
  dvec: Float64Array;
  BO: Float64Array; BO_s: Float64Array; BO_pi: Float64Array; BO_pi2: Float64Array;
  Cdbo: Float64Array; Cdbopi: Float64Array; Cdbopi2: Float64Array;
  C1dbo: Float64Array; C2dbo: Float64Array; C3dbo: Float64Array;
  C1dbopi: Float64Array; C2dbopi: Float64Array; C3dbopi: Float64Array; C4dbopi: Float64Array;
  C1dbopi2: Float64Array; C2dbopi2: Float64Array; C3dbopi2: Float64Array; C4dbopi2: Float64Array;
  dBOp: Float64Array; dln_BOp_s: Float64Array; dln_BOp_pi: Float64Array; dln_BOp_pi2: Float64Array;
  count: number;
}

/** Per-atom workspace (reaxff_types.h storage) over owned and ghost atoms. */
export interface Workspace {
  total_bond_order: Float64Array;
  Deltap: Float64Array; Deltap_boc: Float64Array;
  Delta: Float64Array; Delta_lp: Float64Array; Delta_lp_temp: Float64Array; Delta_e: Float64Array;
  Delta_boc: Float64Array; Delta_val: Float64Array;
  dDelta_lp: Float64Array; dDelta_lp_temp: Float64Array;
  nlp: Float64Array; nlp_temp: Float64Array; Clp: Float64Array; vlpex: Float64Array;
  dDeltap_self: Float64Array;
  bond_mark: Int32Array;
  CdDelta: Float64Array;
  f: Float64Array;
}

/** The state of one evaluation: LAMMPS reax_system + the lists it fills. */
export interface RxSystem {
  ctrl: ControlParams;
  param: ReaxParam;
  /** owned and total (owned + ghost) atom counts. */
  n: number;
  N: number;
  /** element index per atom (-1 for NULL types). */
  type: Int32Array;
  /** atom ids (orig_id) of owned and ghost atoms. */
  orig: Int32Array;
  x: Float64Array<ArrayBufferLike>;
  q: Float64Array<ArrayBufferLike>;
  Hindex: Int32Array;
  numH: number;
  Tap: Float64Array<ArrayBufferLike>;
  far: FarList;
  bonds: BondList;
  ws: Workspace;
  en: Energies;
  /** Three-body list (thb_intrs): per bond index [start, end). */
  thb: ThreeBodyList;
  /** Hydrogen bond list: per H index [start, end). */
  hb: HBondList;
}

export interface ThreeBodyList {
  start: Int32Array;
  end: Int32Array;
  thb: Int32Array;
  pthb: Int32Array;
  theta: Float64Array;
  dcos_di: Float64Array;
  dcos_dj: Float64Array;
  dcos_dk: Float64Array;
  count: number;
}

export interface HBondList {
  start: Int32Array;
  end: Int32Array;
  nbr: Int32Array;
  scl: Int32Array;
  /** index of the far neighbor entry the hydrogen bond is taken from. */
  ptr: Int32Array;
  count: number;
}

export const newWorkspace = (N: number): Workspace => {
  const d = () => new Float64Array(N);
  return {
    total_bond_order: d(), Deltap: d(), Deltap_boc: d(), Delta: d(), Delta_lp: d(), Delta_lp_temp: d(),
    Delta_e: d(), Delta_boc: d(), Delta_val: d(), dDelta_lp: d(), dDelta_lp_temp: d(), nlp: d(),
    nlp_temp: d(), Clp: d(), vlpex: d(), dDeltap_self: new Float64Array(3 * N), bond_mark: new Int32Array(N),
    CdDelta: d(), f: new Float64Array(3 * N),
  };
};

const newBondList = (N: number, cap: number): BondList => {
  const c = Math.max(cap, 1);
  const d = () => new Float64Array(c);
  const v = () => new Float64Array(3 * c);
  return {
    start: new Int32Array(N), end: new Int32Array(N), nbr: new Int32Array(c), sym: new Int32Array(c),
    dbond: new Int32Array(c), d: d(), dvec: v(),
    BO: d(), BO_s: d(), BO_pi: d(), BO_pi2: d(), Cdbo: d(), Cdbopi: d(), Cdbopi2: d(),
    C1dbo: d(), C2dbo: d(), C3dbo: d(), C1dbopi: d(), C2dbopi: d(), C3dbopi: d(), C4dbopi: d(),
    C1dbopi2: d(), C2dbopi2: d(), C3dbopi2: d(), C4dbopi2: d(),
    dBOp: v(), dln_BOp_s: v(), dln_BOp_pi: v(), dln_BOp_pi2: v(), count: 0,
  };
};

/** Taper coefficients (Init_Taper): the seventh-order polynomial that goes from 1 at swa to 0 at swb. */
export const taperCoefficients = (nonbLow: number, nonbCut: number): Float64Array<ArrayBufferLike> => {
  if (nonbCut < 0.0) throw new StyleError('Negative upper Taper-radius cutoff');
  const swa = nonbLow, swb = nonbCut;
  const d1 = swb - swa;
  const d7 = Math.pow(d1, 7.0);
  const swa2 = swa * swa, swa3 = swa * swa * swa;
  const swb2 = swb * swb, swb3 = swb * swb * swb;
  const T = new Float64Array(8);
  T[7] = 20.0 / d7;
  T[6] = (-70.0 * (swa + swb)) / d7;
  T[5] = (84.0 * (swa2 + 3.0 * swa * swb + swb2)) / d7;
  T[4] = (-35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3)) / d7;
  T[3] = (140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3)) / d7;
  T[2] = (-210.0 * (swa3 * swb2 + swa2 * swb3)) / d7;
  T[1] = (140.0 * swa3 * swb3) / d7;
  T[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2 - 7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb) / d7;
  return T;
};

/**
 * Builds the far neighbor list (write_reax_lists over the half newton-off list): for owned i,
 * every j > i (owned or ghost) with d <= nonb_cut; for ghost i, every ghost j > i with d <= bond_cut.
 * Positions are x (owned then ghost); a binned search over all N atoms.
 */
export const buildFarList = (ctrl: ControlParams, n: number, N: number, x: Float64Array<ArrayBufferLike>, type: Int32Array): FarList => {
  const nonb = ctrl.nonbCut, bond = ctrl.bondCut;
  const cs = Math.max(nonb, bond) > 0 ? Math.max(nonb, bond) : 1;
  const start = new Int32Array(N), end = new Int32Array(N);
  if (N === 0) return { start, end, nbr: new Int32Array(0), d: new Float64Array(0), dvec: new Float64Array(0), count: 0 };
  let xmin = Infinity, ymin = Infinity, zmin = Infinity, xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
  for (let a = 0; a < N; a++) {
    const px = x[3 * a], py = x[3 * a + 1], pz = x[3 * a + 2];
    if (px < xmin) xmin = px;
    if (px > xmax) xmax = px;
    if (py < ymin) ymin = py;
    if (py > ymax) ymax = py;
    if (pz < zmin) zmin = pz;
    if (pz > zmax) zmax = pz;
  }
  const nx = Math.floor((xmax - xmin) / cs) + 1;
  const ny = Math.floor((ymax - ymin) / cs) + 1;
  const nz = Math.floor((zmax - zmin) / cs) + 1;
  const ncell = nx * ny * nz;
  const cstart = new Int32Array(ncell + 1);
  const cellOf = new Int32Array(N);
  const cidx = (a: number) => {
    const cx = Math.floor((x[3 * a] - xmin) / cs);
    const cy = Math.floor((x[3 * a + 1] - ymin) / cs);
    const cz = Math.floor((x[3 * a + 2] - zmin) / cs);
    return [cx, cy, cz];
  };
  for (let a = 0; a < N; a++) {
    const [cx, cy, cz] = cidx(a);
    const c = cx + nx * (cy + ny * cz);
    cellOf[a] = c;
    cstart[c + 1]++;
  }
  for (let c = 0; c < ncell; c++) cstart[c + 1] += cstart[c];
  const fill = cstart.slice(0, ncell);
  const cellAtoms = new Int32Array(N);
  for (let a = 0; a < N; a++) cellAtoms[fill[cellOf[a]]++] = a;
  const out: number[] = [];
  const outD: number[] = [];
  const outV: number[] = [];
  for (let i = 0; i < N; i++) {
    start[i] = out.length;
    if (type[i] < 0) { end[i] = out.length; continue; }
    const cutoffSq = (i < n ? nonb : bond) ** 2;
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const [cx, cy, cz] = cidx(i);
    for (let dz = -1; dz <= 1; dz++) {
      const z = cz + dz;
      if (z < 0 || z >= nz) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const y = cy + dy;
        if (y < 0 || y >= ny) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = cx + dx;
          if (xx < 0 || xx >= nx) continue;
          const c = xx + nx * (y + ny * z);
          for (let q = cstart[c]; q < cstart[c + 1]; q++) {
            const j = cellAtoms[q];
            if (j <= i) continue;
            const dvx = x[3 * j] - xi, dvy = x[3 * j + 1] - yi, dvz = x[3 * j + 2] - zi;
            const dsq = dvx * dvx + dvy * dvy + dvz * dvz;
            if (dsq <= cutoffSq) {
              out.push(j);
              outD.push(Math.sqrt(dsq));
              outV.push(dvx, dvy, dvz);
            }
          }
        }
      }
    }
    end[i] = out.length;
  }
  return {
    start, end, nbr: Int32Array.from(out), d: Float64Array.from(outD), dvec: Float64Array.from(outV),
    count: out.length,
  };
};

/** Reset_Atoms: hydrogen indexes of the owned atoms (only for hbonds). */
export const resetHydrogenIndex = (type: Int32Array, n: number, param: ReaxParam, hbondCut: number): { Hindex: Int32Array; numH: number } => {
  const Hindex = new Int32Array(n).fill(-1);
  let numH = 0;
  if (hbondCut > 0) {
    for (let i = 0; i < n; i++) {
      if (type[i] < 0) continue;
      if (param.sbp[type[i]].p_hbond === 1) Hindex[i] = numH++;
      else Hindex[i] = -1;
    }
  }
  return { Hindex, numH };
};

/**
 * Init_Forces_noQEq: bond list and hydrogen bond list from the far neighbor list (the bond
 * creation BOp runs here, for every far pair within bond_cut).
 */
export const initForces = (sys: RxSystem): void => {
  const { ctrl, param, far, ws, n, N, type, x } = sys;
  const bonds = sys.bonds;
  const sbp: SingleBody[] = param.sbp;
  void x;
  for (let i = 0; i < n; i++) ws.bond_mark[i] = 0;
  for (let i = n; i < N; i++) ws.bond_mark[i] = 1000;
  // hydrogen-bond region of an H: the entries come from the owned atoms' far pairs
  let btopI = 0;
  for (let i = 0; i < N; i++) {
    const typeI = type[i];
    if (typeI < 0) continue;
    const startI = far.start[i];
    const endI = far.end[i];
    btopI = bonds.end[i];
    const sbpI = sbp[typeI];
    const local = i < n;
    const cutoff = local ? Math.max(ctrl.hbondCut, ctrl.bondCut) : ctrl.bondCut;
    let ihb = -1;
    let ihbTop = -1;
    if (local && ctrl.hbondCut > 0) {
      ihb = sbpI.p_hbond;
      ihbTop = ihb === 1 ? sys.hb.end[sys.Hindex[i]] : -1;
    }
    for (let pj = startI; pj < endI; pj++) {
      const j = far.nbr[pj];
      const dj = far.d[pj];
      if (!(dj <= cutoff)) continue;
      const typeJ = type[j];
      if (typeJ < 0) continue;
      const sbpJ = sbp[typeJ];
      const twbp = param.tbp[typeI * param.numAtomTypes + typeJ];
      if (local) {
        // hydrogen bond lists
        if (ctrl.hbondCut > 0 && (ihb === 1 || ihb === 2) && dj <= ctrl.hbondCut) {
          const jhb = sbpJ.p_hbond;
          if (ihb === 1 && jhb === 2) {
            const h = sys.hb;
            h.nbr[ihbTop] = j;
            h.scl[ihbTop] = 1;
            h.ptr[ihbTop] = pj;
            ++ihbTop;
            h.count++;
          } else if (j < n && ihb === 2 && jhb === 1) {
            const hj = sys.Hindex[j];
            const jhbTop = sys.hb.end[hj];
            const h = sys.hb;
            h.nbr[jhbTop] = i;
            h.scl[jhbTop] = -1;
            h.ptr[jhbTop] = pj;
            sys.hb.end[hj] = jhbTop + 1;
            h.count++;
          }
        }
      }
      if (dj <= ctrl.bondCut && bondOrderPair(sys, i, btopI, pj, sbpI, sbpJ, twbp)) {
        btopI++;
        if (ws.bond_mark[j] > ws.bond_mark[i] + 1) ws.bond_mark[j] = ws.bond_mark[i] + 1;
        else if (ws.bond_mark[i] > ws.bond_mark[j] + 1) ws.bond_mark[i] = ws.bond_mark[j] + 1;
      }
    }
    bonds.end[i] = btopI;
    if (local && ihb === 1) sys.hb.end[sys.Hindex[i]] = ihbTop;
  }
};

/**
 * BOp (reaxff_bond_orders.cpp): creates the bond i-j and j-i when the uncorrected bond order is
 * at least bo_cut. The bond of i goes at btopI in i's region, the bond of j at the end of j's region.
 */
const bondOrderPair = (
  sys: RxSystem, i: number, btopI: number, pj: number, sbpI: SingleBody, sbpJ: SingleBody, twbp: TwoBody,
): boolean => {
  const { far, bonds: b, ws } = sys;
  const boCut = sys.ctrl.boCut;
  const j = far.nbr[pj];
  const d = far.d[pj];
  const rr2 = 1.0 / (d * d);
  let BO_s = 0, C12 = 0, BO_pi = 0, C34 = 0, BO_pi2 = 0, C56 = 0;
  if (sbpI.r_s > 0.0 && sbpJ.r_s > 0.0) {
    C12 = twbp.p_bo1 * Math.pow(d / twbp.r_s, twbp.p_bo2);
    BO_s = (1.0 + boCut) * Math.exp(C12);
  }
  if (sbpI.r_pi > 0.0 && sbpJ.r_pi > 0.0) {
    C34 = twbp.p_bo3 * Math.pow(d / twbp.r_p, twbp.p_bo4);
    BO_pi = Math.exp(C34);
  }
  if (sbpI.r_pi_pi > 0.0 && sbpJ.r_pi_pi > 0.0) {
    C56 = twbp.p_bo5 * Math.pow(d / twbp.r_pp, twbp.p_bo6);
    BO_pi2 = Math.exp(C56);
  }
  // Initially BO values are the uncorrected ones, page 1
  const BO = BO_s + BO_pi + BO_pi2;
  if (!(BO >= boCut)) return false;
  const ib = btopI;
  const btopJ = b.end[j];
  const jb = btopJ;
  b.nbr[ib] = j;
  b.nbr[jb] = i;
  b.d[ib] = d;
  b.d[jb] = d;
  for (let c = 0; c < 3; c++) {
    b.dvec[3 * ib + c] = far.dvec[3 * pj + c];
    b.dvec[3 * jb + c] = -far.dvec[3 * pj + c];
  }
  b.dbond[ib] = btopI;
  b.dbond[jb] = btopI;
  b.sym[ib] = btopJ;
  b.sym[jb] = btopI;
  b.end[j] = btopJ + 1;
  b.count++;
  // bond order page 2-3: derivative of the total bond order prime
  const ClnBOpS = twbp.p_bo2 * C12 * rr2;
  const ClnBOpPi = twbp.p_bo4 * C34 * rr2;
  const ClnBOpPi2 = twbp.p_bo6 * C56 * rr2;
  b.BO[ib] = b.BO[jb] = BO;
  b.BO_s[ib] = b.BO_s[jb] = BO_s;
  b.BO_pi[ib] = b.BO_pi[jb] = BO_pi;
  b.BO_pi2[ib] = b.BO_pi2[jb] = BO_pi2;
  // only dln_BOp_xx wrt. dr_i is stored here (dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j)
  for (let c = 0; c < 3; c++) {
    const dv = b.dvec[3 * ib + c];
    b.dln_BOp_s[3 * ib + c] = -b.BO_s[ib] * ClnBOpS * dv;
    b.dln_BOp_pi[3 * ib + c] = -b.BO_pi[ib] * ClnBOpPi * dv;
    b.dln_BOp_pi2[3 * ib + c] = -b.BO_pi2[ib] * ClnBOpPi2 * dv;
    b.dln_BOp_s[3 * jb + c] = -1.0 * b.dln_BOp_s[3 * ib + c];
    b.dln_BOp_pi[3 * jb + c] = -1.0 * b.dln_BOp_pi[3 * ib + c];
    b.dln_BOp_pi2[3 * jb + c] = -1.0 * b.dln_BOp_pi2[3 * ib + c];
  }
  const scale = -(b.BO_s[ib] * ClnBOpS + b.BO_pi[ib] * ClnBOpPi + b.BO_pi2[ib] * ClnBOpPi2);
  for (let c = 0; c < 3; c++) {
    b.dBOp[3 * ib + c] = scale * b.dvec[3 * ib + c];
    b.dBOp[3 * jb + c] = -1.0 * b.dBOp[3 * ib + c];
    ws.dDeltap_self[3 * i + c] += b.dBOp[3 * ib + c];
    ws.dDeltap_self[3 * j + c] += b.dBOp[3 * jb + c];
  }
  b.BO_s[ib] -= boCut;
  b.BO[ib] -= boCut;
  b.BO_s[jb] -= boCut;
  b.BO[jb] -= boCut;
  ws.total_bond_order[i] += b.BO[ib];
  ws.total_bond_order[j] += b.BO[jb];
  b.Cdbo[ib] = b.Cdbopi[ib] = b.Cdbopi2[ib] = 0.0;
  b.Cdbo[jb] = b.Cdbopi[jb] = b.Cdbopi2[jb] = 0.0;
  return true;
};

/** BO: corrected bond orders, their derivative coefficients and the atom valence terms. */
export const computeBondOrders = (sys: RxSystem): void => {
  const { ws, param, type, N, bonds: b } = sys;
  const sbp = param.sbp;
  const tbp = param.tbp;
  const nt = param.numAtomTypes;
  const gp = param.gpL;
  const p_boc1 = gp[0];
  const p_boc2 = gp[1];
  // Deltaprime, Deltaprime_boc values
  for (let i = 0; i < N; i++) {
    const ti = type[i];
    if (ti < 0) continue;
    const s = sbp[ti];
    ws.Deltap[i] = ws.total_bond_order[i] - s.valency;
    ws.Deltap_boc[i] = ws.total_bond_order[i] - s.valency_val;
    ws.total_bond_order[i] = 0;
  }
  // corrected bond order calculations
  for (let i = 0; i < N; i++) {
    const ti = type[i];
    if (ti < 0) continue;
    const sbpI = sbp[ti];
    const val_i = sbpI.valency;
    const Deltap_i = ws.Deltap[i];
    const Deltap_boc_i = ws.Deltap_boc[i];
    const startI = b.start[i];
    const endI = b.end[i];
    for (let pj = startI; pj < endI; pj++) {
      const j = b.nbr[pj];
      const tj = type[j];
      if (tj < 0) continue;
      if (i < j || ws.bond_mark[j] > 3) {
        const twbp = tbp[ti * nt + tj];
        if (twbp.ovc < 0.001 && twbp.v13cor < 0.001) {
          b.C1dbo[pj] = 1.0; b.C2dbo[pj] = 0.0; b.C3dbo[pj] = 0.0;
          b.C1dbopi[pj] = 1.0; b.C2dbopi[pj] = 0.0; b.C3dbopi[pj] = 0.0; b.C4dbopi[pj] = 0.0;
          b.C1dbopi2[pj] = 1.0; b.C2dbopi2[pj] = 0.0; b.C3dbopi2[pj] = 0.0; b.C4dbopi2[pj] = 0.0;
        } else {
          const val_j = sbp[tj].valency;
          const Deltap_j = ws.Deltap[j];
          const Deltap_boc_j = ws.Deltap_boc[j];
          let f1 = 1.0, Cf1_ij = 0.0, Cf1_ji = 0.0;
          // on page 1
          if (twbp.ovc >= 0.001) {
            // correction for overcoordination
            const exp_p1i = Math.exp(-p_boc1 * Deltap_i);
            const exp_p2i = Math.exp(-p_boc2 * Deltap_i);
            const exp_p1j = Math.exp(-p_boc1 * Deltap_j);
            const exp_p2j = Math.exp(-p_boc2 * Deltap_j);
            const f2 = exp_p1i + exp_p1j;
            const f3 = (-1.0 / p_boc2) * Math.log(0.5 * (exp_p2i + exp_p2j));
            f1 = 0.5 * ((val_i + f2) / (val_i + f2 + f3) + (val_j + f2) / (val_j + f2 + f3));
            const temp = f2 + f3;
            const u1_ij = val_i + temp;
            const u1_ji = val_j + temp;
            const Cf1A_ij = 0.5 * f3 * (1.0 / (u1_ij * u1_ij) + 1.0 / (u1_ji * u1_ji));
            const Cf1B_ij = -0.5 * ((u1_ij - f3) / (u1_ij * u1_ij) + (u1_ji - f3) / (u1_ji * u1_ji));
            Cf1_ij = 0.5 * (-p_boc1 * exp_p1i / u1_ij -
              ((val_i + f2) / (u1_ij * u1_ij)) * (-p_boc1 * exp_p1i + exp_p2i / (exp_p2i + exp_p2j)) +
              -p_boc1 * exp_p1i / u1_ji -
              ((val_j + f2) / (u1_ji * u1_ji)) * (-p_boc1 * exp_p1i + exp_p2i / (exp_p2i + exp_p2j)));
            Cf1_ji = -Cf1A_ij * p_boc1 * exp_p1j + Cf1B_ij * exp_p2j / (exp_p2i + exp_p2j);
          }
          let f4 = 1.0, f5 = 1.0, f4f5 = 1.0, Cf45_ij = 0.0, Cf45_ji = 0.0;
          if (twbp.v13cor >= 0.001) {
            // correction for 1-3 bond orders
            const BOij = b.BO[pj];
            const exp_f4 = Math.exp(-(twbp.p_boc4 * BOij * BOij - Deltap_boc_i) * twbp.p_boc3 + twbp.p_boc5);
            const exp_f5 = Math.exp(-(twbp.p_boc4 * BOij * BOij - Deltap_boc_j) * twbp.p_boc3 + twbp.p_boc5);
            f4 = 1.0 / (1.0 + exp_f4);
            f5 = 1.0 / (1.0 + exp_f5);
            f4f5 = f4 * f5;
            // bond order pages 8-9: derivative of f4 and f5
            Cf45_ij = -f4 * exp_f4;
            Cf45_ji = -f5 * exp_f5;
          }
          // bond order page 10: derivative of total bond order
          const BO = b.BO[pj];
          const A0_ij = f1 * f4f5;
          const A1_ij = -2 * twbp.p_boc3 * twbp.p_boc4 * BO * (Cf45_ij + Cf45_ji);
          const A2_ij = Cf1_ij / f1 + twbp.p_boc3 * Cf45_ij;
          const A2_ji = Cf1_ji / f1 + twbp.p_boc3 * Cf45_ji;
          const A3_ij = A2_ij + Cf1_ij / f1;
          const A3_ji = A2_ji + Cf1_ji / f1;
          // find corrected bond orders and their derivative coefficients
          b.BO[pj] = BO * A0_ij;
          b.BO_pi[pj] = b.BO_pi[pj] * A0_ij * f1;
          b.BO_pi2[pj] = b.BO_pi2[pj] * A0_ij * f1;
          b.BO_s[pj] = b.BO[pj] - (b.BO_pi[pj] + b.BO_pi2[pj]);
          b.C1dbo[pj] = A0_ij + b.BO[pj] * A1_ij;
          b.C2dbo[pj] = b.BO[pj] * A2_ij;
          b.C3dbo[pj] = b.BO[pj] * A2_ji;
          b.C1dbopi[pj] = f1 * f1 * f4 * f5;
          b.C2dbopi[pj] = b.BO_pi[pj] * A1_ij;
          b.C3dbopi[pj] = b.BO_pi[pj] * A3_ij;
          b.C4dbopi[pj] = b.BO_pi[pj] * A3_ji;
          b.C1dbopi2[pj] = f1 * f1 * f4 * f5;
          b.C2dbopi2[pj] = b.BO_pi2[pj] * A1_ij;
          b.C3dbopi2[pj] = b.BO_pi2[pj] * A3_ij;
          b.C4dbopi2[pj] = b.BO_pi2[pj] * A3_ji;
        }
        // neglect bonds that are < 1e-10
        if (b.BO[pj] < 1e-10) b.BO[pj] = 0.0;
        if (b.BO_s[pj] < 1e-10) b.BO_s[pj] = 0.0;
        if (b.BO_pi[pj] < 1e-10) b.BO_pi[pj] = 0.0;
        if (b.BO_pi2[pj] < 1e-10) b.BO_pi2[pj] = 0.0;
        ws.total_bond_order[i] += b.BO[pj];
      } else {
        // only bond orders are taken from the bond j-i; everything else is set above
        const sym = b.sym[pj];
        b.BO[pj] = b.BO[sym];
        b.BO_s[pj] = b.BO_s[sym];
        b.BO_pi[pj] = b.BO_pi[sym];
        b.BO_pi2[pj] = b.BO_pi2[sym];
        ws.total_bond_order[i] += b.BO[pj];
      }
    }
  }
  const p_lp1 = gp[15];
  for (let j = 0; j < N; j++) {
    const tj = type[j];
    if (tj < 0) continue;
    const s = sbp[tj];
    ws.Delta[j] = ws.total_bond_order[j] - s.valency;
    ws.Delta_e[j] = ws.total_bond_order[j] - s.valency_e;
    ws.Delta_boc[j] = ws.total_bond_order[j] - s.valency_boc;
    ws.Delta_val[j] = ws.total_bond_order[j] - s.valency_val;
    ws.vlpex[j] = ws.Delta_e[j] - 2.0 * Math.trunc(ws.Delta_e[j] / 2.0);
    const explp1 = Math.exp(-p_lp1 * (2.0 + ws.vlpex[j]) * (2.0 + ws.vlpex[j]));
    ws.nlp[j] = explp1 - Math.trunc(ws.Delta_e[j] / 2.0);
    ws.Delta_lp[j] = s.nlp_opt - ws.nlp[j];
    ws.Clp[j] = 2.0 * p_lp1 * explp1 * (2.0 + ws.vlpex[j]);
    ws.dDelta_lp[j] = ws.Clp[j];
    if (s.mass > 21.0) {
      ws.nlp_temp[j] = 0.5 * (s.valency_e - s.valency);
      ws.Delta_lp_temp[j] = s.nlp_opt - ws.nlp_temp[j];
      ws.dDelta_lp_temp[j] = 0.0;
    } else {
      ws.nlp_temp[j] = ws.nlp[j];
      ws.Delta_lp_temp[j] = s.nlp_opt - ws.nlp_temp[j];
      ws.dDelta_lp_temp[j] = ws.Clp[j];
    }
  }
};

/** Add_dBond_to_Forces: the force of one bond (pj of atom i, with j = its neighbor) from the derivatives. */
export const addDBondToForces = (sys: RxSystem, i: number, pj: number): void => {
  const b = sys.bonds;
  const ws = sys.ws;
  const j = b.nbr[pj];
  const ji = b.sym[pj];
  const f = ws.f;
  let c = b.Cdbo[pj] + b.Cdbo[ji];
  const C1dbo = b.C1dbo[pj] * c, C2dbo = b.C2dbo[pj] * c, C3dbo = b.C3dbo[pj] * c;
  c = b.Cdbopi[pj] + b.Cdbopi[ji];
  const C1dbopi = b.C1dbopi[pj] * c, C2dbopi = b.C2dbopi[pj] * c, C3dbopi = b.C3dbopi[pj] * c, C4dbopi = b.C4dbopi[pj] * c;
  c = b.Cdbopi2[pj] + b.Cdbopi2[ji];
  const C1dbopi2 = b.C1dbopi2[pj] * c, C2dbopi2 = b.C2dbopi2[pj] * c, C3dbopi2 = b.C3dbopi2[pj] * c, C4dbopi2 = b.C4dbopi2[pj] * c;
  c = ws.CdDelta[i] + ws.CdDelta[j];
  const C1dDelta = b.C1dbo[pj] * c, C2dDelta = b.C2dbo[pj] * c, C3dDelta = b.C3dbo[pj] * c;
  const temp = [0, 0, 0];
  // forces on i
  let cc = C1dbo + C1dDelta + C2dbopi + C2dbopi2;
  for (let k = 0; k < 3; k++) temp[k] = cc * b.dBOp[3 * pj + k];
  cc = C2dbo + C2dDelta + C3dbopi + C3dbopi2;
  for (let k = 0; k < 3; k++) temp[k] += cc * ws.dDeltap_self[3 * i + k];
  for (let k = 0; k < 3; k++) temp[k] += C1dbopi * b.dln_BOp_pi[3 * pj + k];
  for (let k = 0; k < 3; k++) temp[k] += C1dbopi2 * b.dln_BOp_pi2[3 * pj + k];
  for (let k = 0; k < 3; k++) f[3 * i + k] += temp[k];
  // forces on j
  cc = -(C1dbo + C1dDelta + C2dbopi + C2dbopi2);
  for (let k = 0; k < 3; k++) temp[k] = cc * b.dBOp[3 * pj + k];
  cc = C3dbo + C3dDelta + C4dbopi + C4dbopi2;
  for (let k = 0; k < 3; k++) temp[k] += cc * ws.dDeltap_self[3 * j + k];
  for (let k = 0; k < 3; k++) temp[k] += -C1dbopi * b.dln_BOp_pi[3 * pj + k];
  for (let k = 0; k < 3; k++) temp[k] += -C1dbopi2 * b.dln_BOp_pi2[3 * pj + k];
  for (let k = 0; k < 3; k++) f[3 * j + k] += temp[k];
  // forces on k: i neighbor
  for (let pk = b.start[i]; pk < b.end[i]; pk++) {
    const k = b.nbr[pk];
    const ck = -(C2dbo + C2dDelta + C3dbopi + C3dbopi2);
    for (let q = 0; q < 3; q++) f[3 * k + q] += ck * b.dBOp[3 * pk + q];
  }
  // forces on k: j neighbor
  for (let pk = b.start[j]; pk < b.end[j]; pk++) {
    const k = b.nbr[pk];
    const ck = -(C3dbo + C3dDelta + C4dbopi + C4dbopi2);
    for (let q = 0; q < 3; q++) f[3 * k + q] += ck * b.dBOp[3 * pk + q];
  }
};

/** Compute_Total_Force: every bond with i < nbr adds its derivative contribution. */
export const computeTotalForce = (sys: RxSystem): void => {
  const b = sys.bonds;
  for (let i = 0; i < sys.N; i++) {
    for (let pj = b.start[i]; pj < b.end[i]; pj++) {
      if (i < b.nbr[pj]) addDBondToForces(sys, i, pj);
    }
  }
};

/** Reset_Workspace: the per-step accumulators (bond order sums, derivatives, forces). */
export const resetWorkspace = (ws: Workspace): void => {
  ws.total_bond_order.fill(0);
  ws.dDeltap_self.fill(0);
  ws.CdDelta.fill(0);
  ws.f.fill(0);
};

/** Sizes of the per-atom regions (the list capacities of Estimate_Storages, without the reallocation). */
export const regionCounts = (far: FarList, N: number, n: number, type: Int32Array, Hindex: Int32Array, numH: number, param: ReaxParam, hbondCut: number): {
  bondCount: Int32Array; hbCount: Int32Array;
} => {
  const bondCount = new Int32Array(N);
  const hbCount = new Int32Array(Math.max(numH, 0));
  for (let i = 0; i < N; i++) {
    for (let p = far.start[i]; p < far.end[i]; p++) {
      const j = far.nbr[p];
      bondCount[i]++;
      bondCount[j]++;
      if (hbondCut > 0) {
        if (i < n && type[i] >= 0 && param.sbp[type[i]].p_hbond === 1) hbCount[Hindex[i]]++;
        if (j < n && type[j] >= 0 && param.sbp[type[j]].p_hbond === 1) hbCount[Hindex[j]]++;
      }
    }
  }
  return { bondCount, hbCount };
};

export const allocBondList = (N: number, far: FarList, bondCount: Int32Array): BondList => {
  // every far pair makes at most two bond slots (one in each atom's region)
  const cap = 2 * far.count;
  const b = newBondList(N, cap);
  let top = 0;
  for (let i = 0; i < N; i++) {
    b.start[i] = top;
    b.end[i] = top;
    top += bondCount[i];
  }
  return b;
};

export const ensureHBondList = (N: number, hbCount: Int32Array, numH: number): HBondList => {
  const start = new Int32Array(Math.max(numH, 1));
  const end = new Int32Array(Math.max(numH, 1));
  let top = 0;
  for (let h = 0; h < numH; h++) {
    start[h] = top;
    end[h] = top;
    top += hbCount[h];
  }
  const cap = Math.max(top, 1);
  return { start, end, nbr: new Int32Array(cap), scl: new Int32Array(cap), ptr: new Int32Array(cap), count: 0 };
};
