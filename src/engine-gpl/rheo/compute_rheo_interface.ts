/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/RHEO/compute_rheo_interface.cpp and compute_rheo_interface.h (ComputeRHEOInterface: the fluid-solid
 * interface reconstruction of fix rheo interface/reconstruct) at commit b332f28. RHEO package authors: Joel Clemmer
 * (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU). Copyright Sandia Corporation (GNU General Public License); this file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_rheo.html): "interface/reconstruct" reconstructs the density and velocity of solid
 * particles at a fluid interface. Over the half list (as native REQ_DEFAULT, each owned pair once and a ghost partner
 * read through its owner), with the quintic kernel W (scaled by the active kernel's pre_w, as native's calc_w_quintic):
 * chi_i = sum_{j same phase} W / sum_j W, and for a non-fluid atom with fluid neighbors the density of its pressure
 * extrapolated from theirs, rho_i = calc_rho(sum_j W (P_j - rho_j (a_i - a_j) . r_ij) / sum_j W), at least 0.1 rho0;
 * a non-fluid atom without fluid neighbors takes rho0. a is the non-pressure acceleration of the last step, kept per
 * atom in the property "fp_store" (native fix property/atom d2_fp_store 3): pair rheo stores the pressure force there,
 * and store_forces turns it into (f - f_pressure) / m (f / m for an atom of a fix setforce group).
 */
import type { SimState } from '../../engine/types';
import type { Neighbor } from '../../engine/neighbor';
import { NEIGHMASK } from '../../engine/neighbor';
import type { Geometry } from '../../engine/domain';
import type { Fix } from '../../engine/fix/fix';
import { massOf, RHEO_RHO, RHEO_STATUS } from '../../engine/atoms';
import { StyleError } from '../../engine/force/types';
import { PHASECHECK, rheoHalfPairs, rheoProp, type RheoModel } from './model';

/** The per-atom property of the stored forces (native atom->find_custom("fp_store"), 3 columns). */
export const FP_STORE = 'fp_store';

/** fp_store of the owned atoms (3 per atom), created on first use and kept with the atoms like other custom props. */
export const rheoFpStore = (s: SimState): Float64Array => {
  let c = s.custom.get(FP_STORE);
  if (!c) {
    c = { int: false, cols: 3, data: new Float64Array(3 * s.n) };
    s.custom.set(FP_STORE, c);
  }
  return c.data;
};

/** compute_peratom: chi of every owned atom and the reconstructed density of the non-fluid ones. */
export const computeRheoInterface = (s: SimState, nb: Neighbor, m: RheoModel, geom: Geometry): void => {
  const eos = m.pressureEos;
  if (!eos) throw new StyleError('Missing fix rheo/pressure');
  const n = s.n;
  if (m.chi.length < n) m.chi = new Float64Array(n);
  const chi = m.chi;
  const norm = new Float64Array(n), normwf = new Float64Array(n);
  const rho = rheoProp(s, RHEO_RHO), status = rheoProp(s, RHEO_STATUS);
  const fp = rheoFpStore(s);
  const x = nb.xall, owner = nb.owner;
  for (let i = 0; i < n; i++) {
    if (status[i] & PHASECHECK) rho[i] = 0.0;
    chi[i] = 0.0;
  }
  const dx = [0, 0, 0];
  const pairs = rheoHalfPairs(s, nb, geom);
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p];
    const j = pairs[p + 1] & NEIGHMASK;
    const oj = owner[j];
    dx[0] = x[3 * i] - x[3 * j];
    dx[1] = x[3 * i + 1] - x[3 * j + 1];
    dx[2] = x[3 * i + 2] - x[3 * j + 2];
    const rsq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
    if (!(rsq < m.cutsq)) continue;
    const fluidi = !(status[i] & PHASECHECK);
    const fluidj = !(status[oj] & PHASECHECK);
    const w = m.wQuintic(Math.sqrt(rsq));
    norm[i] += w;
    const statusMatch = (fluidi && fluidj) || (!fluidi && !fluidj);
    if (statusMatch) {
      chi[i] += w;
    } else if (!fluidi) {
      let dot = 0;
      for (let a = 0; a < 3; a++) dot += (-fp[3 * oj + a] + fp[3 * i + a]) * dx[a];
      rho[i] += w * (eos.calcPressure(rho[oj], oj) - rho[oj] * dot);
      normwf[i] += w;
    }
    // the owned partner (native "newton || j < nlocal" with the newton-off half list)
    if (j < n) {
      norm[j] += w;
      if (statusMatch) {
        chi[j] += w;
      } else if (!fluidj) {
        let dot = 0;
        for (let a = 0; a < 3; a++) dot += (-fp[3 * i + a] + fp[3 * j + a]) * dx[a];
        rho[j] += w * (eos.calcPressure(rho[i], i) + rho[i] * dot);
        normwf[j] += w;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    if (norm[i] !== 0.0) chi[i] /= norm[i];
    // recalculate rho for non-fluid particles: 1 + Pw of Adami, Hu and Adams 2012, capped at a tenth of equilibrium
    if (status[i] & PHASECHECK) {
      const t = s.type[i];
      if (normwf[i] !== 0.0) rho[i] = Math.max(0.1 * m.rho0[t], eos.calcRho(rho[i] / normwf[i], i));
      else rho[i] = m.rho0[t];
    }
  }
};

/**
 * store_forces: fp_store holds the pressure force of the last pair evaluation; it becomes the non-pressure force per
 * mass, (f - fp_store) / m, or f / m for an atom in the group of a fix setforce (its forces were overwritten).
 */
export const storeRheoForces = (s: SimState, fixes: readonly Fix[]): void => {
  const fp = rheoFpStore(s);
  const f = s.f;
  // native get_fix_by_style("setforce"): every fix whose style contains setforce
  const setforceBits = fixes.filter((fx) => fx.style.includes('setforce')).map((fx) => fx.groupBit);
  for (let i = 0; i < s.n; i++) {
    const minv = 1.0 / massOf(s, i);
    let skip = false;
    for (const bit of setforceBits) if (s.mask[i] & bit) skip = true;
    if (skip) for (let a = 0; a < 3; a++) fp[3 * i + a] = f[3 * i + a] * minv;
    else for (let a = 0; a < 3; a++) fp[3 * i + a] = (f[3 * i + a] - fp[3 * i + a]) * minv;
  }
};
