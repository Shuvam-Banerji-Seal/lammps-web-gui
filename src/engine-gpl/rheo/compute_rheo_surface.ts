/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/RHEO/compute_rheo_surface.cpp and compute_rheo_surface.h (ComputeRHEOSurface: free-surface detection and
 * the surface normal) and src/RHEO/compute_rheo_rho_sum.cpp (ComputeRHEORhoSum: density by kernel summation), at commit
 * b332f28. RHEO package authors: Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU). Copyright Sandia
 * Corporation (GNU General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_rheo.html): "surface/detection detects free-surfaces with an absence of particles ... values
 * = sdstyle limit limit/splash ... sdstyle = coordination or divergence ... limit = threshold for surface particles ...
 * limit/splash = threshold for splash particles (unitless)", and "rho/sum density evolution performed by a kernel
 * summation ... optional args = self/mass ... self/mass values = none, a particle uses its own mass in summation".
 * Both styles are supported: coordination tests the neighbor count against limit, divergence the position divergence
 * divr (compute_rheo_surface.cpp threshold_style DIVR); the splash test is the coordination one in both.
 *
 * The native sums run over the newton-off half list (REQ_DEFAULT; newton off, so no reverse communication of the ghost
 * contributions). The color gradient and the layer test follow that list in native's pair order (rheoHalfPairs), so the
 * per-atom sums add in native's order. Status bits are those of fix_rheo.h (STATUS_BULK 4, LAYER 8, SURFACE 16, SPLASH
 * 32); SURFACEMASK clears the four of them before the new values are set.
 */
import type { SimState } from '../../engine/types';
import type { Neighbor } from '../../engine/neighbor';
import { NEIGHMASK } from '../../engine/neighbor';
import { massOf, RHEO_RHO, RHEO_STATUS } from '../../engine/atoms';
import type { Geometry } from '../../engine/domain';
import { PHASECHECK, STATUS_SPLASH, STATUS_SURFACE, interfaceRho, rheoHalfPairs, rheoProp, type RheoModel } from './model';

export const STATUS_BULK = 1 << 2;
export const STATUS_LAYER = 1 << 3;
export { STATUS_SURFACE, STATUS_SPLASH } from './model';
/** fix_rheo.h: "SURFACEMASK = 0xFFFFFFC3" (clears bits 2 to 5). */
const SURFACEMASK = ~0x3c;
/** compute_rheo_surface.cpp: "static constexpr double EPSILON = 1e-10;" */
const EPSILON = 1e-10;

const ensure = (m: RheoModel, n: number): void => {
  const dim = m.dim;
  if (m.nsurface.length < n * dim) m.nsurface = new Float64Array(n * dim);
  if (m.gradC.length < n * dim) m.gradC = new Float64Array(n * dim);
  if (m.rsurface.length < n) m.rsurface = new Float64Array(n);
  if (m.divr.length < n) m.divr = new Float64Array(n);
};

/**
 * compute_rheo_rho_sum.cpp compute_peratom(): rho_i = w(0) m_i + sum_j w(r_ij) m_j over the neighbors within the cutoff
 * (m_i instead of m_j with self/mass). Ghosts take their owner's mass.
 */
export const computeRheoRhoSum = (s: SimState, nb: Neighbor, m: RheoModel): void => {
  const n = s.n;
  const full = nb.full;
  if (!full) throw new Error('rheo rho/sum needs a full neighbor list');
  const x = nb.xall, owner = nb.owner, nbr = full.neighbors;
  const rho = rheoProp(s, RHEO_RHO);
  // compute_rheo_rho_sum.cpp: calc_w_self() and calc_w() (the wendland/c4 kernel, else the quintic one)
  const wself = m.calcW(0);
  for (let i = 0; i < n; i++) {
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    rho[i] = wself * massOf(s, i);
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (!(rsq < m.cutsq)) continue;
      const w = m.calcW(Math.sqrt(rsq));
      const o = owner[j];
      rho[i] += m.rhosumSelfMass ? w * massOf(s, i) : w * massOf(s, o);
    }
  }
};

/**
 * compute_rheo_surface.cpp compute_peratom(). The color gradient gradC_i = sum_j dW_ij V_j of each owned atom gives the
 * normal nsurface = -gradC / |gradC| (group atoms), and the position divergence divr_i = -sum_j dW_ij . r_ij V_j. Each
 * group atom is a surface atom when its coordination is below zmin_surface (coordination style), or its divergence below
 * divr_surface (divergence style); then a splash atom when its coordination is below zmin_splash, and bulk otherwise; a
 * non-fluid atom is bulk. A bulk fluid atom with a surface neighbor becomes a layer atom. Normals of atoms that are not
 * surface atoms are zero.
 */
export const computeRheoSurface = (s: SimState, nb: Neighbor, m: RheoModel, groupBit: number, geom: Geometry): void => {
  const n = s.n, dim = m.dim;
  ensure(m, n);
  const full = nb.full;
  if (!full) throw new Error('rheo surface needs a full neighbor list');
  if (!nb.half) throw new Error('rheo surface needs a half neighbor list');
  const x = nb.xall, owner = nb.owner, nbr = full.neighbors;
  const rho = rheoProp(s, RHEO_RHO), status = rheoProp(s, RHEO_STATUS);
  const mask = s.mask;
  const gradC = m.gradC, nsurface = m.nsurface, divr = m.divr;
  gradC.fill(0, 0, n * dim);
  divr.fill(0, 0, n);
  const dW = [0, 0, 0];
  // the color gradient in native order: the half pairs of newton off (each owned pair once, from its lower native
  // index, every ghost image of the owned atom), each pair adding dWij Volj to i and, for an owned j, -dWij Voli to j
  const pairs = rheoHalfPairs(s, nb, geom);
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p];
    const j = pairs[p + 1] & NEIGHMASK;
    const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
    const rsq = dx * dx + dy * dy + dz * dz;
    if (!(rsq < m.cutsq)) continue;
    // calc_dw_quintic(dx, r, dWij, dWji): dW = del * wp / r, dWji = -dWij
    const r = Math.sqrt(rsq);
    const wp = m.dwScalar(r);
    const wprinv = wp / r;
    dW[0] = dx * wprinv;
    dW[1] = dy * wprinv;
    dW[2] = dz * wprinv;
    const o = owner[j];
    // compute_rheo_surface.cpp: "Add corrections for walls" (interface/reconstruct)
    let rhoi = rho[i], rhoj = rho[o];
    if (m.interfaceFlag) [rhoi, rhoj] = interfaceRho(m, s.type, !(status[i] & PHASECHECK), !(status[o] & PHASECHECK), rhoi, rhoj, i, o);
    const Volj = massOf(s, o) / rhoj;
    const del = [dx, dy, dz];
    for (let a = 0; a < dim; a++) {
      divr[i] -= dW[a] * del[a] * Volj;
      gradC[i * dim + a] += dW[a] * Volj;
    }
    if (j < n) {
      const Voli = massOf(s, i) / rhoi;
      for (let a = 0; a < dim; a++) {
        divr[j] += -dW[a] * del[a] * Voli;
        gradC[j * dim + a] += -dW[a] * Voli;
      }
    }
  }
  // normals of the group atoms
  for (let i = 0; i < n; i++) {
    if (!(mask[i] & groupBit)) continue;
    let mag2 = 0;
    for (let a = 0; a < dim; a++) mag2 += gradC[i * dim + a] * gradC[i * dim + a];
    const maggC = 1.0 / (Math.sqrt(mag2) + EPSILON);
    for (let a = 0; a < dim; a++) nsurface[i * dim + a] = -gradC[i * dim + a] * maggC;
  }
  // surface status of the group atoms (non-group atoms keep only the cleared bits)
  const coord = m.coord;
  for (let i = 0; i < n; i++) {
    status[i] = (status[i] & SURFACEMASK) >>> 0;
    if (!(mask[i] & groupBit)) continue;
    let test = m.surfaceDivr ? divr[i] < m.divrSurface : coord[i] < m.zminSurface;
    if (status[i] & PHASECHECK) test = false;
    if (test) {
      status[i] = (status[i] | (coord[i] < m.zminSplash ? STATUS_SPLASH : STATUS_SURFACE)) >>> 0;
    } else {
      status[i] = (status[i] | STATUS_BULK) >>> 0;
    }
  }
  // layer atoms (compute_rheo_surface.cpp, the first pair loop): a fluid bulk atom i takes the layer status when a
  // surface atom is its partner in the native half list, i.e. an owned partner with a higher native index or a ghost
  // image. The owned-owned pair is visited from its lower index only, and its other end gets no layer status
  // (native's branch for the partner needs a bulk fluid atom with the PHASECHECK bits set, which never holds), so
  // the lower-indexed atom alone is tested; measured with native LAMMPS (black box) on the icecubes case.
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p];
    const j = pairs[p + 1] & NEIGHMASK;
    if (status[i] & PHASECHECK) continue;
    const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
    if (!(dx * dx + dy * dy + dz * dz < m.cutsq)) continue;
    if ((status[i] & STATUS_BULK) && (status[owner[j]] & STATUS_SURFACE)) {
      status[i] = ((status[i] & SURFACEMASK) | STATUS_LAYER) >>> 0;
    }
  }
  // surface distance (compute_rheo_surface.cpp): bulk group atoms start at the cutoff, surface and splash atoms at zero,
  // and a fluid atom takes the smallest distance to a surface neighbor (the native pair loop, summed over the full list)
  const rsurface = m.rsurface;
  for (let i = 0; i < n; i++) {
    rsurface[i] = (mask[i] & groupBit) && !(status[i] & (STATUS_SURFACE | STATUS_SPLASH)) ? m.cut : 0.0;
  }
  for (let i = 0; i < n; i++) {
    if (status[i] & PHASECHECK) continue;
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (!(rsq < m.cutsq)) continue;
      if (status[owner[j]] & STATUS_SURFACE) rsurface[i] = Math.min(rsurface[i], Math.sqrt(rsq));
    }
  }
  // normals are zero for atoms that are not surface atoms
  for (let i = 0; i < n; i++) {
    if (!(mask[i] & groupBit)) continue;
    if (!(status[i] & STATUS_SURFACE)) for (let a = 0; a < dim; a++) nsurface[i * dim + a] = 0.0;
  }
};
