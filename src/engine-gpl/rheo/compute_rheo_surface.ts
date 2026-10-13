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
 * Only the coordination style is supported by the browser engine (divergence is refused by name in fix rheo).
 *
 * The native sums run over the half list and reverse-communicate the ghost contributions; the port sums each owned atom
 * over its full neighbor list (the ghost neighbors read their owner's mass, density and status), which gives the same
 * per-atom sums. Status bits are those of fix_rheo.h (STATUS_BULK 4, LAYER 8, SURFACE 16, SPLASH 32); SURFACEMASK
 * clears the four of them before the new values are set.
 */
import type { SimState } from '../../engine/types';
import type { Neighbor } from '../../engine/neighbor';
import { NEIGHMASK } from '../../engine/neighbor';
import { massOf, RHEO_RHO, RHEO_STATUS } from '../../engine/atoms';
import { PHASECHECK, STATUS_SPLASH, STATUS_SURFACE, rheoProp, type RheoModel } from './model';

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
  const wself = m.wQuintic(0);
  for (let i = 0; i < n; i++) {
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    rho[i] = wself * massOf(s, i);
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (!(rsq < m.cutsq)) continue;
      const w = m.wQuintic(Math.sqrt(rsq));
      const o = owner[j];
      rho[i] += m.rhosumSelfMass ? w * massOf(s, i) : w * massOf(s, o);
    }
  }
};

/**
 * compute_rheo_surface.cpp compute_peratom() for the coordination style. The color gradient gradC_i = sum_j dW_ij V_j of
 * each owned atom gives the normal nsurface = -gradC / |gradC| (group atoms). Each group atom is a surface atom when its
 * coordination is below zmin_surface, a splash atom below zmin_splash, and bulk otherwise; a non-fluid atom is bulk. A
 * bulk fluid atom with a surface neighbor becomes a layer atom. Normals of atoms that are not surface atoms are zero.
 */
export const computeRheoSurface = (s: SimState, nb: Neighbor, m: RheoModel, groupBit: number): void => {
  const n = s.n, dim = m.dim;
  ensure(m, n);
  const full = nb.full;
  if (!full) throw new Error('rheo surface needs a full neighbor list');
  const x = nb.xall, owner = nb.owner, nbr = full.neighbors;
  const rho = rheoProp(s, RHEO_RHO), status = rheoProp(s, RHEO_STATUS);
  const mask = s.mask;
  const gradC = m.gradC, nsurface = m.nsurface;
  gradC.fill(0, 0, n * dim);
  const dW = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (!(rsq < m.cutsq)) continue;
      const o = owner[j];
      const Volj = massOf(s, o) / rho[o];
      // calc_dw_quintic(dx, r, dWij): dW = del * wp / r
      const r = Math.sqrt(rsq);
      const wp = m.dwScalar(r);
      const wprinv = wp / r;
      dW[0] = dx * wprinv;
      dW[1] = dy * wprinv;
      dW[2] = dz * wprinv;
      for (let a = 0; a < dim; a++) gradC[i * dim + a] += dW[a] * Volj;
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
    let test = coord[i] < m.zminSurface;
    if (status[i] & PHASECHECK) test = false;
    if (test) {
      status[i] = (status[i] | (coord[i] < m.zminSplash ? STATUS_SPLASH : STATUS_SURFACE)) >>> 0;
    } else {
      status[i] = (status[i] | STATUS_BULK) >>> 0;
    }
  }
  // layer atoms: a fluid bulk atom next to a surface atom
  for (let i = 0; i < n; i++) {
    if (status[i] & PHASECHECK) continue;
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      if (!(dx * dx + dy * dy + dz * dz < m.cutsq)) continue;
      if ((status[i] & STATUS_BULK) && (status[owner[j]] & STATUS_SURFACE)) {
        status[i] = ((status[i] & SURFACEMASK) | STATUS_LAYER) >>> 0;
      }
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
