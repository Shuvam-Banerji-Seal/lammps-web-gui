/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/RHEO/compute_rheo_kernel.cpp and compute_rheo_kernel.h (ComputeRHEOKernel: the reproducing kernel
 * corrections), at commit b332f28. RHEO package authors: Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU).
 * Copyright Sandia Corporation (GNU General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_rheo.html): "RK0, RK1, and RK2, are zeroth, first, and second order reproducing. To
 * generate a reproducing kernel, a particle must have sufficient neighbors inside the kernel cutoff distance (a
 * coordination number) to accurately calculate moments. This threshold is set by zmin. If reproducing kernels are
 * requested but a particle has fewer neighbors, then it will revert to a non-reproducing quintic kernel until it gains
 * more neighbors." The RK1 and RK2 corrections are ported here; RK0's (C0 = 1 / sum_j W_j V_j) scales only the kernel
 * values Wij and Wji that no caller reads, so it is not computed.
 *
 * The RK1 and RK2 corrections: the moment matrix M = sum_j H_j H_j^T W_j V_j over the neighbors (H = (1, x/cut, y/cut,
 * (z/cut)), and for RK2 also x^2/2, y^2/2, (z^2/2), xy, (xz, yz) over cut^2, and V_j = m_j / rho_j) is factored by a
 * Cholesky decomposition, inverted, and its inverse columns give the coefficients C of the corrected kernel
 * W_ij = (C_i . H_ij) W_ij^quintic and dW_ij = (C_i,grad . H_ij) W_ij^quintic (the second-derivative coefficients of
 * RK2 are not read by any caller, so only the value and gradient rows are kept). Native uses
 * LAPACK dpotrf and dpotri (upper triangle); the port does the same factorization and inversion in plain loops, so the
 * values agree to rounding.
 */
import type { SimState } from '../../engine/types';
import type { Neighbor } from '../../engine/neighbor';
import { NEIGHMASK } from '../../engine/neighbor';
import { massOf, RHEO_RHO, RHEO_STATUS } from '../../engine/atoms';
import { PHASECHECK, rheoProp, type RheoModel } from './model';

/** Grows the per-atom arrays of the corrected kernels to n owned atoms (C: 1 + dim rows of Mdim per atom). */
const ensureArrays = (m: RheoModel, n: number): void => {
  const rows = (m.dim + 1) * m.mdim;
  if (m.coord.length < n) m.coord = new Int32Array(n);
  if (m.corrOK.length < n) m.corrOK = new Uint8Array(n);
  if (m.C.length < n * rows) m.C = new Float64Array(n * rows);
};

/**
 * compute_rheo_kernel.cpp compute_coordination(): coord_i = the number of full-list neighbors of owned atom i within
 * the kernel cutoff. Ghost atoms use their owner's count (native forward-communicates it).
 */
export const computeRheoCoordination = (s: SimState, nb: Neighbor, m: RheoModel): void => {
  const n = s.n;
  ensureArrays(m, n);
  const full = nb.full;
  if (!full) throw new Error('rheo kernel needs a full neighbor list');
  const x = nb.xall, nbr = full.neighbors;
  for (let i = 0; i < n; i++) {
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    let c = 0;
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      if (dx * dx + dy * dy + dz * dz < m.cutsq) c++;
    }
    m.coord[i] = c;
  }
};

/**
 * compute_rheo_kernel.cpp compute_peratom() for RK1: the moment matrix of each owned atom with enough neighbors, its
 * Cholesky factorization (upper triangle, M = U^T U), the inverse (M^-1 = U^-1 U^-T) and the packed coefficients
 * C[k][a]: row 0 is column 0 of M^-1, row 1+b is -(column 1+b of M^-1) / cut. An atom whose factorization fails keeps
 * corrOK = 0 and uses the quintic kernel (native: "Revert to uncorrected SPH for this particle").
 */
export const computeRheoCorrections = (s: SimState, nb: Neighbor, m: RheoModel): void => {
  const n = s.n, dim = m.dim, N = m.mdim;
  ensureArrays(m, n);
  const full = nb.full;
  if (!full) throw new Error('rheo kernel needs a full neighbor list');
  const x = nb.xall, owner = nb.owner, nbr = full.neighbors;
  const rho = rheoProp(s, RHEO_RHO), status = rheoProp(s, RHEO_STATUS);
  const M = new Float64Array(N * N);
  const U = new Float64Array(N * N);
  const Ui = new Float64Array(N * N);
  const Minv = new Float64Array(N * N);
  for (let i = 0; i < n; i++) {
    m.corrOK[i] = 0;
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    M.fill(0);
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (!(rsq < m.cutsq)) continue;
      const r = Math.sqrt(rsq);
      const w = m.wQuintic(r);
      const oj = owner[j];
      // compute_rheo_kernel.cpp: with interface/reconstruct, a non-fluid neighbor takes correct_rho
      let rhoj = rho[oj];
      if (m.interfaceFlag && (status[oj] & PHASECHECK)) rhoj = m.correctRho(rhoj, s.type[oj]);
      const vj = massOf(s, oj) / rhoj;
      const H = m.basis(dx, dy, dz);
      for (let a = 0; a < N; a++) {
        for (let b = a; b < N; b++) M[a * N + b] += H[a] * H[b] * w * vj;
      }
    }
    // Skip if undercoordinated (the matrix is still formed, as native does)
    if (m.coord[i] < m.zmin) continue;
    // dpotrf (upper): U^T U = M, column by column
    U.fill(0);
    for (let a = 0; a < N; a++) for (let b = a; b < N; b++) U[a * N + b] = M[a * N + b];
    let ok = true;
    for (let j = 0; j < N; j++) {
      let d = U[j * N + j];
      for (let k = 0; k < j; k++) d -= U[k * N + j] * U[k * N + j];
      if (!(d > 0)) { ok = false; break; }
      const ajj = Math.sqrt(d);
      U[j * N + j] = ajj;
      for (let c = j + 1; c < N; c++) {
        let t = U[j * N + c];
        for (let k = 0; k < j; k++) t -= U[k * N + j] * U[k * N + c];
        U[j * N + c] = t / ajj;
      }
    }
    if (!ok) continue;
    // dtrtri (upper): Ui = U^-1, by back substitution on the columns
    Ui.fill(0);
    for (let j = 0; j < N; j++) {
      Ui[j * N + j] = 1.0 / U[j * N + j];
      for (let a = j - 1; a >= 0; a--) {
        let t = 0;
        for (let k = a + 1; k <= j; k++) t += U[a * N + k] * Ui[k * N + j];
        Ui[a * N + j] = -t / U[a * N + a];
      }
    }
    // dlauum: M^-1 = Ui Ui^T (symmetric; the upper triangle is formed and mirrored)
    for (let a = 0; a < N; a++) {
      for (let b = a; b < N; b++) {
        let t = 0;
        for (let k = b; k < N; k++) t += Ui[a * N + k] * Ui[b * N + k];
        Minv[a * N + b] = t;
        Minv[b * N + a] = t;
      }
    }
    // Pack the correction coefficients (compute_rheo_kernel.cpp: C[i][0][a] = M[a][0];
    // C[i][1 + b][a] = -M[a][b + 1] * cutinv)
    const base = i * (dim + 1) * N;
    for (let a = 0; a < N; a++) {
      m.C[base + a] = Minv[a * N + 0];
      for (let b = 0; b < dim; b++) m.C[base + (1 + b) * N + a] = -Minv[a * N + b + 1] * m.cutinv;
    }
    m.corrOK[i] = 1;
  }
};
