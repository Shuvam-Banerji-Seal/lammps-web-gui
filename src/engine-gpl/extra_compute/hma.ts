/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_hma.cpp (ComputeHMA::compute_vector, the computeCv branch: the phiSum loop and
 * the Cv formula), src/EXTRA-PAIR/pair_lj_smooth_linear.cpp (PairLJSmoothLinear::single_hessian) and src/pair.cpp
 * (Pair::hessian_twobody) at commit b332f28. Copyright the LAMMPS development team (Sandia Corporation, GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The cv keyword of compute hma reaches this module through the lazy HOOKS entry 'hma' (lazyStyles.ts group gpl_hma);
 * the core compute (src/engine/compute/hma.ts) calls it with gplHook('hma').
 */
import type { HmaHook, HmaPhiSumInput, LjSmoothLinearHessianTables } from '../../engine/compute/hma';

/**
 * PairLJSmoothLinear::single_hessian at b332f28. Returns fforce (the force divided by r, times factor_lj) and writes the
 * six Hessian components (xx, xy, xz, yy, yz, zz) into d2u, as Pair::hessian_twobody does.
 */
export function ljSmoothLinearSingleHessian(
  t: LjSmoothLinearHessianTables, itype: number, jtype: number, rsq: number,
  delr: ArrayLike<number>, factorLJ: number, d2u: Float64Array,
): number {
  const k = itype * t.nt + jtype;
  const r2inv = 1.0 / rsq;
  const r6inv = r2inv * r2inv * r2inv;
  const rinv = Math.sqrt(r2inv);
  let forcelj = r6inv * (t.lj1[k] * r6inv - t.lj2[k]);
  forcelj = rinv * forcelj - t.dljcut[k];
  const fforce = factorLJ * forcelj * rinv;
  const d2r = factorLJ * r6inv * (13.0 * t.lj1[k] * r6inv - 7.0 * t.lj2[k]) / rsq;
  const dfac = -(fforce + d2r) / rsq;
  // Pair::hessian_twobody(fforce, dfac, delr, d2u)
  let m = 0;
  for (let a = 0; a < 3; a++) {
    d2u[m] = fforce;
    for (let b = a; b < 3; b++) {
      if (b > a) d2u[m] = 0;
      d2u[m++] += delr[a] * delr[b] * dfac;
    }
  }
  return fforce;
}

/** The phiSum of compute hma cv (compute_hma.cpp, computeCv branch), for a half list (native newton_pair on). */
export function hmaPhiSum(p: HmaPhiSumInput): number {
  const d2u = new Float64Array(6);
  const delr = [0, 0, 0];
  let phiSum = 0;
  for (let i = 0; i < p.inum; i++) {
    const k0 = p.firstneigh[i], k1 = k0 + p.numneigh[i];
    const itype = p.typeall[i];
    for (let kk = k0; kk < k1; kk++) {
      const entry = p.neighbors[kk];
      const factorLJ = p.specialLJ[entry >>> p.sbBits];
      const j = entry & p.neighMask;
      for (let c = 0; c < 3; c++) delr[c] = p.xall[3 * i + c] - p.xall[3 * j + c];
      const rsq = delr[0] * delr[0] + delr[1] * delr[1] + delr[2] * delr[2];
      const jtype = p.typeall[j];
      if (!(rsq < p.cutsq[itype * p.tables.nt + jtype])) continue;
      ljSmoothLinearSingleHessian(p.tables, itype, jtype, rsq, delr, factorLJ, d2u);
      const o = p.owner[j];
      let m = 0;
      for (let k = 0; k < 3; k++) {
        let a = 1.0;
        for (let l = k; l < 3; l++) {
          const idk = p.deltaR[3 * i + k], idl = p.deltaR[3 * i + l];
          const jdk = p.deltaR[3 * o + k], jdl = p.deltaR[3 * o + l];
          phiSum += a * (idk * jdl + jdk * idl) * d2u[m];
          phiSum -= a * (idk * idl * d2u[m] + jdk * jdl * d2u[m]);
          m++;
          if (k === l) a *= 2;
        }
      }
    }
  }
  return phiSum;
}

export const HOOKS: Record<string, HmaHook> = { hma: { phiSum: hmaPhiSum } };
