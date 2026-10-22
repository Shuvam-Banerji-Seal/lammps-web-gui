/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/compute_group_group.cpp (ComputeGroupGroup::init, kspace_contribution and kspace_correction; contributing
 * author Naveen Michaud-Agrawal (Johns Hopkins U), K-space terms added by Stan Moore (BYU)), src/KSPACE/ewald.cpp
 * (Ewald::compute_group_group, slabcorr_groups) and src/KSPACE/pppm.cpp (PPPM::compute_group_group, poisson_groups,
 * slabcorr_groups; contributing authors Roy Pollock (LLNL), Paul Crozier (SNL)) at commit b332f28. Copyright (2003)
 * Sandia Corporation, GNU General Public License. This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The long-range part of compute group/group (kspace yes). The structure factors (ewald) and mesh densities (pppm,
 * pppm/cg) of the two groups come from the engine's solvers (KSpaceEwald.groupStructureFactors,
 * KSpacePPPM.groupDensities); this file combines them as native does: twice the A-B term of the atoms of either group,
 * minus the term of the atoms in both groups (so that the energy counts each pair once, as the real-space part does),
 * minus their self energy and, with boundary yes, the k = 0 term of the net charges.
 */
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { KSpaceBase, KSpaceEwald } from '../../engine/force/kspace/ewald';
import { KSpacePPPM, KSpacePPPMCG } from '../../engine/force/kspace/pppm';
import type { GroupGroupKspaceHook, GroupGroupKspaceTerms } from '../../engine/compute/tally';
import { KSpacePPPMAd } from '../pppm/kspace_ad';

/** compute_group_group.cpp SMALL: a larger k = 0 term gives the net-charge warning. */
const SMALL = 0.00001;
/** math_const.h MY_PIS, sqrt(pi). */
const MY_PIS = 1.77245385090551602729;
/** math_const.h MY_PI2, pi/2. */
const MY_PI2 = 1.57079632679489661923;

/** force->qqrd2e: the Coulomb constant of the units over the dielectric constant (and the pair style's own scale). */
const qqrd2e = (sys: System): number => sys.state.units.qqr2e * (sys.ff.pair?.coulConstScale ?? 1) / sys.ff.dielectric;

/** The kspace style; init has checked that it computes group/group terms. */
const solver = (sys: System): KSpaceBase => sys.ff.kspace as KSpaceBase;

/**
 * Ewald::compute_group_group or PPPM::compute_group_group: e2group and f2group of groups A and B; with aa only the
 * atoms in both groups count (for both groups).
 */
const groupGroup = (sys: System, bitA: number, bitB: number, aa: boolean): { e: number; f: number[] } => {
  const k = solver(sys);
  const s = sys.state;
  const mask = s.mask;
  const both = (i: number): boolean => (mask[i] & bitA) !== 0 && (mask[i] & bitB) !== 0;
  const inA = aa ? both : (i: number): boolean => (mask[i] & bitA) !== 0;
  const inB = aa ? both : (i: number): boolean => (mask[i] & bitB) !== 0;
  const qscale = qqrd2e(sys) * k.scale;
  let e = 0;
  const f = [0, 0, 0];
  let volume: number;
  if (k instanceof KSpaceEwald) {
    const r = k.groupStructureFactors(s, sys.geom, inA, inB);
    volume = r.volume;
    // ug[k] = 4 pi / V exp(-k^2 / 4 g^2) / k^2 and eg[k] = 2 ug[k] k over the half-space vectors
    const preu = 4 * Math.PI / volume;
    const nk = r.kvecs.length / 4;
    for (let v = 0; v < nk; v++) {
      const ug = preu * r.kvecs[4 * v + 3];
      e += ug * (r.sA[2 * v] * r.sB[2 * v] + r.sA[2 * v + 1] * r.sB[2 * v + 1]);
      const partial = r.sA[2 * v + 1] * r.sB[2 * v] - r.sA[2 * v] * r.sB[2 * v + 1];
      for (let d = 0; d < 3; d++) f[d] += 2 * ug * r.kvecs[4 * v + d] * partial;
    }
    e *= qscale;
    for (let d = 0; d < 3; d++) f[d] *= qscale;
  } else if (k instanceof KSpacePPPM) {
    if (k instanceof KSpacePPPMAd) throw new StyleError('Cannot (yet) use kspace_modify diff ad with compute group/group');
    // As native pppm/cg does (src/KSPACE/pppm_cg.cpp PPPMCG::make_rho_groups at b332f28): its A-A pass leaves out the
    // atoms in both groups and keeps those in one of them (pppm keeps only the atoms in both)
    const cgAA = aa && k instanceof KSpacePPPMCG;
    const r = cgAA
      ? k.groupDensities(s, sys.geom, (i) => (mask[i] & bitA) !== 0 && !both(i), (i) => (mask[i] & bitB) !== 0 && !both(i))
      : k.groupDensities(s, sys.geom, inA, inB);
    volume = r.volume;
    // poisson_groups: the energy sum of G Re(A B*) and, for A and B, the force sum of k G Im-part; in the engine's mesh
    // normalization (charges per cell, not per volume) native's s2 and volume factors become 1 / (2 V) and 1 / V
    const ng = r.green.length;
    for (let i = 0; i < ng; i++) e += r.green[i] * (r.rhoA[2 * i] * r.rhoB[2 * i] + r.rhoA[2 * i + 1] * r.rhoB[2 * i + 1]);
    e *= qscale / (2 * volume);
    if (!aa) {
      for (let i = 0; i < ng; i++) {
        const partial = r.green[i] * (r.rhoA[2 * i] * r.rhoB[2 * i + 1] - r.rhoA[2 * i + 1] * r.rhoB[2 * i]);
        f[0] += r.kx[i] * partial;
        f[1] += r.ky[i] * partial;
        f[2] += r.kz[i] * partial;
      }
      for (let d = 0; d < 3; d++) f[d] *= qscale / volume;
    }
  } else if (k.name === 'ewald/electrode') {
    // ewald_electrode.cpp (EwaldElectrode::compute_group_group at b332f28)
    throw new StyleError('Cannot (yet) use ewald/electrode style with compute group/group');
  } else {
    throw new StyleError('Kspace style does not support compute group/group');
  }
  if (k.slab > 1) {
    // slabcorr_groups: the dipole terms of the slab correction between the two groups
    const zprdSlab = sys.geom.lz * k.slab;
    let qsumA = 0, qsumB = 0, dipA = 0, dipB = 0, dipR2A = 0, dipR2B = 0;
    for (let i = 0; i < s.n; i++) {
      const z = s.x[3 * i + 2], q = s.q[i];
      if (inA(i)) { qsumA += q; dipA += q * z; dipR2A += q * z * z; }
      if (inB(i)) { qsumB += q; dipB += q * z; dipR2B += q * z * z; }
    }
    const efact = qscale * 2 * Math.PI / volume;
    e += efact * (dipA * dipB - 0.5 * (qsumA * dipR2B + qsumB * dipR2A) - qsumA * qsumB * zprdSlab * zprdSlab / 12);
    f[2] += qscale * (-4 * Math.PI / volume) * (qsumA * dipB - qsumB * dipA);
  }
  return { e, f };
};

export const GROUP_GROUP_KSPACE: GroupGroupKspaceHook = {
  init(sys: System): void {
    const k = sys.ff.kspace;
    if (!k) throw new StyleError('No Kspace style defined for compute group/group');
    if (!k.groupGroupEnable) throw new StyleError('Kspace style does not support compute group/group');
    // native pppm/tip4p keeps PPPM's group_group_enable and its make_rho_groups, which spreads each oxygen charge from
    // the grid cell of its M site (pppm_tip4p.cpp particle_map); the engine does not reproduce that mix
    if (k.name === 'pppm/tip4p') {
      throw new StyleError('compute group/group kspace yes with kspace_style pppm/tip4p is not supported by the browser engine (native spreads the oxygen charges from the grid cells of the M sites)');
    }
  },

  /** kspace_correction: the self energy of the atoms in both groups and the k = 0 term of the net charges. */
  terms(sys: System, bitA: number, bitB: number): GroupGroupKspaceTerms {
    const k = solver(sys);
    const s = sys.state;
    const mask = s.mask, q = s.q;
    let qsqsumGroup = 0, qsumA = 0, qsumB = 0;
    for (let i = 0; i < s.n; i++) {
      if ((mask[i] & bitA) && (mask[i] & bitB)) qsqsumGroup += q[i] * q[i];
      if (mask[i] & bitA) qsumA += q[i];
      if (mask[i] & bitB) qsumB += q[i];
    }
    const g = k.gEwald;
    const qscale = qqrd2e(sys);
    const eSelf = qscale * g * qsqsumGroup / MY_PIS;
    let eCorrection = 2 * qsumA * qsumB;
    // the extra AA terms
    qsumA = qsumB = 0;
    for (let i = 0; i < s.n; i++) {
      if (!((mask[i] & bitA) && (mask[i] & bitB))) continue;
      qsumA += q[i];
      qsumB += q[i];
    }
    eCorrection -= qsumA * qsumB;
    eCorrection *= qscale * MY_PI2 / (g * g);
    if (Math.abs(eCorrection) > SMALL) {
      sys.warn('Both groups in compute group/group have a net charge; the Kspace boundary correction to energy will be non-zero');
    }
    return { eSelf, eCorrection };
  },

  contribution(sys: System, bitA: number, bitB: number, terms: GroupGroupKspaceTerms, boundary: boolean): { e: number; f: number[] } {
    const all = groupGroup(sys, bitA, bitB, false);
    // the A <-> A part of the atoms in both groups is subtracted, so the energy matches the real-space style
    const aa = groupGroup(sys, bitA, bitB, true);
    let e = 2 * all.e - aa.e - terms.eSelf;
    if (boundary) {
      const g = sys.geom;
      e -= terms.eCorrection / (g.lx * g.ly * g.lz * solver(sys).slab);
    }
    return { e, f: all.f };
  },
};
